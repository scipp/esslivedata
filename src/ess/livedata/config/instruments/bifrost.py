# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Bifrost spectrometer configuration.

Bifrost has 5 arcs (fixed analyzers), each labeled by its energy transfer:
2.7, 3.2, 3.8, 4.4, and 5.0 meV. Each arc consists of all pixels at a given
arc index, spanning 3 tubes, 9 channels, and 100 pixels per channel.

See https://backend.orbit.dtu.dk/ws/portalfiles/portal/409340969/RSI25-AR-00125.pdf
for full instrument details.
"""

from collections.abc import Generator
from enum import Enum
from typing import NewType

import numpy as np
import pydantic
import scipp as sc
from scippnexus import NXdetector

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.env import StreamingEnv
from ess.livedata.config.workflows import (
    CurrentRun,
    CustomMonitor,
    MonitorCountsInInterval,
    TimeseriesAccumulator,
    register_monitor_timeseries_workflows,
)
from ess.livedata.handlers.detector_data_handler import (
    DetectorLogicalView,
    LogicalViewConfig,
    get_nexus_geometry_filename,
)
from ess.livedata.handlers.monitor_data_handler import register_monitor_workflows
from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow
from ess.livedata.handlers.timeseries_handler import register_timeseries_workflows
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping
from ess.livedata.parameter_models import TOARange, WavelengthRange
from ess.reduce.nexus.types import (
    CalibratedBeamline,
    DetectorData,
    Filename,
    MonitorData,
    NeXusData,
    NeXusName,
    SampleRun,
)
from ess.reduce.streaming import EternalAccumulator
from ess.reduce.time_of_flight import GenericTofWorkflow
from ess.spectroscopy.indirect.time_of_flight import TofWorkflow

from ._bifrost_qmap import register_qmap_workflows
from ._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping


def _to_flat_detector_view(obj: sc.Variable | sc.DataArray) -> sc.DataArray:
    da = sc.DataArray(obj) if isinstance(obj, sc.Variable) else obj
    da = da.to(dtype='float32')
    # Padding between channels to make gaps visible
    pad_pix = 10
    da = sc.concat([da, sc.full_like(da['pixel', :pad_pix], value=np.nan)], dim='pixel')
    # Padding between arc to make gaps visible
    pad_tube = 1
    da = sc.concat([da, sc.full_like(da['tube', :pad_tube], value=np.nan)], dim='tube')
    da = da.flatten(dims=('arc', 'tube'), to='arc/tube').flatten(
        dims=('channel', 'pixel'), to='channel/pixel'
    )
    # Remove last padding
    return da['channel/pixel', :-pad_pix]['arc/tube', :-pad_tube]


detector_number = sc.arange('detector_number', 1, 5 * 3 * 9 * 100 + 1, unit=None).fold(
    dim='detector_number', sizes={'arc': 5, 'tube': 3, 'channel': 9, 'pixel': 100}
)
detectors_config = {}
# Each NXdetetor is a He3 tube triplet with shape=(3, 100). Detector numbers in triplet
# are *not* consecutive:
# 1...900 with increasing angle (across all channels)
# 901 is back to first channel and detector, second tube
_unified_detector_view = LogicalViewConfig(
    name='unified_detector_view',
    title='Unified detector view',
    description='All banks merged into a single detector view.',
    source_names=['unified_detector'],
    transform=_to_flat_detector_view,
)


def _bifrost_generator() -> Generator[tuple[str, tuple[int, int]]]:
    # BEWARE! There are gaps in the detector_number per bank, which would usually get
    # dropped when mapping to pixels. BUT we merge banks for Bifrost, before mapping to
    # pixels, so the generated fake events in the wrong bank will end up in the right
    # bank. As a consequence we do not lose any fake events, but the travel over Kafka
    # with the wrong source_name.
    start = 125
    ntube = 3
    for channel in range(1, 10):
        for arc in range(1, 6):
            # Note: Actual start is at base + 100 * (channel - 1), but we start earlier
            # to get consistent counts across all banks, relating to comment above.
            base = ntube * 900 * (arc - 1)
            yield (
                f'{start}_channel_{channel}_{arc}_triplet',
                (base + 1, base + 2700),
            )
            start += 4
        start += 1


detectors_config['fakes'] = dict(_bifrost_generator())

# Would like to use a 2-D scipp.Variable, but GenericNeXusWorkflow does not accept
# detector names as scalar variables.
_detector_names = [
    f'{123 + 4 * (arc - 1) + (5 * 4 + 1) * (channel - 1)}'
    f'_channel_{channel}_{arc}_triplet'
    for arc in range(1, 6)
    for channel in range(1, 10)
]


def _combine_banks(*bank: sc.DataArray) -> sc.DataArray:
    return (
        sc.concat(bank, dim='')
        .fold('', sizes={'arc': 5, 'channel': 9})
        .rename_dims(dim_0='tube', dim_1='pixel')
        # Order with consecutive detector_number
        .transpose(('arc', 'tube', 'channel', 'pixel'))
        .copy()
    )


SpectrumView = NewType('SpectrumView', sc.DataArray)
DetectorCounts = NewType('DetectorCounts', sc.DataArray)
BankCounts = NewType('BankCounts', sc.DataArray)
AnalyzerCounts = NewType('AnalyzerCounts', sc.DataArray)
SpectrumViewTimeBins = NewType('SpectrumViewTimeBins', int)
SpectrumViewPixelsPerTube = NewType('SpectrumViewPixelsPerTube', int)


def _make_spectrum_view(
    data: DetectorData[SampleRun],
    time_bins: SpectrumViewTimeBins,
    pixels_per_tube: SpectrumViewPixelsPerTube,
) -> SpectrumView:
    edges = sc.linspace(
        'event_time_offset', 0, 71_000_000, num=time_bins + 1, unit='ns'
    )
    # Combine, e.g., 10 pixels into 1, so we have tubes with 10 pixels each
    # Preserve arc dimension to allow per-arc visualization
    return SpectrumView(
        data.fold('pixel', sizes={'pixel': pixels_per_tube, 'subpixel': -1})
        .drop_coords(tuple(data.coords))
        .bins.concat('subpixel')
        .flatten(dims=('tube', 'channel', 'pixel'), to='position')
        .hist(event_time_offset=edges)
        .assign_coords(event_time_offset=edges.to(unit='ms'))
    )


_reduction_workflow = TofWorkflow(run_types=(SampleRun,), monitor_types=())
_reduction_workflow[Filename[SampleRun]] = get_nexus_geometry_filename('bifrost')
_reduction_workflow[CalibratedBeamline[SampleRun]] = (
    _reduction_workflow[CalibratedBeamline[SampleRun]]
    .map({NeXusName[NXdetector]: _detector_names})
    .reduce(func=_combine_banks)
)

_reduction_workflow[SpectrumViewTimeBins] = 500
_reduction_workflow[SpectrumViewPixelsPerTube] = 10
_reduction_workflow.insert(_make_spectrum_view)

_source_names = ('unified_detector',)


class SpectrumViewParams(pydantic.BaseModel):
    time_bins: int = pydantic.Field(
        title='Time bins',
        description='Number of time bins for the spectrum view.',
        default=500,
        ge=1,
        le=10000,
    )
    pixels_per_tube: int = pydantic.Field(
        title='Pixels per tube',
        description='Number of pixels per tube for the spectrum view.',
        default=10,
    )

    @pydantic.field_validator('pixels_per_tube')
    @classmethod
    def pixels_per_tube_must_be_divisor_of_100(cls, v):
        if 100 % v != 0:
            raise ValueError('pixels_per_tube must be a divisor of 100')
        return v


class BifrostWorkflowParams(pydantic.BaseModel):
    spectrum_view: SpectrumViewParams = pydantic.Field(
        title='Spectrum view parameters', default_factory=SpectrumViewParams
    )


# Arc energies in meV
class ArcEnergy(str, Enum):
    """Arc energy transfer values."""

    ARC_2_7 = '2.7'
    ARC_3_2 = '3.2'
    ARC_3_8 = '3.8'
    ARC_4_4 = '4.4'
    ARC_5_0 = '5.0'


_arc_energy_to_index = {
    ArcEnergy.ARC_2_7: 0,
    ArcEnergy.ARC_3_2: 1,
    ArcEnergy.ARC_3_8: 2,
    ArcEnergy.ARC_4_4: 3,
    ArcEnergy.ARC_5_0: 4,
}


class DetectorRatemeterRegionParams(pydantic.BaseModel):
    """Parameters for detector ratemeter region."""

    arc: ArcEnergy = pydantic.Field(
        title='Arc',
        description='Select arc by its energy transfer (meV).',
        default=ArcEnergy.ARC_5_0,
    )
    pixel_start: int = pydantic.Field(
        title='Pixel start',
        description='Starting pixel index along the arc (0-899).',
        default=0,
        ge=0,
        le=899,
    )
    pixel_stop: int = pydantic.Field(
        title='Pixel stop',
        description='Stopping pixel index along the arc (1-900).',
        default=900,
        ge=1,
        le=900,
    )

    @pydantic.model_validator(mode='after')
    def pixel_range_valid(self):
        if self.pixel_start >= self.pixel_stop:
            raise ValueError('pixel_start must be less than pixel_stop')
        return self


class DetectorRatemeterParams(pydantic.BaseModel):
    """Parameters for detector ratemeter workflow."""

    region: DetectorRatemeterRegionParams = pydantic.Field(
        title='Ratemeter region parameters',
        default_factory=DetectorRatemeterRegionParams,
    )


DetectorRegionCounts = NewType('DetectorRegionCounts', sc.DataArray)


def _detector_ratemeter(
    data: DetectorData[SampleRun], region: DetectorRatemeterRegionParams
) -> DetectorRegionCounts:
    """Calculate detector count rate for selected arc and pixel range."""
    arc_idx = _arc_energy_to_index[region.arc]
    # Select the arc
    arc_data = data['arc', arc_idx]
    # Flatten channel and pixel dimensions into 900 positions along the arc
    flat = arc_data.flatten(dims=('channel', 'pixel'), to='position')
    # Select pixel range
    selected = flat['position', region.pixel_start : region.pixel_stop]
    # Sum over all tubes, positions, and events
    counts = selected.sum()
    time = selected.bins.coords['event_time_zero'].min()
    counts.coords['time'] = time
    counts.variances = counts.values  # Poisson statistics
    return DetectorRegionCounts(counts)


# Bragg-peak monitor timeseries with wavelength or time-of-arrival selection


class IntervalMode(str, Enum):
    """Mode for selecting interval: time-of-arrival or wavelength."""

    TIME_OF_ARRIVAL = 'time_of_arrival'
    WAVELENGTH = 'wavelength'


class BraggPeakMonitorParams(pydantic.BaseModel):
    """Parameters for Bragg-peak monitor timeseries workflow."""

    interval_mode: IntervalMode = pydantic.Field(
        title='Interval Mode',
        description='Select interval by time-of-arrival or wavelength.',
        default=IntervalMode.TIME_OF_ARRIVAL,
    )
    toa_range: TOARange = pydantic.Field(
        title='Time of Arrival Range',
        description=(
            'Time of arrival range to include (used when mode is time_of_arrival).'
        ),
        default_factory=TOARange,
    )
    wavelength_range: WavelengthRange = pydantic.Field(
        title='Wavelength Range',
        description='Wavelength range to include (used when mode is wavelength).',
        default_factory=WavelengthRange,
    )


def _get_monitor_interval_by_toa(
    data: MonitorData[CurrentRun, CustomMonitor], range: TOARange
) -> MonitorCountsInInterval:
    """Get monitor counts in a time-of-arrival interval."""
    start, stop = range.range_ns
    if data.bins is not None:
        counts = data.bins['event_time_offset', start:stop].sum()
        counts.coords['time'] = data.coords['event_time_zero'][0]
    else:
        counts = data['time', start:stop].sum()
        counts.coords['time'] = data.coords['frame_time'][0]
    return MonitorCountsInInterval(counts)


def _get_monitor_interval_by_wavelength(
    data: MonitorData[CurrentRun, CustomMonitor], range: WavelengthRange
) -> MonitorCountsInInterval:
    """Get monitor counts in a wavelength interval."""
    start, stop = range.range_m
    if data.bins is not None:
        # Convert wavelength edges to meters for slicing
        counts = data.bins['wavelength', start:stop].sum()
        counts.coords['time'] = data.coords['event_time_zero'][0]
    else:
        counts = data['wavelength', start:stop].sum()
        counts.coords['time'] = data.coords['frame_time'][0]
    return MonitorCountsInInterval(counts)


# Monitor names matching group names in Nexus files
monitor_names = [
    '007_frame_0',
    '090_frame_1',
    '097_frame_2',
    '110_frame_3',
    '111_psd0',
    '113_psd1',
]

# Some example motions used for testing, probably not reflecting reality
f144_attribute_registry = {
    'detector_rotation': {'units': 'deg'},
    'sample_rotation': {'units': 'deg'},
    'sample_temperature': {'units': 'K'},
}

instrument = Instrument(name='bifrost', f144_attribute_registry=f144_attribute_registry)

register_monitor_workflows(instrument=instrument, source_names=monitor_names)
register_timeseries_workflows(instrument, source_names=list(f144_attribute_registry))
instrument.add_detector('unified_detector', detector_number=detector_number)
instrument_registry.register(instrument)
_logical_view = DetectorLogicalView(
    instrument=instrument, config=_unified_detector_view
)


@instrument.register_workflow(
    name='spectrum_view',
    version=1,
    title='Spectrum view',
    description='Spectrum view with configurable time bins and pixels per tube.',
    source_names=_source_names,
)
def _spectrum_view(params: BifrostWorkflowParams) -> StreamProcessorWorkflow:
    wf = _reduction_workflow.copy()
    view_params = params.spectrum_view
    wf[SpectrumViewTimeBins] = view_params.time_bins
    wf[SpectrumViewPixelsPerTube] = view_params.pixels_per_tube
    return StreamProcessorWorkflow(
        wf,
        dynamic_keys={'unified_detector': NeXusData[NXdetector, SampleRun]},
        target_keys=(SpectrumView, AnalyzerCounts, BankCounts, DetectorCounts),
        accumulators={
            SpectrumView: EternalAccumulator,
            AnalyzerCounts: TimeseriesAccumulator,
        },
    )


@instrument.register_workflow(
    name='detector_ratemeter',
    version=1,
    title='Detector Ratemeter',
    description='Count rate for a selected arc and pixel range.',
    source_names=_source_names,
)
def _detector_ratemeter_workflow(
    params: DetectorRatemeterParams,
) -> StreamProcessorWorkflow:
    wf = _reduction_workflow.copy()
    wf[DetectorRatemeterRegionParams] = params.region
    wf.insert(_detector_ratemeter)
    return StreamProcessorWorkflow(
        wf,
        dynamic_keys={'unified_detector': NeXusData[NXdetector, SampleRun]},
        target_keys=(DetectorRegionCounts,),
        accumulators={DetectorRegionCounts: TimeseriesAccumulator},
    )


register_qmap_workflows(instrument)
register_monitor_timeseries_workflows(instrument, source_names=monitor_names)


# Register Bragg-peak monitor workflow with wavelength support
@instrument.register_workflow(
    name='bragg_peak_monitor',
    version=1,
    title='Bragg-peak Monitor Rate',
    description=(
        'Monitor count rate with selectable time-of-arrival or wavelength interval.'
    ),
    source_names=monitor_names,
    aux_source_names=[],
)
def _bragg_peak_monitor_workflow(
    source_name: str, params: BraggPeakMonitorParams
) -> StreamProcessorWorkflow:
    """Bragg-peak monitor workflow supporting both TOA and wavelength selection."""
    wf = GenericTofWorkflow(run_types=[CurrentRun], monitor_types=[CustomMonitor])
    wf[Filename[CurrentRun]] = instrument.nexus_file
    wf[NeXusName[CustomMonitor]] = source_name

    # Insert the appropriate interval function based on mode
    if params.interval_mode == IntervalMode.TIME_OF_ARRIVAL:
        wf[TOARange] = params.toa_range
        wf.insert(_get_monitor_interval_by_toa)
    else:  # WAVELENGTH
        wf[WavelengthRange] = params.wavelength_range
        wf.insert(_get_monitor_interval_by_wavelength)

    return StreamProcessorWorkflow(
        base_workflow=wf,
        dynamic_keys={source_name: NeXusData[CustomMonitor, CurrentRun]},
        target_keys=(MonitorCountsInInterval,),
        accumulators={MonitorCountsInInterval: TimeseriesAccumulator},
    )


def _make_bifrost_detectors() -> StreamLUT:
    """
    Bifrost detector mapping.

    Input keys based on
    https://confluence.ess.eu/display/ECDC/Kafka+Topics+Overview+for+Instruments
    """
    # Source names have the format `arc=[0-4];triplet=[0-8]`.
    return {
        InputStreamKey(
            topic='bifrost_detector', source_name=f'arc={arc};triplet={triplet}'
        ): f'arc{arc}_triplet{triplet}'
        for arc in range(5)
        for triplet in range(9)
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'bifrost',
        detector_names=list(detectors_config['fakes']),
        monitor_names=monitor_names,
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(
            instrument='bifrost', monitor_names=monitor_names
        ),
        detectors=_make_bifrost_detectors(),
    ),
}
