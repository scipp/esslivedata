# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Bifrost with all banks merged into a single one.
"""

from collections.abc import Generator
from typing import NewType

import numpy as np
import pydantic
import scipp as sc
from scippnexus import NXdetector

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.env import StreamingEnv
from ess.livedata.config.workflows import (
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
from ess.reduce.nexus.types import (
    CalibratedBeamline,
    DetectorData,
    Filename,
    NeXusData,
    NeXusName,
    SampleRun,
)
from ess.reduce.streaming import EternalAccumulator
from ess.spectroscopy.indirect.time_of_flight import TofWorkflow

from ._bifrost_qmap import register_qmap_workflows
from ._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping


def _to_flat_detector_view(obj: sc.Variable | sc.DataArray) -> sc.DataArray:
    da = sc.DataArray(obj) if isinstance(obj, sc.Variable) else obj
    # Add fake coords until we can serialize string labels, or plot without coords.
    da = da.to(dtype='float32')
    # Padding between sectors to make gaps visible
    pad_pix = 10
    da = sc.concat([da, sc.full_like(da['pixel', :pad_pix], value=np.nan)], dim='pixel')
    # Padding between analyzers to make gaps visible
    pad_tube = 1
    da = sc.concat([da, sc.full_like(da['tube', :pad_tube], value=np.nan)], dim='tube')
    coords = {dim: sc.arange(dim, size) for dim, size in da.sizes.items()}
    da = (
        da.assign_coords(
            {
                'analyzer/tube': da.sizes['tube'] * coords['analyzer'] + coords['tube'],
                'sector/pixel': da.sizes['pixel'] * coords['sector'] + coords['pixel'],
            }
        )
        .flatten(dims=('analyzer', 'tube'), to='analyzer/tube')
        .flatten(dims=('sector', 'pixel'), to='sector/pixel')
    )
    # Remove last padding
    return da['sector/pixel', :-pad_pix]['analyzer/tube', :-pad_tube]


detector_number = sc.arange('detector_number', 1, 5 * 3 * 9 * 100 + 1, unit=None).fold(
    dim='detector_number', sizes={'analyzer': 5, 'tube': 3, 'sector': 9, 'pixel': 100}
)
detectors_config = {}
# Each NXdetetor is a He3 tube triplet with shape=(3, 100). Detector numbers in triplet
# are *not* consecutive:
# 1...900 with increasing angle (across all sectors)
# 901 is back to first sector and detector, second tube
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
    for sector in range(1, 10):
        for analyzer in range(1, 6):
            # Note: Actual start is at base + 100 * (sector - 1), but we start earlier
            # to get consistent counts across all banks, relating to comment above.
            base = ntube * 900 * (analyzer - 1)
            yield (
                f'{start}_channel_{sector}_{analyzer}_triplet',
                (base + 1, base + 2700),
            )
            start += 4
        start += 1


detectors_config['fakes'] = dict(_bifrost_generator())

# Would like to use a 2-D scipp.Variable, but GenericNeXusWorkflow does not accept
# detector names as scalar variables.
_detector_names = [
    f'{123 + 4 * (analyzer - 1) + (5 * 4 + 1) * (sector - 1)}'
    f'_channel_{sector}_{analyzer}_triplet'
    for analyzer in range(1, 6)
    for sector in range(1, 10)
]


def _combine_banks(*bank: sc.DataArray) -> sc.DataArray:
    return (
        sc.concat(bank, dim='')
        .fold('', sizes={'analyzer': 5, 'sector': 9})
        .rename_dims(dim_0='tube', dim_1='pixel')
        .transpose(('analyzer', 'tube', 'sector', 'pixel'))
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
    return SpectrumView(
        data.fold('pixel', sizes={'pixel': pixels_per_tube, 'subpixel': -1})
        .drop_coords(tuple(data.coords))
        .bins.concat('subpixel')
        .flatten(to='analyzer/tube/sector/pixel')
        .hist(event_time_offset=edges)
        .assign_coords(event_time_offset=edges.to(unit='ms'))
    )


def _analyzer_counts(data: DetectorData[SampleRun]) -> AnalyzerCounts:
    time = data.bins.coords['event_time_zero'].min()
    counts = data.sum(dim=('pixel', 'tube'))
    counts.variances = counts.values  # Poisson statistics
    return AnalyzerCounts(counts.assign_coords(time=time))


def _bank_counts(analyzer_counts: AnalyzerCounts) -> BankCounts:
    return BankCounts(analyzer_counts.sum(dim='analyzer'))


def _detector_counts(bank_counts: BankCounts) -> DetectorCounts:
    return DetectorCounts(bank_counts.sum())


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
_reduction_workflow.insert(_analyzer_counts)
_reduction_workflow.insert(_bank_counts)
_reduction_workflow.insert(_detector_counts)

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


register_qmap_workflows(instrument)
register_monitor_timeseries_workflows(instrument, source_names=monitor_names)


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
