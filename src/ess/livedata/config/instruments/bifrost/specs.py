# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Bifrost spectrometer spec registration

Bifrost has 5 arcs (fixed analyzers), each labeled by its energy transfer:
2.7, 3.2, 3.8, 4.4, and 5.0 meV. Each arc consists of all pixels at a given
arc index, spanning 3 tubes, 9 channels, and 100 pixels per channel.

See https://backend.orbit.dtu.dk/ws/portalfiles/portal/409340969/RSI25-AR-00125.pdf
for full instrument details.
"""

from enum import StrEnum
from typing import ClassVar, Literal

import pydantic
import scipp as sc

from ess.livedata.config import (
    Instrument,
    filter_authorized_streams,
    instrument_registry,
    name_streams,
)
from ess.livedata.config.workflow_spec import OutputView, WorkflowOutputsBase
from ess.livedata.handlers.detector_view_specs import SpectrumViewSpec
from ess.livedata.handlers.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)
from ess.livedata.parameter_models import EnergyEdges, QEdges

from .streams_parsed import PARSED_STREAMS

#: Disk choppers feeding the wavelength-LUT cascade, in beam order
#: (source -> sample): pulse-shaping, frame-overlap, bandwidth.
BIFROST_CHOPPERS = [
    'pulse_shaping_chopper_1',
    'pulse_shaping_chopper_2',
    'frame_overlap_chopper_1',
    'frame_overlap_chopper_2',
    'bandwidth_chopper_1',
    'bandwidth_chopper_2',
]


# Arc energies in meV
class ArcEnergy(StrEnum):
    """Arc energy transfer values."""

    ARC_2_7 = '2.7 meV'
    ARC_3_2 = '3.2 meV'
    ARC_3_8 = '3.8 meV'
    ARC_4_4 = '4.4 meV'
    ARC_5_0 = '5.0 meV'


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


def _make_3d_template() -> sc.DataArray:
    """Create an empty 3D template for 3D output data."""
    return sc.DataArray(
        sc.zeros(dims=['dim_0', 'dim_1', 'dim_2'], shape=[0, 0, 0], unit='counts')
    )


class DetectorRatemeterOutputs(WorkflowOutputsBase):
    """Outputs for detector ratemeter workflow."""

    output_views: ClassVar[tuple[OutputView, ...]] = (
        OutputView(
            name='detector_region_counts',
            title='Detector Region Counts',
            fields={
                'since_start': 'detector_region_counts_cumulative',
                'per_update': 'detector_region_counts',
            },
            description=(
                'Counts for the selected arc and pixel range. With "since run '
                'start" shows the count accumulated since the start of the run; '
                'with "latest update" or a window, shows recent counts. Display '
                'as a rate (counts/s) via the plot Rate option.'
            ),
            params=('region',),
        ),
    )

    detector_region_counts: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.scalar(0, unit='counts'),
            coords={'time': sc.scalar(0, unit='ns')},
        ),
        title='Detector Region Counts',
        description=(
            'Counts for the selected arc and pixel range, for the latest update '
            'interval only. Resets each update interval.'
        ),
    )
    detector_region_counts_cumulative: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0, unit='counts')),
        title='Detector Region Counts',
        description=(
            'Counts for the selected arc and pixel range, accumulated since the '
            'start of the run.'
        ),
    )


class DetectorRatemeterParams(pydantic.BaseModel):
    """Parameters for detector ratemeter workflow."""

    region: DetectorRatemeterRegionParams = pydantic.Field(
        title='Ratemeter region parameters',
        description=(
            'Detector region to count events in: an arc (selected by its energy '
            'transfer) and a pixel range along that arc.'
        ),
        default_factory=DetectorRatemeterRegionParams,
    )


QMAX_DEFAULT = 3.0
QBIN_DEFAULT = 100


class BifrostQMapParams(pydantic.BaseModel):
    q_edges: QEdges = pydantic.Field(
        default=QEdges(start=0.5, stop=QMAX_DEFAULT, num_bins=QBIN_DEFAULT),
        description=(
            "Bin edges for the magnitude of momentum transfer Q (in 1/Å), the "
            "scattering vector length probed by the measurement."
        ),
    )
    energy_edges: EnergyEdges = pydantic.Field(
        default=EnergyEdges(start=-1.0, stop=1.0, num_bins=QBIN_DEFAULT),
        description="Energy transfer bin edges.",
    )


class QAxisOption(StrEnum):
    Qx = 'Qx'
    Qy = 'Qy'
    Qz = 'Qz'


class QAxisSelection(pydantic.BaseModel):
    axis: QAxisOption = pydantic.Field(
        description=(
            "Momentum-transfer component (Qx, Qy, or Qz) that this axis of the "
            "Q-space map spans."
        )
    )


class QAxisParams(QEdges, QAxisSelection):
    pass


class BifrostElasticQMapParams(pydantic.BaseModel):
    axis1: QAxisParams = pydantic.Field(
        default=QAxisParams(
            axis=QAxisOption.Qx,
            start=-QMAX_DEFAULT,
            stop=QMAX_DEFAULT,
            num_bins=QBIN_DEFAULT,
        ),
        description=(
            "Horizontal axis of the 2D Q-space map: which momentum-transfer "
            "component (Qx, Qy, or Qz) it spans, and its bin edges."
        ),
    )
    axis2: QAxisParams = pydantic.Field(
        default=QAxisParams(
            axis=QAxisOption.Qz,
            start=-QMAX_DEFAULT,
            stop=QMAX_DEFAULT,
            num_bins=QBIN_DEFAULT,
        ),
        description=(
            "Vertical axis of the 2D Q-space map: which momentum-transfer "
            "component (Qx, Qy, or Qz) it spans, and its bin edges."
        ),
    )


class CustomQAxis(pydantic.BaseModel):
    qx: float = pydantic.Field(
        default=0, title='Qx', description="Custom x component of the cut axis."
    )
    qy: float = pydantic.Field(
        default=0, title='Qy', description="Custom y component of the cut axis."
    )
    qz: float = pydantic.Field(
        default=0, title='Qz', description="Custom z component of the cut axis."
    )


class BifrostCustomElasticQMapParams(pydantic.BaseModel):
    axis1: CustomQAxis = pydantic.Field(
        default=CustomQAxis(qx=1, qy=0, qz=0),
        description="Custom vector for the first cut axis.",
    )
    axis1_edges: QEdges = pydantic.Field(
        default=QEdges(start=-QMAX_DEFAULT, stop=QMAX_DEFAULT, num_bins=QBIN_DEFAULT),
        description="First cut axis edges.",
    )
    axis2: CustomQAxis = pydantic.Field(
        default=CustomQAxis(qx=0, qy=0, qz=1),
        description="Custom vector for the second cut axis.",
    )
    axis2_edges: QEdges = pydantic.Field(
        default=QEdges(start=-QMAX_DEFAULT, stop=QMAX_DEFAULT, num_bins=QBIN_DEFAULT),
        description="Second cut axis edges.",
    )


class QMapOutputs(WorkflowOutputsBase):
    """Outputs for Bifrost Q-map workflows."""

    cut_data: sc.DataArray = pydantic.Field(
        default_factory=_make_3d_template,
        title='Cut Data',
        description='Q-space or Q-E space intensity per arc (arc x axis1 x axis2).',
    )


# Monitor names matching group names in Nexus files. Order matches the Kafka
# ``cbm1``..``cbm5`` source names (see ``_make_cbm_monitors``).
monitors = [
    'psc_monitor',  # cbm1
    'overlap_monitor',  # cbm2
    'bandwidth_monitor',  # cbm3
    'normalization_monitor',  # cbm4
    'elastic_monitor',  # cbm5
]


streams = name_streams(filter_authorized_streams(PARSED_STREAMS))

# Create instrument
instrument = Instrument(
    name='bifrost',
    detector_names=['unified_detector'],
    monitors=monitors,
    choppers=BIFROST_CHOPPERS,
    streams=streams,
)

# Register instrument
instrument_registry.register(instrument)

# Register monitor workflow spec (TOA-only, no TOF lookup tables)
register_monitor_workflow_specs(instrument, monitors, params=TOAOnlyMonitorDataParams)


def _logical_view(obj: sc.Variable | sc.DataArray, source_name: str) -> sc.DataArray:
    """Reshape raw detector data into the 2D ``(arc/tube, channel/pixel)`` image.

    The 5 arcs and 3 tubes are stacked along the vertical ``arc/tube`` axis
    (15 rows) and the 9 channels and 100 pixels along the horizontal
    ``channel/pixel`` axis (900 columns). This matches the static triplet-bounds
    grid overlaid on the unified detector view.
    """
    da = sc.DataArray(obj) if isinstance(obj, sc.Variable) else obj
    return (
        da.to(dtype='float32')
        .flatten(dims=('arc', 'tube'), to='arc/tube')
        .flatten(dims=('channel', 'pixel'), to='channel/pixel')
    )


class BifrostSpectrumParams(pydantic.BaseModel):
    """Runtime parameters for the Bifrost spectrum-view transform."""

    pixels_per_tube: Literal[1, 2, 4, 5, 10, 20, 25, 50, 100] = pydantic.Field(
        default=10,
        title='Pixels per tube',
        description='Number of output pixels per tube.',
    )


def _bifrost_spectrum_transform(
    histogram: sc.DataArray, params: BifrostSpectrumParams
) -> sc.DataArray:
    """Reshape the cumulative histogram into ``(arc, detector_number, toa)``.

    The 2D image stacks ``(arc, tube)`` and ``(channel, pixel)``; unfold both,
    group adjacent pixels within a tube into ``pixels_per_tube`` output pixels
    (summing the intervening ``subpixel`` axis), and flatten the per-arc
    ``(tube, channel, pixel)`` axes into a single ``detector_number`` axis.
    """
    subpixel = 100 // params.pixels_per_tube
    grouped = (
        histogram.fold('arc/tube', sizes={'arc': 5, 'tube': 3})
        .fold('channel/pixel', sizes={'channel': 9, 'pixel': 100})
        .fold('pixel', sizes={'pixel': params.pixels_per_tube, 'subpixel': subpixel})
        .sum('subpixel')
    )
    return grouped.flatten(dims=('tube', 'channel', 'pixel'), to='detector_number')


# Register unified detector view with embedded spectrum output.
unified_detector_view_handle = instrument.add_logical_view(
    name='unified_detector_view',
    title='Unified detector view',
    description='All banks merged into a single detector view.',
    source_names=['unified_detector'],
    transform=_logical_view,
    output_ndim=2,
    spectrum_view=SpectrumViewSpec(
        transform=_bifrost_spectrum_transform,
        output_dims=['arc', 'detector_number'],
        extra_description=(
            'Per-arc, per-pixel spectrum. Adjacent pixels within a tube are '
            'grouped so each tube yields ``pixels_per_tube`` output pixels.'
        ),
        params_model=BifrostSpectrumParams,
    ),
)

detector_ratemeter_handle = instrument.register_spec(
    name='detector_ratemeter',
    version=1,
    title='Detector Ratemeter',
    description=(
        'Counts for a selected arc and pixel range, as a current (per-update) '
        'and a cumulative output. Both carry the time bounds needed to display '
        'a rate (counts/s) via the plot Rate option. The cumulative count '
        'resets only on a run transition.'
    ),
    source_names=['unified_detector'],
    params=DetectorRatemeterParams,
    outputs=DetectorRatemeterOutputs,
)

# Register Q-map workflow specs
qmap_handle = instrument.register_spec(
    name='qmap',
    version=1,
    title='Q map',
    description='Map of scattering intensity as function of Q and energy transfer.',
    source_names=['unified_detector'],
    params=BifrostQMapParams,
    outputs=QMapOutputs,
)

elastic_qmap_handle = instrument.register_spec(
    name='elastic_qmap',
    version=1,
    title='Elastic Q map',
    description='Elastic Q map with predefined axes.',
    source_names=['unified_detector'],
    params=BifrostElasticQMapParams,
    outputs=QMapOutputs,
)

elastic_qmap_custom_handle = instrument.register_spec(
    name='elastic_qmap_custom',
    version=1,
    title='Elastic Q map (custom)',
    description='Elastic Q map with custom axes.',
    source_names=['unified_detector'],
    params=BifrostCustomElasticQMapParams,
    outputs=QMapOutputs,
)
