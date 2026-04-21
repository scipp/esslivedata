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

import pydantic
import scipp as sc

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.workflow_spec import AuxInput, AuxSources, WorkflowOutputsBase
from ess.livedata.handlers.detector_view_specs import SpectrumViewSpec
from ess.livedata.handlers.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)
from ess.livedata.parameter_models import EnergyEdges, QEdges


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

    detector_region_counts: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.scalar(0, unit='counts'),
            coords={'time': sc.scalar(0, unit='ns')},
        ),
        title='Detector Region Counts',
        description='Total counts for the selected arc and pixel range.',
    )


class DetectorRatemeterParams(pydantic.BaseModel):
    """Parameters for detector ratemeter workflow."""

    region: DetectorRatemeterRegionParams = pydantic.Field(
        title='Ratemeter region parameters',
        default_factory=DetectorRatemeterRegionParams,
    )


QMAX_DEFAULT = 3.0
QBIN_DEFAULT = 100


class BifrostQMapParams(pydantic.BaseModel):
    q_edges: QEdges = pydantic.Field(
        default=QEdges(start=0.5, stop=QMAX_DEFAULT, num_bins=QBIN_DEFAULT),
        description="Q bin edges.",
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
    axis: QAxisOption = pydantic.Field(description="Cut axis.")


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
        description="First cut axis.",
    )
    axis2: QAxisParams = pydantic.Field(
        default=QAxisParams(
            axis=QAxisOption.Qz,
            start=-QMAX_DEFAULT,
            stop=QMAX_DEFAULT,
            num_bins=QBIN_DEFAULT,
        ),
        description="Second cut axis.",
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


bifrost_aux_sources = AuxSources(
    {
        'detector_rotation': AuxInput(
            choices=('detector_rotation',),
            default='detector_rotation',
            title='Detector Rotation',
            description='Detector bank rotation angle.',
        ),
        'sample_rotation': AuxInput(
            choices=('sample_rotation',),
            default='sample_rotation',
            title='Sample Rotation',
            description='Sample rotation angle.',
        ),
    }
)


class QMapOutputs(WorkflowOutputsBase):
    """Outputs for Bifrost Q-map workflows."""

    cut_data: sc.DataArray = pydantic.Field(
        default_factory=_make_3d_template,
        title='Cut Data',
        description='Q-space or Q-E space intensity per arc (arc x axis1 x axis2).',
    )


# Monitor names matching group names in Nexus files
monitors = [
    '090_frame_1',
    '097_frame_2',
    '110_frame_3',
    '111_psd0',
    'bragg_peak_monitor',
]

# Combined f144 log stream configuration.
# Maps internal name -> {source: Kafka source name, units: unit string, topic: topic}
# Generated using: python -m ess.livedata.nexus_helpers <file> --generate --topic <t>
f144_log_streams: dict[str, dict[str, str]] = {
    # Motion streams (topic: bifrost_motion)
    'attenuator_1': {
        'source': 'BIFRO-AttChg:MC-Pne-01:ShtAuxBits07',
        'units': 'dimensionless',
        'topic': 'bifrost_motion',
    },
    'attenuator_2': {
        'source': 'BIFRO-AttChg:MC-Pne-02:ShtAuxBits07',
        'units': 'dimensionless',
        'topic': 'bifrost_motion',
    },
    'attenuator_3': {
        'source': 'BIFRO-AttChg:MC-Pne-03:ShtAuxBits07',
        'units': 'dimensionless',
        'topic': 'bifrost_motion',
    },
    'detector_rotation': {
        'source': 'BIFRO-DtCar:MC-RotZ-01:Mtr.RBV',
        'units': 'deg',
        'topic': 'bifrost_motion',
    },
    'get_lost_tube': {
        'source': 'BIFRO-InBm:MC-Pne-01:ShtAuxBits07',
        'units': 'dimensionless',
        'topic': 'bifrost_motion',
    },
    'goniometer_x': {
        'source': 'BIFRO-SpGon:MC-RotX-01:Mtr.RBV',
        'units': 'deg',
        'topic': 'bifrost_motion',
    },
    'goniometer_y': {
        'source': 'BIFRO-SpGon:MC-RotY-01:Mtr.RBV',
        'units': 'deg',
        'topic': 'bifrost_motion',
    },
    'sample_rotation': {
        'source': 'BIFRO-SpRot:MC-RotZ-01:Mtr.RBV',
        'units': 'deg',
        'topic': 'bifrost_motion',
    },
    'slit_bottom': {
        'source': 'BIFRO-SpSl1:MC-SlZm-01:PzMtr.RBV',
        'units': 'mm',
        'topic': 'bifrost_motion',
    },
    'slit_left': {
        'source': 'BIFRO-SpSl1:MC-SlYp-01:PzMtr.RBV',
        'units': 'mm',
        'topic': 'bifrost_motion',
    },
    'slit_right': {
        'source': 'BIFRO-SpSl1:MC-SlYm-01:PzMtr.RBV',
        'units': 'mm',
        'topic': 'bifrost_motion',
    },
    'slit_top': {
        'source': 'BIFRO-SpSl1:MC-SlZp-01:PzMtr.RBV',
        'units': 'mm',
        'topic': 'bifrost_motion',
    },
    'slit_position': {
        'source': 'BIFRO-SpSl1:MC-LinX-01:PzMtr-PosReadback',
        'units': 'mm',
        'topic': 'bifrost_motion',
    },
    # Sample environment streams (topic: bifrost_sample_env)
    'heater_1': {
        'source': 'YMIR-SEE:SE-LS336-004:HTR1',
        'units': 'W',
        'topic': 'bifrost_sample_env',
    },
    'heater_2': {
        'source': 'YMIR-SEE:SE-LS336-004:HTR2',
        'units': 'W',
        'topic': 'bifrost_sample_env',
    },
    'temperature_0': {
        'source': 'YMIR-SEE:SE-LS336-004:KRDG0',
        'units': 'K',
        'topic': 'bifrost_sample_env',
    },
    'temperature_1': {
        'source': 'YMIR-SEE:SE-LS336-004:KRDG1',
        'units': 'K',
        'topic': 'bifrost_sample_env',
    },
    'temperature_2': {
        'source': 'YMIR-SEE:SE-LS336-004:KRDG2',
        'units': 'K',
        'topic': 'bifrost_sample_env',
    },
    'temperature_3': {
        'source': 'YMIR-SEE:SE-LS336-004:KRDG3',
        'units': 'K',
        'topic': 'bifrost_sample_env',
    },
    'temperature_setpoint': {
        'source': 'YMIR-SEE:SE-LS336-004:SETP_S1',
        'units': 'K',
        'topic': 'bifrost_sample_env',
    },
    'sensor_0': {
        'source': 'YMIR-SEE:SE-LS336-004:SRDG0',
        'units': 'V',
        'topic': 'bifrost_sample_env',
    },
    'sensor_1': {
        'source': 'YMIR-SEE:SE-LS336-004:SRDG1',
        'units': 'V',
        'topic': 'bifrost_sample_env',
    },
    'sensor_2': {
        'source': 'YMIR-SEE:SE-LS336-004:SRDG2',
        'units': 'V',
        'topic': 'bifrost_sample_env',
    },
    'sensor_3': {
        'source': 'YMIR-SEE:SE-LS336-004:SRDG3',
        'units': 'V',
        'topic': 'bifrost_sample_env',
    },
}

# Derived from f144_log_streams for use by the Instrument
f144_attribute_registry = {
    name: {'units': info['units']} for name, info in f144_log_streams.items()
}

# Create instrument
instrument = Instrument(
    name='bifrost',
    detector_names=['unified_detector'],
    monitors=monitors,
    f144_attribute_registry=f144_attribute_registry,
)

# Register instrument
instrument_registry.register(instrument)

# Register monitor workflow spec (TOA-only, no TOF lookup tables)
monitor_handle = register_monitor_workflow_specs(
    instrument, monitors, params=TOAOnlyMonitorDataParams
)


def _logical_view(obj: sc.Variable | sc.DataArray, source_name: str) -> sc.DataArray:
    """Reshape raw detector data into ``(arc, detector_number_full)``.

    ``arc`` is preserved so the spectrum transform can operate per-arc; the
    remaining (tube, channel, pixel) axes are flattened into a single
    ``detector_number_full`` dim of size 2700.
    """
    da = sc.DataArray(obj) if isinstance(obj, sc.Variable) else obj
    return da.to(dtype='float32').flatten(
        dims=('tube', 'channel', 'pixel'), to='detector_number_full'
    )


def _bifrost_spectrum_transform(histogram: sc.DataArray, rebin: int) -> sc.DataArray:
    """Reshape the cumulative histogram into ``(arc, detector_number, toa)``.

    ``detector_number_full`` is folded back into an outer detector_number
    axis and an inner ``subpixel`` axis of size ``rebin``; summing over
    ``subpixel`` yields ``100 // rebin`` pixels per tube.
    """
    folded = histogram.fold(
        'detector_number_full',
        sizes={'detector_number': -1, 'subpixel': rebin},
    )
    return folded.sum('subpixel')


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
        output_dims=['arc', 'detector_number', 'time_of_arrival'],
        output_title='Spectrum view',
        output_description=(
            'Per-arc, per-pixel spectrum over time-of-arrival, with adjacent '
            'pixels within a tube grouped by the rebin factor.'
        ),
        default_rebin_factor=10,
    ),
)

detector_ratemeter_handle = instrument.register_spec(
    name='detector_ratemeter',
    version=1,
    title='Detector Ratemeter',
    description='Counts for a selected arc and pixel range.',
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
    aux_sources=bifrost_aux_sources,
    outputs=QMapOutputs,
)

elastic_qmap_handle = instrument.register_spec(
    name='elastic_qmap',
    version=1,
    title='Elastic Q map',
    description='Elastic Q map with predefined axes.',
    source_names=['unified_detector'],
    params=BifrostElasticQMapParams,
    aux_sources=bifrost_aux_sources,
    outputs=QMapOutputs,
)

elastic_qmap_custom_handle = instrument.register_spec(
    name='elastic_qmap_custom',
    version=1,
    title='Elastic Q map (custom)',
    description='Elastic Q map with custom axes.',
    source_names=['unified_detector'],
    params=BifrostCustomElasticQMapParams,
    aux_sources=bifrost_aux_sources,
    outputs=QMapOutputs,
)
