# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Bifrost spectrometer spec registration (lightweight).

This module registers workflow specs WITHOUT heavy dependencies.
Frontend loads this module to access workflow specs.
Backend services must also import .streams for stream mapping configuration.

Bifrost has 5 arcs (fixed analyzers), each labeled by its energy transfer:
2.7, 3.2, 3.8, 4.4, and 5.0 meV. Each arc consists of all pixels at a given
arc index, spanning 3 tubes, 9 channels, and 100 pixels per channel.

See https://backend.orbit.dtu.dk/ws/portalfiles/portal/409340969/RSI25-AR-00125.pdf
for full instrument details.
"""

from enum import Enum

import pydantic
import scipp as sc

from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.workflow_spec import AuxSourcesBase, WorkflowOutputsBase
from ess.livedata.handlers.detector_view_specs import DetectorViewParams
from ess.livedata.handlers.monitor_workflow_specs import register_monitor_workflow_specs
from ess.livedata.handlers.timeseries_workflow_specs import (
    register_timeseries_workflow_specs,
)
from ess.livedata.parameter_models import EnergyEdges, QEdges


# Arc energies in meV
class ArcEnergy(str, Enum):
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


class SpectrumViewOutputs(WorkflowOutputsBase):
    spectrum_view: sc.DataArray = pydantic.Field(
        title='Spectrum View',
        description='Spectrum view showing time-of-flight vs. detector position.',
    )


class DetectorRatemeterParams(pydantic.BaseModel):
    """Parameters for detector ratemeter workflow."""

    region: DetectorRatemeterRegionParams = pydantic.Field(
        title='Ratemeter region parameters',
        default_factory=DetectorRatemeterRegionParams,
    )


# Q-map parameters (from _bifrost_qmap.py)
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


class QAxisOption(str, Enum):
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


class BifrostAuxSources(AuxSourcesBase):
    """Auxiliary source names for Bifrost Q-map workflows."""

    detector_rotation: str = pydantic.Field(
        default='detector_rotation', description='Detector bank rotation angle.'
    )
    sample_rotation: str = pydantic.Field(
        default='sample_rotation', description='Sample rotation angle.'
    )


class QMapOutputs(WorkflowOutputsBase):
    """Outputs for Bifrost Q-map workflows."""

    cut_data: sc.DataArray = pydantic.Field(
        title='Cut Data',
        description='2D cut of intensity in Q-space or Q-E space.',
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

# Create instrument
instrument = Instrument(name='bifrost', f144_attribute_registry=f144_attribute_registry)

# Register lightweight workflows (no heavy dependencies)
monitor_workflow_handle = register_monitor_workflow_specs(
    instrument=instrument, source_names=monitor_names
)
timeseries_workflow_handle = register_timeseries_workflow_specs(
    instrument, source_names=list(f144_attribute_registry)
)

# Register instrument
instrument_registry.register(instrument)

# Register detector view spec (no factory yet)
unified_detector_view_handle = instrument.register_spec(
    namespace='detector_data',
    name='unified_detector_view',
    version=1,
    title='Unified detector view',
    description='All banks merged into a single detector view.',
    source_names=['unified_detector'],
    params=DetectorViewParams,
)

# Register spectroscopy workflow specs
spectrum_view_handle = instrument.register_spec(
    name='spectrum_view',
    version=1,
    title='Spectrum view',
    description='Spectrum view with configurable time bins and pixels per tube.',
    source_names=['unified_detector'],
    params=BifrostWorkflowParams,
    outputs=SpectrumViewOutputs,
)

detector_ratemeter_handle = instrument.register_spec(
    name='detector_ratemeter',
    version=1,
    title='Detector Ratemeter',
    description='Counts for a selected arc and pixel range.',
    source_names=['unified_detector'],
    params=DetectorRatemeterParams,
)

# Register Q-map workflow specs
qmap_handle = instrument.register_spec(
    name='qmap',
    version=1,
    title='Q map',
    description='Map of scattering intensity as function of Q and energy transfer.',
    source_names=['unified_detector'],
    params=BifrostQMapParams,
    aux_sources=BifrostAuxSources,
    outputs=QMapOutputs,
)

elastic_qmap_handle = instrument.register_spec(
    name='elastic_qmap',
    version=1,
    title='Elastic Q map',
    description='Elastic Q map with predefined axes.',
    source_names=['unified_detector'],
    params=BifrostElasticQMapParams,
    aux_sources=BifrostAuxSources,
    outputs=QMapOutputs,
)

elastic_qmap_custom_handle = instrument.register_spec(
    name='elastic_qmap_custom',
    version=1,
    title='Elastic Q map (custom)',
    description='Elastic Q map with custom axes.',
    source_names=['unified_detector'],
    params=BifrostCustomElasticQMapParams,
    aux_sources=BifrostAuxSources,
    outputs=QMapOutputs,
)
