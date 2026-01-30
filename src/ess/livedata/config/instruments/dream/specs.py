# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DREAM instrument spec registration.
"""

from enum import StrEnum
from typing import Literal

import pydantic
import scipp as sc

from ess.livedata import parameter_models
from ess.livedata.config import Instrument, SourceMetadata, instrument_registry
from ess.livedata.config.workflow_spec import AuxSourcesBase, WorkflowOutputsBase
from ess.livedata.handlers.detector_view_specs import (
    DetectorROIAuxSources,
    DetectorViewOutputs,
    DetectorViewParams,
)
from ess.livedata.handlers.monitor_workflow_specs import (
    MonitorDataParams,
    register_monitor_workflow_specs,
)

from .streams import detector_names, monitor_names
from .views import get_mantle_front_layer, get_strip_view, get_wire_view


# Pydantic models for DREAM instrument configuration
# (defined early for use in Instrument)
class InstrumentConfigurationEnum(StrEnum):
    """
    Chopper configuration options for DREAM.

    Mirrors ess.dream.InstrumentConfiguration enum for UI generation.
    """

    high_flux_BC215 = 'high_flux_BC215'
    high_flux_BC240 = 'high_flux_BC240'
    high_resolution = 'high_resolution'


class InstrumentConfiguration(pydantic.BaseModel):
    """
    Instrument configuration for DREAM.
    """

    value: InstrumentConfigurationEnum = pydantic.Field(
        default=InstrumentConfigurationEnum.high_flux_BC240,
        description='Chopper settings determining TOA to TOF conversion.',
    )

    @pydantic.model_validator(mode="after")
    def check_high_resolution_not_implemented(self):
        if self.value == InstrumentConfigurationEnum.high_resolution:
            raise pydantic.ValidationError.from_exception_data(
                "ValidationError",
                [
                    {
                        "type": "value_error",
                        "loc": ("value",),
                        "input": self.value,
                        "ctx": {
                            "error": "The 'high_resolution' setting is not available."
                        },
                    }
                ],
            )
        return self


class DreamMonitorDataParams(MonitorDataParams):
    """DREAM-specific monitor parameters with chopper settings."""

    instrument_configuration: InstrumentConfiguration = pydantic.Field(
        title="Instrument Configuration",
        description="Chopper configuration for TOF mode lookup table selection.",
        default_factory=InstrumentConfiguration,
    )


# Create instrument
instrument = Instrument(
    name='dream',
    detector_names=detector_names,
    monitors=monitor_names,
    source_metadata={
        'mantle_detector': SourceMetadata(
            title='Mantle',
            description='Main cylindrical detector covering the mantle region',
        ),
        'endcap_backward_detector': SourceMetadata(
            title='Backward Endcap',
            description='Endcap detector at backward scattering angles',
        ),
        'endcap_forward_detector': SourceMetadata(
            title='Forward Endcap',
            description='Endcap detector at forward scattering angles',
        ),
        'high_resolution_detector': SourceMetadata(
            title='High Resolution',
            description='High-resolution detector bank',
        ),
        'sans_detector': SourceMetadata(
            title='SANS',
            description='Small-angle neutron scattering detector',
        ),
        'monitor_bunker': SourceMetadata(title='Bunker Monitor'),
        'monitor_cave': SourceMetadata(title='Cave Monitor'),
    },
)

# Register instrument
instrument_registry.register(instrument)

# Register monitor workflow spec with DREAM-specific params for TOF mode support
monitor_handle = register_monitor_workflow_specs(
    instrument, monitor_names, params=DreamMonitorDataParams
)

# Register logical detector views
instrument.add_logical_view(
    name='mantle_front_layer',
    title='Mantle front layer',
    description='All voxels of the front layer of the mantle detector.',
    source_names=['mantle_detector'],
    transform=get_mantle_front_layer,
)
instrument.add_logical_view(
    name='wire_view',
    title='Wire view',
    description='Sum over strips to show counts per wire.',
    source_names=[
        'mantle_detector',
        'endcap_backward_detector',
        'endcap_forward_detector',
    ],
    transform=get_wire_view,
    roi_support=False,
    reduction_dim='strip',
)
instrument.add_logical_view(
    name='strip_view',
    title='Strip view',
    description='Sum over all dimensions except strip to show counts per strip.',
    source_names=detector_names,
    transform=get_strip_view,
    output_ndim=1,
    roi_support=False,
    reduction_dim='other',
)

# Mapping of detector names to their projection types
_projections: dict[str, str] = {
    'mantle_detector': 'cylinder_mantle_z',
    'endcap_backward_detector': 'xy_plane',
    'endcap_forward_detector': 'xy_plane',
    'high_resolution_detector': 'xy_plane',
    'sans_detector': 'xy_plane',
}


class DreamDetectorViewParams(DetectorViewParams):
    """DREAM-specific detector view parameters with chopper settings."""

    instrument_configuration: InstrumentConfiguration = pydantic.Field(
        title="Instrument Configuration",
        description="Chopper configuration for TOF mode lookup table selection.",
        default_factory=InstrumentConfiguration,
    )


# Register detector projection spec with DreamDetectorViewParams for TOF mode.
# Replaces both the legacy DetectorProjection and the Sciline detector view
# registrations.
projection_handle = instrument.register_spec(
    namespace='detector_data',
    name='detector_projection',
    version=1,
    title='Detector Projection',
    description=(
        'Projection of detector banks onto 2D planes. '
        'Uses the appropriate projection for each detector.'
    ),
    source_names=list(_projections.keys()),
    aux_sources=DetectorROIAuxSources,
    params=DreamDetectorViewParams,
    outputs=DetectorViewOutputs,
)


class DreamAuxSources(AuxSourcesBase):
    """Auxiliary source names for DREAM powder workflows."""

    cave_monitor: Literal['monitor_cave'] = pydantic.Field(
        default='monitor_cave',
        description='Cave monitor for normalization.',
    )


class PowderWorkflowParams(pydantic.BaseModel):
    """Parameters for DREAM powder reduction workflows."""

    dspacing_edges: parameter_models.DspacingEdges = pydantic.Field(
        title='d-spacing bins',
        description='Define the bin edges for binning in d-spacing.',
        default=parameter_models.DspacingEdges(
            start=0.4,
            stop=3.5,
            num_bins=500,
            unit=parameter_models.DspacingUnit.ANGSTROM,
        ),
    )
    two_theta_edges: parameter_models.TwoTheta = pydantic.Field(
        title='Two-theta bins',
        description='Define the bin edges for binning in 2-theta.',
        default=parameter_models.TwoTheta(
            start=0.4, stop=3.1415, num_bins=100, unit=parameter_models.AngleUnit.RADIAN
        ),
    )
    wavelength_range: parameter_models.WavelengthRange = pydantic.Field(
        title='Wavelength range',
        description='Range of wavelengths to include in the reduction.',
        default=parameter_models.WavelengthRange(
            start=1.1, stop=4.5, unit=parameter_models.WavelengthUnit.ANGSTROM
        ),
    )
    instrument_configuration: InstrumentConfiguration = pydantic.Field(
        title='Instrument configuration',
        description='Chopper settings determining TOA to TOF conversion.',
        default=InstrumentConfiguration(),
    )


class PowderReductionOutputs(WorkflowOutputsBase):
    """Outputs for DREAM powder reduction workflow."""

    focussed_data_dspacing: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['dspacing'], shape=[0], unit='dimensionless'),
            coords={'dspacing': sc.arange('dspacing', 0, unit='angstrom')},
        ),
        title='I(d)',
        description='Focussed intensity as a function of d-spacing.',
    )
    focussed_data_dspacing_two_theta: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(
                dims=['dspacing', 'two_theta'], shape=[0, 0], unit='dimensionless'
            ),
            coords={
                'dspacing': sc.arange('dspacing', 0, unit='angstrom'),
                'two_theta': sc.arange('two_theta', 0, unit='rad'),
            },
        ),
        title='I(d, 2θ)',
        description='Focussed intensity as a function of d-spacing and two-theta.',
    )


class PowderReductionWithVanadiumOutputs(PowderReductionOutputs):
    """Outputs for DREAM powder reduction workflow with vanadium normalization."""

    i_of_dspacing: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['dspacing'], shape=[0], unit='counts'),
            coords={'dspacing': sc.arange('dspacing', 0, unit='angstrom')},
        ),
        title='Normalized I(d)',
        description=(
            'Normalized intensity as a function of d-spacing (vanadium-corrected).'
        ),
    )
    i_of_dspacing_two_theta: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['dspacing', 'two_theta'], shape=[0, 0], unit='counts'),
            coords={
                'dspacing': sc.arange('dspacing', 0, unit='angstrom'),
                'two_theta': sc.arange('two_theta', 0, unit='rad'),
            },
        ),
        title='Normalized I(d, 2θ)',
        description=(
            'Normalized intensity as a function of d-spacing and two-theta '
            '(vanadium-corrected).'
        ),
    )


# Register powder reduction workflow specs
_powder_detector_names = [
    name for name in instrument.detector_names if 'sans' not in name
]

powder_reduction_handle = instrument.register_spec(
    name='powder_reduction',
    version=1,
    title='Powder reduction',
    description='Powder reduction without vanadium normalization.',
    source_names=_powder_detector_names,
    aux_sources=DreamAuxSources,
    outputs=PowderReductionOutputs,
    params=PowderWorkflowParams,
)

powder_reduction_with_vanadium_handle = instrument.register_spec(
    name='powder_reduction_with_vanadium',
    version=1,
    title='Powder reduction (with vanadium)',
    description='Powder reduction with vanadium normalization.',
    source_names=_powder_detector_names,
    aux_sources=DreamAuxSources,
    outputs=PowderReductionWithVanadiumOutputs,
    params=PowderWorkflowParams,
)
