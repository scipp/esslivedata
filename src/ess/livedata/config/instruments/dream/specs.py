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
from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.workflow_spec import AuxSourcesBase, WorkflowOutputsBase
from ess.livedata.handlers.detector_view_specs import (
    DetectorViewParams,
    register_detector_view_spec,
)

# Detector names for DREAM data reduction workflows
detector_names = [
    'mantle_detector',
    'endcap_backward_detector',
    'endcap_forward_detector',
    'high_resolution_detector',
    'sans_detector',
]

# Create instrument
instrument = Instrument(
    name='dream',
    detector_names=detector_names,
    monitors=['monitor1', 'monitor2'],
)

# Register instrument
instrument_registry.register(instrument)

# Detector names for XY projection (endcap and high-res detectors)
_xy_projection_detectors = [
    'endcap_backward_detector',
    'endcap_forward_detector',
    'high_resolution_detector',
]

# Detector names for cylinder projection (mantle detector)
_cylinder_projection_detectors = [
    'mantle_detector',
]

# Register detector view specs for cylinder projection
cylinder_handles = register_detector_view_spec(
    instrument=instrument,
    projection='cylinder_mantle_z',
    source_names=_cylinder_projection_detectors,
)

# Register detector view specs for XY projection
xy_handles = register_detector_view_spec(
    instrument=instrument,
    projection='xy_plane',
    source_names=_xy_projection_detectors,
)

# Extract individual handles for use in factories.py
cylinder_view_handle = cylinder_handles['view']
cylinder_roi_handle = cylinder_handles['roi']
xy_view_handle = xy_handles['view']
xy_roi_handle = xy_handles['roi']

# Register logical view specs (mantle front layer and wire view)
# These don't use the standard projection pattern, so we register them directly
mantle_front_layer_handle = instrument.register_spec(
    namespace='detector_data',
    name='mantle_front_layer',
    version=1,
    title='Mantle front layer',
    description='All voxels of the front layer of the mantle detector.',
    source_names=['mantle_detector'],
    params=DetectorViewParams,
)

mantle_wire_view_handle = instrument.register_spec(
    namespace='detector_data',
    name='mantle_wire_view',
    version=1,
    title='Mantle wire view',
    description='Sum over strips to show counts per wire in the mantle detector.',
    source_names=['mantle_detector'],
    params=DetectorViewParams,
)


# Pydantic models for DREAM workflows
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


class DreamAuxSources(AuxSourcesBase):
    """Auxiliary source names for DREAM powder workflows."""

    cave_monitor: Literal['monitor1'] = pydantic.Field(
        default='monitor1',
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
        title='I(d)',
        description='Focussed intensity as a function of d-spacing.',
    )
    focussed_data_dspacing_two_theta: sc.DataArray = pydantic.Field(
        title='I(d, 2θ)',
        description='Focussed intensity as a function of d-spacing and two-theta.',
    )


class PowderReductionWithVanadiumOutputs(PowderReductionOutputs):
    """Outputs for DREAM powder reduction workflow with vanadium normalization."""

    i_of_dspacing: sc.DataArray = pydantic.Field(
        title='Normalized I(d)',
        description=(
            'Normalized intensity as a function of d-spacing (vanadium-corrected).'
        ),
    )
    i_of_dspacing_two_theta: sc.DataArray = pydantic.Field(
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
