# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
LOKI instrument spec registration (lightweight).

This module registers workflow specs WITHOUT heavy dependencies.
Frontend loads this module to access workflow specs.
Backend services must also import .streams for stream mapping configuration.
"""

from typing import Literal

import pydantic
import scipp as sc

from ess.livedata import parameter_models
from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.workflow_spec import AuxSourcesBase, WorkflowOutputsBase
from ess.livedata.handlers.detector_view_specs import register_detector_view_specs
from ess.livedata.handlers.monitor_data_handler import register_monitor_workflows


class SansWorkflowOptions(pydantic.BaseModel):
    use_transmission_run: bool = pydantic.Field(
        title='Use transmission run',
        description='Use transmission run instead of monitor readings of sample run',
        default=False,
    )


class LokiAuxSources(AuxSourcesBase):
    """Auxiliary source names for LOKI SANS workflows."""

    incident_monitor: Literal['monitor1'] = pydantic.Field(
        default='monitor1',
        description='Incident beam monitor for normalization.',
    )
    transmission_monitor: Literal['monitor2'] = pydantic.Field(
        default='monitor2',
        description='Transmission monitor for sample transmission calculation.',
    )


class IofQOutputs(WorkflowOutputsBase):
    """Outputs for the basic I(Q) workflow."""

    i_of_q: sc.DataArray = pydantic.Field(
        title='I(Q)',
        description='Scattered intensity as a function of momentum transfer Q.',
    )


class IofQWithTransmissionOutputs(IofQOutputs):
    """Outputs for I(Q) workflow with transmission from current run."""

    transmission_fraction: sc.DataArray = pydantic.Field(
        title='Transmission Fraction',
        description='Sample transmission fraction calculated from current run.',
    )


class SansWorkflowParams(pydantic.BaseModel):
    q_edges: parameter_models.QEdges = pydantic.Field(
        title='Q bins',
        description='Define the bin edges for binning in Q.',
        default=parameter_models.QEdges(
            start=0.01,
            stop=0.3,
            num_bins=20,
            unit=parameter_models.QUnit.INVERSE_ANGSTROM,
        ),
    )
    wavelength_edges: parameter_models.WavelengthEdges = pydantic.Field(
        title='Wavelength bins',
        description='Define the bin edges for binning in wavelength.',
        default=parameter_models.WavelengthEdges(
            start=1.0,
            stop=13.0,
            num_bins=100,
            unit=parameter_models.WavelengthUnit.ANGSTROM,
        ),
    )
    options: SansWorkflowOptions = pydantic.Field(
        title='Options',
        description='Options for the SANS workflow.',
        default=SansWorkflowOptions(),
    )


# Detector names for LOKI
detector_names = [f'loki_detector_{bank}' for bank in range(9)]

# Create instrument
instrument = Instrument(name='loki')

# Register monitor workflows (lightweight)
register_monitor_workflows(instrument=instrument, source_names=['monitor1', 'monitor2'])

# Register instrument
instrument_registry.register(instrument)

# Register detector view specs (lightweight, no factories yet)
xy_projection_handles = register_detector_view_specs(
    instrument=instrument,
    projections=['xy_plane'],
    source_names=detector_names,
)

# Register I(Q) workflow spec (basic)
i_of_q_handle = instrument.register_spec(
    name='i_of_q',
    version=1,
    title='I(Q)',
    source_names=detector_names,
    aux_sources=LokiAuxSources,
    outputs=IofQOutputs,
)

# Register I(Q) workflow spec (with params)
i_of_q_with_params_handle = instrument.register_spec(
    name='i_of_q_with_params',
    version=1,
    title='I(Q) with params',
    description='I(Q) reduction with configurable parameters.',
    source_names=detector_names,
    aux_sources=LokiAuxSources,
    outputs=IofQWithTransmissionOutputs,
    params=SansWorkflowParams,
)
