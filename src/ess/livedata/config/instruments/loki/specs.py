# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
LOKI instrument spec registration.
"""

from enum import StrEnum
from typing import Literal

import pydantic
import scipp as sc

from ess.livedata import parameter_models
from ess.livedata.config import Instrument, SourceMetadata, instrument_registry
from ess.livedata.config.workflow_spec import AuxSourcesBase, WorkflowOutputsBase
from ess.livedata.handlers.detector_view_specs import register_detector_view_spec
from ess.livedata.handlers.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)

from .views import get_wire_view


class TransmissionMode(StrEnum):
    """Transmission correction mode for SANS reduction."""

    constant = 'constant'
    current_run = 'current_run'


class TransmissionOptions(pydantic.BaseModel):
    """Transmission correction configuration.

    The standard SANS transmission calculation normalizes sample monitors
    against an empty-beam measurement. In live mode only simplified modes
    are available that do not require a separate empty-beam run.
    """

    mode: TransmissionMode = pydantic.Field(
        title='Mode',
        description=(
            '"constant": no correction (transmission fraction = 1).'
            ' "current_run": estimate transmission from the ratio of'
            ' transmission to incident monitor in the sample run.'
        ),
        default=TransmissionMode.current_run,
    )


class BeamCenterXY(pydantic.BaseModel):
    """Beam center position in detector coordinates."""

    x: float = pydantic.Field(
        default=0.0,
        title='X',
        description='Beam center X coordinate.',
    )
    y: float = pydantic.Field(
        default=0.0,
        title='Y',
        description='Beam center Y coordinate.',
    )
    unit: parameter_models.LengthUnit = pydantic.Field(
        default=parameter_models.LengthUnit.METER,
        description='Unit of the beam center coordinates.',
    )

    def get_vector(self) -> sc.Variable:
        """Get the beam center as a 3D scipp vector."""
        return sc.vector([self.x, self.y, 0.0], unit=self.unit.value)


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


def _make_1d_q_template() -> sc.DataArray:
    """Create an empty 1D template for I(Q) output."""
    return sc.DataArray(
        sc.zeros(dims=['Q'], shape=[0], unit='dimensionless'),
        coords={'Q': sc.arange('Q', 0, unit='1/angstrom')},
    )


def _make_1d_wavelength_template() -> sc.DataArray:
    """Create an empty 1D template for wavelength-binned output."""
    return sc.DataArray(
        sc.zeros(dims=['wavelength'], shape=[0], unit='dimensionless'),
        coords={'wavelength': sc.arange('wavelength', 0, unit='angstrom')},
    )


class IofQOutputs(WorkflowOutputsBase):
    """Outputs for the I(Q) workflow."""

    i_of_q: sc.DataArray = pydantic.Field(
        default_factory=_make_1d_q_template,
        title='I(Q)',
        description='Scattered intensity as a function of momentum transfer Q.',
    )
    transmission_fraction: sc.DataArray = pydantic.Field(
        default_factory=_make_1d_wavelength_template,
        title='Transmission Fraction',
        description='Sample transmission fraction calculated from monitor ratio.',
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
    beam_center: BeamCenterXY = pydantic.Field(
        title='Beam center',
        description='Beam center position in the detector plane.',
        default_factory=BeamCenterXY,
    )
    transmission: TransmissionOptions = pydantic.Field(
        title='Transmission',
        description=(
            'Transmission correction settings.'
            ' The standard SANS transmission calculation normalizes sample'
            ' monitors against an empty-beam measurement;'
            ' this is not available in live mode.'
        ),
        default_factory=TransmissionOptions,
    )


# Detector names for LOKI
detector_names = [f'loki_detector_{bank}' for bank in range(9)]

# Create instrument
instrument = Instrument(
    name='loki',
    detector_names=detector_names,
    monitors=['monitor1', 'monitor2'],
    source_metadata={
        'monitor1': SourceMetadata(title='Incident Monitor'),
        'monitor2': SourceMetadata(title='Transmission Monitor'),
    },
)

# Register instrument
instrument_registry.register(instrument)

# Register monitor workflow spec (TOA-only, no TOF lookup tables)
monitor_handle = register_monitor_workflow_specs(
    instrument, ['monitor1', 'monitor2'], params=TOAOnlyMonitorDataParams
)

xy_projection_handle = register_detector_view_spec(
    instrument=instrument,
    projection='xy_plane',
    source_names=detector_names,
)

# Register wire view for all detector banks
instrument.add_logical_view(
    name='wire_view',
    title='Wire View',
    description='Sum over straw and pixel dimensions to show layer x tube counts.',
    source_names=detector_names,
    transform=get_wire_view,
    output_ndim=2,
    reduction_dim=['straw', 'pixel'],
)

# Register I(Q) workflow spec
i_of_q_handle = instrument.register_spec(
    name='i_of_q',
    version=1,
    title='I(Q)',
    description=(
        'SANS I(Q) reduction for LOKI. Converts detector event data into'
        ' scattered intensity as a function of momentum transfer Q.'
        ' Direct-beam normalization (flat/efficiency correction) is not applied and '
        'currently the transmission does not take into a account and empty run.'
    ),
    source_names=detector_names,
    aux_sources=LokiAuxSources,
    outputs=IofQOutputs,
    params=SansWorkflowParams,
)
