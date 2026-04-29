# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Spec registration for the wavelength lookup-table workflow."""

from __future__ import annotations

import pydantic
import scipp as sc

from ..config.instrument import Instrument
from ..config.workflow_spec import WorkflowOutputsBase
from ..handlers.workflow_factory import SpecHandle
from ..parameter_models import (
    DistanceResolution,
    LtotalRange,
    PulsePeriod,
    TimeResolution,
)

#: Logical primary stream name for the synthesized chopper-cascade tick. The
#: synthesizer emits a single ``LogData`` message with this stream name once
#: all chopper setpoints are reached (vacuously true for chopperless
#: instruments). The workflow uses presence of this signal as its trigger.
CHOPPER_CASCADE_SOURCE = 'chopper_cascade'

#: Output key returned by the workflow's ``finalize`` and the field name on
#: :class:`WavelengthLutOutputs`. Also the namespace/name used in the spec.
WAVELENGTH_LUT_OUTPUT = 'wavelength_lut'


class Simulation(pydantic.BaseModel):
    """Simulation knobs that do not carry units."""

    pulse_stride: int = pydantic.Field(
        default=1,
        ge=1,
        description="Pulse stride (1 unless pulse-skipping is used).",
    )
    num_simulated_neutrons: int = pydantic.Field(
        default=1_000_000,
        ge=1_000,
        description=(
            "Neutrons in simulation. Lower for faster turnaround during "
            "commissioning or tests."
        ),
    )


class WavelengthLutParams(pydantic.BaseModel):
    """User-facing parameters for the wavelength lookup-table workflow."""

    pulse_period: PulsePeriod = pydantic.Field(
        title='Pulse period',
        description='Source pulse period.',
        default_factory=PulsePeriod,
    )
    distance_resolution: DistanceResolution = pydantic.Field(
        title='Distance resolution',
        description='Resolution of the distance axis in the lookup table.',
        default_factory=DistanceResolution,
    )
    time_resolution: TimeResolution = pydantic.Field(
        title='Time resolution',
        description='Resolution of the event-time-offset axis in the lookup table.',
        default_factory=TimeResolution,
    )
    Ltotal_range: LtotalRange = pydantic.Field(
        title='L_total range',
        description='Range of total flight-path lengths covered by the table.',
        default_factory=LtotalRange,
    )
    simulation: Simulation = pydantic.Field(
        title='Simulation',
        description='Simulation knobs (pulse stride, number of neutrons).',
        default_factory=Simulation,
    )


def _empty_wavelength_lut_template() -> sc.DataArray:
    """Empty placeholder used by the spec until the first computation."""
    return sc.DataArray(
        sc.zeros(dims=['distance', 'event_time_offset'], shape=[0, 0], unit='angstrom'),
        coords={
            'distance': sc.array(dims=['distance'], values=[], unit='m'),
            'event_time_offset': sc.array(
                dims=['event_time_offset'], values=[], unit='us'
            ),
        },
    )


class WavelengthLutOutputs(WorkflowOutputsBase):
    """Outputs of the wavelength lookup-table workflow."""

    wavelength_lut: sc.DataArray = pydantic.Field(
        default_factory=_empty_wavelength_lut_template,
        title='Wavelength lookup table',
        description=(
            'Wavelength as a function of distance and event-time-offset, '
            'computed from the current chopper cascade.'
        ),
    )


def register_wavelength_lut_workflow_spec(
    instrument: Instrument,
    *,
    params: type[WavelengthLutParams] = WavelengthLutParams,
) -> SpecHandle:
    """Register the wavelength lookup-table workflow spec for ``instrument``.

    The workflow's only ``source_name`` is the synthetic ``chopper_cascade``
    stream emitted by ``ChopperSynthesizer``. The factory must be attached
    later via the returned handle.
    """
    return instrument.register_spec(
        namespace='wavelength_lut',
        name='wavelength_lut',
        version=1,
        title='Wavelength lookup table',
        description=(
            'Compute a wavelength lookup table from the current chopper-cascade '
            'configuration. Refires when chopper setpoints change.'
        ),
        source_names=[CHOPPER_CASCADE_SOURCE],
        params=params,
        outputs=WavelengthLutOutputs,
        reset_on_run_transition=False,
    )
