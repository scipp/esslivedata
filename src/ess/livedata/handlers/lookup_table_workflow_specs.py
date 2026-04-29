# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Spec registration for the TOF lookup-table workflow."""

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


class LookupTableParams(pydantic.BaseModel):
    """User-facing parameters for the TOF lookup-table workflow."""

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


def _empty_lookup_table_template() -> sc.DataArray:
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


class LookupTableOutputs(WorkflowOutputsBase):
    """Outputs of the TOF lookup-table workflow."""

    lookup_table: sc.DataArray = pydantic.Field(
        default_factory=_empty_lookup_table_template,
        title='TOF lookup table',
        description=(
            'Wavelength as a function of distance and event-time-offset, '
            'computed from the current chopper cascade.'
        ),
    )


def register_lookup_table_workflow_spec(
    instrument: Instrument,
    *,
    params: type[LookupTableParams] = LookupTableParams,
) -> SpecHandle:
    """Register the TOF lookup-table workflow spec for ``instrument``.

    The workflow's only ``source_name`` is the synthetic ``chopper_cascade``
    stream emitted by ``ChopperSynthesizer``. The factory must be attached
    later via the returned handle.
    """
    return instrument.register_spec(
        namespace='tof_table',
        name='lookup_table',
        version=1,
        title='TOF lookup table',
        description=(
            'Compute a wavelength lookup table from the current chopper-cascade '
            'configuration. Refires when chopper setpoints change.'
        ),
        source_names=[CHOPPER_CASCADE_SOURCE],
        params=params,
        outputs=LookupTableOutputs,
        reset_on_run_transition=False,
    )
