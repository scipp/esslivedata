# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Spec registration for the wavelength lookup-table workflow."""

from __future__ import annotations

import pydantic
import scipp as sc

from ..config.instrument import Instrument
from ..config.workflow_spec import WorkflowOutputsBase
from ..handlers.workflow_factory import SpecHandle
from ..parameter_models import LengthUnit, RangeModel, TimeUnit

#: Logical primary stream name for the synthesized chopper-cascade tick. The
#: synthesizer emits a single ``LogData`` message with this stream name once
#: all chopper setpoints are reached (vacuously true for chopperless
#: instruments). The workflow uses presence of this signal as its trigger.
CHOPPER_CASCADE_SOURCE = 'chopper_cascade'

#: Output key returned by the workflow's ``finalize`` and the field name on
#: :class:`WavelengthLutOutputs`. Also the workflow ``name`` in the spec.
WAVELENGTH_LUT_OUTPUT = 'wavelength_lut'


class Pulse(pydantic.BaseModel):
    """Source pulse properties.

    The unit of ``frequency`` is hardcoded to Hz: there is no realistic case
    where a neutron-source pulse frequency would be expressed in kHz/MHz, so
    we skip the dropdown and keep the model tight.
    """

    frequency: float = pydantic.Field(
        default=14.0, gt=0.0, description="Pulse frequency in Hz."
    )
    stride: int = pydantic.Field(
        default=1,
        ge=1,
        description="Pulse stride (1 unless pulse-skipping is used).",
    )

    def get_period(self) -> sc.Variable:
        """Return the pulse period as a scipp scalar."""
        return sc.scalar(1.0 / self.frequency, unit='s')


class DistanceResolution(pydantic.BaseModel):
    """Resolution of the distance axis in the lookup table."""

    value: float = pydantic.Field(default=0.1, description="Distance bin resolution.")
    unit: LengthUnit = pydantic.Field(default=LengthUnit.METER, description="Unit.")

    def get(self) -> sc.Variable:
        return sc.scalar(self.value, unit=self.unit.value)


class TimeResolution(pydantic.BaseModel):
    """Resolution of the event-time-offset axis in the lookup table."""

    value: float = pydantic.Field(default=250.0, description="Time bin resolution.")
    unit: TimeUnit = pydantic.Field(default=TimeUnit.MICROSECOND, description="Unit.")

    def get(self) -> sc.Variable:
        return sc.scalar(self.value, unit=self.unit.value)


class LtotalRange(RangeModel):
    """Range of total flight paths covered by the lookup table."""

    start: float = pydantic.Field(default=5.0, description="Shortest L_total.")
    stop: float = pydantic.Field(default=30.0, description="Longest L_total.")
    unit: LengthUnit = pydantic.Field(default=LengthUnit.METER, description="Unit.")


class Simulation(pydantic.BaseModel):
    """Simulation knobs."""

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

    pulse: Pulse = pydantic.Field(
        title='Source pulse',
        description='Source pulse frequency and stride.',
        default_factory=Pulse,
    )
    distance_range: LtotalRange = pydantic.Field(
        title='Distance range',
        description='Range of total flight-path lengths covered by the table.',
        default_factory=LtotalRange,
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
    simulation: Simulation = pydantic.Field(
        title='Simulation',
        description='Simulation knobs.',
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

    Hosted by the ``timeseries`` service alongside per-source timeseries
    workflows: both are f144-driven, share the same source/preprocessor
    plumbing, and (in v1) the chopper PVs feeding the synthesizer are
    themselves f144 streams that the timeseries service can plot. The
    ``ChopperSynthesizer`` emitting the synthetic primary trigger is a
    temporary stand-in for an upstream-side ``chopper_cascade_reached``
    f144 stream; once the producer publishes that directly, the wrapper
    drops out and this is just another timeseries workflow.

    The workflow's only ``source_name`` is the synthetic ``chopper_cascade``
    stream emitted by ``ChopperSynthesizer``. The factory must be attached
    later via the returned handle.
    """
    return instrument.register_spec(
        namespace='timeseries',
        name=WAVELENGTH_LUT_OUTPUT,
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
