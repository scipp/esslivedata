# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Spec registration for the wavelength lookup-table workflow."""

from __future__ import annotations

import pydantic
import scipp as sc

from ..config.instrument import Instrument
from ..config.workflow_spec import TIMESERIES, AuxSources, WorkflowOutputsBase
from ..handlers.workflow_factory import SpecHandle
from ..kafka.stream_mapping import InputStreamKey, StreamLUT
from ..parameter_models import LengthUnit, RangeModel, TimeUnit

#: Logical primary stream name for the synthesized chopper-cascade tick. The
#: synthesizer emits a single ``LogData`` message with this stream name once
#: all chopper setpoints are reached (vacuously true for chopperless
#: instruments). The workflow uses presence of this signal as its trigger.
CHOPPER_CASCADE_SOURCE = 'chopper_cascade'

#: Output key returned by the workflow's ``finalize`` and the field name on
#: :class:`WavelengthLutOutputs`. Also the workflow ``name`` in the spec.
WAVELENGTH_LUT_OUTPUT = 'wavelength_lut'


#: Naming convention for per-chopper aux input streams. The synthesizer
#: emits these stream names; the spec routes them to the workflow as aux
#: inputs; the workflow consumes them by the same name.
def speed_setpoint_input(chopper: str) -> str:
    """Logical aux-input name for a chopper's rotation-speed setpoint."""
    return f'{chopper}_rotation_speed_setpoint'


def delay_setpoint_input(chopper: str) -> str:
    """Logical aux-input name for a chopper's delay setpoint.

    ``DiskChopper.from_nexus`` derives the chopper's phase relative to the
    source pulse from ``delay`` and ``rotation_speed_setpoint`` as
    ``phase = 2*pi*frequency*delay``. The synthesizer plateau-detects on
    the noisy ``<chopper>_delay`` readback and emits this stream when
    stable; production NeXus files carry no ``phase`` field.
    """
    return f'{chopper}_delay_setpoint'


def chopper_topic(instrument: str) -> str:
    """Kafka topic carrying chopper PV streams for ``instrument``."""
    return f'{instrument}_choppers'


#: ECDC PV-name suffix per chopper f144 stream. Convention is consistent
#: across ESS chopper integrations; readback delay is in nanoseconds and
#: speed setpoints in hertz.
_CHOPPER_PV_SUFFIXES: dict[str, tuple[str, str]] = {
    'delay': (':TotDly', 'ns'),
    'rotation_speed_setpoint': (':Spd_S', 'Hz'),
}


def make_chopper_attribute_registry(
    chopper_names: list[str],
) -> dict[str, dict[str, str]]:
    """Return f144 attribute-registry entries for chopper streams.

    Includes the in-process synthesized ``<chopper>_delay_setpoint`` so the
    log-data preprocessor accepts those messages.
    """
    out: dict[str, dict[str, str]] = {}
    for chopper in chopper_names:
        for stream_kind, (_, units) in _CHOPPER_PV_SUFFIXES.items():
            out[f'{chopper}_{stream_kind}'] = {'units': units}
        out[f'{chopper}_delay_setpoint'] = {'units': 'ns'}
    return out


def make_chopper_stream_lut(instrument: str, pv_prefixes: dict[str, str]) -> StreamLUT:
    """Return ``(topic, source_name) -> internal_name`` for chopper PV streams.

    Synthesized ``_delay_setpoint`` is omitted: the synthesizer emits it
    in-process and it never arrives from Kafka.
    """
    topic = chopper_topic(instrument)
    return {
        InputStreamKey(topic=topic, source_name=f'{prefix}{suffix}'): (
            f'{chopper}_{stream_kind}'
        )
        for chopper, prefix in pv_prefixes.items()
        for stream_kind, (suffix, _) in _CHOPPER_PV_SUFFIXES.items()
    }


def make_chopper_log_topic_for_stream(
    instrument: str, chopper_names: list[str]
) -> dict[str, str]:
    """Per-stream topic override for outbound f144 publishing.

    Used by the dev-mode log-producer widget so chopper sliders publish to
    the same topic the timeseries service subscribes to in production.
    """
    topic = chopper_topic(instrument)
    return {
        f'{chopper}_{stream_kind}': topic
        for chopper in chopper_names
        for stream_kind in (*_CHOPPER_PV_SUFFIXES, 'delay_setpoint')
    }


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


def _chopper_aux_sources(chopper_names: list[str]) -> AuxSources | None:
    """Build aux-source mappings for a list of choppers.

    Each chopper contributes a ``rotation_speed_setpoint`` and a
    ``phase_setpoint`` aux input; both bind to identically-named streams
    emitted by :class:`ChopperSynthesizer`. Returns ``None`` for chopperless
    instruments — the workflow has no aux inputs.
    """
    if not chopper_names:
        return None
    inputs = {
        stream: stream
        for name in chopper_names
        for stream in (speed_setpoint_input(name), delay_setpoint_input(name))
    }
    return AuxSources(inputs)


def register_wavelength_lut_workflow_spec(
    instrument: Instrument,
    *,
    params: type[WavelengthLutParams] = WavelengthLutParams,
) -> SpecHandle:
    """Register the wavelength lookup-table workflow spec for ``instrument``.

    Hosted by the ``timeseries`` service alongside per-source timeseries
    workflows: both are f144-driven, share the same source/preprocessor
    plumbing, and the chopper PVs feeding the synthesizer are themselves
    f144 streams that the timeseries service can plot. The
    ``ChopperSynthesizer`` emitting the synthetic primary trigger is a
    temporary stand-in for an upstream-side ``chopper_cascade_reached``
    f144 stream; once the producer publishes that directly, the wrapper
    drops out and this is just another timeseries workflow.

    The workflow's only ``source_name`` is the synthetic ``chopper_cascade``
    stream emitted by ``ChopperSynthesizer``. Aux sources are derived from
    ``instrument.choppers``: empty ⇒ chopperless. The factory must be
    attached later via the returned handle.
    """
    handle = instrument.register_spec(
        group=TIMESERIES,
        name=WAVELENGTH_LUT_OUTPUT,
        version=1,
        title='Wavelength lookup table',
        description=(
            'Compute a wavelength lookup table from the current chopper-cascade '
            'configuration. Refires when chopper setpoints change.'
        ),
        source_names=[CHOPPER_CASCADE_SOURCE],
        aux_sources=_chopper_aux_sources(instrument.choppers),
        params=params,
        outputs=WavelengthLutOutputs,
        reset_on_run_transition=False,
    )

    @handle.attach_factory()
    def _factory(params: WavelengthLutParams):
        # Lazy import: keep heavy scipp/sciline/ess.reduce graph construction
        # out of spec registration so module import stays cheap.
        from .wavelength_lut_workflow import create_wavelength_lut_workflow

        nexus_filename = instrument.nexus_file if instrument.choppers else None
        return create_wavelength_lut_workflow(
            params=params,
            chopper_names=instrument.choppers,
            nexus_filename=nexus_filename,
        )

    return handle
