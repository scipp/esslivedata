# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Synthesize a chopper-cascade trigger from chopper PV streams.

The synthesizer is a :class:`MessageSource` decorator: it wraps an
already-adapted source (i.e. domain-level :class:`Message` objects) and:

- Forwards every wrapped message verbatim.
- Tracks per-chopper ``<chopper>_rotation_speed_setpoint`` (clean upstream
  f144) and ``<chopper>_phase`` (noisy readback) values.
- Runs a rolling-window stability detector on each chopper's phase samples;
  when stable and the value differs from the cached locked value, emits a
  synthetic ``<chopper>_phase_setpoint`` f144 message.
- When every configured chopper has both a cached
  ``rotation_speed_setpoint`` and a stable ``phase_setpoint``, emits a
  synthetic primary tick on the ``chopper_cascade`` logical stream — but
  only on cycles where one of those inputs actually changed.

For chopperless instruments (empty ``chopper_names``) it emits exactly one
vacuous ``chopper_cascade`` tick on the first call to ``get_messages`` and
is otherwise a passthrough.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import structlog

from ..core.message import Message, MessageSource, StreamId, StreamKind
from ..core.timestamp import Timestamp
from ..handlers.accumulators import LogData
from ..handlers.wavelength_lut_workflow_specs import CHOPPER_CASCADE_SOURCE

logger = structlog.get_logger(__name__)

CHOPPER_CASCADE_STREAM = StreamId(kind=StreamKind.LOG, name=CHOPPER_CASCADE_SOURCE)


def _phase_stream_name(chopper: str) -> str:
    return f'{chopper}_phase'


def _speed_setpoint_stream_name(chopper: str) -> str:
    return f'{chopper}_rotation_speed_setpoint'


def _phase_setpoint_stream_name(chopper: str) -> str:
    return f'{chopper}_phase_setpoint'


def _make_chopper_cascade_message() -> Message[LogData]:
    """Build the synthetic 'chopper cascade reached setpoints' tick."""
    now = Timestamp.now()
    return Message(
        timestamp=now,
        stream=CHOPPER_CASCADE_STREAM,
        value=LogData(time=now.to_ns(), value=1),
    )


def _make_phase_setpoint_message(
    chopper: str, value: float, time_ns: int
) -> Message[LogData]:
    now = Timestamp.from_ns(time_ns)
    return Message(
        timestamp=now,
        stream=StreamId(kind=StreamKind.LOG, name=_phase_setpoint_stream_name(chopper)),
        value=LogData(time=time_ns, value=value),
    )


class _StabilityDetector:
    """Rolling-window stability detector.

    Holds the most recent ``window_size`` samples. A "lock" is acquired when
    the window's standard deviation is below ``atol``; the locked value is
    the window mean. The same ``atol`` decides whether a new mean has
    drifted far enough from the previous lock to count as a new setpoint —
    so noise rejection and change detection share one knob.
    """

    def __init__(self, *, window_size: int, atol: float) -> None:
        self._buffer: deque[float] = deque(maxlen=window_size)
        self._atol = atol
        self._locked: float | None = None

    def add(self, sample: float) -> float | None:
        """Append a sample. Return a new locked value if it changed, else None."""
        self._buffer.append(sample)
        if len(self._buffer) < self._buffer.maxlen:
            return None
        arr = np.fromiter(self._buffer, dtype=float)
        if arr.std() >= self._atol:
            return None
        mean = float(arr.mean())
        if self._locked is None or abs(mean - self._locked) > self._atol:
            self._locked = mean
            return mean
        return None

    @property
    def locked(self) -> float | None:
        return self._locked


@dataclass
class _ChopperState:
    detector: _StabilityDetector
    speed_setpoint: float | None = None
    phase_setpoint: float | None = None

    def is_locked(self) -> bool:
        return self.speed_setpoint is not None and self.phase_setpoint is not None


@dataclass
class _ChopperSynthesizerConfig:
    chopper_names: tuple[str, ...] = ()
    phase_window_size: int = 10
    phase_atol_deg: float = 1.0


class ChopperSynthesizer(MessageSource[Message]):
    """Decorator that injects synthetic chopper-cascade triggers.

    Two modes:

    - ``chopper_names`` empty (chopperless instrument): emit one vacuous
      ``chopper_cascade`` tick on the first ``get_messages`` call, then
      passthrough.
    - ``chopper_names`` non-empty: per-chopper plateau detection on phase,
      pass-through-with-cache for rotation_speed_setpoint, conditional
      ``chopper_cascade`` emission once all choppers locked.
    """

    def __init__(
        self,
        wrapped: MessageSource[Message],
        *,
        chopper_names: Sequence[str] = (),
        phase_window_size: int = 2,
        phase_atol_deg: float = 1.0,
    ) -> None:
        self._wrapped = wrapped
        self._config = _ChopperSynthesizerConfig(
            chopper_names=tuple(chopper_names),
            phase_window_size=phase_window_size,
            phase_atol_deg=phase_atol_deg,
        )
        self._states: dict[str, _ChopperState] = {
            name: _ChopperState(
                detector=_StabilityDetector(
                    window_size=phase_window_size, atol=phase_atol_deg
                )
            )
            for name in chopper_names
        }
        self._phase_streams = {_phase_stream_name(name): name for name in chopper_names}
        self._speed_streams = {
            _speed_setpoint_stream_name(name): name for name in chopper_names
        }
        self._emitted_initial_tick = False
        self._was_all_locked = False
        if chopper_names:
            logger.info(
                'chopper_synthesizer_configured',
                choppers=list(chopper_names),
                phase_window_size=phase_window_size,
                phase_atol_deg=phase_atol_deg,
            )

    def get_messages(self) -> Sequence[Message]:
        synthetic: list[Message] = []
        forwarded: list[Message] = []

        if not self._config.chopper_names and not self._emitted_initial_tick:
            self._emitted_initial_tick = True
            synthetic.append(_make_chopper_cascade_message())
            logger.info('chopper_cascade_initial_tick_emitted')

        any_input_changed = False
        for msg in self._wrapped.get_messages():
            forwarded.append(msg)
            if self._handle_chopper_message(msg, synthetic):
                any_input_changed = True

        if self._config.chopper_names:
            all_locked = all(s.is_locked() for s in self._states.values())
            if any_input_changed and all_locked:
                synthetic.append(_make_chopper_cascade_message())
                if not self._was_all_locked:
                    logger.info(
                        'chopper_cascade_all_locked',
                        choppers=list(self._config.chopper_names),
                    )
                else:
                    logger.info('chopper_cascade_emitted')
            self._was_all_locked = all_locked

        return [*synthetic, *forwarded]

    def _handle_chopper_message(self, msg: Message, synthetic: list[Message]) -> bool:
        """Update chopper state from ``msg``. Return True if an input changed."""
        name = msg.stream.name
        chopper = self._phase_streams.get(name)
        if chopper is not None:
            return self._handle_phase_sample(chopper, msg, synthetic)
        chopper = self._speed_streams.get(name)
        if chopper is not None:
            return self._handle_speed_setpoint(chopper, msg)
        return False

    def _handle_phase_sample(
        self, chopper: str, msg: Message, synthetic: list[Message]
    ) -> bool:
        sample = float(msg.value.value)
        new_setpoint = self._states[chopper].detector.add(sample)
        if new_setpoint is None:
            return False
        synthetic.append(
            _make_phase_setpoint_message(chopper, new_setpoint, msg.value.time)
        )
        self._states[chopper].phase_setpoint = new_setpoint
        logger.info('chopper_phase_locked', chopper=chopper, setpoint=new_setpoint)
        return True

    def _handle_speed_setpoint(self, chopper: str, msg: Message) -> bool:
        new_speed = float(msg.value.value)
        state = self._states[chopper]
        if state.speed_setpoint == new_speed:
            return False
        state.speed_setpoint = new_speed
        logger.info('chopper_speed_setpoint_updated', chopper=chopper, value=new_speed)
        return True
