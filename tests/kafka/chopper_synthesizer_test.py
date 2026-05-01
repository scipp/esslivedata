# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the ChopperSynthesizer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from ess.livedata.core.message import Message, MessageSource, StreamId, StreamKind
from ess.livedata.handlers.accumulators import LogData
from ess.livedata.kafka.chopper_synthesizer import (
    CHOPPER_CASCADE_STREAM,
    ChopperSynthesizer,
)


class FakeSource(MessageSource[Message]):
    def __init__(self, batches: list[list[Message]] | None = None) -> None:
        self._batches = list(batches or [])

    def get_messages(self) -> Sequence[Message]:
        if not self._batches:
            return []
        return self._batches.pop(0)


def _phase_msg(chopper: str, value: float, time_ns: int = 0) -> Message:
    return Message(
        stream=StreamId(kind=StreamKind.LOG, name=f'{chopper}_phase'),
        value=LogData(time=time_ns, value=value),
    )


def _speed_setpoint_msg(chopper: str, value: float, time_ns: int = 0) -> Message:
    return Message(
        stream=StreamId(kind=StreamKind.LOG, name=f'{chopper}_rotation_speed_setpoint'),
        value=LogData(time=time_ns, value=value),
    )


def _stream_names(messages: Sequence[Message]) -> list[str]:
    return [m.stream.name for m in messages]


# ---------------------------------------------------------------------------
# Chopperless mode (existing behaviour)
# ---------------------------------------------------------------------------


def test_chopperless_emits_initial_tick_on_first_call_only():
    src = ChopperSynthesizer(FakeSource([[], [], []]))

    first = list(src.get_messages())
    second = list(src.get_messages())
    third = list(src.get_messages())

    assert len(first) == 1
    assert first[0].stream == CHOPPER_CASCADE_STREAM
    assert isinstance(first[0].value, LogData)
    assert first[0].value.value == 1
    assert second == []
    assert third == []


def test_chopperless_passes_through_with_initial_tick_first():
    other = Message(
        stream=StreamId(kind=StreamKind.LOG, name='other_pv'),
        value=LogData(time=0, value=42),
    )
    src = ChopperSynthesizer(FakeSource([[other]]))

    out = list(src.get_messages())

    assert out[0].stream == CHOPPER_CASCADE_STREAM
    assert other in out


def test_chopperless_passthrough_after_initial_tick():
    later = Message(
        stream=StreamId(kind=StreamKind.LOG, name='other_pv'),
        value=LogData(time=1, value=2),
    )
    src = ChopperSynthesizer(FakeSource([[], [later]]))

    _ = list(src.get_messages())
    second = list(src.get_messages())

    assert second == [later]


# ---------------------------------------------------------------------------
# Single-chopper mode
# ---------------------------------------------------------------------------


@pytest.fixture
def single_chopper():
    """Synthesizer config: one chopper, small window, modest tolerance."""
    return {
        'chopper_names': ['c1'],
        'phase_window_size': 5,
        'phase_atol_deg': 0.5,
    }


def test_single_chopper_no_initial_tick(single_chopper):
    src = ChopperSynthesizer(FakeSource([[]]), **single_chopper)
    assert list(src.get_messages()) == []


def test_single_chopper_speed_setpoint_passes_through_no_cascade(single_chopper):
    msg = _speed_setpoint_msg('c1', 14.0)
    src = ChopperSynthesizer(FakeSource([[msg]]), **single_chopper)

    out = list(src.get_messages())

    # Speed alone does not trigger cascade — phase still missing.
    assert out == [msg]


def test_single_chopper_short_phase_burst_no_emit(single_chopper):
    # Window size 5; only 3 samples — not enough to lock.
    batch = [_phase_msg('c1', 90.0, time_ns=i) for i in range(3)]
    src = ChopperSynthesizer(FakeSource([batch]), **single_chopper)

    out = list(src.get_messages())

    # Originals pass through, no synthetic emissions.
    assert out == batch


def test_single_chopper_locks_and_emits_cascade_after_speed(single_chopper):
    # 5 stable phase samples → lock; followed by speed setpoint → cascade.
    phase_batch = [_phase_msg('c1', 90.0, time_ns=i) for i in range(5)]
    speed_batch = [_speed_setpoint_msg('c1', 14.0, time_ns=10)]
    src = ChopperSynthesizer(FakeSource([phase_batch, speed_batch]), **single_chopper)

    first = list(src.get_messages())
    streams_first = _stream_names(first)

    # Phase locked → synthetic phase_setpoint emitted; cascade not yet (no speed).
    assert 'c1_phase_setpoint' in streams_first
    assert 'chopper_cascade' not in streams_first
    # Originals still flow through.
    assert all(m in first for m in phase_batch)

    second = list(src.get_messages())
    streams_second = _stream_names(second)

    # Speed setpoint arrived → cascade fires (transition to all-locked).
    assert 'chopper_cascade' in streams_second
    assert all(m in second for m in speed_batch)


def test_single_chopper_phase_setpoint_value_matches_window_mean(single_chopper):
    samples = [89.9, 90.1, 90.0, 90.05, 89.95]
    batch = [_phase_msg('c1', v, time_ns=i) for i, v in enumerate(samples)]
    src = ChopperSynthesizer(FakeSource([batch]), **single_chopper)

    out = list(src.get_messages())

    [setpoint] = [m for m in out if m.stream.name == 'c1_phase_setpoint']
    assert setpoint.value.value == pytest.approx(np.mean(samples))


def test_single_chopper_noisy_window_no_lock(single_chopper):
    # Std exceeds atol (0.5 deg) — no lock.
    samples = [85.0, 95.0, 88.0, 92.0, 90.0]
    batch = [_phase_msg('c1', v, time_ns=i) for i, v in enumerate(samples)]
    src = ChopperSynthesizer(FakeSource([batch]), **single_chopper)

    out = list(src.get_messages())

    assert all(m.stream.name != 'c1_phase_setpoint' for m in out)
    assert all(m.stream.name != 'chopper_cascade' for m in out)


def test_single_chopper_setpoint_change_re_emits(single_chopper):
    # Lock at 90, then operator nudges to 100 — second lock emits a new setpoint
    # and a new cascade tick.
    first_window = [_phase_msg('c1', 90.0, time_ns=i) for i in range(5)]
    speed = [_speed_setpoint_msg('c1', 14.0, time_ns=10)]
    second_window = [_phase_msg('c1', 100.0, time_ns=20 + i) for i in range(5)]
    src = ChopperSynthesizer(
        FakeSource([first_window, speed, second_window]), **single_chopper
    )

    list(src.get_messages())  # first lock at 90
    list(src.get_messages())  # speed → first cascade
    third = list(src.get_messages())

    setpoints = [m for m in third if m.stream.name == 'c1_phase_setpoint']
    cascades = [m for m in third if m.stream.name == 'chopper_cascade']
    assert len(setpoints) == 1
    assert setpoints[0].value.value == pytest.approx(100.0)
    assert len(cascades) == 1


def test_single_chopper_no_cascade_when_input_unchanged(single_chopper):
    # Once locked, a batch with an unrelated message must not re-emit cascade.
    phase_batch = [_phase_msg('c1', 90.0, time_ns=i) for i in range(5)]
    speed_batch = [_speed_setpoint_msg('c1', 14.0, time_ns=10)]
    unrelated = Message(
        stream=StreamId(kind=StreamKind.LOG, name='other_pv'),
        value=LogData(time=20, value=1),
    )
    src = ChopperSynthesizer(
        FakeSource([phase_batch, speed_batch, [unrelated]]), **single_chopper
    )

    list(src.get_messages())  # lock phase
    list(src.get_messages())  # speed → cascade
    third = list(src.get_messages())

    assert third == [unrelated]


def test_single_chopper_speed_change_re_emits_cascade(single_chopper):
    phase_batch = [_phase_msg('c1', 90.0, time_ns=i) for i in range(5)]
    speed1 = [_speed_setpoint_msg('c1', 14.0, time_ns=10)]
    speed2 = [_speed_setpoint_msg('c1', 28.0, time_ns=20)]
    src = ChopperSynthesizer(
        FakeSource([phase_batch, speed1, speed2]), **single_chopper
    )

    list(src.get_messages())  # phase lock
    list(src.get_messages())  # first cascade
    third = list(src.get_messages())

    assert any(m.stream == CHOPPER_CASCADE_STREAM for m in third)
