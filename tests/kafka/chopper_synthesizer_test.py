# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the chopperless ChopperSynthesizer (v0)."""

from __future__ import annotations

from collections.abc import Sequence

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


def test_emits_initial_tick_on_first_call_only():
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


def test_passes_through_wrapped_messages_alongside_initial_tick():
    other = Message(
        stream=StreamId(kind=StreamKind.LOG, name='other_pv'),
        value=LogData(time=0, value=42),
    )
    src = ChopperSynthesizer(FakeSource([[other]]))

    out = list(src.get_messages())
    streams = [m.stream for m in out]

    assert CHOPPER_CASCADE_STREAM in streams
    assert other.stream in streams
    # Initial tick comes first so the context accumulator caches the trigger
    # before any later-batch consumer races against it.
    assert out[0].stream == CHOPPER_CASCADE_STREAM


def test_passthrough_after_initial_tick():
    later = Message(
        stream=StreamId(kind=StreamKind.LOG, name='other_pv'),
        value=LogData(time=1, value=2),
    )
    src = ChopperSynthesizer(FakeSource([[], [later]]))

    _ = list(src.get_messages())
    second = list(src.get_messages())

    assert second == [later]
