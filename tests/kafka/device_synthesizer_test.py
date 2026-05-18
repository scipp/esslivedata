# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for :class:`DeviceSynthesizer`."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ess.livedata.config.stream import Device
from ess.livedata.core.message import Message, MessageSource, StreamId, StreamKind
from ess.livedata.handlers.accumulators import DeviceSample, LogData
from ess.livedata.kafka.device_synthesizer import DeviceSynthesizer


class FakeSource(MessageSource[Message]):
    def __init__(self) -> None:
        self._batches: list[list[Message]] = []

    def queue(self, messages: list[Message]) -> None:
        self._batches.append(messages)

    def get_messages(self) -> Sequence[Message]:
        if not self._batches:
            return []
        return self._batches.pop(0)


def _log(name: str, time: int, value: float) -> Message[LogData]:
    return Message(
        stream=StreamId(kind=StreamKind.LOG, name=name),
        value=LogData(time=time, value=value),
    )


def _device(
    *,
    value: str,
    target: str | None = None,
    settled: str | None = None,
    units: str | None = 'mm',
) -> Device:
    return Device(value=value, target=target, settled=settled, units=units)


def test_passes_through_unrelated_messages_unchanged() -> None:
    src = FakeSource()
    other = _log('other_pv', time=10, value=1.0)
    src.queue([other])
    syn = DeviceSynthesizer(src, devices={})
    out = list(syn.get_messages())
    assert out == [other]


def test_no_emit_before_bootstrap_completes() -> None:
    src = FakeSource()
    syn = DeviceSynthesizer(
        src,
        devices={'m': _device(value='m_value', target='m_target', settled='m_settled')},
    )
    src.queue([_log('m_value', time=1, value=5.0)])
    src.queue([_log('m_target', time=2, value=5.0)])

    assert list(syn.get_messages()) == []
    assert list(syn.get_messages()) == []


def test_emits_after_all_substreams_observed() -> None:
    src = FakeSource()
    syn = DeviceSynthesizer(
        src,
        devices={'m': _device(value='m_value', target='m_target', settled='m_settled')},
    )
    src.queue(
        [
            _log('m_value', time=1, value=5.0),
            _log('m_target', time=2, value=10.0),
            _log('m_settled', time=3, value=1),
        ]
    )

    out = list(syn.get_messages())
    assert len(out) == 1
    msg = out[0]
    assert msg.stream == StreamId(kind=StreamKind.DEVICE, name='m')
    sample = msg.value
    assert isinstance(sample, DeviceSample)
    # Bootstrap emits a single sample with max-time = 3 (DMOV time).
    assert sample.time.to_ns() == 3
    assert sample.value == 5.0
    assert sample.target == 10.0
    assert sample.settled is True


def test_suppresses_configured_substreams() -> None:
    src = FakeSource()
    syn = DeviceSynthesizer(
        src,
        devices={'m': _device(value='m_value', target='m_target', settled='m_settled')},
    )
    other = _log('other_pv', time=10, value=99.0)
    src.queue(
        [
            _log('m_value', time=1, value=5.0),
            _log('m_target', time=2, value=10.0),
            _log('m_settled', time=3, value=1),
            other,
        ]
    )
    out = list(syn.get_messages())
    streams = [m.stream for m in out]
    assert other.stream in streams
    assert StreamId(kind=StreamKind.LOG, name='m_value') not in streams
    assert StreamId(kind=StreamKind.LOG, name='m_target') not in streams
    assert StreamId(kind=StreamKind.LOG, name='m_settled') not in streams


def test_max_time_policy_across_substreams() -> None:
    src = FakeSource()
    syn = DeviceSynthesizer(
        src, devices={'m': _device(value='m_value', target='m_target')}
    )
    # Bootstrap two substreams.
    src.queue(
        [
            _log('m_value', time=5, value=1.0),
            _log('m_target', time=2, value=2.0),
        ]
    )
    out = list(syn.get_messages())
    assert len(out) == 1
    assert out[0].value.time.to_ns() == 5

    # Late VAL update with earlier time still updates state but does not move max.
    src.queue([_log('m_target', time=3, value=2.5)])
    out = list(syn.get_messages())
    assert len(out) == 1
    assert out[0].value.time.to_ns() == 5
    assert out[0].value.target == 2.5


def test_partial_device_without_settled() -> None:
    src = FakeSource()
    syn = DeviceSynthesizer(
        src, devices={'m': _device(value='m_value', target='m_target')}
    )
    src.queue(
        [
            _log('m_value', time=1, value=1.0),
            _log('m_target', time=2, value=2.0),
        ]
    )
    out = list(syn.get_messages())
    sample = out[0].value
    assert sample.settled is None


def test_union_anchored_emits_on_each_input_after_bootstrap() -> None:
    src = FakeSource()
    syn = DeviceSynthesizer(
        src, devices={'m': _device(value='m_value', target='m_target')}
    )
    src.queue(
        [
            _log('m_value', time=1, value=1.0),
            _log('m_target', time=2, value=2.0),
        ]
    )
    # Bootstrap: one emit.
    assert len(list(syn.get_messages())) == 1
    # Three subsequent inputs → three emits.
    src.queue(
        [
            _log('m_value', time=3, value=1.5),
            _log('m_target', time=4, value=2.5),
            _log('m_value', time=5, value=1.7),
        ]
    )
    out = list(syn.get_messages())
    assert len(out) == 3


def test_substream_collision_across_devices_raises() -> None:
    src = FakeSource()
    with pytest.raises(ValueError, match="configured for both devices"):
        DeviceSynthesizer(
            src,
            devices={
                'a': _device(value='shared'),
                'b': _device(value='shared'),
            },
        )
