# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test: DeviceSynthesizer + ToNXlog produce the merged DataArray.

Exercises the full in-process path that all four services share: real synthesizer,
real accumulator, fake message source feeding LogData substream messages.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipp as sc

from ess.livedata.config.stream import Device
from ess.livedata.core.message import Message, MessageSource, StreamId, StreamKind
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.handlers.accumulators import LogData
from ess.livedata.handlers.to_nxlog import ToNXlog
from ess.livedata.kafka.device_synthesizer import DeviceSynthesizer


class _Source(MessageSource[Message]):
    def __init__(self, messages: list[Message]) -> None:
        self._messages = list(messages)

    def get_messages(self) -> Sequence[Message]:
        out, self._messages = self._messages, []
        return out


def _log(name: str, time: int, value: float) -> Message[LogData]:
    return Message(
        stream=StreamId(kind=StreamKind.LOG, name=name),
        value=LogData(time=time, value=value),
    )


def test_synthesizer_plus_accumulator_produces_merged_dataarray() -> None:
    device = Device(value='m_v', target='m_t', idle='m_s', units='mm')
    syn = DeviceSynthesizer(
        _Source(
            [
                # Bootstrap: one event per substream.
                _log('m_v', time=100, value=1.0),
                _log('m_t', time=200, value=5.0),
                _log('m_s', time=300, value=0),
                # Motion in progress: RBV updates, DMOV unchanged.
                _log('m_v', time=400, value=2.0),
                _log('m_v', time=500, value=3.0),
                # Motion completes: DMOV flips, then RBV settles at target.
                _log('m_s', time=600, value=1),
                _log('m_v', time=700, value=5.0),
            ]
        ),
        devices={'m': device},
    )
    acc = ToNXlog(attrs={'units': 'mm'}, has_target=True, has_idle=True)
    for msg in syn.get_messages():
        assert isinstance(msg.value, LogData)
        acc.add(Timestamp.from_ns(0), msg.value)

    result = acc.get()
    # 5 emits after bootstrap (300, 400, 500, 600, 700).
    assert result.sizes == {'time': 5}
    np.testing.assert_array_equal(result.data.values, [1.0, 2.0, 3.0, 3.0, 5.0])
    np.testing.assert_array_equal(
        result.coords['target'].values, [5.0, 5.0, 5.0, 5.0, 5.0]
    )
    np.testing.assert_array_equal(
        result.coords['idle'].values, [False, False, False, True, True]
    )
    assert result.data.unit == sc.Unit('mm')
    assert result.coords['target'].unit == sc.Unit('mm')
