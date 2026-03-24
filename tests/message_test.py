# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import time

from ess.livedata import Message, StreamId, StreamKind, compact_messages
from ess.livedata.core.timestamp import Timestamp


class TestMessage:
    def test_auto_generates_timestamp_when_not_provided(self) -> None:
        before = Timestamp(time.time_ns())
        msg = Message(stream=StreamId(kind=StreamKind.LOG, name="test"), value="data")
        after = Timestamp(time.time_ns())

        assert before <= msg.timestamp <= after

    def test_preserves_explicitly_provided_timestamp(self) -> None:
        explicit_timestamp = Timestamp(12345)
        msg = Message(
            timestamp=explicit_timestamp,
            stream=StreamId(kind=StreamKind.LOG, name="test"),
            value="data",
        )

        assert msg.timestamp == explicit_timestamp

    def test_multiple_messages_have_different_auto_timestamps(self) -> None:
        msg1 = Message(
            stream=StreamId(kind=StreamKind.LOG, name="test1"), value="data1"
        )
        time.sleep(0.001)  # Small delay to ensure different timestamps
        msg2 = Message(
            stream=StreamId(kind=StreamKind.LOG, name="test2"), value="data2"
        )

        assert msg1.timestamp != msg2.timestamp
        assert msg1.timestamp < msg2.timestamp


def test_comparison_compares_timestamps() -> None:
    msg1 = Message(timestamp=Timestamp(1), stream="key1", value="value1")
    msg2 = Message(timestamp=Timestamp(2), stream="key1", value="value2")
    msg3 = Message(timestamp=Timestamp(3), stream="key0", value="value3")
    msg4 = Message(timestamp=Timestamp(3), stream="key1", value="value3")
    assert msg1 < msg2
    assert msg2 < msg3
    assert msg1 < msg3
    assert not msg1 < msg1
    assert not msg3 < msg4  # key is less but only timestamps compared


def test_compact_messages_drops_all_but_latest_for_each_key() -> None:
    msg1 = Message(timestamp=Timestamp(1), stream="key1", value="value1")
    msg2 = Message(timestamp=Timestamp(2), stream="key1", value="value2")
    msg3 = Message(timestamp=Timestamp(1), stream="key0", value="value3")
    msg4 = Message(timestamp=Timestamp(3), stream="key1", value="value3")
    assert compact_messages([msg1, msg2, msg3, msg4]) == sorted([msg3, msg4])
    assert compact_messages([msg1, msg2, msg3]) == sorted([msg2, msg3])
    assert compact_messages([msg1, msg2]) == [msg2]
    assert compact_messages([msg1]) == [msg1]
    assert not compact_messages([])
    assert compact_messages([msg1, msg2, msg1, msg2]) == [msg2]
