# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import time

import scipp as sc

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.http_transport.sink import QueueBasedMessageSink


def test_sink_stores_and_retrieves_messages():
    """Test basic publish and get operations."""
    sink = QueueBasedMessageSink[sc.DataArray](max_size=10)

    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test')
    da = sc.DataArray(sc.scalar(1.0))

    messages = [
        Message(timestamp=1000, stream=stream, value=da),
        Message(timestamp=2000, stream=stream, value=da * 2),
    ]

    sink.publish_messages(messages)
    retrieved = sink.get_messages()

    assert len(retrieved) == 2
    assert retrieved[0].timestamp == 1000
    assert retrieved[1].timestamp == 2000


def test_sink_clears_after_get():
    """Test that get_messages clears the queue."""
    sink = QueueBasedMessageSink[sc.DataArray](max_size=10)

    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test')
    messages = [
        Message(timestamp=1000, stream=stream, value=sc.DataArray(sc.scalar(1.0)))
    ]

    sink.publish_messages(messages)
    first_get = sink.get_messages()
    second_get = sink.get_messages()

    assert len(first_get) == 1
    assert len(second_get) == 0


def test_sink_drops_oldest_when_full():
    """Test that oldest messages are dropped when max_size is reached."""
    sink = QueueBasedMessageSink[sc.DataArray](max_size=3)

    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test')

    messages = [
        Message(
            timestamp=i * 1000, stream=stream, value=sc.DataArray(sc.scalar(float(i)))
        )
        for i in range(5)
    ]

    sink.publish_messages(messages)
    retrieved = sink.get_messages()

    assert len(retrieved) == 3
    assert retrieved[0].timestamp == 2000
    assert retrieved[1].timestamp == 3000
    assert retrieved[2].timestamp == 4000


def test_sink_filters_expired_messages():
    """Test that messages older than max_age_seconds are filtered."""
    sink = QueueBasedMessageSink[sc.DataArray](max_size=10, max_age_seconds=0.1)

    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test')
    current_time_ns = int(time.time() * 1_000_000_000)

    old_message = Message(
        timestamp=current_time_ns - 200_000_000,  # 200ms ago
        stream=stream,
        value=sc.DataArray(sc.scalar(1.0)),
    )
    new_message = Message(
        timestamp=current_time_ns - 50_000_000,  # 50ms ago
        stream=stream,
        value=sc.DataArray(sc.scalar(2.0)),
    )

    sink.publish_messages([old_message, new_message])
    retrieved = sink.get_messages()

    assert len(retrieved) == 1
    assert sc.identical(retrieved[0].value, sc.DataArray(sc.scalar(2.0)))


def test_sink_peek_does_not_clear():
    """Test that peek_messages does not remove messages."""
    sink = QueueBasedMessageSink[sc.DataArray](max_size=10)

    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test')
    messages = [
        Message(timestamp=1000, stream=stream, value=sc.DataArray(sc.scalar(1.0)))
    ]

    sink.publish_messages(messages)
    peeked = sink.peek_messages()
    retrieved = sink.get_messages()

    assert len(peeked) == 1
    assert len(retrieved) == 1


def test_sink_size_property():
    """Test that size property returns correct count."""
    sink = QueueBasedMessageSink[sc.DataArray](max_size=10)

    assert sink.size == 0

    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test')
    messages = [
        Message(
            timestamp=i * 1000, stream=stream, value=sc.DataArray(sc.scalar(float(i)))
        )
        for i in range(3)
    ]

    sink.publish_messages(messages)
    assert sink.size == 3

    sink.get_messages()
    assert sink.size == 0
