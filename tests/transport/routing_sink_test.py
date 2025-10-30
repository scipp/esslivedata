# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import logging
from typing import Any

import pytest

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.transport.routing_sink import RoutingSink


class FakeSink:
    """Simple fake sink that records published messages."""

    def __init__(self) -> None:
        self.published: list[Message[Any]] = []

    def publish_messages(self, messages: list[Message[Any]]) -> None:
        self.published.extend(messages)


def test_routing_sink_routes_messages_by_stream_kind():
    detector_sink = FakeSink()
    monitor_sink = FakeSink()
    config_sink = FakeSink()

    routing_sink = RoutingSink(
        {
            StreamKind.DETECTOR_EVENTS: detector_sink,
            StreamKind.MONITOR_EVENTS: monitor_sink,
            StreamKind.LIVEDATA_CONFIG: config_sink,
        }
    )

    detector_msg = Message(
        timestamp=100,
        stream=StreamId(kind=StreamKind.DETECTOR_EVENTS, name='detector1'),
        value='detector_data',
    )
    monitor_msg = Message(
        timestamp=200,
        stream=StreamId(kind=StreamKind.MONITOR_EVENTS, name='monitor1'),
        value='monitor_data',
    )
    config_msg = Message(
        timestamp=300,
        stream=StreamId(kind=StreamKind.LIVEDATA_CONFIG, name='config1'),
        value='config_data',
    )

    routing_sink.publish_messages([detector_msg, monitor_msg, config_msg])

    assert len(detector_sink.published) == 1
    assert detector_sink.published[0] == detector_msg

    assert len(monitor_sink.published) == 1
    assert monitor_sink.published[0] == monitor_msg

    assert len(config_sink.published) == 1
    assert config_sink.published[0] == config_msg


def test_routing_sink_groups_multiple_messages_of_same_kind():
    detector_sink = FakeSink()
    monitor_sink = FakeSink()

    routing_sink = RoutingSink(
        {
            StreamKind.DETECTOR_EVENTS: detector_sink,
            StreamKind.MONITOR_EVENTS: monitor_sink,
        }
    )

    detector_msg1 = Message(
        timestamp=100,
        stream=StreamId(kind=StreamKind.DETECTOR_EVENTS, name='detector1'),
        value='data1',
    )
    detector_msg2 = Message(
        timestamp=200,
        stream=StreamId(kind=StreamKind.DETECTOR_EVENTS, name='detector2'),
        value='data2',
    )
    monitor_msg = Message(
        timestamp=300,
        stream=StreamId(kind=StreamKind.MONITOR_EVENTS, name='monitor1'),
        value='monitor_data',
    )

    routing_sink.publish_messages([detector_msg1, detector_msg2, monitor_msg])

    assert len(detector_sink.published) == 2
    assert detector_sink.published == [detector_msg1, detector_msg2]

    assert len(monitor_sink.published) == 1
    assert monitor_sink.published[0] == monitor_msg


def test_routing_sink_skips_messages_with_no_route(caplog):
    detector_sink = FakeSink()

    routing_sink = RoutingSink(
        {StreamKind.DETECTOR_EVENTS: detector_sink},
        logger=logging.getLogger(__name__),
    )

    detector_msg = Message(
        timestamp=100,
        stream=StreamId(kind=StreamKind.DETECTOR_EVENTS, name='detector1'),
        value='detector_data',
    )
    monitor_msg = Message(
        timestamp=200,
        stream=StreamId(kind=StreamKind.MONITOR_EVENTS, name='monitor1'),
        value='monitor_data',
    )

    with caplog.at_level(logging.WARNING):
        routing_sink.publish_messages([detector_msg, monitor_msg])

    assert len(detector_sink.published) == 1
    assert detector_sink.published[0] == detector_msg

    assert any(
        'No route configured' in record.message and 'monitor_events' in record.message
        for record in caplog.records
    )


def test_routing_sink_handles_empty_message_list():
    detector_sink = FakeSink()
    routing_sink = RoutingSink({StreamKind.DETECTOR_EVENTS: detector_sink})

    routing_sink.publish_messages([])

    assert len(detector_sink.published) == 0


def test_routing_sink_preserves_message_order_within_groups():
    detector_sink = FakeSink()

    routing_sink = RoutingSink({StreamKind.DETECTOR_EVENTS: detector_sink})

    messages = [
        Message(
            timestamp=100 + i,
            stream=StreamId(kind=StreamKind.DETECTOR_EVENTS, name=f'detector{i}'),
            value=f'data{i}',
        )
        for i in range(5)
    ]

    routing_sink.publish_messages(messages)

    assert len(detector_sink.published) == 5
    assert detector_sink.published == messages


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
