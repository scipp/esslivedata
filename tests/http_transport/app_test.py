# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc
from fastapi.testclient import TestClient

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.http_transport.app import create_message_api
from ess.livedata.http_transport.serialization import DA00MessageSerializer
from ess.livedata.http_transport.sink import QueueBasedMessageSink


def test_api_returns_messages():
    """Test that GET /messages returns messages from the sink."""
    sink = QueueBasedMessageSink[sc.DataArray](max_size=10)
    serializer = DA00MessageSerializer()
    app = create_message_api(sink, serializer)
    client = TestClient(app)

    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test')
    messages = [
        Message(timestamp=1000, stream=stream, value=sc.DataArray(sc.scalar(1.0))),
        Message(timestamp=2000, stream=stream, value=sc.DataArray(sc.scalar(2.0))),
    ]

    sink.publish_messages(messages)

    response = client.get('/messages')

    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/json'

    deserialized = serializer.deserialize(response.content)
    assert len(deserialized) == 2
    assert deserialized[0].timestamp == 1000
    assert deserialized[1].timestamp == 2000


def test_api_returns_no_content_when_empty():
    """Test that GET /messages returns 204 when no messages available."""
    sink = QueueBasedMessageSink[sc.DataArray](max_size=10)
    serializer = DA00MessageSerializer()
    app = create_message_api(sink, serializer)
    client = TestClient(app)

    response = client.get('/messages')

    assert response.status_code == 204


def test_api_clears_queue_after_get():
    """Test that messages are cleared after being retrieved."""
    sink = QueueBasedMessageSink[sc.DataArray](max_size=10)
    serializer = DA00MessageSerializer()
    app = create_message_api(sink, serializer)
    client = TestClient(app)

    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test')
    messages = [
        Message(timestamp=1000, stream=stream, value=sc.DataArray(sc.scalar(1.0)))
    ]

    sink.publish_messages(messages)

    first_response = client.get('/messages')
    second_response = client.get('/messages')

    assert first_response.status_code == 200
    assert second_response.status_code == 204


def test_health_endpoint():
    """Test that /health endpoint returns status."""
    sink = QueueBasedMessageSink[sc.DataArray](max_size=10)
    serializer = DA00MessageSerializer()
    app = create_message_api(sink, serializer)
    client = TestClient(app)

    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test')
    messages = [
        Message(timestamp=1000, stream=stream, value=sc.DataArray(sc.scalar(1.0)))
    ]
    sink.publish_messages(messages)

    response = client.get('/health')

    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['queue_size'] == 1
