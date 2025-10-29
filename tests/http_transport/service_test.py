# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: S104  # Binding to 0.0.0.0 is intentional for HTTP tests
"""Tests for HTTPMultiEndpointSink."""

import pytest
import requests

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.http_transport import (
    DA00MessageSerializer,
    GenericJSONMessageSerializer,
    HTTPMultiEndpointSink,
    StatusMessageSerializer,
)


def test_http_multi_endpoint_sink_starts_and_is_immediately_available():
    """Test that HTTP server is available immediately after start() returns."""
    sink = HTTPMultiEndpointSink(
        instrument='dummy',
        stream_serializers={
            StreamKind.LIVEDATA_DATA: DA00MessageSerializer(),
            StreamKind.LIVEDATA_STATUS: StatusMessageSerializer(),
            StreamKind.LIVEDATA_CONFIG: GenericJSONMessageSerializer(),
        },
        host='0.0.0.0',
        port=9876,  # Use unique port for test
    )

    try:
        sink.start()

        # Immediately try to connect to endpoints - should work
        response = requests.get('http://localhost:9876/livedata_data', timeout=1.0)
        assert response.status_code in (200, 204)

        response = requests.get('http://localhost:9876/livedata_heartbeat', timeout=1.0)
        assert response.status_code in (200, 204)

        response = requests.get('http://localhost:9876/livedata_commands', timeout=1.0)
        assert response.status_code in (200, 204)

    finally:
        sink.stop()


def test_http_multi_endpoint_sink_routes_messages_correctly():
    """Test that messages are routed to correct endpoints based on StreamKind."""
    sink = HTTPMultiEndpointSink(
        instrument='dummy',
        stream_serializers={
            StreamKind.LIVEDATA_DATA: GenericJSONMessageSerializer(),
            StreamKind.LIVEDATA_STATUS: GenericJSONMessageSerializer(),
            StreamKind.LIVEDATA_CONFIG: GenericJSONMessageSerializer(),
        },
        host='0.0.0.0',
        port=9877,  # Use unique port for test
    )

    try:
        sink.start()

        # Publish messages to different streams
        data_msg = Message(
            stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name='test'),
            value={'type': 'data', 'value': 42},
        )
        status_msg = Message(
            stream=StreamId(kind=StreamKind.LIVEDATA_STATUS, name=''),
            value={'type': 'status', 'running': True},
        )
        config_msg = Message(
            stream=StreamId(kind=StreamKind.LIVEDATA_CONFIG, name=''),
            value={'type': 'config', 'key': 'test_key'},
        )

        sink.publish_messages([data_msg, status_msg, config_msg])

        # Check data endpoint
        response = requests.get('http://localhost:9877/livedata_data', timeout=1.0)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['value']['type'] == 'data'

        # Check status endpoint
        response = requests.get('http://localhost:9877/livedata_heartbeat', timeout=1.0)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['value']['type'] == 'status'

        # Check config endpoint
        response = requests.get('http://localhost:9877/livedata_commands', timeout=1.0)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['value']['type'] == 'config'

    finally:
        sink.stop()


def test_http_multi_endpoint_sink_with_subset_of_streams():
    """Test sink with only subset of stream kinds configured."""
    sink = HTTPMultiEndpointSink(
        instrument='dummy',
        stream_serializers={
            StreamKind.LIVEDATA_CONFIG: GenericJSONMessageSerializer(),
        },
        host='0.0.0.0',
        port=9878,  # Use unique port for test
    )

    try:
        sink.start()

        # Only /livedata_commands endpoint should exist
        response = requests.get('http://localhost:9878/livedata_commands', timeout=1.0)
        assert response.status_code in (200, 204)

        # Publish config message - should work
        config_msg = Message(
            stream=StreamId(kind=StreamKind.LIVEDATA_CONFIG, name=''),
            value={'key': 'value'},
        )
        sink.publish_messages([config_msg])

        response = requests.get('http://localhost:9878/livedata_commands', timeout=1.0)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1

        # Publish data message - should be dropped with warning
        data_msg = Message(
            stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name='test'),
            value={'data': 'test'},
        )
        sink.publish_messages([data_msg])

        # Should not appear on any endpoint (if it existed)
        # Data endpoint doesn't exist since we didn't configure LIVEDATA_DATA

    finally:
        sink.stop()


def test_http_multi_endpoint_sink_context_manager():
    """Test HTTPMultiEndpointSink as context manager."""
    with HTTPMultiEndpointSink(
        instrument='dummy',
        stream_serializers={
            StreamKind.LIVEDATA_DATA: GenericJSONMessageSerializer(),
        },
        host='0.0.0.0',
        port=9879,  # Use unique port for test
    ) as sink:
        # Server should be ready
        response = requests.get('http://localhost:9879/livedata_data', timeout=1.0)
        assert response.status_code in (200, 204)

        # Publish and retrieve
        msg = Message(
            stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name='test'),
            value={'x': 1},
        )
        sink.publish_messages([msg])

        response = requests.get('http://localhost:9879/livedata_data', timeout=1.0)
        assert response.status_code == 200

    # After exit, connection should fail (server stopped)
    with pytest.raises(requests.ConnectionError):
        requests.get('http://localhost:9879/livedata_data', timeout=0.5)


def test_http_multi_endpoint_sink_multiple_streams_same_topic():
    """Test that multiple StreamKinds mapping to same topic share endpoint."""
    # MONITOR_COUNTS and MONITOR_EVENTS both map to beam_monitor topic
    # They must use the same serializer instance
    serializer = GenericJSONMessageSerializer()
    sink = HTTPMultiEndpointSink(
        instrument='dummy',
        stream_serializers={
            StreamKind.MONITOR_COUNTS: serializer,
            StreamKind.MONITOR_EVENTS: serializer,
        },
        host='0.0.0.0',
        port=9880,  # Use unique port for test
    )

    try:
        sink.start()

        # Both message types should route to same endpoint
        counts_msg = Message(
            stream=StreamId(kind=StreamKind.MONITOR_COUNTS, name='mon1'),
            value={'type': 'counts'},
        )
        events_msg = Message(
            stream=StreamId(kind=StreamKind.MONITOR_EVENTS, name='mon1'),
            value={'type': 'events'},
        )

        sink.publish_messages([counts_msg, events_msg])

        # Both should appear on /beam_monitor endpoint
        response = requests.get('http://localhost:9880/beam_monitor', timeout=1.0)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    finally:
        sink.stop()


def test_http_multi_endpoint_sink_endpoint_naming():
    """Test that endpoints are correctly named after removing instrument prefix."""
    sink = HTTPMultiEndpointSink(
        instrument='bifrost',  # Different instrument
        stream_serializers={
            StreamKind.LIVEDATA_ROI: GenericJSONMessageSerializer(),
        },
        host='0.0.0.0',
        port=9881,  # Use unique port for test
    )

    try:
        sink.start()

        # Endpoint should be /livedata_roi (not /bifrost_livedata_roi)
        response = requests.get('http://localhost:9881/livedata_roi', timeout=1.0)
        assert response.status_code in (200, 204)

    finally:
        sink.stop()
