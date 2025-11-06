# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for NullTransport and null implementations."""

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.dashboard.transport import (
    NullMessageSink,
    NullMessageSource,
    NullTransport,
)


def test_null_message_source_returns_empty_list():
    """NullMessageSource always returns empty list."""
    source = NullMessageSource()
    assert source.get_messages() == []
    assert source.get_messages() == []  # Multiple calls also return empty


def test_null_message_sink_discards_messages():
    """NullMessageSink accepts messages but does nothing with them."""
    sink = NullMessageSink()
    messages = [
        Message(
            value="test",
            stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name="test"),
        )
    ]
    # Should not raise
    sink.publish_messages(messages)
    sink.publish_messages([])


def test_null_transport_provides_null_resources():
    """NullTransport returns resources with null implementations."""
    with NullTransport() as resources:
        # Should have all required attributes
        assert hasattr(resources, 'message_source')
        assert hasattr(resources, 'command_sink')
        assert hasattr(resources, 'roi_sink')

        # Message source returns no messages
        assert resources.message_source.get_messages() == []

        # Sinks accept messages without error
        test_msg = Message(
            value="test",
            stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name="test"),
        )
        resources.command_sink.publish_messages([test_msg])
        resources.roi_sink.publish_messages([test_msg])


def test_null_transport_context_manager_works():
    """NullTransport can be used as context manager."""
    transport = NullTransport()

    # Enter returns resources
    resources = transport.__enter__()
    assert resources is not None

    # Exit doesn't raise
    transport.__exit__(None, None, None)


def test_null_transport_start_stop():
    """NullTransport start() and stop() are no-ops."""
    transport = NullTransport()

    # start() and stop() should not raise
    transport.start()
    transport.stop()

    # Can call multiple times
    transport.start()
    transport.start()
    transport.stop()
    transport.stop()


def test_null_transport_lifecycle():
    """Test complete lifecycle with context manager and start/stop."""
    with NullTransport() as resources:
        # Resources should be available
        assert resources is not None
        assert resources.message_source is not None
        assert resources.command_sink is not None
        assert resources.roi_sink is not None

    # Can create and use again
    transport = NullTransport()
    with transport:
        transport.start()
        # Should be able to use resources
        transport.stop()
