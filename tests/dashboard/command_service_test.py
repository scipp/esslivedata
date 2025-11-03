# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.config.models import ConfigKey
from ess.livedata.core.message import COMMANDS_STREAM_ID, Message
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.handlers.config_handler import ConfigUpdate


class FakeMessageSink:
    """Fake message sink that preserves batch boundaries for testing.

    Unlike the generic FakeMessageSink in fakes.py, this implementation stores
    each publish_messages() call as a separate batch. This is necessary to verify
    batching behavior in CommandService (e.g., that send() creates one batch,
    send_batch() creates one batch, and empty batches don't publish).
    """

    def __init__(self):
        self.published_messages = []

    def publish_messages(self, messages: list[Message]) -> None:
        """Record published messages as a batch."""
        self.published_messages.append(messages)


@pytest.fixture
def fake_sink() -> FakeMessageSink:
    """Create a fake message sink."""
    return FakeMessageSink()


@pytest.fixture
def command_service(fake_sink: FakeMessageSink) -> CommandService:
    """Create a command service with fake sink."""
    return CommandService(sink=fake_sink)


class TestCommandService:
    def test_send_single_command(
        self, command_service: CommandService, fake_sink: FakeMessageSink
    ):
        """Test sending a single command."""
        key = ConfigKey(key="test_key", source_name="test_source")
        value = {"param": "value"}

        command_service.send(key, value)

        # Should have published exactly one batch
        assert len(fake_sink.published_messages) == 1

        # Batch should contain one message
        messages = fake_sink.published_messages[0]
        assert len(messages) == 1

        # Verify message content
        msg = messages[0]
        assert msg.stream == COMMANDS_STREAM_ID
        assert isinstance(msg.value, ConfigUpdate)
        assert msg.value.config_key == key
        assert msg.value.value == value

    def test_send_batch(
        self, command_service: CommandService, fake_sink: FakeMessageSink
    ):
        """Test sending a batch of commands."""
        commands = [
            (ConfigKey(key="key1", source_name="source1"), {"param1": "value1"}),
            (ConfigKey(key="key2", source_name="source2"), {"param2": "value2"}),
            (ConfigKey(key="key3", source_name="source3"), {"param3": "value3"}),
        ]

        command_service.send_batch(commands)

        # Should have published exactly one batch
        assert len(fake_sink.published_messages) == 1

        # Batch should contain all messages
        messages = fake_sink.published_messages[0]
        assert len(messages) == 3

        # Verify each message
        for i, (key, value) in enumerate(commands):
            msg = messages[i]
            assert msg.stream == COMMANDS_STREAM_ID
            assert isinstance(msg.value, ConfigUpdate)
            assert msg.value.config_key == key
            assert msg.value.value == value

    def test_send_empty_batch(
        self, command_service: CommandService, fake_sink: FakeMessageSink
    ):
        """Test sending an empty batch doesn't publish."""
        command_service.send_batch([])

        # Should not have published anything
        assert len(fake_sink.published_messages) == 0

    def test_multiple_sends_create_multiple_batches(
        self, command_service: CommandService, fake_sink: FakeMessageSink
    ):
        """Test that multiple send() calls create separate batches."""
        key1 = ConfigKey(key="key1", source_name="source1")
        key2 = ConfigKey(key="key2", source_name="source2")

        command_service.send(key1, {"value": 1})
        command_service.send(key2, {"value": 2})

        # Should have published two separate batches
        assert len(fake_sink.published_messages) == 2

        # Each batch should contain one message
        assert len(fake_sink.published_messages[0]) == 1
        assert len(fake_sink.published_messages[1]) == 1
