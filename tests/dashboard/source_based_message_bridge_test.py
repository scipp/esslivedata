# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import json
from typing import Any
from unittest.mock import Mock

from confluent_kafka import KafkaError

from ess.livedata.config.models import ConfigKey
from ess.livedata.dashboard.source_based_message_bridge import (
    SourceBasedMessageBridge,
)


class FakeKafkaMessage:
    """Fake Kafka message for testing."""

    def __init__(self, key: bytes, value: bytes, error_code: int | None = None):
        self._key = key
        self._value = value
        self._error_code = error_code

    def key(self):
        return self._key

    def value(self):
        return self._value

    def error(self):
        if self._error_code is None:
            return None
        mock_error = Mock()
        mock_error.code.return_value = self._error_code
        return mock_error


class FakeBackgroundMessageSource:
    """Fake BackgroundMessageSource for testing."""

    def __init__(self):
        self.started = False
        self.stopped = False
        self.messages: list[FakeKafkaMessage] = []

    def start(self):
        self.started = True
        self.stopped = False

    def stop(self):
        self.stopped = True
        self.started = False

    def get_messages(self):
        messages = self.messages.copy()
        self.messages.clear()
        return messages

    def add_message(self, key: str, value: dict[str, Any]):
        """Helper to add a message to be consumed."""
        msg = FakeKafkaMessage(
            key=key.encode('utf-8'), value=json.dumps(value).encode('utf-8')
        )
        self.messages.append(msg)


class FakeProducer:
    """Fake Kafka producer for testing."""

    def __init__(self):
        self.produced_messages: list[dict] = []
        self.should_fail = False

    def produce(self, topic: str, key: bytes, value: bytes, **kwargs):
        if self.should_fail:
            raise RuntimeError("Producer failed")
        self.produced_messages.append(
            {'topic': topic, 'key': key.decode('utf-8'), 'value': value.decode('utf-8')}
        )

    def poll(self, timeout: float):
        pass  # Non-blocking poll


class TestSourceBasedMessageBridge:
    """Test the SourceBasedMessageBridge class."""

    def test_start_starts_background_source(self):
        """Test that start() starts the BackgroundMessageSource."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        bridge.start()

        assert source.started
        assert not source.stopped

    def test_stop_stops_background_source(self):
        """Test that stop() stops the BackgroundMessageSource."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        bridge.start()
        bridge.stop()

        assert source.stopped
        assert not source.started

    def test_publish_sends_message_to_kafka(self):
        """Test that publish sends messages through the producer."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "commands-topic")

        bridge.start()
        key = ConfigKey(source_name="test", service_name="service", key="param")
        bridge.publish(key, {"value": 123})

        assert len(producer.produced_messages) == 1
        msg = producer.produced_messages[0]
        assert msg['topic'] == "commands-topic"
        assert msg['key'] == str(key)
        assert json.loads(msg['value']) == {"value": 123}

    def test_publish_when_not_running_warns_and_does_not_send(self):
        """Test that publishing when not running doesn't crash."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        key = ConfigKey(source_name="test", service_name="service", key="param")
        bridge.publish(key, {"value": 123})

        # Should not send anything
        assert len(producer.produced_messages) == 0

    def test_pop_all_returns_empty_dict_when_no_messages(self):
        """Test that pop_all returns empty dict when no messages available."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        bridge.start()
        result = bridge.pop_all()

        assert result == {}

    def test_pop_all_decodes_and_returns_messages(self):
        """Test that pop_all decodes messages from BackgroundMessageSource."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        bridge.start()

        # Add message to source
        key = ConfigKey(source_name="test", service_name="service", key="param")
        source.add_message(str(key), {"value": 456})

        result = bridge.pop_all()

        assert key in result
        assert result[key] == {"value": 456}

    def test_pop_all_deduplicates_by_key(self):
        """Test that pop_all naturally deduplicates by key (last write wins)."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        bridge.start()

        # Add multiple messages with same key
        key = ConfigKey(source_name="test", service_name="service", key="param")
        source.add_message(str(key), {"value": 1})
        source.add_message(str(key), {"value": 2})
        source.add_message(str(key), {"value": 3})

        result = bridge.pop_all()

        # Should only have the latest value
        assert key in result
        assert result[key] == {"value": 3}

    def test_pop_all_handles_multiple_different_keys(self):
        """Test that pop_all handles multiple different keys correctly."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        bridge.start()

        key1 = ConfigKey(source_name="test1", service_name="service", key="param1")
        key2 = ConfigKey(source_name="test2", service_name="service", key="param2")
        source.add_message(str(key1), {"value": "first"})
        source.add_message(str(key2), {"value": "second"})

        result = bridge.pop_all()

        assert len(result) == 2
        assert result[key1] == {"value": "first"}
        assert result[key2] == {"value": "second"}

    def test_pop_all_skips_messages_with_partition_eof_error(self):
        """Test that messages with PARTITION_EOF error are skipped."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        bridge.start()

        # Add message with PARTITION_EOF error
        eof_msg = FakeKafkaMessage(
            key=b"test", value=b"{}", error_code=KafkaError._PARTITION_EOF
        )
        source.messages.append(eof_msg)

        # Add valid message
        key = ConfigKey(source_name="test", service_name="service", key="param")
        source.add_message(str(key), {"value": 123})

        result = bridge.pop_all()

        # Should only have the valid message
        assert len(result) == 1
        assert result[key] == {"value": 123}

    def test_pop_all_handles_decode_errors_gracefully(self):
        """Test that messages that fail to decode are logged and skipped."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        bridge.start()

        # Add invalid message (bad JSON)
        bad_msg = FakeKafkaMessage(key=b"bad-key", value=b"not-json")
        source.messages.append(bad_msg)

        # Add valid message
        key = ConfigKey(source_name="test", service_name="service", key="param")
        source.add_message(str(key), {"value": 123})

        result = bridge.pop_all()

        # Should only have the valid message
        assert len(result) == 1
        assert result[key] == {"value": 123}

    def test_publish_handles_producer_errors_gracefully(self):
        """Test that producer errors are logged and don't crash."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        producer.should_fail = True
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        bridge.start()
        key = ConfigKey(source_name="test", service_name="service", key="param")

        # Should not crash despite producer failure
        bridge.publish(key, {"value": 123})

        # Should not have sent anything
        assert len(producer.produced_messages) == 0

    def test_multiple_start_calls_are_safe(self):
        """Test that calling start multiple times is safe."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        bridge.start()
        bridge.start()
        bridge.start()

        # Should still be started
        assert source.started

    def test_multiple_stop_calls_are_safe(self):
        """Test that calling stop multiple times is safe."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        bridge.start()
        bridge.stop()
        bridge.stop()
        bridge.stop()

        # Should still be stopped
        assert source.stopped

    def test_start_stop_cycle(self):
        """Test that bridge can be started and stopped multiple times."""
        source = FakeBackgroundMessageSource()
        producer = FakeProducer()
        bridge = SourceBasedMessageBridge(source, producer, "test-topic")

        for i in range(3):
            bridge.start()
            assert source.started

            key = ConfigKey(source_name="test", service_name="service", key=f"cycle{i}")
            bridge.publish(key, {"cycle": i})

            bridge.stop()
            assert source.stopped

        # Should have published all messages
        assert len(producer.produced_messages) == 3
