# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import time
from queue import Queue

import pytest

from ess.livedata.kafka.message_adapter import FakeKafkaMessage
from ess.livedata.kafka.source import BackgroundMessageSource, KafkaMessageSource


class FakeKafkaConsumer:
    def consume(self, num_messages: int, timeout: float) -> list[FakeKafkaMessage]:
        return [
            FakeKafkaMessage(value='abc', topic="topic1"),
            FakeKafkaMessage(value='def', topic="topic2"),
            FakeKafkaMessage(value='xyz', topic="topic1"),
        ][:num_messages]


class ControllableKafkaConsumer:
    """A fake consumer that allows controlling what messages are returned."""

    def __init__(self):
        self.message_queue: Queue = Queue()
        self.consume_calls = 0
        self.should_raise = False
        self.exception_to_raise: Exception | None = None
        self.consume_delay = 0.0

    def add_messages(self, messages: list[FakeKafkaMessage]) -> None:
        """Add messages to be returned by consume."""
        for msg in messages:
            self.message_queue.put(msg)

    def consume(self, num_messages: int, timeout: float) -> list[FakeKafkaMessage]:
        self.consume_calls += 1

        if self.consume_delay > 0:
            time.sleep(self.consume_delay)

        if self.should_raise and self.exception_to_raise:
            raise self.exception_to_raise

        messages = []
        for _ in range(num_messages):
            try:
                msg = self.message_queue.get_nowait()
                messages.append(msg)
            except Exception:
                break
        return messages


def test_get_messages_returns_multiple() -> None:
    source = KafkaMessageSource(consumer=FakeKafkaConsumer())
    messages = source.get_messages()
    assert len(messages) == 3
    assert messages[0].topic() == "topic1"
    assert messages[0].value() == "abc"
    assert messages[1].topic() == "topic2"
    assert messages[1].value() == "def"
    assert messages[2].topic() == "topic1"
    assert messages[2].value() == "xyz"


def test_get_messages_returns_results_of_consume() -> None:
    source = KafkaMessageSource(consumer=FakeKafkaConsumer())
    messages = source.get_messages()
    # The FakeKafkaConsumer returns the same messages every time
    assert messages == source.get_messages()
    assert messages == source.get_messages()


def test_limit_number_of_consumed_messages() -> None:
    source = KafkaMessageSource(consumer=FakeKafkaConsumer(), num_messages=2)
    messages1 = source.get_messages()
    assert len(messages1) == 2
    assert messages1[0].topic() == "topic1"
    assert messages1[0].value() == "abc"
    assert messages1[1].topic() == "topic2"
    assert messages1[1].value() == "def"
    messages2 = source.get_messages()
    # The FakeKafkaConsumer returns the same messages every time
    assert messages1 == messages2


class TestBackgroundMessageSource:
    def test_context_manager_starts_and_stops_background_consumption(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that context manager properly starts and stops background
        consumption."""
        consumer = ControllableKafkaConsumer()
        test_messages = [FakeKafkaMessage(value=b'test', topic="topic1")]
        consumer.add_messages(test_messages)

        # Before context: no consumption should happen
        initial_calls = consumer.consume_calls
        time.sleep(0.02)
        assert consumer.consume_calls == initial_calls

        with BackgroundMessageSource(consumer, timeout=0.01) as source:
            # During context: consumption should be happening
            time.sleep(0.02)
            assert consumer.consume_calls > initial_calls

            # Should get the messages
            messages = source.get_messages()
            assert len(messages) == 1
            assert messages[0].value() == b'test'

        # After context: consumption should stop
        calls_at_exit = consumer.consume_calls
        time.sleep(0.02)
        # Should not have made more consume calls after exit
        assert consumer.consume_calls == calls_at_exit

        # Verify logging messages (structlog writes to stdout)
        captured = capsys.readouterr()
        assert "background_consumer_started" in captured.out
        assert "background_consumer_stopped" in captured.out

    def test_manual_start_stop_controls_background_consumption(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test manual start and stop of background consumption."""
        consumer = ControllableKafkaConsumer()
        source = BackgroundMessageSource(consumer, timeout=0.01)

        # Initially no consumption
        initial_calls = consumer.consume_calls
        time.sleep(0.02)
        assert consumer.consume_calls == initial_calls

        # After start: consumption begins
        source.start()
        time.sleep(0.02)
        calls_after_start = consumer.consume_calls
        assert calls_after_start > initial_calls
        captured = capsys.readouterr()
        assert "background_consumer_started" in captured.out

        # After stop: consumption ends
        source.stop()
        time.sleep(0.02)
        calls_after_stop = consumer.consume_calls
        # Should not have made significantly more calls after stop
        assert (
            calls_after_stop <= calls_after_start + 1
        )  # Allow for one more due to timing
        captured = capsys.readouterr()
        assert "background_consumer_stopped" in captured.out

    def test_multiple_starts_are_safe(self) -> None:
        """Test that calling start multiple times doesn't cause issues."""
        consumer = ControllableKafkaConsumer()
        source = BackgroundMessageSource(consumer, timeout=0.01)

        source.start()
        source.start()  # Should be safe
        source.start()  # Should be safe

        time.sleep(0.01)
        # Should still work normally
        assert consumer.consume_calls > 0

        source.stop()

    def test_multiple_stops_are_safe(self) -> None:
        """Test that calling stop multiple times doesn't cause issues."""
        consumer = ControllableKafkaConsumer()
        source = BackgroundMessageSource(consumer, timeout=0.01)

        source.start()
        time.sleep(0.01)

        source.stop()
        source.stop()  # Should be safe
        source.stop()  # Should be safe

    def test_get_messages_starts_consumption_automatically(self) -> None:
        """Test that get_messages starts background consumption if not started."""
        consumer = ControllableKafkaConsumer()
        test_messages = [FakeKafkaMessage(value=b'auto_start', topic="topic1")]
        consumer.add_messages(test_messages)

        source = BackgroundMessageSource(consumer, timeout=0.01)

        # get_messages should start consumption automatically
        initial_calls = consumer.consume_calls
        source.get_messages()

        # Wait a bit for background consumption to start
        time.sleep(0.02)
        assert consumer.consume_calls > initial_calls

        source.stop()

    def test_background_consumption_queues_messages(self) -> None:
        """Test that messages are consumed and queued in the background."""
        consumer = ControllableKafkaConsumer()
        test_messages = [
            FakeKafkaMessage(value=b'msg1', topic="topic1"),
            FakeKafkaMessage(value=b'msg2', topic="topic2"),
        ]
        consumer.add_messages(test_messages)

        with BackgroundMessageSource(consumer, timeout=0.01) as source:
            # Wait for background consumption
            time.sleep(0.02)

            messages = source.get_messages()
            assert len(messages) == 2
            assert messages[0].value() == b'msg1'
            assert messages[1].value() == b'msg2'

    def test_get_messages_returns_accumulated_batches(self) -> None:
        """Test that get_messages returns all batches accumulated since last call."""
        consumer = ControllableKafkaConsumer()

        # Add messages that will be consumed in separate batches
        batch1 = [FakeKafkaMessage(value=b'batch1_msg1', topic="topic1")]
        batch2 = [FakeKafkaMessage(value=b'batch2_msg1', topic="topic2")]
        consumer.add_messages(batch1 + batch2)

        with BackgroundMessageSource(consumer, num_messages=1, timeout=0.001) as source:
            # Wait for background consumption of both batches
            time.sleep(0.01)

            # Should get all messages from accumulated batches
            messages = source.get_messages()
            assert len(messages) == 2

            # Second call should return empty list as no new messages
            messages = source.get_messages()
            assert len(messages) == 0

    def test_queue_overflow_behavior(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test behavior when message queue overflows."""
        consumer = ControllableKafkaConsumer()

        # Create many messages to potentially overflow the queue
        many_messages = [
            FakeKafkaMessage(value=f'msg{i}', topic="topic1") for i in range(20)
        ]
        consumer.add_messages(many_messages)

        # Use small queue size to force overflow
        with BackgroundMessageSource(
            consumer, max_queue_size=2, num_messages=3, timeout=0.001
        ) as source:
            time.sleep(0.01)  # Let it consume and potentially overflow

            # Should still be able to get some messages
            messages = source.get_messages()
            # Should have gotten some messages, but maybe not all due to overflow
            assert len(messages) > 0

        # Check for overflow warning messages (structlog writes to stdout)
        captured = capsys.readouterr()
        assert "message_queue_full" in captured.out

    def test_error_handling_continues_consumption(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that errors in consume don't permanently stop consumption."""
        consumer = ControllableKafkaConsumer()

        # Set up to raise error initially
        consumer.should_raise = True
        consumer.exception_to_raise = RuntimeError("Test error")

        with BackgroundMessageSource(consumer, timeout=0.01) as source:
            # Wait for error to occur
            time.sleep(0.05)

            # Stop raising errors and add a message
            consumer.should_raise = False
            consumer.add_messages(
                [FakeKafkaMessage(value=b'after_error', topic="topic1")]
            )

            # Wait for recovery and message consumption
            time.sleep(0.01)

            # Should eventually get the message despite earlier errors
            messages = source.get_messages()
            # May take multiple attempts due to timing
            for _ in range(5):
                if messages:
                    break
                time.sleep(0.01)
                messages = source.get_messages()

            assert any(msg.value() == b'after_error' for msg in messages)

        # Verify error was logged (structlog writes to stdout)
        captured = capsys.readouterr()
        assert "background_consumer_error" in captured.out

    def test_no_messages_available_returns_empty_list(self) -> None:
        """Test behavior when no messages are available."""
        consumer = ControllableKafkaConsumer()
        # Don't add any messages

        with BackgroundMessageSource(consumer, timeout=0.01) as source:
            time.sleep(0.01)  # Wait a bit

            messages = source.get_messages()
            assert len(messages) == 0

    def test_context_manager_cleanup_on_exception(self) -> None:
        """Test that context manager properly cleans up even if exception occurs."""
        consumer = ControllableKafkaConsumer()

        calls_during = 0
        try:
            with BackgroundMessageSource(consumer, timeout=0.01):
                # Verify consumption started
                time.sleep(0.01)
                calls_during = consumer.consume_calls
                raise ValueError("Test exception")
        except ValueError:
            pass

        # After exception, consumption should have stopped
        time.sleep(0.01)
        calls_after = consumer.consume_calls
        # Should not make significantly more calls after context exit
        assert calls_after <= calls_during + 1  # Allow for timing

    def test_consume_calls_happen_in_background(self) -> None:
        """Test that consume is continuously called in the background."""
        consumer = ControllableKafkaConsumer()
        # Add a small delay to make consume calls more predictable
        consumer.consume_delay = 0.01

        with BackgroundMessageSource(consumer, timeout=0.005):
            initial_calls = consumer.consume_calls
            time.sleep(0.05)  # Let background thread run

            # Should have made multiple consume calls
            assert consumer.consume_calls > initial_calls + 3

    def test_custom_parameters_affect_behavior(self) -> None:
        """Test that custom parameters are respected."""
        consumer = ControllableKafkaConsumer()

        # Test with specific num_messages parameter
        consumer.add_messages(
            [FakeKafkaMessage(value=f'msg{i}', topic="topic1") for i in range(10)]
        )

        with BackgroundMessageSource(consumer, num_messages=3, timeout=0.01) as source:
            time.sleep(0.02)

            # Should consume in batches of 3 (though exact verification
            # depends on timing)
            messages = source.get_messages()
            # Should get some messages
            assert len(messages) > 0
