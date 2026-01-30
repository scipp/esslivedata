# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import time
from queue import Queue

import pytest

from ess.livedata.kafka.message_adapter import FakeKafkaMessage
from ess.livedata.kafka.source import (
    BackgroundMessageSource,
    ConsumerHealthStatus,
    KafkaMessageSource,
    MultiConsumer,
)


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
            # Wait for at least one error to occur
            time.sleep(0.05)

            # Stop raising errors and add a message
            consumer.should_raise = False
            consumer.add_messages(
                [FakeKafkaMessage(value=b'after_error', topic="topic1")]
            )

            # Wait for recovery - backoff after first error is 0.5s
            messages = source.get_messages()
            for _ in range(20):  # Wait up to 1 second
                if messages:
                    break
                time.sleep(0.05)
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

    def test_circuit_breaker_stops_after_max_errors(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that circuit breaker stops consumption after max consecutive errors."""
        consumer = ControllableKafkaConsumer()
        consumer.should_raise = True
        consumer.exception_to_raise = RuntimeError("Persistent error")

        max_errors = 3
        with BackgroundMessageSource(
            consumer, timeout=0.01, max_consecutive_errors=max_errors
        ) as source:
            # Wait for circuit breaker to trigger
            # With backoff, this takes: 0.5 + 1.0 + 1.5 = 3 seconds minimum
            # Use shorter timeout since we're testing the circuit breaker
            time.sleep(0.5)  # Initial errors happen quickly

            # Check health status shows failure
            status = source.get_health_status()
            # Thread should have stopped due to circuit breaker
            # Give it time to stop
            for _ in range(50):
                if not status.thread_alive:
                    break
                time.sleep(0.1)
                status = source.get_health_status()

            assert status.consecutive_errors >= max_errors
            assert status.failure_reason is not None
            assert "Circuit breaker" in status.failure_reason

        # Verify circuit breaker was logged
        captured = capsys.readouterr()
        assert "consumer_circuit_breaker_triggered" in captured.out

    def test_circuit_breaker_resets_on_success(self) -> None:
        """Test that consecutive error count resets after successful consume."""
        consumer = ControllableKafkaConsumer()

        with BackgroundMessageSource(
            consumer, timeout=0.01, max_consecutive_errors=5
        ) as source:
            # Cause some errors
            consumer.should_raise = True
            consumer.exception_to_raise = RuntimeError("Temporary error")
            time.sleep(0.05)  # Let at least one error happen

            # Now stop errors and add messages
            consumer.should_raise = False
            consumer.add_messages([FakeKafkaMessage(value=b'success', topic="topic1")])

            # Wait for recovery - backoff after first error is 0.5s
            for _ in range(20):  # Wait up to 1 second
                status = source.get_health_status()
                if status.total_messages_consumed > 0:
                    break
                time.sleep(0.05)

            # Error count should have reset after successful consume
            status = source.get_health_status()
            assert status.consecutive_errors == 0
            assert status.is_healthy
            assert status.total_messages_consumed > 0

    def test_circuit_breaker_disabled_when_max_errors_zero(self) -> None:
        """Test that circuit breaker can be disabled by setting max_errors to 0."""
        consumer = ControllableKafkaConsumer()
        consumer.should_raise = True
        consumer.exception_to_raise = RuntimeError("Error")

        # Disable circuit breaker
        with BackgroundMessageSource(
            consumer, timeout=0.01, max_consecutive_errors=0
        ) as source:
            time.sleep(0.1)

            # Thread should still be alive despite errors
            status = source.get_health_status()
            assert status.thread_alive
            assert status.consecutive_errors > 0
            # No failure reason because circuit breaker is disabled
            assert status.failure_reason is None

    def test_is_healthy_returns_true_when_not_started(self) -> None:
        """Test that is_healthy returns True before starting."""
        consumer = ControllableKafkaConsumer()
        source = BackgroundMessageSource(consumer, timeout=0.01)

        assert source.is_healthy()

    def test_is_healthy_returns_true_during_normal_operation(self) -> None:
        """Test that is_healthy returns True during normal consumption."""
        consumer = ControllableKafkaConsumer()
        consumer.add_messages([FakeKafkaMessage(value=b'test', topic="topic1")])

        with BackgroundMessageSource(consumer, timeout=0.01) as source:
            time.sleep(0.02)
            assert source.is_healthy()

    def test_is_healthy_returns_false_after_circuit_breaker(self) -> None:
        """Test that is_healthy returns False after circuit breaker triggers."""
        consumer = ControllableKafkaConsumer()
        consumer.should_raise = True
        consumer.exception_to_raise = RuntimeError("Error")

        with BackgroundMessageSource(
            consumer, timeout=0.01, max_consecutive_errors=2
        ) as source:
            # Wait for circuit breaker
            time.sleep(0.5)

            # Eventually should be unhealthy
            for _ in range(20):
                if not source.is_healthy():
                    break
                time.sleep(0.05)

            assert not source.is_healthy()

    def test_is_healthy_returns_false_after_health_timeout(self) -> None:
        """Test that is_healthy returns False if no consume for too long."""
        consumer = ControllableKafkaConsumer()

        # Very short health timeout for testing
        with BackgroundMessageSource(
            consumer,
            timeout=0.01,
            health_timeout=0.1,
            max_consecutive_errors=0,  # Disable circuit breaker
        ) as source:
            time.sleep(0.02)  # Let first consume happen
            assert source.is_healthy()

            # Now cause errors so consume stops succeeding
            consumer.should_raise = True
            consumer.exception_to_raise = RuntimeError("Error")

            # Wait for health timeout
            time.sleep(0.2)
            assert not source.is_healthy()

    def test_get_health_status_returns_complete_status(self) -> None:
        """Test that get_health_status returns all expected fields."""
        consumer = ControllableKafkaConsumer()
        consumer.add_messages([FakeKafkaMessage(value=b'test', topic="topic1")])

        with BackgroundMessageSource(consumer, timeout=0.01) as source:
            time.sleep(0.02)

            status = source.get_health_status()

            assert isinstance(status, ConsumerHealthStatus)
            assert status.is_healthy is True
            assert status.thread_alive is True
            assert status.seconds_since_last_consume is not None
            assert status.seconds_since_last_consume < 1.0
            assert status.consecutive_errors == 0
            assert status.queue_depth >= 0
            assert status.total_messages_consumed >= 0
            assert status.total_batches_dropped >= 0
            assert status.failure_reason is None

    def test_get_health_status_tracks_totals(self) -> None:
        """Test that health status tracks total messages and dropped batches."""
        consumer = ControllableKafkaConsumer()

        # Add many messages to force queue overflow
        for i in range(30):
            consumer.add_messages([FakeKafkaMessage(value=f'msg{i}', topic="topic1")])

        with BackgroundMessageSource(
            consumer, max_queue_size=2, num_messages=1, timeout=0.001
        ) as source:
            time.sleep(0.05)

            status = source.get_health_status()
            assert status.total_messages_consumed > 0
            # With small queue size and many messages, some batches should be dropped
            assert status.total_batches_dropped > 0

    def test_get_consumer_lag_returns_none_for_simple_consumer(self) -> None:
        """Test that get_consumer_lag returns None for consumers without lag support."""
        consumer = ControllableKafkaConsumer()

        with BackgroundMessageSource(consumer, timeout=0.01) as source:
            # ControllableKafkaConsumer doesn't have assignment/get_watermark_offsets
            lag = source.get_consumer_lag()
            assert lag is None

    def test_error_handling_includes_consecutive_error_count_in_logs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that error logs include consecutive error count."""
        consumer = ControllableKafkaConsumer()
        consumer.should_raise = True
        consumer.exception_to_raise = RuntimeError("Test error")

        with BackgroundMessageSource(consumer, timeout=0.01, max_consecutive_errors=5):
            time.sleep(0.1)
            consumer.should_raise = False

        captured = capsys.readouterr()
        assert "consecutive_errors" in captured.out


class FakeTopicPartition:
    """Fake TopicPartition for testing lag support."""

    def __init__(self, topic: str, partition: int, offset: int = -1):
        self.topic = topic
        self.partition = partition
        self.offset = offset

    def __eq__(self, other):
        if not isinstance(other, FakeTopicPartition):
            return False
        return self.topic == other.topic and self.partition == other.partition

    def __hash__(self):
        return hash((self.topic, self.partition))


class LagSupportingConsumer:
    """A fake consumer that supports lag monitoring methods."""

    def __init__(self, topic: str, partitions: list[int]):
        self._topic = topic
        self._partitions = partitions
        self._positions: dict[tuple[str, int], int] = {}
        self._watermarks: dict[tuple[str, int], tuple[int, int]] = {}

        for p in partitions:
            self._positions[(topic, p)] = 0
            self._watermarks[(topic, p)] = (0, 100)

    def set_position(self, partition: int, offset: int) -> None:
        self._positions[(self._topic, partition)] = offset

    def set_watermarks(self, partition: int, low: int, high: int) -> None:
        self._watermarks[(self._topic, partition)] = (low, high)

    def consume(self, num_messages: int, timeout: float) -> list:
        return []

    def assignment(self) -> list[FakeTopicPartition]:
        return [FakeTopicPartition(self._topic, p) for p in self._partitions]

    def get_watermark_offsets(
        self, tp: FakeTopicPartition, timeout: float = 1.0
    ) -> tuple[int, int]:
        return self._watermarks[(tp.topic, tp.partition)]

    def position(
        self, partitions: list[FakeTopicPartition]
    ) -> list[FakeTopicPartition]:
        result = []
        for tp in partitions:
            offset = self._positions[(tp.topic, tp.partition)]
            result.append(FakeTopicPartition(tp.topic, tp.partition, offset))
        return result


class TestMultiConsumer:
    def test_assignment_aggregates_from_all_consumers(self) -> None:
        consumer1 = LagSupportingConsumer("topic1", [0, 1])
        consumer2 = LagSupportingConsumer("topic2", [0])

        multi = MultiConsumer([consumer1, consumer2])
        assignment = multi.assignment()

        assert len(assignment) == 3
        topics = {(tp.topic, tp.partition) for tp in assignment}
        assert topics == {("topic1", 0), ("topic1", 1), ("topic2", 0)}

    def test_get_watermark_offsets_delegates_to_owning_consumer(self) -> None:
        consumer1 = LagSupportingConsumer("topic1", [0])
        consumer1.set_watermarks(0, 10, 200)
        consumer2 = LagSupportingConsumer("topic2", [0])
        consumer2.set_watermarks(0, 5, 50)

        multi = MultiConsumer([consumer1, consumer2])

        tp1 = FakeTopicPartition("topic1", 0)
        tp2 = FakeTopicPartition("topic2", 0)

        assert multi.get_watermark_offsets(tp1) == (10, 200)
        assert multi.get_watermark_offsets(tp2) == (5, 50)

    def test_get_watermark_offsets_raises_for_unknown_partition(self) -> None:
        consumer = LagSupportingConsumer("topic1", [0])
        multi = MultiConsumer([consumer])

        unknown_tp = FakeTopicPartition("unknown", 0)
        with pytest.raises(ValueError, match="No consumer found"):
            multi.get_watermark_offsets(unknown_tp)

    def test_position_delegates_to_owning_consumers(self) -> None:
        consumer1 = LagSupportingConsumer("topic1", [0])
        consumer1.set_position(0, 42)
        consumer2 = LagSupportingConsumer("topic2", [0])
        consumer2.set_position(0, 99)

        multi = MultiConsumer([consumer1, consumer2])

        partitions = [FakeTopicPartition("topic1", 0), FakeTopicPartition("topic2", 0)]
        positions = multi.position(partitions)

        assert len(positions) == 2
        assert positions[0].offset == 42
        assert positions[1].offset == 99

    def test_consume_aggregates_from_all_consumers(self) -> None:
        consumer1 = ControllableKafkaConsumer()
        consumer1.add_messages([FakeKafkaMessage(value=b'msg1', topic="topic1")])
        consumer2 = ControllableKafkaConsumer()
        consumer2.add_messages([FakeKafkaMessage(value=b'msg2', topic="topic2")])

        multi = MultiConsumer([consumer1, consumer2])
        messages = multi.consume(10, 0.01)

        assert len(messages) == 2

    def test_works_with_mixed_consumers(self) -> None:
        """MultiConsumer works even if some consumers don't support lag methods."""
        lag_consumer = LagSupportingConsumer("topic1", [0])
        simple_consumer = ControllableKafkaConsumer()

        multi = MultiConsumer([lag_consumer, simple_consumer])

        # assignment should only include partitions from lag-supporting consumer
        assignment = multi.assignment()
        assert len(assignment) == 1
        assert assignment[0].topic == "topic1"


class TestBackgroundMessageSourceWithMultiConsumer:
    def test_get_consumer_lag_works_through_multi_consumer(self) -> None:
        """Test that get_consumer_lag works through MultiConsumer wrapper."""
        consumer1 = LagSupportingConsumer("topic1", [0])
        consumer1.set_position(0, 50)
        consumer1.set_watermarks(0, 0, 100)

        consumer2 = LagSupportingConsumer("topic2", [0])
        consumer2.set_position(0, 25)
        consumer2.set_watermarks(0, 0, 75)

        multi = MultiConsumer([consumer1, consumer2])

        with BackgroundMessageSource(multi, timeout=0.01) as source:
            time.sleep(0.02)
            lag = source.get_consumer_lag()

            assert lag is not None
            assert lag["total_lag"] == (100 - 50) + (75 - 25)  # 50 + 50 = 100
            assert "topic1:0" in lag
            assert lag["topic1:0"]["lag"] == 50
            assert "topic2:0" in lag
            assert lag["topic2:0"]["lag"] == 50
