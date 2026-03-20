# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from unittest.mock import patch

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.core.message_batcher import (
    DEESCALATION_HEADROOM_RATIO,
    DEESCALATION_IDLE_WINDOWS,
    DEESCALATION_UNDERLOAD_THRESHOLD,
    ESCALATION_OVERLOAD_THRESHOLD,
    AdaptiveMessageBatcher,
    SimpleMessageBatcher,
)


def make_message(timestamp_ns: int, value: str = "test") -> Message[str]:
    """Helper to create test messages with specific timestamps."""
    stream = StreamId(kind=StreamKind.DETECTOR_EVENTS, name="test")
    return Message(timestamp=timestamp_ns, stream=stream, value=value)


class TestSimpleMessageBatcher:
    def test_empty_messages_returns_none(self):
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        result = batcher.batch([])
        assert result is None

    def test_single_message_creates_initial_batch(self):
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        msg = make_message(1000)

        batch = batcher.batch([msg])

        assert batch is not None
        assert batch.start_time == 1000
        assert batch.end_time == 1000
        assert batch.messages == [msg]

    def test_multiple_messages_same_time_creates_initial_batch(self):
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        messages = [make_message(1000, f"msg{i}") for i in range(3)]

        batch = batcher.batch(messages)

        assert batch is not None
        assert batch.start_time == 1000
        assert batch.end_time == 1000
        assert batch.messages == messages

    def test_initial_batch_spans_all_timestamps(self):
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        messages = [
            make_message(1000, "early"),
            make_message(2000, "middle"),
            make_message(3000, "late"),
        ]

        batch = batcher.batch(messages)

        assert batch is not None
        assert batch.start_time == 1000
        assert batch.end_time == 3000
        assert len(batch.messages) == 3

    def test_second_batch_call_with_no_future_messages_returns_none(self):
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        # Create initial batch
        batcher.batch([make_message(1000)])

        # Second call with messages in current window (within 1s after end time 1000)
        result = batcher.batch(
            [make_message(1000 + 500_000_000)]
        )  # 0.5s after end time

        assert result is None

    def test_batch_alignment_after_initial_batch(self):
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        initial_end = 1000
        batch_length_ns = 1_000_000_000

        # Create initial batch ending at 1000
        batcher.batch([make_message(initial_end)])

        # Add message after batch boundary
        future_msg = make_message(initial_end + batch_length_ns + 100)
        batch = batcher.batch([future_msg])

        assert batch is not None
        assert batch.start_time == initial_end
        assert batch.end_time == initial_end + batch_length_ns
        assert batch.messages == []  # Empty batch, future message goes to next

    def test_nearly_ordered_messages_basic_case(self):
        """Test the core assumption: messages are nearly ordered."""
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        batch_length_ns = 1_000_000_000

        # Initial batch with end time at 1400
        initial_msgs = [make_message(1000 + i * 100) for i in range(5)]
        batch1 = batcher.batch(initial_msgs)

        # Nearly ordered follow-up messages (small out-of-order variations)
        # Active batch boundary is at 1400 + 1_000_000_000
        follow_up_msgs = [
            make_message(1400 + batch_length_ns + 50),  # Slightly future
            make_message(
                1400 + batch_length_ns - 10
            ),  # Slightly late (within active batch)
            make_message(1400 + batch_length_ns + 100),  # More future
        ]
        batch2 = batcher.batch(follow_up_msgs)

        assert batch1 is not None
        assert batch2 is not None
        # Late message should be included in returned batch (the completed active batch)
        assert len(batch2.messages) == 1  # Only the late message
        assert batch2.messages[0].timestamp == 1400 + batch_length_ns - 10

    def test_late_arriving_messages_included_in_current_batch(self):
        """
        Late messages go into the current batch even if timestamp suggests otherwise.
        """
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        batch_length_ns = 1_000_000_000

        # Initial batch
        batcher.batch([make_message(1000)])

        # Messages: one very late, one future
        messages = [
            make_message(500),  # Very late - should go to current batch
            make_message(1000 + batch_length_ns + 100),  # Future
        ]
        batch = batcher.batch(messages)

        assert batch is not None
        assert len(batch.messages) == 1
        assert batch.messages[0].timestamp == 500  # Late message included

    def test_batch_length_respected(self):
        batch_length_s = 2.0
        batcher = SimpleMessageBatcher(batch_length_s=batch_length_s)
        batch_length_ns = int(batch_length_s * 1_000_000_000)

        # Initial batch
        batcher.batch([make_message(1000)])

        # Message just at boundary
        boundary_msg = make_message(1000 + batch_length_ns)
        batch = batcher.batch([boundary_msg])

        assert batch is not None
        assert batch.end_time == 1000 + batch_length_ns

    def test_invalid_timestamp_filtered_out(self):
        """Messages with non-numeric timestamps should be filtered out."""
        batcher = SimpleMessageBatcher(batch_length_s=1.0)

        # Create message with invalid timestamp manually
        stream = StreamId(kind=StreamKind.DETECTOR_EVENTS, name="test")
        valid_msg = make_message(1000)
        invalid_msg = Message(timestamp="invalid", stream=stream, value="test")  # type: ignore[arg-type]

        batch = batcher.batch([valid_msg, invalid_msg])

        assert batch is not None
        assert len(batch.messages) == 1
        assert batch.messages[0] == valid_msg

    def test_all_invalid_timestamps_returns_none(self):
        """If all messages have invalid timestamps, should return None."""
        batcher = SimpleMessageBatcher(batch_length_s=1.0)

        stream = StreamId(kind=StreamKind.DETECTOR_EVENTS, name="test")
        invalid_msgs = [
            Message(timestamp="invalid1", stream=stream, value="test1"),  # type: ignore[arg-type]
            Message(timestamp=None, stream=stream, value="test2"),  # type: ignore[arg-type]
        ]

        batch = batcher.batch(invalid_msgs)
        assert batch is None

    def test_multiple_batches_progression(self):
        """Test progression through multiple batches."""
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        batch_length_ns = 1_000_000_000

        # Initial batch
        batch1 = batcher.batch([make_message(1000)])
        assert batch1 is not None

        # Second batch triggered by future message
        batch2 = batcher.batch([make_message(1000 + batch_length_ns + 100)])
        assert batch2 is not None

        # Third batch
        batch3 = batcher.batch([make_message(1000 + 2 * batch_length_ns + 100)])
        assert batch3 is not None

        assert all(b is not None for b in [batch1, batch2, batch3])
        assert batch1.end_time == 1000
        assert batch2.start_time == 1000
        assert batch2.end_time == 1000 + batch_length_ns
        assert batch3.start_time == 1000 + batch_length_ns
        assert batch3.end_time == 1000 + 2 * batch_length_ns

    def test_messages_accumulate_in_active_batch(self):
        """Messages within batch window should accumulate."""
        batcher = SimpleMessageBatcher(batch_length_s=1.0)

        # Initial batch
        batcher.batch([make_message(1000)])

        # Add messages to active batch (within window)
        batcher.batch([make_message(1000 + 500_000_000)])  # 0.5s later
        result = batcher.batch([make_message(1000 + 800_000_000)])  # 0.8s later

        # Should return None as no future messages trigger batch completion
        assert result is None

    def test_exact_boundary_conditions(self):
        """Test messages exactly at batch boundaries."""
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        batch_length_ns = 1_000_000_000

        # Initial batch
        batcher.batch([make_message(1000)])

        # Message exactly at boundary (should be future)
        exact_boundary = make_message(1000 + batch_length_ns)
        batch = batcher.batch([exact_boundary])

        assert batch is not None
        assert len(batch.messages) == 0  # Empty batch, boundary message goes to next

    def test_large_time_gaps(self):
        """Test behavior with large gaps in time."""
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        batch_length_ns = 1_000_000_000

        # Initial batch
        batcher.batch([make_message(1000)])

        # Message far in the future
        far_future = make_message(1000 + 10 * batch_length_ns)
        batch = batcher.batch([far_future])

        assert batch is not None
        assert batch.start_time == 1000
        assert batch.end_time == 1000 + batch_length_ns

    def test_zero_timestamp_messages(self):
        """Test behavior with zero timestamps."""
        batcher = SimpleMessageBatcher(batch_length_s=1.0)

        msg = make_message(0)
        batch = batcher.batch([msg])

        assert batch is not None
        assert batch.start_time == 0
        assert batch.end_time == 0
        assert batch.messages == [msg]

    def test_negative_timestamp_messages(self):
        """Test behavior with negative timestamps."""
        batcher = SimpleMessageBatcher(batch_length_s=1.0)

        msg = make_message(-1000)
        batch = batcher.batch([msg])

        assert batch is not None
        assert batch.start_time == -1000
        assert batch.end_time == -1000

    def test_very_small_batch_length(self):
        """Test with very small batch length."""
        batcher = SimpleMessageBatcher(batch_length_s=0.001)  # 1ms
        batch_length_ns = 1_000_000  # 1ms in ns

        # Initial batch
        batcher.batch([make_message(1000)])

        # Message just over boundary
        future_msg = make_message(1000 + batch_length_ns + 1)
        batch = batcher.batch([future_msg])

        assert batch is not None
        assert batch.end_time == 1000 + batch_length_ns

    def test_mixed_early_and_late_messages(self):
        """Test complex scenario with mix of early, on-time, and late messages."""
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        batch_length_ns = 1_000_000_000

        # Initial batch
        batcher.batch([make_message(5000)])

        # Mixed messages: some late, some future
        mixed_msgs = [
            make_message(4000),  # Late
            make_message(5000 + batch_length_ns - 100),  # Just before boundary
            make_message(3000),  # Very late
            make_message(5000 + batch_length_ns + 100),  # Future
            make_message(4500),  # Somewhat late
        ]

        batch = batcher.batch(mixed_msgs)

        assert batch is not None
        # Should include all late messages
        late_timestamps = [msg.timestamp for msg in batch.messages]
        expected_late = [4000, 5000 + batch_length_ns - 100, 3000, 4500]
        assert sorted(late_timestamps) == sorted(expected_late)

    def test_large_time_gaps_with_empty_batches(self):
        """Test that large time gaps produce all expected empty batches."""
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        batch_length_ns = 1_000_000_000

        # Initial batch ending at timestamp 1000
        batch1 = batcher.batch([make_message(1000)])
        assert batch1 is not None
        assert batch1.end_time == 1000

        # Message 5 batch lengths in the future
        # This should trigger 5 empty batches before the message is processed
        gap_message = make_message(1000 + 5 * batch_length_ns + 100)

        # First call should return first empty batch
        batch2 = batcher.batch([gap_message])
        assert batch2 is not None
        assert batch2.start_time == 1000
        assert batch2.end_time == 1000 + batch_length_ns
        assert len(batch2.messages) == 0

        # Subsequent calls should return more empty batches
        batch3 = batcher.batch([])
        assert batch3 is not None
        assert batch3.start_time == 1000 + batch_length_ns
        assert batch3.end_time == 1000 + 2 * batch_length_ns
        assert len(batch3.messages) == 0

        batch4 = batcher.batch([])
        assert batch4 is not None
        assert batch4.start_time == 1000 + 2 * batch_length_ns
        assert batch4.end_time == 1000 + 3 * batch_length_ns
        assert len(batch4.messages) == 0

        batch5 = batcher.batch([])
        assert batch5 is not None
        assert batch5.start_time == 1000 + 3 * batch_length_ns
        assert batch5.end_time == 1000 + 4 * batch_length_ns
        assert len(batch5.messages) == 0

        batch6 = batcher.batch([])
        assert batch6 is not None
        assert batch6.start_time == 1000 + 4 * batch_length_ns
        assert batch6.end_time == 1000 + 5 * batch_length_ns
        assert len(batch6.messages) == 0

        # Next call should return None (no more empty batches)
        # The gap message should now be in the active batch
        result = batcher.batch([])
        assert result is None
        batch7 = batcher.batch([make_message(1000 + 6 * batch_length_ns + 100)])
        assert batch7 is not None
        assert batch7.start_time == 1000 + 5 * batch_length_ns
        assert batch7.end_time == 1000 + 6 * batch_length_ns
        assert len(batch7.messages) == 1
        assert batch7.messages[0].timestamp == 1000 + 5 * batch_length_ns + 100

    def test_large_gap_single_call_returns_first_empty_batch(self):
        """Single call with a large gap message returns only the first empty batch."""
        batcher = SimpleMessageBatcher(batch_length_s=1.0)
        batch_length_ns = 1_000_000_000

        # Initial batch
        batcher.batch([make_message(1000)])

        # Message far in the future - should only return first empty batch
        far_future = make_message(1000 + 10 * batch_length_ns)
        batch = batcher.batch([far_future])

        assert batch is not None
        assert batch.start_time == 1000
        assert batch.end_time == 1000 + batch_length_ns
        assert len(batch.messages) == 0

        # The far future message should still be waiting
        # Subsequent empty calls should produce more empty batches
        next_batch = batcher.batch([])
        assert next_batch is not None
        assert next_batch.start_time == 1000 + batch_length_ns
        assert next_batch.end_time == 1000 + 2 * batch_length_ns
        assert len(next_batch.messages) == 0


class FakeClock:
    """Fake monotonic clock for testing time-based de-escalation."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _escalate_to_level(batcher: AdaptiveMessageBatcher, level: int) -> None:
    """Drive the batcher to the given level by reporting overloaded batches."""
    while batcher.state.level < level:
        window = batcher.batch_length_s
        for _ in range(ESCALATION_OVERLOAD_THRESHOLD):
            batcher.report_batch(100, processing_time_s=window * 1.5)


class TestAdaptiveMessageBatcher:
    def test_initial_state_is_level_zero(self):
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)
        assert batcher.state.level == 0
        assert batcher.state.batch_length_s == 1.0

    def test_delegates_to_inner_batcher(self):
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0)
        msg = make_message(1000)
        batch = batcher.batch([msg])
        assert batch is not None
        assert batch.messages == [msg]

    def test_escalates_after_consecutive_overloaded_batches(self):
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)

        for _ in range(ESCALATION_OVERLOAD_THRESHOLD):
            batcher.report_batch(100, processing_time_s=1.5)

        assert batcher.state.level == 1
        assert batcher.state.batch_length_s == 2.0

    def test_does_not_escalate_before_threshold(self):
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)

        for _ in range(ESCALATION_OVERLOAD_THRESHOLD - 1):
            batcher.report_batch(100, processing_time_s=1.5)

        assert batcher.state.level == 0

    def test_does_not_escalate_when_processing_fits(self):
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)

        for _ in range(20):
            batcher.report_batch(100, processing_time_s=0.8)

        assert batcher.state.level == 0

    def test_escalation_capped_at_max_level(self):
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)

        _escalate_to_level(batcher, 2)
        assert batcher.state.level == 2
        assert batcher.state.batch_length_s == 4.0

        # Further overloaded batches should not exceed max
        for _ in range(ESCALATION_OVERLOAD_THRESHOLD * 2):
            batcher.report_batch(100, processing_time_s=10.0)
        assert batcher.state.level == 2

    def test_deescalates_after_idle_duration(self):
        clock = FakeClock()
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)

        with patch('ess.livedata.core.message_batcher.time.monotonic', clock):
            _escalate_to_level(batcher, 1)
            assert batcher.state.level == 1

            # Idle for just under the threshold — no de-escalation
            clock.advance(DEESCALATION_IDLE_WINDOWS * 2.0 - 0.1)
            batcher.report_batch(None)
            assert batcher.state.level == 1

            # Cross the threshold
            clock.advance(0.2)
            batcher.report_batch(None)
            assert batcher.state.level == 0

    def test_does_not_deescalate_below_zero(self):
        clock = FakeClock()
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)

        with patch('ess.livedata.core.message_batcher.time.monotonic', clock):
            clock.advance(100.0)
            batcher.report_batch(None)
            assert batcher.state.level == 0

    def test_underloaded_batch_resets_overload_counter(self):
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)

        # Almost reach escalation threshold
        for _ in range(ESCALATION_OVERLOAD_THRESHOLD - 1):
            batcher.report_batch(100, processing_time_s=1.5)

        # One underloaded batch resets the overload counter
        batcher.report_batch(100, processing_time_s=0.3)

        # Need full threshold again
        for _ in range(ESCALATION_OVERLOAD_THRESHOLD - 1):
            batcher.report_batch(100, processing_time_s=1.5)
        assert batcher.state.level == 0

    def test_idle_cycles_do_not_reset_overload_counter(self):
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)

        # Almost reach escalation threshold
        for _ in range(ESCALATION_OVERLOAD_THRESHOLD - 1):
            batcher.report_batch(100, processing_time_s=1.5)

        # Idle cycles (polling between batches) do not reset counters
        batcher.report_batch(None)

        # One more overloaded batch completes the threshold
        batcher.report_batch(100, processing_time_s=1.5)
        assert batcher.state.level == 1

    def test_non_empty_batch_resets_idle_timer(self):
        clock = FakeClock()
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)

        with patch('ess.livedata.core.message_batcher.time.monotonic', clock):
            _escalate_to_level(batcher, 1)
            assert batcher.state.level == 1

            # Almost reach de-escalation time
            clock.advance(DEESCALATION_IDLE_WINDOWS * 2.0 - 0.1)
            batcher.report_batch(None)
            assert batcher.state.level == 1

            # A non-empty batch resets the idle timer
            batcher.report_batch(100, processing_time_s=1.5)

            # Now need the full idle duration again
            clock.advance(DEESCALATION_IDLE_WINDOWS * 2.0 - 0.1)
            batcher.report_batch(None)
            assert batcher.state.level == 1

    def test_empty_batches_excluded_from_counters(self):
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)

        # Interleave empty batches with overloaded — should not reset counter
        for _ in range(ESCALATION_OVERLOAD_THRESHOLD - 1):
            batcher.report_batch(100, processing_time_s=1.5)
            batcher.report_batch(0)

        batcher.report_batch(100, processing_time_s=1.5)
        assert batcher.state.level == 1

    def test_empty_batches_do_not_contribute_to_escalation(self):
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)

        for _ in range(ESCALATION_OVERLOAD_THRESHOLD * 3):
            batcher.report_batch(0)
        assert batcher.state.level == 0

    def test_deescalates_under_sustained_light_load(self):
        """De-escalation via underload: processing uses less than headroom ratio."""
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)
        _escalate_to_level(batcher, 1)
        assert batcher.state.level == 1

        # Report underloaded batches (processing < 70% of 2s window)
        underloaded_time = batcher.batch_length_s * DEESCALATION_HEADROOM_RATIO - 0.1
        for _ in range(DEESCALATION_UNDERLOAD_THRESHOLD):
            batcher.report_batch(100, processing_time_s=underloaded_time)

        assert batcher.state.level == 0

    def test_does_not_deescalate_without_enough_headroom(self):
        """No de-escalation when processing uses most of the window."""
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)
        _escalate_to_level(batcher, 1)
        assert batcher.state.level == 1

        # Processing at 80% of 2s window — above headroom threshold
        for _ in range(DEESCALATION_UNDERLOAD_THRESHOLD * 3):
            batcher.report_batch(100, processing_time_s=1.6)

        assert batcher.state.level == 1

    def test_multi_level_escalation_and_deescalation(self):
        clock = FakeClock()
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=3)

        with patch('ess.livedata.core.message_batcher.time.monotonic', clock):
            _escalate_to_level(batcher, 2)
            assert batcher.state.level == 2
            assert batcher.state.batch_length_s == 4.0

            # De-escalate via idle — one level at a time
            clock.advance(DEESCALATION_IDLE_WINDOWS * 4.0)
            batcher.report_batch(None)
            assert batcher.state.level == 1
            assert batcher.state.batch_length_s == 2.0

            clock.advance(DEESCALATION_IDLE_WINDOWS * 2.0)
            batcher.report_batch(None)
            assert batcher.state.level == 0
            assert batcher.state.batch_length_s == 1.0

    def test_state_reflects_custom_base_length(self):
        batcher = AdaptiveMessageBatcher(base_batch_length_s=0.5, max_level=2)
        assert batcher.state.batch_length_s == 0.5

        _escalate_to_level(batcher, 1)
        assert batcher.state.batch_length_s == 1.0

        _escalate_to_level(batcher, 2)
        assert batcher.state.batch_length_s == 2.0

    def test_no_oscillation_when_barely_keeping_up(self):
        """At 8s window, rapid idle cycles between batches should not de-escalate."""
        clock = FakeClock()
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=3)

        with patch('ess.livedata.core.message_batcher.time.monotonic', clock):
            _escalate_to_level(batcher, 3)
            assert batcher.state.level == 3

            # Simulate "barely keeping up": process batch in 7s, then 1s of idle
            for _ in range(10):
                clock.advance(7.0)
                batcher.report_batch(100, processing_time_s=7.0)
                for _ in range(10):
                    clock.advance(0.1)
                    batcher.report_batch(None)

            assert batcher.state.level == 3

    def test_overload_resets_underload_counter(self):
        batcher = AdaptiveMessageBatcher(base_batch_length_s=1.0, max_level=2)
        _escalate_to_level(batcher, 1)

        # Almost enough underloaded batches
        underloaded_time = batcher.batch_length_s * DEESCALATION_HEADROOM_RATIO - 0.1
        for _ in range(DEESCALATION_UNDERLOAD_THRESHOLD - 1):
            batcher.report_batch(100, processing_time_s=underloaded_time)

        # One overloaded batch resets the counter
        batcher.report_batch(100, processing_time_s=batcher.batch_length_s + 0.1)

        # Need full threshold again
        for _ in range(DEESCALATION_UNDERLOAD_THRESHOLD - 1):
            batcher.report_batch(100, processing_time_s=underloaded_time)
        assert batcher.state.level == 1
