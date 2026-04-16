# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for rate-aware message batcher.

Tests use 'seconds' as the natural unit: timestamps are in seconds (converted
to nanoseconds via helpers), batch lengths are in seconds.
"""

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.core.rate_aware_batcher import (
    MIN_BATCHES_FOR_GATE,
    RateAwareMessageBatcher,
)
from ess.livedata.core.timestamp import Timestamp

# --- Helpers ---

DETECTOR = StreamId(kind=StreamKind.DETECTOR_EVENTS, name="det")
MONITOR = StreamId(kind=StreamKind.MONITOR_EVENTS, name="mon")


def ts(seconds: float) -> Timestamp:
    return Timestamp.from_ns(int(seconds * 1e9))


def msg(t: float, stream: StreamId = DETECTOR, value: str = "") -> Message[str]:
    return Message(timestamp=ts(t), stream=stream, value=value)


def msgs_at(
    rate_hz: float,
    start: float,
    duration: float,
    stream: StreamId = DETECTOR,
) -> list[Message[str]]:
    """Generate perfectly-spaced messages for a stream."""
    count = round(rate_hz * duration)
    period = 1.0 / rate_hz
    return [msg(start + i * period, stream) for i in range(count)]


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_converged_batcher(
    rate_hz: float = 14.0,
    batch_length_s: float = 1.0,
    streams: dict[StreamId, float] | None = None,
    timeout_s: float | None = None,
    clock: FakeClock | None = None,
) -> tuple[RateAwareMessageBatcher, float]:
    """Create a batcher with converged rate estimates.

    Returns (batcher, next_batch_start) where next_batch_start is the
    start time of the currently active (empty) batch, so callers can
    generate aligned messages.

    Parameters
    ----------
    streams:
        Map of stream_id -> rate_hz. If None, uses a single DETECTOR stream
        at ``rate_hz``.
    """
    if streams is None:
        streams = {DETECTOR: rate_hz}
    if clock is None:
        clock = FakeClock()

    batcher = RateAwareMessageBatcher(
        batch_length_s=batch_length_s,
        clock=clock,
        ema_alpha=1.0,
        timeout_s=timeout_s if timeout_s is not None else batch_length_s * 1.5,
    )

    # Initial batch seeds the timeline
    initial: list[Message[str]] = []
    for sid, r in streams.items():
        initial.extend(msgs_at(r, start=0.0, duration=batch_length_s, stream=sid))
    batcher.batch(initial)

    # The initial batch end_time is max(timestamps).
    # For a single stream at rate R, that's (round(R*T)-1)/R.
    max_ts = max(m.timestamp for m in initial)
    batch_start = max_ts.to_ns() / 1e9

    # Feed batches using timeout to build up observation count
    for i in range(MIN_BATCHES_FOR_GATE):
        t0 = batch_start + i * batch_length_s
        batch_msgs: list[Message[str]] = []
        for sid, r in streams.items():
            batch_msgs.extend(msgs_at(r, start=t0, duration=batch_length_s, stream=sid))
        batcher.batch(batch_msgs)
        clock.advance(batcher._timeout_s + 0.01)
        result = batcher.batch([])
        assert result is not None, f"Timeout batch {i} should have closed"

    next_batch_start = batch_start + MIN_BATCHES_FOR_GATE * batch_length_s
    return batcher, next_batch_start


# --- Tests ---


class TestEmptyInput:
    def test_no_messages_returns_none(self):
        batcher = RateAwareMessageBatcher()
        assert batcher.batch([]) is None

    def test_repeated_empty_returns_none(self):
        batcher = RateAwareMessageBatcher()
        for _ in range(10):
            assert batcher.batch([]) is None


class TestInitialBatch:
    def test_first_messages_returned_immediately(self):
        batcher = RateAwareMessageBatcher()
        messages = msgs_at(14.0, start=0.0, duration=1.0)
        batch = batcher.batch(messages)
        assert batch is not None
        assert len(batch.messages) == 14

    def test_initial_batch_timestamps(self):
        batcher = RateAwareMessageBatcher()
        messages = [msg(1.0), msg(1.5), msg(2.0)]
        batch = batcher.batch(messages)
        assert batch is not None
        assert batch.start_time == ts(1.0)
        assert batch.end_time == ts(2.0)


class TestSingleStreamCompletion:
    """Core behavior: batch completes when last slot is filled."""

    def test_batch_completes_when_last_slot_filled(self):
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        messages = msgs_at(14.0, start=t0, duration=1.0)
        result = batcher.batch(messages)
        assert result is not None
        assert len(result.messages) == 14

    def test_batch_does_not_complete_without_last_slot(self):
        clock = FakeClock()
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0, clock=clock)
        # Only first 6 of 14 messages — well before last slot
        partial = msgs_at(14.0, start=t0, duration=1.0)[:6]
        result = batcher.batch(partial)
        assert result is None

    def test_missing_pulse_does_not_block_completion(self):
        """If pulse 7 is missing, seeing pulse 14 still completes the batch."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        messages = msgs_at(14.0, start=t0, duration=1.0)
        del messages[6]  # Remove pulse 7
        assert len(messages) == 13
        result = batcher.batch(messages)
        assert result is not None
        assert len(result.messages) == 13

    def test_split_message_does_not_cause_premature_completion(self):
        """Two messages with the same early timestamp don't trick the slot check."""
        clock = FakeClock()
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0, clock=clock)
        # Only first 6 messages, but duplicate one of them
        partial = msgs_at(14.0, start=t0, duration=1.0)[:6]
        dup = msg(partial[2].timestamp.to_ns() / 1e9)
        partial.insert(2, dup)
        result = batcher.batch(partial)
        assert result is None

    def test_split_on_last_slot_still_completes(self):
        """Split (duplicate) on the last slot still completes."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        messages = msgs_at(14.0, start=t0, duration=1.0)
        # Duplicate the last message
        messages.append(msg(messages[-1].timestamp.to_ns() / 1e9))
        result = batcher.batch(messages)
        assert result is not None


class TestTimeout:
    def test_timeout_closes_incomplete_batch(self):
        clock = FakeClock()
        batcher = RateAwareMessageBatcher(
            batch_length_s=1.0,
            timeout_s=0.5,
            clock=clock,
            ema_alpha=1.0,
        )
        # Initial batch
        batcher.batch([msg(0.0)])

        # Feed some messages but not enough to fill last slot
        batcher.batch([msg(1.1), msg(1.2)])
        assert batcher.batch([]) is None

        # Advance past timeout
        clock.advance(0.6)
        batch = batcher.batch([])
        assert batch is not None
        assert len(batch.messages) == 2


class TestMultiStream:
    def test_waits_for_all_gated_streams(self):
        clock = FakeClock()
        streams = {DETECTOR: 14.0, MONITOR: 1.0}
        batcher, t0 = make_converged_batcher(
            streams=streams, timeout_s=999.0, clock=clock
        )

        # Feed only detector messages — should not complete
        det_full = msgs_at(14.0, start=t0, duration=1.0, stream=DETECTOR)
        result = batcher.batch(det_full)
        assert result is None

        # Feed monitor message → now all streams satisfied → complete
        mon_full = msgs_at(1.0, start=t0, duration=1.0, stream=MONITOR)
        result = batcher.batch(mon_full)
        assert result is not None


class TestOverflow:
    def test_overflow_messages_appear_in_next_batch(self):
        """Messages past the last slot go to the next batch."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0)

        # 14 messages for this batch + 3 for the next
        current = msgs_at(14.0, start=t0, duration=1.0)
        future = msgs_at(14.0, start=t0 + 1.0, duration=1.0)[:3]
        batch1 = batcher.batch(current + future)

        assert batch1 is not None
        assert len(batch1.messages) == 14

        # Overflow should be available for the next batch
        remaining = msgs_at(14.0, start=t0 + 1.0, duration=1.0)[3:]
        batch2 = batcher.batch(remaining)
        assert batch2 is not None
        assert len(batch2.messages) == 14


class TestPhaseOffset:
    """Streams whose first pulse doesn't align with batch_start."""

    def _converge_with_offset(
        self,
        rate_hz: float = 14.0,
        offset: float = 0.04,
        batch_length_s: float = 1.0,
        clock: FakeClock | None = None,
        timeout_s: float = 999.0,
    ) -> tuple[RateAwareMessageBatcher, float]:
        """Converge a single DETECTOR stream that has a phase offset.

        Returns (batcher, next_batch_start).
        """
        if clock is None:
            clock = FakeClock()
        period = 1.0 / rate_hz
        count = round(rate_hz * batch_length_s)

        batcher = RateAwareMessageBatcher(
            batch_length_s=batch_length_s,
            clock=clock,
            ema_alpha=1.0,
            timeout_s=timeout_s,
        )

        def stream_msgs(batch_start: float) -> list[Message[str]]:
            return [msg(batch_start + offset + i * period) for i in range(count)]

        # Initial batch
        initial = stream_msgs(0.0)
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        # Converge via timeout
        for i in range(MIN_BATCHES_FOR_GATE):
            t0 = batch_start + i * batch_length_s
            batcher.batch(stream_msgs(t0))
            clock.advance(timeout_s + 0.01)
            result = batcher.batch([])
            assert result is not None, f"Convergence batch {i} should close"

        next_start = batch_start + MIN_BATCHES_FOR_GATE * batch_length_s
        return batcher, next_start

    def test_offset_does_not_cause_overflow(self):
        """14 Hz stream with 0.04s offset (~56% of period): all 14 in one batch."""
        offset = 0.04
        batcher, t0 = self._converge_with_offset(offset=offset)
        period = 1.0 / 14.0
        test_msgs = [msg(t0 + offset + i * period) for i in range(14)]
        result = batcher.batch(test_msgs)
        assert result is not None, "Batch should complete via slot gate"
        assert len(result.messages) == 14

    def test_two_streams_different_offsets(self):
        """Two gated streams with different phase offsets complete independently."""
        clock = FakeClock()
        det_offset = 0.04
        mon_offset = 0.01
        det_rate = 14.0
        mon_rate = 5.0

        batcher = RateAwareMessageBatcher(
            batch_length_s=1.0, clock=clock, ema_alpha=1.0, timeout_s=999.0
        )

        def det_msgs(batch_start: float) -> list[Message[str]]:
            return [
                msg(batch_start + det_offset + i / det_rate, DETECTOR)
                for i in range(round(det_rate))
            ]

        def mon_msgs(batch_start: float) -> list[Message[str]]:
            return [
                msg(batch_start + mon_offset + i / mon_rate, MONITOR)
                for i in range(round(mon_rate))
            ]

        # Initial batch
        initial = det_msgs(0.0) + mon_msgs(0.0)
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        # Converge via timeout
        for i in range(MIN_BATCHES_FOR_GATE):
            t0 = batch_start + i * 1.0
            batcher.batch(det_msgs(t0) + mon_msgs(t0))
            clock.advance(1000.0)
            result = batcher.batch([])
            assert result is not None

        # Post-convergence: both streams present
        t_test = batch_start + MIN_BATCHES_FOR_GATE * 1.0
        result = batcher.batch(det_msgs(t_test) + mon_msgs(t_test))
        assert result is not None
        assert len(result.messages) == 14 + 5

    def test_offset_near_half_period(self):
        """Phase offset at exactly half a pulse period — worst case for rounding."""
        rate = 14.0
        period = 1.0 / rate
        # Just under half period to avoid ambiguity
        offset = period * 0.49
        batcher, t0 = self._converge_with_offset(rate_hz=rate, offset=offset)
        test_msgs = [msg(t0 + offset + i * period) for i in range(14)]
        result = batcher.batch(test_msgs)
        assert result is not None
        assert len(result.messages) == 14


class TestJitterResilience:
    """Verify slot assignment is stable under realistic timestamp jitter."""

    def test_moderate_jitter_preserves_message_count(self):
        """14 Hz with +/- 10ms jitter: all 14 messages land in one batch."""
        import random

        rng = random.Random(42)
        rate = 14.0
        period = 1.0 / rate
        jitter_max = 0.010  # 10ms — ~14% of period

        batcher, t0 = make_converged_batcher(rate_hz=rate)
        test_msgs = [
            msg(t0 + i * period + rng.uniform(-jitter_max, jitter_max))
            for i in range(14)
        ]
        result = batcher.batch(test_msgs)
        assert result is not None
        assert len(result.messages) == 14

    def test_average_count_over_many_batches(self):
        """Over 50 batches with jitter, average message count is 14."""
        import random

        rng = random.Random(123)
        rate = 14.0
        period = 1.0 / rate
        jitter_max = 0.010
        n_batches = 50

        clock = FakeClock()
        batcher, t0 = make_converged_batcher(rate_hz=rate, timeout_s=999.0, clock=clock)

        counts: list[int] = []
        for b in range(n_batches):
            batch_t0 = t0 + b * 1.0
            batch_msgs = [
                msg(batch_t0 + i * period + rng.uniform(-jitter_max, jitter_max))
                for i in range(14)
            ]
            result = batcher.batch(batch_msgs)
            if result is not None:
                counts.append(len(result.messages))
            else:
                # Force close via timeout so we can continue
                clock.advance(1000.0)
                result = batcher.batch([])
                if result is not None:
                    counts.append(len(result.messages))

        avg = sum(counts) / len(counts)
        assert abs(avg - 14.0) < 0.5, f"Average {avg} too far from 14.0"


class TestOneHzEdgeCase:
    """1 Hz monitor with 1s batch: single-slot edge case."""

    def test_single_message_completes_batch(self):
        batcher, t0 = make_converged_batcher(rate_hz=1.0)
        result = batcher.batch([msg(t0)])
        assert result is not None
        assert len(result.messages) == 1

    def test_overflow_at_slot_1(self):
        """Message from next batch period goes to overflow, not current batch."""
        batcher, t0 = make_converged_batcher(rate_hz=1.0)
        result = batcher.batch([msg(t0), msg(t0 + 1.0)])
        assert result is not None
        assert len(result.messages) == 1


class TestTimeGaps:
    """Stream pauses and resumes — batch window must advance past gaps."""

    def test_gap_of_5_batches_recovers_without_timeout(self):
        """After a 5s gap, the batcher advances and delivers the next batch."""
        clock = FakeClock()
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0, clock=clock)

        # Skip 5 batch periods, then send normal-rate messages
        gap_batches = 5
        t_resume = t0 + gap_batches * 1.0
        resume_msgs = msgs_at(14.0, start=t_resume, duration=1.0)
        result = batcher.batch(resume_msgs)

        # Should produce a batch with the resumed data, not require timeout
        assert result is not None
        assert len(result.messages) == 14

    def test_gap_preserves_batch_continuity(self):
        """After a gap, the next batch starts at or before the resumed data."""
        clock = FakeClock()
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0, clock=clock)

        gap_batches = 10
        t_resume = t0 + gap_batches * 1.0
        resume_msgs = msgs_at(14.0, start=t_resume, duration=1.0)
        result = batcher.batch(resume_msgs)
        assert result is not None
        # Batch start should be at or before the first resumed message
        assert result.start_time <= resume_msgs[0].timestamp


class TestStreamLifecycle:
    """Streams can appear mid-run or disappear; eviction prevents blocking."""

    def test_new_stream_joins_gate_after_convergence(self):
        """A stream that appears after initial convergence eventually gates."""
        clock = FakeClock()
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=1.5, clock=clock)

        # Introduce a second stream (MONITOR at 5 Hz)
        for i in range(MIN_BATCHES_FOR_GATE):
            batch_t = t0 + i * 1.0
            det = msgs_at(14.0, start=batch_t, duration=1.0, stream=DETECTOR)
            mon = msgs_at(5.0, start=batch_t, duration=1.0, stream=MONITOR)
            batcher.batch(det + mon)
            clock.advance(2.0)
            result = batcher.batch([])
            assert result is not None

        # Now both streams should be gated — sending only detector should NOT
        # complete the batch (monitor is missing)
        t_test = t0 + MIN_BATCHES_FOR_GATE * 1.0
        result = batcher.batch(
            msgs_at(14.0, start=t_test, duration=1.0, stream=DETECTOR)
        )
        assert result is None

    def test_disappeared_stream_evicted_after_n_batches(self):
        """A stream absent for several batches is evicted from the gate."""
        from ess.livedata.core.rate_aware_batcher import ABSENT_BATCHES_FOR_EVICTION

        clock = FakeClock()
        streams = {DETECTOR: 14.0, MONITOR: 5.0}
        batcher, t0 = make_converged_batcher(
            streams=streams, timeout_s=1.5, clock=clock
        )

        # Close batches with only detector present. Each closes via timeout
        # since monitor is still gated and blocking slot-based completion.
        for i in range(ABSENT_BATCHES_FOR_EVICTION + 1):
            det = msgs_at(14.0, start=t0 + i * 1.0, duration=1.0, stream=DETECTOR)
            batcher.batch(det)
            clock.advance(2.0)
            batcher.batch([])

        # Batch window may not align with simple arithmetic (extra closes
        # once monitor is evicted). Use a future timestamp — gap recovery
        # will advance the window.
        t_test = t0 + 100.0
        result = batcher.batch(
            msgs_at(14.0, start=t_test, duration=1.0, stream=DETECTOR)
        )
        assert result is not None, (
            "Detector-only batch should complete after monitor eviction"
        )

    def test_evicted_stream_reappears(self):
        """An evicted stream can re-enter the gate after re-convergence."""
        from ess.livedata.core.rate_aware_batcher import ABSENT_BATCHES_FOR_EVICTION

        clock = FakeClock()
        streams = {DETECTOR: 14.0, MONITOR: 5.0}
        batcher, t0 = make_converged_batcher(
            streams=streams, timeout_s=1.5, clock=clock
        )

        # Evict monitor
        for i in range(ABSENT_BATCHES_FOR_EVICTION + 1):
            batcher.batch(msgs_at(14.0, start=t0 + i, duration=1.0, stream=DETECTOR))
            clock.advance(2.0)
            batcher.batch([])

        # Re-introduce monitor. Use gap recovery to align.
        t = t0 + 200.0
        for i in range(MIN_BATCHES_FOR_GATE):
            det = msgs_at(14.0, start=t + i, duration=1.0, stream=DETECTOR)
            mon = msgs_at(5.0, start=t + i, duration=1.0, stream=MONITOR)
            batcher.batch(det + mon)
            clock.advance(2.0)
            batcher.batch([])

        # Now both should be gated again — detector-only should NOT complete
        t_test = t + MIN_BATCHES_FOR_GATE + 100.0
        result = batcher.batch(
            msgs_at(14.0, start=t_test, duration=1.0, stream=DETECTOR)
        )
        assert result is None, "Monitor should be re-gated after reappearance"


class TestDriftCorrection:
    """Source rate drift should not cause systematic message loss."""

    def test_slight_drift_over_100_batches(self):
        """13.98 Hz source (expected 14): no systematic loss over 100 batches."""
        actual_rate = 13.98
        period = 1.0 / actual_rate
        n_batches = 100

        clock = FakeClock()
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0, clock=clock)

        total_messages_in = 0
        total_messages_out = 0
        for b in range(n_batches):
            batch_t0 = t0 + b * 1.0
            count = round(actual_rate * 1.0)  # 14 messages per batch
            batch_msgs = [msg(batch_t0 + i * period) for i in range(count)]
            total_messages_in += len(batch_msgs)
            result = batcher.batch(batch_msgs)
            if result is not None:
                total_messages_out += len(result.messages)
            else:
                clock.advance(1000.0)
                result = batcher.batch([])
                if result is not None:
                    total_messages_out += len(result.messages)

        # Should not lose more than ~1% of messages
        loss_rate = 1.0 - total_messages_out / total_messages_in
        assert loss_rate < 0.01, f"Lost {loss_rate:.1%} of messages"

    def test_abrupt_rate_change(self):
        """Rate drops from 14 Hz to 7 Hz: converges after a few batches."""
        clock = FakeClock()
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=1.5, clock=clock)

        # Run at 7 Hz for MIN_BATCHES_FOR_GATE + extra batches
        # First few may timeout (rate mismatch), but should converge
        n_batches = MIN_BATCHES_FOR_GATE + 5
        for b in range(n_batches):
            batch_t0 = t0 + b * 1.0
            batch_msgs = msgs_at(7.0, start=batch_t0, duration=1.0)
            result = batcher.batch(batch_msgs)
            if result is None:
                clock.advance(2.0)
                batcher.batch([])

        # After convergence, a 7 Hz batch should complete via slot gate
        t_test = t0 + n_batches * 1.0 + 100.0  # gap recovery
        result = batcher.batch(msgs_at(7.0, start=t_test, duration=1.0))
        assert result is not None
        assert len(result.messages) == 7


class TestSetBatchLength:
    """set_batch_length() for AdaptiveMessageBatcher integration."""

    def test_increase_batch_length(self):
        """Batch length 1s -> 2s: next batch gets ~28 messages at 14 Hz."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0)

        # Change batch length for next batch
        batcher.set_batch_length(2.0)

        # Complete the current (1s) batch to trigger the switch
        current = msgs_at(14.0, start=t0, duration=1.0)
        batch1 = batcher.batch(current)
        assert batch1 is not None

        # Next batch should be 2s long
        assert batcher.batch_length_s == 2.0
        next_start = batch1.end_time.to_ns() / 1e9
        next_msgs = msgs_at(14.0, start=next_start, duration=2.0)
        batch2 = batcher.batch(next_msgs)
        assert batch2 is not None
        assert len(batch2.messages) == 28

    def test_decrease_batch_length(self):
        """Batch length 1s -> 0.5s: next batch gets ~7 messages at 14 Hz."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        batcher.set_batch_length(0.5)

        current = msgs_at(14.0, start=t0, duration=1.0)
        batch1 = batcher.batch(current)
        assert batch1 is not None

        assert batcher.batch_length_s == 0.5
        next_start = batch1.end_time.to_ns() / 1e9
        next_msgs = msgs_at(14.0, start=next_start, duration=0.5)
        batch2 = batcher.batch(next_msgs)
        assert batch2 is not None
        assert len(batch2.messages) == 7


class TestNonGatedStreams:
    """Non-gated streams (e.g., log) are included but don't affect the gate."""

    LOG = StreamId(kind=StreamKind.LOG, name="log")

    def test_included_in_current_batch(self):
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        log_msg = msg(t0 + 0.5, stream=self.LOG)
        det_msgs = msgs_at(14.0, start=t0, duration=1.0)
        result = batcher.batch([*det_msgs, log_msg])
        assert result is not None
        assert log_msg in result.messages
        assert len(result.messages) == 15  # 14 detector + 1 log

    def test_does_not_affect_gate(self):
        clock = FakeClock()
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0, clock=clock)
        # Only log messages, no detector — gate not satisfied
        result = batcher.batch([msg(t0 + 0.5, stream=self.LOG)])
        assert result is None
