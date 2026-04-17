# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for rate-aware message batcher.

Tests use 'seconds' as the natural unit: timestamps are in seconds (converted
to nanoseconds via helpers), batch lengths are in seconds.

The batcher uses a logical clock (high-water mark of observed message
timestamps) rather than wall time.  Where tests need to trigger the timeout
fallback, they feed a non-gated "trigger" message whose timestamp advances the
high-water mark past the threshold.
"""

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.core.rate_aware_batcher import (
    MIN_BATCHES_FOR_GATE,
    RateAwareMessageBatcher,
)
from ess.livedata.core.timestamp import Duration, Timestamp

# --- Helpers ---

DETECTOR = StreamId(kind=StreamKind.DETECTOR_EVENTS, name="det")
MONITOR = StreamId(kind=StreamKind.MONITOR_EVENTS, name="mon")
# Non-gated stream used to advance the logical clock without affecting gates.
_HWM = StreamId(kind=StreamKind.LOG, name="_hwm")


def ts(seconds: float) -> Timestamp:
    return Timestamp.from_ns(int(seconds * 1e9))


def msg(t: float, stream: StreamId = DETECTOR, value: str = "") -> Message[str]:
    return Message(timestamp=ts(t), stream=stream, value=value)


def hwm_trigger(t: float) -> Message[str]:
    """Non-gated message that advances the logical clock to *t*."""
    return Message(timestamp=ts(t), stream=_HWM, value="")


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


def make_converged_batcher(
    rate_hz: float = 14.0,
    batch_length_s: float = 1.0,
    streams: dict[StreamId, float] | None = None,
    timeout_s: float | None = None,
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

    # Use a small timeout for convergence (logical-clock triggers close
    # each batch), then set the caller's requested timeout afterwards.
    convergence_timeout = batch_length_s * 0.8
    batcher = RateAwareMessageBatcher(
        batch_length_s=batch_length_s,
        ema_alpha=1.0,
        timeout_s=convergence_timeout,
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

    # Feed batches; a trigger message advances the logical clock past the
    # convergence timeout to close each batch.
    for i in range(MIN_BATCHES_FOR_GATE):
        t0 = batch_start + i * batch_length_s
        batch_msgs: list[Message[str]] = []
        for sid, r in streams.items():
            batch_msgs.extend(msgs_at(r, start=t0, duration=batch_length_s, stream=sid))
        batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
        result = batcher.batch(batch_msgs)
        assert result is not None, f"Timeout batch {i} should have closed"

    # Set the caller's requested timeout for post-convergence use.
    if timeout_s is not None:
        batcher._timeout = Duration.from_seconds(timeout_s)
    # else leave the convergence timeout (0.8 * batch_length) — the default

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
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0)
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
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0)
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
        batcher = RateAwareMessageBatcher(
            batch_length_s=1.0,
            timeout_s=0.5,
            ema_alpha=1.0,
        )
        # Initial batch
        batcher.batch([msg(0.0)])

        # Feed messages within timeout threshold (batch_start=0.0, threshold=0.5)
        result = batcher.batch([msg(0.1), msg(0.2)])
        assert result is None

        # Advance logical clock past timeout
        batch = batcher.batch([hwm_trigger(0.6)])
        assert batch is not None
        assert len(batch.messages) == 3  # 2 detector + 1 hwm trigger


class TestMultiStream:
    def test_waits_for_all_gated_streams(self):
        streams = {DETECTOR: 14.0, MONITOR: 1.0}
        batcher, t0 = make_converged_batcher(streams=streams, timeout_s=999.0)

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

    def test_overflow_completes_batch_with_missing_last_slot(self):
        """A future message proves the batch window passed, even if the last
        slot was never delivered. Completion should be immediate (slot-based),
        not delayed until the high-water-mark timeout fires."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0)

        # All 14 messages except the last one, plus one future message
        all_msgs = msgs_at(14.0, start=t0, duration=1.0)
        missing_last = all_msgs[:-1]  # 13 messages, last slot absent
        future = [msg(t0 + 1.5)]  # clearly in next batch
        result = batcher.batch(missing_last + future)
        assert result is not None, (
            "Overflow message should trigger slot-based completion "
            "even when the last slot is missing"
        )
        assert len(result.messages) == 13


class TestPhaseOffset:
    """Streams whose first pulse doesn't align with batch_start."""

    def _converge_with_offset(
        self,
        rate_hz: float = 14.0,
        offset: float = 0.04,
        batch_length_s: float = 1.0,
        timeout_s: float = 999.0,
    ) -> tuple[RateAwareMessageBatcher, float]:
        """Converge a single DETECTOR stream that has a phase offset.

        Returns (batcher, next_batch_start).
        """
        period = 1.0 / rate_hz
        count = round(rate_hz * batch_length_s)

        convergence_timeout = batch_length_s * 0.8
        batcher = RateAwareMessageBatcher(
            batch_length_s=batch_length_s,
            ema_alpha=1.0,
            timeout_s=convergence_timeout,
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
            batch_msgs = stream_msgs(t0)
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None, f"Convergence batch {i} should close"

        batcher._timeout = Duration.from_seconds(timeout_s)
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
        det_offset = 0.04
        mon_offset = 0.01
        det_rate = 14.0
        mon_rate = 5.0

        convergence_timeout = 0.8
        batcher = RateAwareMessageBatcher(
            batch_length_s=1.0, ema_alpha=1.0, timeout_s=convergence_timeout
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
            batch_msgs = det_msgs(t0) + mon_msgs(t0)
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None

        batcher._timeout = Duration.from_seconds(999.0)

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

        batcher, t0 = make_converged_batcher(rate_hz=rate, timeout_s=999.0)

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
                # Trigger timeout via logical clock
                result = batcher.batch([hwm_trigger(batch_t0 + 999.01)])
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
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0)

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
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0)

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
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=1.5)

        # Introduce a second stream (MONITOR at 5 Hz).
        # Detector is already gated, so its slot gate closes each batch.
        for i in range(MIN_BATCHES_FOR_GATE):
            batch_t = t0 + i * 1.0
            det = msgs_at(14.0, start=batch_t, duration=1.0, stream=DETECTOR)
            mon = msgs_at(5.0, start=batch_t, duration=1.0, stream=MONITOR)
            result = batcher.batch(det + mon)
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

        timeout = 1.5
        streams = {DETECTOR: 14.0, MONITOR: 5.0}
        batcher, t0 = make_converged_batcher(streams=streams, timeout_s=timeout)

        # Close batches with only detector present. Each closes via timeout
        # since monitor is still gated and blocking slot-based completion.
        for i in range(ABSENT_BATCHES_FOR_EVICTION + 1):
            batch_t = t0 + i * 1.0
            det = msgs_at(14.0, start=batch_t, duration=1.0, stream=DETECTOR)
            batcher.batch(det)
            batcher.batch([hwm_trigger(batch_t + timeout + 0.01)])

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

        timeout = 1.5
        streams = {DETECTOR: 14.0, MONITOR: 5.0}
        batcher, t0 = make_converged_batcher(streams=streams, timeout_s=timeout)

        # Evict monitor
        for i in range(ABSENT_BATCHES_FOR_EVICTION + 1):
            batch_t = t0 + i
            batcher.batch(msgs_at(14.0, start=batch_t, duration=1.0, stream=DETECTOR))
            batcher.batch([hwm_trigger(batch_t + timeout + 0.01)])

        # Re-introduce monitor. Use gap recovery to align.
        # Detector is the only gated stream (monitor was evicted), so its
        # slot gate closes each batch — no timeout trigger needed.
        t = t0 + 200.0
        for i in range(MIN_BATCHES_FOR_GATE):
            det = msgs_at(14.0, start=t + i, duration=1.0, stream=DETECTOR)
            mon = msgs_at(5.0, start=t + i, duration=1.0, stream=MONITOR)
            result = batcher.batch(det + mon)
            assert result is not None

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

        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0)

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
                result = batcher.batch([hwm_trigger(batch_t0 + 999.01)])
                if result is not None:
                    total_messages_out += len(result.messages)

        # Should not lose more than ~1% of messages
        loss_rate = 1.0 - total_messages_out / total_messages_in
        assert loss_rate < 0.01, f"Lost {loss_rate:.1%} of messages"

    def test_abrupt_rate_change(self):
        """Rate drops from 14 Hz to 7 Hz: converges after a few batches."""
        timeout = 1.5
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=timeout)

        # Run at 7 Hz for MIN_BATCHES_FOR_GATE + extra batches
        # First few may timeout (rate mismatch), but should converge
        n_batches = MIN_BATCHES_FOR_GATE + 5
        for b in range(n_batches):
            batch_t0 = t0 + b * 1.0
            batch_msgs = msgs_at(7.0, start=batch_t0, duration=1.0)
            result = batcher.batch(batch_msgs)
            if result is None:
                batcher.batch([hwm_trigger(batch_t0 + timeout + 0.01)])

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


class TestEnvelopeBoundaries:
    """Probe the boundaries of the working envelope to find breakdown modes."""

    def test_phase_offset_exactly_half_period(self):
        """Phase offset at 50% of period — Python banker's rounding territory."""
        rate = 14.0
        period = 1.0 / rate
        offset = period * 0.5  # Exactly half

        convergence_timeout = 0.8
        batcher = RateAwareMessageBatcher(
            batch_length_s=1.0, ema_alpha=1.0, timeout_s=convergence_timeout
        )
        count = round(rate * 1.0)

        def stream_msgs(batch_start: float) -> list[Message[str]]:
            return [msg(batch_start + offset + i * period) for i in range(count)]

        initial = stream_msgs(0.0)
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        for i in range(MIN_BATCHES_FOR_GATE):
            t0 = batch_start + i * 1.0
            batch_msgs = stream_msgs(t0)
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None

        batcher._timeout = Duration.from_seconds(999.0)
        t0 = batch_start + MIN_BATCHES_FOR_GATE * 1.0
        result = batcher.batch(stream_msgs(t0))
        assert result is not None
        assert len(result.messages) == 14

    def test_phase_offset_just_over_half_period(self):
        """Phase offset at 51% of period — past the rounding midpoint."""
        rate = 14.0
        period = 1.0 / rate
        offset = period * 0.51

        convergence_timeout = 0.8
        batcher = RateAwareMessageBatcher(
            batch_length_s=1.0, ema_alpha=1.0, timeout_s=convergence_timeout
        )
        count = round(rate * 1.0)

        def stream_msgs(batch_start: float) -> list[Message[str]]:
            return [msg(batch_start + offset + i * period) for i in range(count)]

        initial = stream_msgs(0.0)
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        for i in range(MIN_BATCHES_FOR_GATE):
            t0 = batch_start + i * 1.0
            batch_msgs = stream_msgs(t0)
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None

        batcher._timeout = Duration.from_seconds(999.0)
        t0 = batch_start + MIN_BATCHES_FOR_GATE * 1.0
        result = batcher.batch(stream_msgs(t0))
        assert result is not None
        assert len(result.messages) == 14

    def test_high_jitter_30ms_at_14hz(self):
        """30ms jitter at 14 Hz (~42% of period). Near the breaking point."""
        import random

        rng = random.Random(42)
        rate = 14.0
        period = 1.0 / rate
        jitter_max = 0.030  # 42% of period
        timeout = 999.0

        batcher, t0 = make_converged_batcher(rate_hz=rate, timeout_s=timeout)

        # Run 50 batches, check that we don't systematically lose messages
        counts: list[int] = []
        for b in range(50):
            batch_t0 = t0 + b * 1.0
            batch_msgs = [
                msg(batch_t0 + i * period + rng.uniform(-jitter_max, jitter_max))
                for i in range(14)
            ]
            result = batcher.batch(batch_msgs)
            if result is not None:
                counts.append(len(result.messages))
            else:
                result = batcher.batch([hwm_trigger(batch_t0 + timeout + 0.01)])
                if result is not None:
                    counts.append(len(result.messages))

        avg = sum(counts) / len(counts)
        # At this jitter level, some batches may get 13 or 15 instead of 14
        # but the average should still be close
        assert abs(avg - 14.0) < 1.0, f"Average {avg} too far from 14.0"

    def test_out_of_order_messages_before_batch_start(self):
        """Messages with timestamps before batch_start (late arrivals)."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0)

        # 14 normal messages + 2 "late" messages from the previous batch
        normal = msgs_at(14.0, start=t0, duration=1.0)
        late = [msg(t0 - 0.05), msg(t0 - 0.02)]
        result = batcher.batch(late + normal)

        # The batch should complete (normal messages fill the last slot).
        # Late messages are included (negative slot < expected, so not overflow).
        assert result is not None
        assert len(result.messages) == 16  # 14 + 2 late

    def test_sub_hz_stream_does_not_gate(self):
        """A sub-Hz stream never gets a grid (zero integer rate) and doesn't gate.

        With a 14 Hz detector also present, the detector gates; the sub-Hz
        stream's messages are included but don't affect completion.
        """
        convergence_timeout = 0.8
        SUB_HZ = StreamId(kind=StreamKind.MONITOR_EVENTS, name="sub_hz")
        batcher = RateAwareMessageBatcher(
            batch_length_s=1.0, ema_alpha=1.0, timeout_s=convergence_timeout
        )
        # Initial batch with both streams
        initial = msgs_at(14.0, start=0.0, duration=1.0, stream=DETECTOR)
        initial.append(msg(0.0, stream=SUB_HZ))
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        # Converge via timeout — sub-Hz gets 1 msg every other batch
        for i in range(MIN_BATCHES_FOR_GATE):
            t0 = batch_start + i * 1.0
            batch_msgs = msgs_at(14.0, start=t0, duration=1.0, stream=DETECTOR)
            if i % 2 == 0:
                batch_msgs.append(msg(t0 + 0.5, stream=SUB_HZ))
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None

        # Post-convergence: detector-only should complete
        # (sub-Hz has no grid, doesn't gate)
        batcher._timeout = Duration.from_seconds(999.0)
        t_test = batch_start + MIN_BATCHES_FOR_GATE * 1.0
        det = msgs_at(14.0, start=t_test, duration=1.0, stream=DETECTOR)
        result = batcher.batch(det)
        assert result is not None
        assert len(result.messages) == 14

    def test_high_rate_short_batch(self):
        """14 Hz with 0.1s batch: slots_per_batch=1, single-slot behavior."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0, batch_length_s=0.1)

        # At 14 Hz with 0.1s batch, slots_per_batch = round(14 * 0.1) = round(1.4) = 1
        # So the batcher expects 1 message per batch.
        # Feeding 1 message should complete immediately.
        result = batcher.batch([msg(t0)])
        assert result is not None
        assert len(result.messages) == 1

    def test_high_rate_short_batch_loses_second_message(self):
        """14 Hz with 0.1s batch: second message in the period overflows."""
        batcher, t0 = make_converged_batcher(
            rate_hz=14.0, batch_length_s=0.1, timeout_s=999.0
        )

        # Two messages within 0.1s at 14 Hz spacing (0.0714s apart)
        # slots_per_batch = 1, so second message overflows
        period = 1.0 / 14.0
        result = batcher.batch([msg(t0), msg(t0 + period)])
        # First message completes batch (slot 0 = last slot for expected=1),
        # second goes to overflow
        assert result is not None
        assert len(result.messages) == 1

    def test_non_integer_rate_times_batch_length(self):
        """14.5 Hz x 1s = 14.5 expected: round to 14, systematic overflow?"""
        rate = 14.5
        period = 1.0 / rate
        n_batches = 100
        timeout = 999.0

        convergence_timeout = 0.8
        batcher = RateAwareMessageBatcher(
            batch_length_s=1.0, ema_alpha=1.0, timeout_s=convergence_timeout
        )

        # Initial batch with 14 or 15 messages
        initial_count = round(rate * 1.0)  # 14 (banker's rounding)
        initial = [msg(i * period) for i in range(initial_count)]
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        # Converge
        for i in range(MIN_BATCHES_FOR_GATE):
            t0 = batch_start + i * 1.0
            batch_msgs = [msg(t0 + j * period) for j in range(initial_count)]
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None

        batcher._timeout = Duration.from_seconds(timeout)

        # Run 100 batches alternating between 14 and 15 messages
        # (simulating a real 14.5 Hz source)
        t_base = batch_start + MIN_BATCHES_FOR_GATE * 1.0
        total_in = 0
        total_out = 0
        for b in range(n_batches):
            # A true 14.5 Hz source produces 14 or 15 messages per second
            # depending on phase alignment
            count = 15 if b % 2 else 14
            batch_msgs = [msg(t_base + b * 1.0 + j * period) for j in range(count)]
            total_in += count
            result = batcher.batch(batch_msgs)
            if result is not None:
                total_out += len(result.messages)
            else:
                result = batcher.batch([hwm_trigger(t_base + b * 1.0 + timeout + 0.01)])
                if result is not None:
                    total_out += len(result.messages)

        loss_rate = 1.0 - total_out / total_in
        assert loss_rate < 0.02, f"Lost {loss_rate:.1%} of messages at 14.5 Hz"

    def test_burst_delivery_pattern(self):
        """All 14 messages arrive in one batch() call with correct timestamps."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        # Same as normal — timestamps are spaced correctly, just all delivered at once
        all_at_once = msgs_at(14.0, start=t0, duration=1.0)
        result = batcher.batch(all_at_once)
        assert result is not None
        assert len(result.messages) == 14

    def test_realistic_ema_alpha_abrupt_rate_change(self):
        """With default alpha=0.05, how many batches to converge after 14→7 Hz?"""
        convergence_timeout = 0.8
        batcher = RateAwareMessageBatcher(
            batch_length_s=1.0, ema_alpha=0.05, timeout_s=convergence_timeout
        )

        # Converge at 14 Hz
        initial = msgs_at(14.0, start=0.0, duration=1.0)
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        for i in range(MIN_BATCHES_FOR_GATE):
            t0 = batch_start + i * 1.0
            batch_msgs = msgs_at(14.0, start=t0, duration=1.0)
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None

        timeout = 1.5
        batcher._timeout = Duration.from_seconds(timeout)

        # Switch to 7 Hz. Count how many batches need timeout vs slot-gate
        t_base = batch_start + MIN_BATCHES_FOR_GATE * 1.0
        timeout_count = 0
        slot_gate_count = 0
        for b in range(100):
            batch_t = t_base + b * 1.0
            result = batcher.batch(msgs_at(7.0, start=batch_t, duration=1.0))
            if result is not None:
                slot_gate_count += 1
            else:
                result = batcher.batch([hwm_trigger(batch_t + timeout + 0.01)])
                if result is not None:
                    timeout_count += 1

        # With alpha=0.05, rate moves slowly. Most batches will close via
        # timeout for a while, then eventually via slot gate.
        # The important thing: no messages are lost.
        total = slot_gate_count + timeout_count
        assert total == 100, f"Expected 100 batches, got {total}"

    def test_jitter_at_50_percent_of_period(self):
        """36ms jitter at 14 Hz (50% of period). At the theoretical limit.

        Note: the phase offset estimation uses a single message per batch,
        so with jitter the estimate drifts. This causes ~50% of batches to
        require timeout rather than slot-gate completion. No messages are
        lost — the average per-batch count remains exactly 14.0.
        """
        import random

        rng = random.Random(99)
        rate = 14.0
        period = 1.0 / rate
        jitter_max = period * 0.50
        timeout = 999.0

        batcher, t0 = make_converged_batcher(rate_hz=rate, timeout_s=timeout)

        counts: list[int] = []
        for b in range(100):
            batch_t0 = t0 + b * 1.0
            batch_msgs = [
                msg(batch_t0 + i * period + rng.uniform(-jitter_max, jitter_max))
                for i in range(14)
            ]
            result = batcher.batch(batch_msgs)
            if result is not None:
                counts.append(len(result.messages))
            else:
                result = batcher.batch([hwm_trigger(batch_t0 + timeout + 0.01)])
                if result is not None:
                    counts.append(len(result.messages))

        avg = sum(counts) / len(counts)
        assert abs(avg - 14.0) < 1.0, f"Average {avg} too far from 14.0"

    def test_extreme_jitter_breaks_gracefully(self):
        """Jitter exceeding 100% of period: verify no crash, just degraded accuracy."""
        import random

        rng = random.Random(77)
        rate = 14.0
        period = 1.0 / rate
        jitter_max = period * 1.5  # 150% of period!
        timeout = 1.5

        batcher, t0 = make_converged_batcher(rate_hz=rate, timeout_s=timeout)

        # Just verify no crash or infinite loop over 20 batches
        for b in range(20):
            batch_t0 = t0 + b * 1.0
            batch_msgs = [
                msg(batch_t0 + i * period + rng.uniform(-jitter_max, jitter_max))
                for i in range(14)
            ]
            result = batcher.batch(batch_msgs)
            if result is None:
                batcher.batch([hwm_trigger(batch_t0 + timeout + 0.01)])

    def test_overflow_does_not_accumulate(self):
        """Over 200 batches, overflow never grows unboundedly."""
        rate = 14.0
        batcher, t0 = make_converged_batcher(rate_hz=rate, timeout_s=999.0)

        max_overflow = 0
        for b in range(200):
            batch_t0 = t0 + b * 1.0
            batch_msgs = msgs_at(rate, start=batch_t0, duration=1.0)
            result = batcher.batch(batch_msgs)
            overflow_size = len(batcher._overflow)
            max_overflow = max(max_overflow, overflow_size)
            if result is None:
                result = batcher.batch([hwm_trigger(batch_t0 + 999.01)])
                if result is not None:
                    overflow_size = len(batcher._overflow)
                    max_overflow = max(max_overflow, overflow_size)

        # Overflow should never accumulate significantly
        assert max_overflow <= 2, f"Max overflow was {max_overflow}"


class TestSlotBoundaryStability:
    """Rate estimate errors must not cause slot-boundary oscillation.

    A fractional-Hz EMA estimate (e.g. 14.003) makes the estimated period
    slightly shorter/longer than the true period. Over 14 slots the error
    accumulates, pushing the boundary message between "included" and
    "overflow" on alternate batches — producing a 13/15/13/15 pattern
    instead of a steady 14.

    The bug requires integer-period timestamp spacing (as Kafka sources
    produce) and tick-based delivery across multiple batch() calls.
    """

    @staticmethod
    def _run_integer_period_stream(
        rate_hz: float = 14.0,
        rate_error: float = 0.0,
        n_batches: int = 40,
    ) -> list[int]:
        """Converge with integer-period delivery, inject a rate error, run.

        Uses integer-nanosecond period spacing (``int(1e9 / rate_hz)``)
        throughout, matching how Kafka sources generate timestamps.
        Delivers messages in 100ms ticks, matching the testbench.
        """
        convergence_timeout = 1.5
        batcher = RateAwareMessageBatcher(
            batch_length_s=1.0,
            ema_alpha=1.0,
            timeout_s=convergence_timeout,
        )

        period_ns = int(1e9 / rate_hz)
        start_ns = 1_000_000_000_000
        next_pulse_ns = start_ns
        # Track logical time for tick delivery
        tick_time = 0.0

        def deliver_ticks(
            n_seconds: float, *, with_triggers: bool = False
        ) -> list[int]:
            """Deliver messages in 100ms ticks, return per-batch counts."""
            nonlocal next_pulse_ns, tick_time
            counts: list[int] = []
            target = tick_time + n_seconds
            while tick_time < target:
                tick_time += 0.1
                now_ns = start_ns + int(tick_time * 1e9)
                msgs: list[Message[str]] = []
                while next_pulse_ns <= now_ns:
                    msgs.append(
                        Message(
                            timestamp=Timestamp.from_ns(next_pulse_ns),
                            stream=DETECTOR,
                            value="",
                        )
                    )
                    next_pulse_ns += period_ns
                if with_triggers:
                    msgs.append(
                        Message(
                            timestamp=Timestamp.from_ns(
                                start_ns + int(tick_time * 1e9)
                            ),
                            stream=_HWM,
                            value="",
                        )
                    )
                result = batcher.batch(msgs) if msgs else batcher.batch([])
                if result is not None:
                    counts.append(len(result.messages))
            return counts

        # Run until converged (initial + MIN_BATCHES_FOR_GATE timeout batches)
        deliver_ticks((MIN_BATCHES_FOR_GATE + 2) * 1.5, with_triggers=True)

        state = batcher._estimators[DETECTOR]
        assert state.converged

        # Inject rate error and freeze it (alpha=0)
        state.rate_hz = rate_hz + rate_error
        batcher._ema_alpha = 0.0

        # Skip a few batches: the abrupt rate injection changes
        # integer_rate_hz, which invalidates the current phase offset.
        # The phase self-corrects within 1-2 batches.
        warmup = 3
        all_counts = deliver_ticks((n_batches + warmup) * 1.1)
        return all_counts[warmup : warmup + n_batches]

    def test_positive_rate_error_no_oscillation(self):
        """rate_hz=14.1: integer rounding absorbs the error."""
        counts = self._run_integer_period_stream(rate_error=+0.1)
        assert all(c == 14 for c in counts), f"Oscillation: {set(counts)}"

    def test_negative_rate_error_no_oscillation(self):
        """rate_hz=13.9: integer rounding absorbs the error."""
        counts = self._run_integer_period_stream(rate_error=-0.1)
        assert all(c == 14 for c in counts), f"Oscillation: {set(counts)}"

    def test_small_positive_error_no_oscillation(self):
        """rate_hz=14.001: even tiny errors used to trigger the bug."""
        counts = self._run_integer_period_stream(rate_error=+0.001)
        assert all(c == 14 for c in counts), f"Oscillation: {set(counts)}"


class TestDefaultTimeoutWithSlotGate:
    """Default timeout must not preempt slot-gate completion.

    When timeout < batch_length, tick-based delivery triggers the timeout
    before the last slot's message arrives — producing short batches, biasing
    the rate estimate downward, and causing oscillation.  This requires
    tick-based delivery (multiple batch() calls per batch window) because
    all-at-once delivery routes every message before the completion check.

    Convergence uses all-at-once delivery (like real Kafka bursts) to get a
    correct rate estimate; the post-convergence phase switches to 100ms ticks.
    """

    @staticmethod
    def _run_tick_delivery(
        batcher: RateAwareMessageBatcher,
        rate_hz: float,
        start: float,
        n_batches: int,
    ) -> list[int]:
        """Deliver messages in 100ms ticks, return per-batch message counts."""
        period_ns = int(1e9 / rate_hz)
        start_ns = int(start * 1e9)
        next_pulse_ns = start_ns
        tick_time_ns = start_ns

        counts: list[int] = []
        target_ns = start_ns + int(n_batches * 1.1 * 1e9)
        while tick_time_ns < target_ns:
            tick_time_ns += 100_000_000  # 100ms ticks
            batch_msgs: list[Message[str]] = []
            while next_pulse_ns <= tick_time_ns:
                batch_msgs.append(
                    Message(
                        timestamp=Timestamp.from_ns(next_pulse_ns),
                        stream=DETECTOR,
                        value="",
                    )
                )
                next_pulse_ns += period_ns
            result = batcher.batch(batch_msgs) if batch_msgs else batcher.batch([])
            if result is not None:
                counts.append(len(result.messages))
        return counts[:n_batches]

    def test_steady_count_at_14hz(self):
        """14 Hz with default timeout: every batch should contain 14 messages."""
        # Converge cleanly with all-at-once delivery
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        # Apply the actual constructor default timeout
        batcher._timeout = RateAwareMessageBatcher()._timeout
        counts = self._run_tick_delivery(batcher, 14.0, t0, n_batches=30)
        assert all(c == 14 for c in counts), f"Unsteady counts: {counts}"

    def test_steady_count_at_7hz(self):
        """7 Hz with default timeout: every batch should contain 7 messages."""
        batcher, t0 = make_converged_batcher(rate_hz=7.0)
        batcher._timeout = RateAwareMessageBatcher()._timeout
        counts = self._run_tick_delivery(batcher, 7.0, t0, n_batches=30)
        assert all(c == 7 for c in counts), f"Unsteady counts: {counts}"


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
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=999.0)
        # Only log messages, no detector — gate not satisfied
        result = batcher.batch([msg(t0 + 0.5, stream=self.LOG)])
        assert result is None
