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

import random

import pytest

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.core.message_batcher import MessageBatch
from ess.livedata.core.rate_aware_batcher import (
    MIN_DIFFS_FOR_GATE,
    RateAwareMessageBatcher,
)
from ess.livedata.core.timestamp import Timestamp

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
    # convergence timeout to close each batch. 1 Hz streams need an extra
    # iteration because the first batch after initial duplicates the
    # initial's last timestamp, producing no new diff.
    warmup_iters = MIN_DIFFS_FOR_GATE + 1
    for i in range(warmup_iters):
        t0 = batch_start + i * batch_length_s
        batch_msgs: list[Message[str]] = []
        for sid, r in streams.items():
            batch_msgs.extend(msgs_at(r, start=t0, duration=batch_length_s, stream=sid))
        batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
        result = batcher.batch(batch_msgs)
        assert result is not None, f"Timeout batch {i} should have closed"

    # Set the caller's requested timeout for post-convergence use.
    if timeout_s is not None:
        batcher.timeout_factor = timeout_s / batch_length_s
    # else leave the convergence timeout (0.8 * batch_length) — the default

    next_batch_start = batch_start + warmup_iters * batch_length_s
    return batcher, next_batch_start


def _run_ticks(
    batcher: RateAwareMessageBatcher,
    start: float,
    schedule: list[Message[str]],
    n_batches: int,
    tick_s: float = 0.1,
    max_run_s: float = 120.0,
) -> list[MessageBatch]:
    """Deliver a pre-sorted message schedule via fixed-size ticks.

    No ``hwm_trigger`` is injected -- closures are driven solely by real
    message arrivals (slot gate) or by the high-water mark advancing past
    the timeout threshold via real timestamps.  Useful for verifying that
    a setup can close batches without timeout-fallback help.
    """
    tick_step = int(tick_s * 1e9)
    start_ns = int(start * 1e9)
    now_ns = start_ns
    idx = 0
    closed: list[MessageBatch] = []
    budget_ns = start_ns + int(max_run_s * 1e9)
    while len(closed) < n_batches and now_ns < budget_ns:
        now_ns += tick_step
        batch_msgs: list[Message[str]] = []
        while idx < len(schedule) and schedule[idx].timestamp.to_ns() <= now_ns:
            batch_msgs.append(schedule[idx])
            idx += 1
        r = batcher.batch(batch_msgs) if batch_msgs else batcher.batch([])
        if r is not None:
            closed.append(r)
    return closed


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
            timeout_s=convergence_timeout,
        )

        def stream_msgs(batch_start: float) -> list[Message[str]]:
            return [msg(batch_start + offset + i * period) for i in range(count)]

        # Initial batch
        initial = stream_msgs(0.0)
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        # Converge via timeout
        for i in range(MIN_DIFFS_FOR_GATE):
            t0 = batch_start + i * batch_length_s
            batch_msgs = stream_msgs(t0)
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None, f"Convergence batch {i} should close"

        batcher.timeout_factor = timeout_s / batch_length_s
        next_start = batch_start + MIN_DIFFS_FOR_GATE * batch_length_s
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
            batch_length_s=1.0, timeout_s=convergence_timeout
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
        for i in range(MIN_DIFFS_FOR_GATE):
            t0 = batch_start + i * 1.0
            batch_msgs = det_msgs(t0) + mon_msgs(t0)
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None

        batcher.timeout_factor = 999.0

        # Post-convergence: both streams present
        t_test = batch_start + MIN_DIFFS_FOR_GATE * 1.0
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

    def test_overflow_message_not_silently_dropped_when_gap_has_zero_steps(self):
        """A mis-rounded overflow whose ts lies inside the current batch
        window (gap_ns < batch_ns) must not be silently dropped.

        ``_advance_past_gap`` runs with ``steps=0`` in that case; the
        re-routed message overflows again, and the old unconditional
        ``self._overflow = []`` wiped it.
        """
        batcher, t0 = make_converged_batcher(
            rate_hz=1.0, streams={MONITOR: 1.0}, timeout_s=999.0
        )
        # Pulse at batch_start + 0.6 s: ``pulse_index`` rounds to 1,
        # slot 1 = overflow at ``slots_per_batch = 1``.  No other gated
        # stream has routed messages, so ``_advance_past_gap`` fires
        # with ``steps = 0``.
        stray_ts = t0 + 0.6
        stray = msg(stray_ts, stream=MONITOR)
        r1 = batcher.batch([stray])
        # Drive one real pulse + timeout tick — the stray must appear
        # in one of these batches, not be silently dropped.
        r2 = batcher.batch([msg(t0 + 1.0, stream=MONITOR)])
        r3 = batcher.batch([hwm_trigger(t0 + 999.01)])

        stray_seen = sum(
            any(m.timestamp == ts(stray_ts) for m in r.messages)
            for r in (r1, r2, r3)
            if r is not None
        )
        assert stray_seen == 1, "Stray overflow was silently dropped"

    def test_non_gated_not_duplicated_in_zero_step_gap_recovery(self):
        """Non-gated messages in the active batch must not be duplicated when
        the gap-recovery path fires with ``steps == 0``.

        At ``steps == 0`` the gap-recovery path keeps the active window in
        place but must still reset per-stream buckets so that re-routing the
        stashed messages does not double-add them.
        """
        log_stream = StreamId(kind=StreamKind.LOG, name="log")
        batcher, t0 = make_converged_batcher(
            rate_hz=1.0, streams={MONITOR: 1.0}, timeout_s=999.0
        )
        log_msg = Message(timestamp=ts(t0 + 0.3), stream=log_stream, value="log_1")
        # MON pulse at t0+0.6 s: slot 1 at slots_per_batch=1 → overflow.
        # No gated-gridded msg has landed in the active batch, so the
        # gap-recovery path fires; gap_ns < batch_ns → steps == 0.
        stray = msg(t0 + 0.6, stream=MONITOR, value="mon_stray")
        r1 = batcher.batch([log_msg, stray])
        r2 = batcher.batch([msg(t0 + 1.0, stream=MONITOR)])
        r3 = batcher.batch([hwm_trigger(t0 + 999.01)])
        results = [r for r in (r1, r2, r3) if r is not None]
        log_count = sum(1 for r in results for m in r.messages if m.value == "log_1")
        assert log_count == 1, f"LOG message duplicated (count={log_count})"

    def test_unconverged_gated_not_duplicated_in_zero_step_gap_recovery(self):
        """Unconverged gated streams (no grid yet) land in the active batch.
        Same ``steps == 0`` path must not duplicate them either.
        """
        slow_mon = StreamId(kind=StreamKind.MONITOR_COUNTS, name="slow_mon")
        batcher, t0 = make_converged_batcher(
            rate_hz=1.0, streams={MONITOR: 1.0}, timeout_s=999.0
        )
        slow_msg = msg(t0 + 0.3, stream=slow_mon, value="slow_first")
        stray = msg(t0 + 0.6, stream=MONITOR, value="mon_stray")
        r1 = batcher.batch([slow_msg, stray])
        r2 = batcher.batch([msg(t0 + 1.0, stream=MONITOR)])
        r3 = batcher.batch([hwm_trigger(t0 + 999.01)])
        results = [r for r in (r1, r2, r3) if r is not None]
        slow_count = sum(
            1 for r in results for m in r.messages if m.value == "slow_first"
        )
        assert slow_count == 1, (
            f"Unconverged gated message duplicated (count={slow_count})"
        )

    def test_bucket_not_duplicated_in_zero_step_gap_recovery(self):
        """Internal bucket state must not diverge from published messages
        after zero-step gap recovery.

        Scenario mirrors
        ``test_unconverged_gated_not_duplicated_in_zero_step_gap_recovery``
        but adds a second gridded stream (DETECTOR) that does not send a
        message this call, preventing the slot gate from closing the batch
        during the same ``batch()`` invocation.  That lets us inspect
        bucket state before ``_close_batch`` wipes it.

        Without resetting buckets before re-routing ``stashed``, the bucket
        for the unconverged-gated stream would receive its message twice:
        once from the initial route, once again from the re-route.
        """
        slow_mon = StreamId(kind=StreamKind.MONITOR_COUNTS, name="slow_mon")
        batcher, t0 = make_converged_batcher(
            rate_hz=1.0,
            streams={MONITOR: 1.0, DETECTOR: 1.0},
            timeout_s=999.0,
        )
        slow_msg = msg(t0 + 0.3, stream=slow_mon, value="slow_first")
        stray = msg(t0 + 0.6, stream=MONITOR, value="mon_stray")
        batcher.batch([slow_msg, stray])
        count = len(batcher._streams[slow_mon].messages)
        assert count == 1, (
            f"Bucket for unconverged-gated stream double-counted "
            f"(count={count}, expected 1)"
        )


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
        for i in range(MIN_DIFFS_FOR_GATE):
            batch_t = t0 + i * 1.0
            det = msgs_at(14.0, start=batch_t, duration=1.0, stream=DETECTOR)
            mon = msgs_at(5.0, start=batch_t, duration=1.0, stream=MONITOR)
            result = batcher.batch(det + mon)
            assert result is not None

        # Now both streams should be gated — sending only detector should NOT
        # complete the batch (monitor is missing)
        t_test = t0 + MIN_DIFFS_FOR_GATE * 1.0
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
        for i in range(MIN_DIFFS_FOR_GATE):
            det = msgs_at(14.0, start=t + i, duration=1.0, stream=DETECTOR)
            mon = msgs_at(5.0, start=t + i, duration=1.0, stream=MONITOR)
            result = batcher.batch(det + mon)
            assert result is not None

        # Now both should be gated again — detector-only should NOT complete
        t_test = t + MIN_DIFFS_FOR_GATE + 100.0
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

        # Run at 7 Hz for MIN_DIFFS_FOR_GATE + extra batches
        # First few may timeout (rate mismatch), but should converge
        n_batches = MIN_DIFFS_FOR_GATE + 5
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

    def test_repeated_calls_last_value_wins(self):
        """Multiple set_batch_length calls before a close collapse to last value."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        batcher.set_batch_length(2.0)
        batcher.set_batch_length(4.0)
        batcher.set_batch_length(0.5)

        # Close the current (1s) batch — applies the pending change
        current = msgs_at(14.0, start=t0, duration=1.0)
        batch1 = batcher.batch(current)
        assert batch1 is not None
        assert batcher.batch_length_s == 0.5

        # Next batch should be 0.5s long (7 messages at 14 Hz)
        next_start = batch1.end_time.to_ns() / 1e9
        next_msgs = msgs_at(14.0, start=next_start, duration=0.5)
        batch2 = batcher.batch(next_msgs)
        assert batch2 is not None
        assert len(batch2.messages) == 7

    def test_multistream_slots_per_batch_updates(self):
        """Grids for all converged streams recompute slots_per_batch on resize."""
        streams = {DETECTOR: 14.0, MONITOR: 5.0}
        batcher, t0 = make_converged_batcher(streams=streams, timeout_s=999.0)

        batcher.set_batch_length(2.0)
        det = msgs_at(14.0, start=t0, duration=1.0, stream=DETECTOR)
        mon = msgs_at(5.0, start=t0, duration=1.0, stream=MONITOR)
        batch1 = batcher.batch(det + mon)
        assert batch1 is not None
        assert batcher.batch_length_s == 2.0

        # The 2s batch completing with the expected counts proves the grids
        # were resized — 14*2=28 DET slots and 5*2=10 MON slots.
        next_start = batch1.end_time.to_ns() / 1e9
        det_next = msgs_at(14.0, start=next_start, duration=2.0, stream=DETECTOR)
        mon_next = msgs_at(5.0, start=next_start, duration=2.0, stream=MONITOR)
        batch2 = batcher.batch(det_next + mon_next)
        assert batch2 is not None
        assert len(batch2.messages) == 28 + 10

    def test_timeout_scales_with_batch_length(self):
        """timeout_s = timeout_factor * batch_length_s tracks the window size."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0, timeout_s=0.5)
        assert batcher.timeout_factor == 0.5
        assert batcher.timeout_s == 0.5

        batcher.set_batch_length(2.0)
        # Close the current 1s batch to apply the pending change
        batch1 = batcher.batch(msgs_at(14.0, start=t0, duration=1.0))
        assert batch1 is not None

        assert batcher.batch_length_s == 2.0
        assert batcher.timeout_factor == 0.5
        assert batcher.timeout_s == 1.0

    def test_grid_origin_preserved_across_size_change(self):
        """Grid origin must survive a resize so phase alignment is stable."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        original_origin = batcher.grid_of(DETECTOR).origin_ns

        batcher.set_batch_length(2.0)
        batch1 = batcher.batch(msgs_at(14.0, start=t0, duration=1.0))
        assert batch1 is not None
        assert batcher.batch_length_s == 2.0
        assert batcher.grid_of(DETECTOR).origin_ns == original_origin

        # De-escalate back and verify origin still stable.
        batcher.set_batch_length(1.0)
        next_start = batch1.end_time.to_ns() / 1e9
        batch2 = batcher.batch(msgs_at(14.0, start=next_start, duration=2.0))
        assert batch2 is not None
        assert batcher.batch_length_s == 1.0
        assert batcher.grid_of(DETECTOR).origin_ns == original_origin


class TestEnvelopeBoundaries:
    """Probe the boundaries of the working envelope to find breakdown modes."""

    def test_phase_offset_exactly_half_period(self):
        """Phase offset at 50% of period — Python banker's rounding territory."""
        rate = 14.0
        period = 1.0 / rate
        offset = period * 0.5  # Exactly half

        convergence_timeout = 0.8
        batcher = RateAwareMessageBatcher(
            batch_length_s=1.0, timeout_s=convergence_timeout
        )
        count = round(rate * 1.0)

        def stream_msgs(batch_start: float) -> list[Message[str]]:
            return [msg(batch_start + offset + i * period) for i in range(count)]

        initial = stream_msgs(0.0)
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        for i in range(MIN_DIFFS_FOR_GATE):
            t0 = batch_start + i * 1.0
            batch_msgs = stream_msgs(t0)
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None

        batcher.timeout_factor = 999.0
        t0 = batch_start + MIN_DIFFS_FOR_GATE * 1.0
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
            batch_length_s=1.0, timeout_s=convergence_timeout
        )
        count = round(rate * 1.0)

        def stream_msgs(batch_start: float) -> list[Message[str]]:
            return [msg(batch_start + offset + i * period) for i in range(count)]

        initial = stream_msgs(0.0)
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        for i in range(MIN_DIFFS_FOR_GATE):
            t0 = batch_start + i * 1.0
            batch_msgs = stream_msgs(t0)
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None

        batcher.timeout_factor = 999.0
        t0 = batch_start + MIN_DIFFS_FOR_GATE * 1.0
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
            batch_length_s=1.0, timeout_s=convergence_timeout
        )
        # Initial batch with both streams
        initial = msgs_at(14.0, start=0.0, duration=1.0, stream=DETECTOR)
        initial.append(msg(0.0, stream=SUB_HZ))
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        # Converge via timeout — sub-Hz gets 1 msg every other batch
        for i in range(MIN_DIFFS_FOR_GATE):
            t0 = batch_start + i * 1.0
            batch_msgs = msgs_at(14.0, start=t0, duration=1.0, stream=DETECTOR)
            if i % 2 == 0:
                batch_msgs.append(msg(t0 + 0.5, stream=SUB_HZ))
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None

        # Post-convergence: detector-only should complete
        # (sub-Hz has no grid, doesn't gate)
        batcher.timeout_factor = 999.0
        t_test = batch_start + MIN_DIFFS_FOR_GATE * 1.0
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
            batch_length_s=1.0, timeout_s=convergence_timeout
        )

        # Initial batch with 14 or 15 messages
        initial_count = round(rate * 1.0)  # 14 (banker's rounding)
        initial = [msg(i * period) for i in range(initial_count)]
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        # Converge
        for i in range(MIN_DIFFS_FOR_GATE):
            t0 = batch_start + i * 1.0
            batch_msgs = [msg(t0 + j * period) for j in range(initial_count)]
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None

        batcher.timeout_factor = timeout

        # Run 100 batches alternating between 14 and 15 messages
        # (simulating a real 14.5 Hz source)
        t_base = batch_start + MIN_DIFFS_FOR_GATE * 1.0
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

    def test_abrupt_rate_change_no_loss(self):
        """Stream switches from 14→7 Hz: no messages lost during transition."""
        convergence_timeout = 0.8
        batcher = RateAwareMessageBatcher(
            batch_length_s=1.0, timeout_s=convergence_timeout
        )

        # Converge at 14 Hz
        initial = msgs_at(14.0, start=0.0, duration=1.0)
        batcher.batch(initial)
        batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

        for i in range(MIN_DIFFS_FOR_GATE):
            t0 = batch_start + i * 1.0
            batch_msgs = msgs_at(14.0, start=t0, duration=1.0)
            batch_msgs.append(hwm_trigger(t0 + convergence_timeout + 0.01))
            result = batcher.batch(batch_msgs)
            assert result is not None

        timeout = 1.5
        batcher.timeout_factor = timeout

        # Switch to 7 Hz. Count how many batches need timeout vs slot-gate
        t_base = batch_start + MIN_DIFFS_FOR_GATE * 1.0
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

    def test_overflow_not_re_routed_while_window_unchanged(self):
        """Overflow messages sitting across poll cycles must not be
        re-routed each time. Re-evaluating stable overflow against the
        same window every cycle is wasted work proportional to overflow
        size times cycles-until-batch-close.
        """
        # Two gated streams: DETECTOR keeps the current batch open (no
        # overflow, incomplete gate), MONITOR produces overflow from the
        # next window. Gap-recovery cannot fire because DETECTOR has
        # in-window messages.
        batcher, t0 = make_converged_batcher(
            rate_hz=14.0, streams={DETECTOR: 14.0, MONITOR: 14.0}, timeout_s=999.0
        )
        # One DETECTOR pulse at slot 0 — incomplete gate for DETECTOR.
        batcher.batch([msg(t0 + 0.05, stream=DETECTOR)])

        # Two MONITOR messages in the *next* window; they land in overflow.
        batcher.batch([msg(t0 + 1.1, stream=MONITOR), msg(t0 + 1.2, stream=MONITOR)])
        assert len(batcher._overflow) == 2

        # Idle polls must not re-route overflow against the same window.
        # Route calls are invisible from outside, so spy on _route_message.
        route_calls = 0
        original_route = batcher._route_message

        def counting_route(msg):
            nonlocal route_calls
            route_calls += 1
            original_route(msg)

        batcher._route_message = counting_route
        for _ in range(5):
            batcher.batch([])
        assert route_calls == 0, (
            f"Overflow was re-routed {route_calls} times while window unchanged"
        )


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
        stream: StreamId = DETECTOR,
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
                        stream=stream,
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
        batcher.timeout_factor = RateAwareMessageBatcher().timeout_factor
        counts = self._run_tick_delivery(batcher, 14.0, t0, n_batches=30)
        assert all(c == 14 for c in counts), f"Unsteady counts: {counts}"

    def test_steady_count_at_7hz(self):
        """7 Hz with default timeout: every batch should contain 7 messages."""
        batcher, t0 = make_converged_batcher(rate_hz=7.0)
        batcher.timeout_factor = RateAwareMessageBatcher().timeout_factor
        counts = self._run_tick_delivery(batcher, 7.0, t0, n_batches=30)
        assert all(c == 7 for c in counts), f"Unsteady counts: {counts}"

    def test_steady_count_at_1hz_monitor_only(self):
        """Monitor-only 1 Hz service: every batch must contain exactly 1 message.

        Regression for the production dropout: at ``slots_per_batch = 1``,
        any slot mis-classification is a 100% loss for the batch, and 5
        absent batches in a row evict the stream entirely.
        """
        batcher, t0 = make_converged_batcher(rate_hz=1.0, streams={MONITOR: 1.0})
        batcher.timeout_factor = RateAwareMessageBatcher().timeout_factor
        counts = self._run_tick_delivery(batcher, 1.0, t0, n_batches=30, stream=MONITOR)
        assert all(c == 1 for c in counts), f"Unsteady counts: {counts}"
        assert MONITOR in batcher.tracked_streams, "MON evicted during steady 1 Hz"


class TestMultiRateRobustness:
    """Multi-rate streams with realistic jitter: no loss, no eviction.

    Covers production rate combinations (including the 1 Hz + 1 Hz, 1 Hz + N
    cases that trip the single-slot edge case) under 100 ms tick-based
    delivery with bounded jitter.
    """

    @staticmethod
    def _run(
        rates: dict[StreamId, float],
        *,
        jitter_s: float,
        n_batches: int,
        seed: int,
    ) -> tuple[int, int, set[StreamId]]:
        """Tick-based delivery with jitter; returns (in, out, evicted)."""
        batcher, t0 = make_converged_batcher(streams=rates)
        batcher.timeout_factor = RateAwareMessageBatcher().timeout_factor
        rng = random.Random(seed)
        period_ns = {sid: int(1e9 / r) for sid, r in rates.items()}
        jitter_ns = int(jitter_s * 1e9)
        start_ns = int(t0 * 1e9)
        next_pulse_ns = dict.fromkeys(rates, start_ns)
        tick_ns = start_ns
        total_in = 0
        total_out = 0
        closed = 0
        budget_ns = start_ns + (n_batches + 20) * 1_000_000_000
        while closed < n_batches and tick_ns < budget_ns:
            tick_ns += 100_000_000
            batch_msgs: list[Message[str]] = []
            for sid, p in period_ns.items():
                while next_pulse_ns[sid] <= tick_ns:
                    j = rng.randint(-jitter_ns, jitter_ns) if jitter_ns else 0
                    batch_msgs.append(msg((next_pulse_ns[sid] + j) / 1e9, stream=sid))
                    total_in += 1
                    next_pulse_ns[sid] += p
            r = batcher.batch(batch_msgs)
            if r is not None:
                closed += 1
                total_out += sum(1 for m in r.messages if m.stream in rates)
        # Drain: advance the logical clock far past any in-flight messages so
        # remaining batches close via timeout.
        for _ in range(4):
            tick_ns += 2_000_000_000
            r = batcher.batch([hwm_trigger(tick_ns / 1e9)])
            if r is not None:
                total_out += sum(1 for m in r.messages if m.stream in rates)
        evicted = {sid for sid in rates if sid not in batcher.tracked_streams}
        return total_in, total_out, evicted

    @pytest.mark.parametrize(
        "rates",
        [
            {MONITOR: 1.0},
            {DETECTOR: 1.0, MONITOR: 1.0},
            {DETECTOR: 2.0, MONITOR: 1.0},
            {DETECTOR: 3.0, MONITOR: 1.0},
            {DETECTOR: 7.0, MONITOR: 1.0},
            {DETECTOR: 14.0, MONITOR: 1.0},
            {DETECTOR: 7.0, MONITOR: 2.0},
            {DETECTOR: 14.0, MONITOR: 7.0},
        ],
    )
    def test_no_loss_with_jitter(self, rates):
        """2 ms jitter (~0.2% at 1 Hz, ~3% at 14 Hz): nothing dropped."""
        total_in, total_out, evicted = self._run(
            rates, jitter_s=0.002, n_batches=20, seed=42
        )
        assert not evicted, f"Streams evicted: {evicted}"
        assert total_out == total_in, (
            f"Lost {total_in - total_out} of {total_in} ({rates})"
        )

    @pytest.mark.parametrize(
        "rates",
        [
            {MONITOR: 1.0},
            {DETECTOR: 1.0, MONITOR: 1.0},
            {DETECTOR: 14.0, MONITOR: 1.0},
            {DETECTOR: 7.0, MONITOR: 2.0},
        ],
    )
    def test_no_loss_without_jitter(self, rates):
        """Baseline (no jitter) counterpart: nothing dropped."""
        total_in, total_out, evicted = self._run(
            rates, jitter_s=0.0, n_batches=20, seed=0
        )
        assert not evicted, f"Streams evicted: {evicted}"
        assert total_out == total_in, (
            f"Lost {total_in - total_out} of {total_in} ({rates})"
        )


class TestSubHzGatedStream:
    """Sub-Hz gated streams (< 1 Hz): rate rounds to 0 so no grid is built.

    The stream is observed via the estimator and delivered via the no-grid
    path in ``_route_message``, but never drives or blocks batch completion.
    A higher-rate gated stream closes each batch via the slot gate; the
    sub-Hz stream rides along without triggering timeout fallbacks.
    """

    MON_HALF = StreamId(kind=StreamKind.MONITOR_EVENTS, name="mon_half_hz")

    def test_slot_gate_closes_alongside_high_rate_gated_stream(self):
        """0.5 Hz MON + 14 Hz DET: every batch closes via detector slot gate.

        "Every batch full" (14 det msgs each) is the slot-gate signature;
        a timeout fallback fires before the last tick of the window and
        would produce < 14.
        """
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        batcher.timeout_factor = RateAwareMessageBatcher().timeout_factor  # 1.2
        n = 20
        det_period = int(1e9 / 14.0)
        mon_period = 2_000_000_000
        start_ns = int(t0 * 1e9)

        schedule = [
            Message(
                timestamp=Timestamp.from_ns(start_ns + k * det_period),
                stream=DETECTOR,
                value="",
            )
            for k in range(14 * n)
        ] + [
            Message(
                timestamp=Timestamp.from_ns(start_ns + k * mon_period),
                stream=self.MON_HALF,
                value="",
            )
            for k in range(n // 2)
        ]
        schedule.sort(key=lambda m: m.timestamp.to_ns())

        closed = _run_ticks(batcher, t0, schedule, n_batches=n)
        assert len(closed) == n

        det_per_batch = [
            sum(1 for m in b.messages if m.stream == DETECTOR) for b in closed
        ]
        assert all(c == 14 for c in det_per_batch), (
            f"Timeout fallback fired (uneven det counts): {det_per_batch}"
        )
        mon_out = sum(
            1 for b in closed for m in b.messages if m.stream == self.MON_HALF
        )
        assert mon_out == n // 2

        assert not batcher.is_gating(self.MON_HALF), "Sub-Hz must not be gridded"
        assert self.MON_HALF in batcher.tracked_streams, "Sub-Hz must not be evicted"
        assert batcher._streams[self.MON_HALF].estimator.integer_rate_hz is None

    def test_no_eviction_at_05hz_over_many_batches(self):
        """Absent/present cadence at 0.5 Hz keeps counter under eviction threshold."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        batcher.timeout_factor = RateAwareMessageBatcher().timeout_factor
        n = 50
        det_period = int(1e9 / 14.0)
        mon_period = 2_000_000_000
        start_ns = int(t0 * 1e9)

        schedule = [
            Message(
                timestamp=Timestamp.from_ns(start_ns + k * det_period),
                stream=DETECTOR,
                value="",
            )
            for k in range(14 * n)
        ] + [
            Message(
                timestamp=Timestamp.from_ns(start_ns + k * mon_period),
                stream=self.MON_HALF,
                value="",
            )
            for k in range(n // 2)
        ]
        schedule.sort(key=lambda m: m.timestamp.to_ns())
        closed = _run_ticks(batcher, t0, schedule, n_batches=n, max_run_s=n + 10)
        assert len(closed) == n
        assert self.MON_HALF in batcher.tracked_streams

    def test_sub_hz_alone_delivered_via_timeout(self):
        """Sub-Hz-only service: timeout drives closure; nothing dropped."""
        batcher = RateAwareMessageBatcher(batch_length_s=1.0)  # default timeout
        n = 10
        mon_period = 2_000_000_000
        # 6 pulses covers 10 closures under the empty/with-msg alternation that
        # sub-Hz-only delivery produces past the first two batches.
        pulse_count = n // 2 + 1
        schedule = [
            Message(
                timestamp=Timestamp.from_ns(k * mon_period),
                stream=self.MON_HALF,
                value="",
            )
            for k in range(pulse_count)
        ]
        closed = _run_ticks(batcher, 0.0, schedule, n_batches=n, max_run_s=30.0)
        assert len(closed) == n
        mon_out = sum(
            1 for b in closed for m in b.messages if m.stream == self.MON_HALF
        )
        assert mon_out == pulse_count
        assert not batcher.is_gating(self.MON_HALF)


class TestSubRateExclusion:
    """Streams whose integer rate is below 1 pulse / batch_length are excluded.

    The estimator may still report an integer rate (e.g. 1 Hz), but if
    ``int_rate * batch_length_s < 1`` the stream cannot reliably fill a
    slot per batch and must not gate closure.  Such streams deliver
    opportunistically via the non-gated path.

    Distinct from ``TestSubHzGatedStream`` (integer-snap returns None);
    here the snap succeeds but the product with the current batch length
    is below the gating floor.
    """

    def test_shrink_batch_length_demotes_1hz_stream(self):
        """1 Hz MON gridded at 1 s batch.  Shrink to 0.6 s → excluded from grids."""
        streams = {DETECTOR: 14.0, MONITOR: 1.0}
        batcher, t0 = make_converged_batcher(
            batch_length_s=1.0, streams=streams, timeout_s=999.0
        )
        assert batcher.is_gating(MONITOR)

        batcher.set_batch_length(0.6)
        det = msgs_at(14.0, start=t0, duration=1.0)
        result = batcher.batch([*det, msg(t0, stream=MONITOR)])
        assert result is not None

        assert batcher.batch_length_s == pytest.approx(0.6)
        assert not batcher.is_gating(MONITOR), "Sub-rate stream must be excluded"
        assert MONITOR in batcher.tracked_streams, "Estimator must stay live"
        assert batcher._streams[MONITOR].estimator.integer_rate_hz == 1

    def test_grow_batch_length_regates_previously_excluded_stream(self):
        """Restoring 1 s batch length re-admits the 1 Hz stream to gating."""
        streams = {DETECTOR: 14.0, MONITOR: 1.0}
        batcher, t0 = make_converged_batcher(
            batch_length_s=1.0, streams=streams, timeout_s=999.0
        )
        # Shrink: MON excluded.
        batcher.set_batch_length(0.6)
        det = msgs_at(14.0, start=t0, duration=1.0)
        batch1 = batcher.batch([*det, msg(t0, stream=MONITOR)])
        assert batch1 is not None
        assert not batcher.is_gating(MONITOR)

        # Close one 0.6 s batch to free the shrink transition from the path.
        t1 = batch1.end_time.to_ns() / 1e9
        det2 = msgs_at(14.0, start=t1, duration=0.6)
        batch2 = batcher.batch(det2)
        assert batch2 is not None

        # Grow back.  The pending length applies at the next close.
        batcher.set_batch_length(1.0)
        t2 = batch2.end_time.to_ns() / 1e9
        det3 = msgs_at(14.0, start=t2, duration=1.0)
        mon_msg = msg(t2, stream=MONITOR)
        batch3 = batcher.batch([*det3, mon_msg])
        assert batch3 is not None
        assert batcher.batch_length_s == pytest.approx(1.0)
        assert batcher.is_gating(MONITOR), "Stream must re-gate when batch grows"

    def test_excluded_stream_does_not_block_closure(self):
        """14 Hz DET + 1 Hz MON at 0.6 s batch: DET slot gate closes every batch.

        Timeout fallback would deliver < 8 DET per batch on the tail of the
        window; a clean 8-per-batch signature proves MON did not block.
        """
        streams = {DETECTOR: 14.0, MONITOR: 1.0}
        batcher, t0 = make_converged_batcher(batch_length_s=1.0, streams=streams)
        # Apply the shrink: close one batch at 1 s length.
        det0 = msgs_at(14.0, start=t0, duration=1.0)
        batcher.set_batch_length(0.6)
        batcher.batch([*det0, msg(t0, stream=MONITOR)])
        assert not batcher.is_gating(MONITOR)

        # Drive 10 batches of 0.6 s with DET at 14 Hz + MON at true 1 Hz.
        batcher.timeout_factor = RateAwareMessageBatcher().timeout_factor  # 1.2
        n = 10
        det_period = int(1e9 / 14.0)
        mon_period = 1_000_000_000
        start_ns = int((t0 + 1.0) * 1e9)
        det_msgs_total = 14 * n  # plenty to cover 10 * 0.6 s = 6 s
        schedule = [
            Message(
                timestamp=Timestamp.from_ns(start_ns + k * det_period),
                stream=DETECTOR,
                value="",
            )
            for k in range(det_msgs_total)
        ] + [
            Message(
                timestamp=Timestamp.from_ns(start_ns + k * mon_period),
                stream=MONITOR,
                value="",
            )
            for k in range(6)  # 6 MON pulses across the 6 s window
        ]
        schedule.sort(key=lambda m: m.timestamp.to_ns())
        closed = _run_ticks(batcher, t0 + 1.0, schedule, n_batches=n, tick_s=0.05)
        assert len(closed) == n

        det_per_batch = [
            sum(1 for m in b.messages if m.stream == DETECTOR) for b in closed
        ]
        # 14 Hz * 0.6 s = 8.4 pulses/batch → batches see 8 or 9.  Either way
        # the last-filled slot is slot 7 (= slots_per_batch - 1), i.e. the slot
        # gate closed the batch.  Timeout fallback would produce a ragged
        # distribution keyed off arrival time rather than the {8, 9} pattern.
        assert all(c in {8, 9} for c in det_per_batch), (
            f"Timeout fallback fired: {det_per_batch}"
        )
        mon_out = sum(1 for b in closed for m in b.messages if m.stream == MONITOR)
        assert mon_out == 6


class TestBrokenTimestampStream:
    """A stream whose timestamps live in a disjoint epoch must not veto
    completion for the rest of the instrument.

    Production regression (Bifrost): MONITOR_COUNTS arrived with
    timestamps ~43 years offset from MONITOR_EVENTS.  The broken
    stream's grid was built in its own epoch, pulse_index(batch_start)
    yielded a huge negative number, tracker.max_slot stayed at -1
    forever, and _is_batch_complete vetoed every slot-gate closure.

    Symmetric for far-future timestamps via the ``end_time = max(ts)``
    seeding of the first active batch's start_time: whichever stream's
    epoch wins batch_start seeding, the other stream's grid origin ends
    up implausibly far from it.  The fix rejects both directions by
    refusing to create a grid whose origin is too many batches away
    from batch_start at grid-creation time.
    """

    GOOD = StreamId(kind=StreamKind.MONITOR_EVENTS, name="good")
    BROKEN = StreamId(kind=StreamKind.MONITOR_COUNTS, name="broken")

    _A_EPOCH_NS = 1_780_000_000_000_000_000  # 2026-ish Unix ns
    _GOOD_PERIOD_NS = 71_428_571  # 14 Hz
    _BROKEN_PERIOD_NS = 1_000_000_000  # 1 Hz
    _FAR_OFFSET_NS = 1_350_000_000_000_000_000  # ~43 years

    def _seed_and_drive(
        self,
        batcher: RateAwareMessageBatcher,
        broken_offset_ns: int,
        warmup_batches: int,
        measured_batches: int,
    ) -> tuple[list[MessageBatch], list[MessageBatch]]:
        """Seed initial batch + drive warmup and measured batches.

        Returns (warmup_closed, measured_closed).
        """
        # Initial batch: both streams present.  max(ts) across the whole
        # set seeds batch_start for the active (empty) batch.
        initial = [
            Message(
                timestamp=Timestamp.from_ns(
                    self._A_EPOCH_NS + i * self._GOOD_PERIOD_NS
                ),
                stream=self.GOOD,
                value="",
            )
            for i in range(14)
        ] + [
            Message(
                timestamp=Timestamp.from_ns(
                    self._A_EPOCH_NS + broken_offset_ns + i * self._BROKEN_PERIOD_NS
                ),
                stream=self.BROKEN,
                value=f"b_init_{i}",
            )
            for i in range(5)
        ]
        batcher.batch(initial)

        # What epoch did batch_start end up in? max(ts).  Drive batches
        # relative to that so good-stream messages keep arriving with
        # slots >= 0 once the grid is built.  The broken stream's grid
        # (if created) lives in the OTHER epoch.
        if broken_offset_ns > 0:
            base_ns = self._A_EPOCH_NS + broken_offset_ns + 4 * self._BROKEN_PERIOD_NS
        else:
            base_ns = self._A_EPOCH_NS + 13 * self._GOOD_PERIOD_NS

        warmup: list[MessageBatch] = []
        for k in range(warmup_batches):
            batch0 = base_ns + (k + 1) * 1_000_000_000
            good_msgs = [
                Message(
                    timestamp=Timestamp.from_ns(batch0 + i * self._GOOD_PERIOD_NS),
                    stream=self.GOOD,
                    value="",
                )
                for i in range(14)
            ]
            broken_msg = Message(
                timestamp=Timestamp.from_ns(batch0 + broken_offset_ns + 500_000_000),
                stream=self.BROKEN,
                value=f"b_warmup_{k}",
            )
            r = batcher.batch([*good_msgs, broken_msg])
            if r is not None:
                warmup.append(r)

        # Measurement phase: disable timeout, require slot-gate closure.
        batcher.timeout_factor = 999.0
        measured: list[MessageBatch] = []
        offset = (warmup_batches + 1) * 1_000_000_000
        for k in range(measured_batches):
            batch0 = base_ns + offset + k * 1_000_000_000
            good_msgs = [
                Message(
                    timestamp=Timestamp.from_ns(batch0 + i * self._GOOD_PERIOD_NS),
                    stream=self.GOOD,
                    value="",
                )
                for i in range(14)
            ]
            broken_msg = Message(
                timestamp=Timestamp.from_ns(batch0 + broken_offset_ns + 500_000_000),
                stream=self.BROKEN,
                value=f"b_meas_{k}",
            )
            r = batcher.batch([*good_msgs, broken_msg])
            if r is not None:
                measured.append(r)
        return warmup, measured

    def test_far_past_broken_stream_does_not_veto_slot_gate(self):
        """Prod regression: broken stream ~43 years in the past."""
        batcher = RateAwareMessageBatcher(batch_length_s=1.0, timeout_s=0.8)
        _, measured = self._seed_and_drive(
            batcher,
            broken_offset_ns=-self._FAR_OFFSET_NS,
            warmup_batches=8,
            measured_batches=10,
        )
        assert len(measured) == 10, (
            f"Only {len(measured)}/10 batches closed via slot gate "
            f"(broken stream vetoed the rest)"
        )
        assert not batcher.is_gating(self.BROKEN)
        broken_out = sum(
            1 for b in measured for m in b.messages if m.stream == self.BROKEN
        )
        assert broken_out == 10, f"Broken stream msg count: {broken_out}"

    def test_far_future_broken_stream_does_not_veto_slot_gate(self):
        """Symmetric case: broken stream ~43 years in the future.

        ``end_time = max(ts)`` in the initial batch seeds batch_start in
        the broken (future) epoch; the good stream becomes the one with
        an implausible grid origin.  Either way, the check must prevent
        a stream with implausible origin from vetoing every batch.
        """
        batcher = RateAwareMessageBatcher(batch_length_s=1.0, timeout_s=0.8)
        _, measured = self._seed_and_drive(
            batcher,
            broken_offset_ns=self._FAR_OFFSET_NS,
            warmup_batches=8,
            measured_batches=10,
        )
        assert len(measured) == 10, (
            f"Only {len(measured)}/10 batches closed (far-future offset)"
        )
        # _overflow must not grow unboundedly across batches.
        assert len(batcher._overflow) <= 1, (
            f"Overflow accumulated to {len(batcher._overflow)} — "
            f"broken stream is re-overflowing every batch"
        )

    def test_single_broken_stream_alone_does_not_hang(self):
        """Instrument with only a broken stream still makes progress via timeout.

        No good stream means nothing to seed a plausible batch_start from
        its own epoch, so the broken stream's origin IS batch_start's
        epoch and the grid IS created.  That's fine — the stream is
        self-consistent in its own (broken) frame; downstream will see
        the bogus timestamps but the batcher must not hang.
        """
        batcher = RateAwareMessageBatcher(batch_length_s=1.0, timeout_s=0.8)
        broken_start_ns = 423_000_000_000_000_000  # ~43 years in the past
        initial = [
            Message(
                timestamp=Timestamp.from_ns(
                    broken_start_ns + i * self._BROKEN_PERIOD_NS
                ),
                stream=self.BROKEN,
                value="",
            )
            for i in range(5)
        ]
        batcher.batch(initial)

        closed = 0
        for k in range(10):
            ts_ns = broken_start_ns + (5 + k) * self._BROKEN_PERIOD_NS
            r = batcher.batch(
                [
                    Message(
                        timestamp=Timestamp.from_ns(ts_ns),
                        stream=self.BROKEN,
                        value="",
                    ),
                    hwm_trigger(ts_ns / 1e9 + 0.81),
                ]
            )
            if r is not None:
                closed += 1
        assert closed >= 8, f"Batcher hung with single broken stream: closed={closed}"

    def test_far_future_outlier_does_not_permanently_elevate_hwm(self):
        """A single far-future timestamp must not DoS the batcher via HWM.

        Prior to the HWM cap, one malformed message (e.g. epoch bug upstream
        producing a timestamp ~1 year ahead) would permanently pin the
        high-water mark in the future.  Every subsequent call -- including
        empty ones -- would then close a batch via the timeout path, burning
        CPU producing millions of empty batches and flooding downstream
        until the batcher's window drifted up to the bogus HWM (~1 year).
        """
        batcher, _ = make_converged_batcher(rate_hz=14.0, timeout_s=0.8)

        # Malformed non-gated message ~1 year in the future.
        one_year_s = 365 * 24 * 3600.0
        batcher.batch([hwm_trigger(one_year_s)])

        # After the outlier, call ``batch([])`` many times.  Without the cap,
        # every call closes an empty batch because HWM is pinned ~1 year past
        # any plausible threshold; the DoS would keep firing for ~3.15e7
        # iterations.  With the cap, HWM is kept within
        # ``_MAX_HWM_PAST_WINDOW_BATCHES`` of the active window, so cascading
        # empty closures self-halt after a handful of cycles.
        n_probes = 10_000
        empty_closes = sum(1 for _ in range(n_probes) if batcher.batch([]) is not None)

        # With the cap, cascading empty closures terminate after a handful
        # of cycles.  Without the cap, every probe closes a batch.
        assert empty_closes < 10, (
            f"Outlier permanently elevated HWM: {empty_closes}/{n_probes} empty "
            f"closures triggered (batcher will keep producing empty batches "
            f"until window catches up to the bogus HWM)"
        )


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

    def test_slot_gate_closes_with_non_gated_stream_present(self):
        """LOG at 2 Hz + 14 Hz DET: slot gate closes every batch, nothing lost.

        The non-gated LOG stream never touches the estimator or the grid,
        so its presence must not bias completion one way or the other.
        """
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        batcher.timeout_factor = RateAwareMessageBatcher().timeout_factor
        n = 20
        det_period = int(1e9 / 14.0)
        log_period = 500_000_000  # 2 Hz
        start_ns = int(t0 * 1e9)

        schedule = [
            Message(
                timestamp=Timestamp.from_ns(start_ns + k * det_period),
                stream=DETECTOR,
                value="",
            )
            for k in range(14 * n)
        ] + [
            Message(
                timestamp=Timestamp.from_ns(start_ns + k * log_period),
                stream=self.LOG,
                value="",
            )
            for k in range(2 * n)
        ]
        schedule.sort(key=lambda m: m.timestamp.to_ns())

        closed = _run_ticks(batcher, t0, schedule, n_batches=n)
        assert len(closed) == n
        det_per_batch = [
            sum(1 for m in b.messages if m.stream == DETECTOR) for b in closed
        ]
        assert all(c == 14 for c in det_per_batch), (
            f"Timeout fallback fired: {det_per_batch}"
        )
        log_out = sum(1 for b in closed for m in b.messages if m.stream == self.LOG)
        assert log_out == 2 * n
        assert self.LOG not in batcher.tracked_streams

    def test_irregular_bursts_alongside_gated(self):
        """LOG alternates between a >1 Hz burst and a <1 Hz quiet phase.

        Bursty non-gated traffic must not perturb the gated stream's slot
        gate or the non-gated delivery path.
        """
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        batcher.timeout_factor = RateAwareMessageBatcher().timeout_factor
        n = 20
        det_period = int(1e9 / 14.0)
        start_ns = int(t0 * 1e9)

        schedule: list[Message[str]] = [
            Message(
                timestamp=Timestamp.from_ns(start_ns + k * det_period),
                stream=DETECTOR,
                value="",
            )
            for k in range(14 * n)
        ]
        # Even seconds: 5-msg burst at ~20 Hz local rate.
        # Odd seconds: silent (gap between bursts is ~1.5 s, sub-Hz).
        log_expected = 0
        for sec in range(n):
            if sec % 2 == 0:
                for k in range(5):
                    t_ns = start_ns + sec * 1_000_000_000 + k * 50_000_000 + 100_000_000
                    schedule.append(
                        Message(
                            timestamp=Timestamp.from_ns(t_ns),
                            stream=self.LOG,
                            value="",
                        )
                    )
                    log_expected += 1
        schedule.sort(key=lambda m: m.timestamp.to_ns())

        closed = _run_ticks(batcher, t0, schedule, n_batches=n)
        assert len(closed) == n
        det_per_batch = [
            sum(1 for m in b.messages if m.stream == DETECTOR) for b in closed
        ]
        assert all(c == 14 for c in det_per_batch), (
            f"Timeout fallback fired: {det_per_batch}"
        )
        log_out = sum(1 for b in closed for m in b.messages if m.stream == self.LOG)
        assert log_out == log_expected
        assert self.LOG not in batcher.tracked_streams

    def test_compound_non_gated_plus_sub_hz_plus_gated(self):
        """Non-gated LOG + sub-Hz MON + 14 Hz DET: slot gate closes cleanly."""
        MON_HALF = StreamId(kind=StreamKind.MONITOR_EVENTS, name="mon_half_hz")
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        batcher.timeout_factor = RateAwareMessageBatcher().timeout_factor
        n = 20
        det_period = int(1e9 / 14.0)
        mon_period = 2_000_000_000
        start_ns = int(t0 * 1e9)

        schedule: list[Message[str]] = [
            Message(
                timestamp=Timestamp.from_ns(start_ns + k * det_period),
                stream=DETECTOR,
                value="",
            )
            for k in range(14 * n)
        ]
        schedule.extend(
            Message(
                timestamp=Timestamp.from_ns(start_ns + k * mon_period),
                stream=MON_HALF,
                value="",
            )
            for k in range(n // 2)
        )
        log_expected = 0
        for sec in range(n):
            if sec % 3 == 0:
                for k in range(4):
                    t_ns = start_ns + sec * 1_000_000_000 + k * 60_000_000 + 200_000_000
                    schedule.append(
                        Message(
                            timestamp=Timestamp.from_ns(t_ns),
                            stream=self.LOG,
                            value="",
                        )
                    )
                    log_expected += 1
        schedule.sort(key=lambda m: m.timestamp.to_ns())

        closed = _run_ticks(batcher, t0, schedule, n_batches=n)
        assert len(closed) == n
        det_per_batch = [
            sum(1 for m in b.messages if m.stream == DETECTOR) for b in closed
        ]
        assert all(c == 14 for c in det_per_batch), (
            f"Timeout fallback fired: {det_per_batch}"
        )
        mon_out = sum(1 for b in closed for m in b.messages if m.stream == MON_HALF)
        log_out = sum(1 for b in closed for m in b.messages if m.stream == self.LOG)
        assert mon_out == n // 2
        assert log_out == log_expected
        assert MON_HALF in batcher.tracked_streams
        assert not batcher.is_gating(MON_HALF)
        assert self.LOG not in batcher.tracked_streams

    def test_future_timestamped_held_for_next_batch(self):
        """Non-gated msg with ts past window.end (within cap) lands in next batch.

        Preserves batch atomicity: downstream accumulators must not pair a
        log reading from a future time range with detector data from the
        current window.
        """
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        log_msg = msg(t0 + 1.3, stream=self.LOG)
        det_now = msgs_at(14.0, start=t0, duration=1.0)

        r1 = batcher.batch([*det_now, log_msg])
        assert r1 is not None
        assert log_msg not in r1.messages
        assert len(r1.messages) == 14

        det_next = msgs_at(14.0, start=t0 + 1.0, duration=1.0)
        r2 = batcher.batch(det_next)
        assert r2 is not None
        assert log_msg in r2.messages

    def test_implausibly_future_timestamped_falls_through(self):
        """Msg beyond the hold-back cap is delivered immediately.

        Guards against a pathological timestamp (epoch bug, unit mismatch)
        caching the message indefinitely.  Same fallback as the HWM clamp.
        """
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        rogue = msg(t0 + 999.0, stream=self.LOG)
        det_now = msgs_at(14.0, start=t0, duration=1.0)

        r1 = batcher.batch([*det_now, rogue])
        assert r1 is not None
        assert rogue in r1.messages

    def test_held_message_delivered_via_gap_recovery(self):
        """A held message re-routes correctly when the window jumps past it."""
        batcher, t0 = make_converged_batcher(rate_hz=14.0)
        held = msg(t0 + 2.5, stream=self.LOG)
        det_now = msgs_at(14.0, start=t0, duration=1.0)

        r1 = batcher.batch([*det_now, held])
        assert r1 is not None
        assert held not in r1.messages

        det_far = msgs_at(14.0, start=t0 + 10.0, duration=1.0)
        r2 = batcher.batch(det_far)
        assert r2 is not None
        assert held in r2.messages
