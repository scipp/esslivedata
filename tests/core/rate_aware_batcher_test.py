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
