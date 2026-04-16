# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Rate-aware message batcher.

Batches messages based on per-stream rate estimation and slot-based completion.
A batch is considered complete for a given stream when a message arrives whose
timestamp falls in the last expected "pulse slot" for that stream within the
batch window — not when a fixed message count is reached.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.core.message_batcher import MessageBatch, MessageBatcher
from ess.livedata.core.timestamp import Duration, Timestamp

GATED_STREAM_KINDS = frozenset(
    {
        StreamKind.DETECTOR_EVENTS,
        StreamKind.MONITOR_EVENTS,
        StreamKind.MONITOR_COUNTS,
        StreamKind.AREA_DETECTOR,
    }
)

MIN_BATCHES_FOR_GATE = 3
ABSENT_BATCHES_FOR_EVICTION = 5


@dataclass
class _StreamState:
    """Per-stream rate tracking and convergence state."""

    rate_hz: float | None = None
    observation_count: int = 0
    phase_offset_ns: int = 0
    absent_batches: int = 0

    @property
    def converged(self) -> bool:
        return (
            self.observation_count >= MIN_BATCHES_FOR_GATE and self.rate_hz is not None
        )

    @property
    def integer_rate_hz(self) -> int | None:
        """Rate rounded to the nearest integer Hz.

        ESS sources publish at integer Hz rates. Rounding eliminates the
        fractional-Hz EMA noise that otherwise causes slot-boundary
        oscillation (±1 message per batch).
        """
        return None if self.rate_hz is None else round(self.rate_hz)

    def expected_count(self, batch_length_s: float) -> int:
        if self.integer_rate_hz is None:
            return 0
        return round(self.integer_rate_hz * batch_length_s)

    def update_rate(
        self, message_count: int, batch_length_s: float, alpha: float
    ) -> None:
        observed = message_count / batch_length_s
        if self.rate_hz is None:
            self.rate_hz = observed
        else:
            self.rate_hz = alpha * observed + (1 - alpha) * self.rate_hz
        self.observation_count += 1


@dataclass
class _BatchStreamTracker:
    """Tracks messages for one stream within the current batch."""

    messages: list[Message[Any]] = field(default_factory=list)
    max_slot: int = -1

    def add(self, msg: Message[Any], slot: int) -> None:
        self.messages.append(msg)
        if slot > self.max_slot:
            self.max_slot = slot

    @property
    def count(self) -> int:
        return len(self.messages)


class RateAwareMessageBatcher(MessageBatcher):
    """A batcher that uses per-stream rate estimation and slot-based completion.

    Completion for each gated stream is determined by whether a message has
    arrived whose timestamp falls in the last expected pulse slot, rather than
    by message count alone. This handles missing pulses (never published) and
    split messages (two messages with the same timestamp) gracefully.

    Streams whose kind is not in ``GATED_STREAM_KINDS`` are included
    opportunistically in whatever batch is active.
    """

    def __init__(
        self,
        batch_length_s: float = 1.0,
        timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        ema_alpha: float = 0.05,
    ) -> None:
        self._batch_length = Duration.from_seconds(batch_length_s)
        self._timeout_s = timeout_s if timeout_s is not None else batch_length_s * 1.5
        self._clock = clock
        self._ema_alpha = ema_alpha

        self._streams: dict[StreamId, _StreamState] = {}

        self._pending_batch_length: Duration | None = None
        self._active_batch: MessageBatch | None = None
        self._batch_trackers: dict[StreamId, _BatchStreamTracker] = {}
        self._batch_start_wall: float | None = None
        self._overflow: list[Message[Any]] = []

    @property
    def batch_length_s(self) -> float:
        return self._batch_length.to_seconds()

    def set_batch_length(self, batch_length_s: float) -> None:
        """Update the batch length for future batches.

        The current active batch completes using its original length.
        The new length takes effect when the next batch starts.
        """
        self._pending_batch_length = Duration.from_seconds(batch_length_s)

    def batch(self, messages: list[Message[Any]]) -> MessageBatch | None:
        messages = [msg for msg in messages if isinstance(msg.timestamp, Timestamp)]

        if self._active_batch is None:
            return self._start_first_batch(messages)

        all_messages = self._overflow + messages
        self._overflow = []

        for msg in all_messages:
            self._route_message(msg)

        if self._overflow and self._active_batch_has_no_gated_messages():
            self._advance_past_gap()
            for msg in self._overflow:
                self._route_message(msg)
            self._overflow = []

        if self._is_batch_complete():
            return self._close_batch()
        return None

    def _start_first_batch(self, messages: list[Message[Any]]) -> MessageBatch | None:
        if not messages:
            return None
        start_time = min(msg.timestamp for msg in messages)
        end_time = max(msg.timestamp for msg in messages)
        batch = MessageBatch(
            start_time=start_time, end_time=end_time, messages=messages
        )

        next_start = end_time
        self._active_batch = MessageBatch(
            start_time=next_start,
            end_time=next_start + self._batch_length,
            messages=[],
        )
        self._batch_start_wall = self._clock()
        for msg in messages:
            self._ensure_stream(msg.stream)
        return batch

    def _ensure_stream(self, stream_id: StreamId) -> _StreamState:
        if stream_id not in self._streams:
            self._streams[stream_id] = _StreamState()
        return self._streams[stream_id]

    def _is_gated(self, stream_id: StreamId) -> bool:
        return stream_id.kind in GATED_STREAM_KINDS

    def _route_message(self, msg: Message[Any]) -> None:
        if self._active_batch is None:
            raise RuntimeError("No active batch when routing message")
        stream_state = self._ensure_stream(msg.stream)

        if not self._is_gated(msg.stream):
            self._add_to_active_batch(msg)
            return

        tracker = self._batch_trackers.get(msg.stream)
        if tracker is None:
            tracker = _BatchStreamTracker()
            self._batch_trackers[msg.stream] = tracker

        # Before convergence or for very low rates: include everything
        if not stream_state.converged:
            self._add_to_active_batch(msg)
            tracker.add(msg, slot=-1)
            return

        expected = stream_state.expected_count(self.batch_length_s)
        if expected <= 0:
            self._add_to_active_batch(msg)
            tracker.add(msg, slot=-1)
            return

        slot = self._slot_index(msg, stream_state)
        if slot >= expected:
            self._overflow.append(msg)
        else:
            self._add_to_active_batch(msg)
            tracker.add(msg, slot)

    def _slot_index(self, msg: Message[Any], stream_state: _StreamState) -> int:
        """Compute which pulse slot a message belongs to within the active batch."""
        if self._active_batch is None:
            raise RuntimeError("No active batch when computing slot index")
        if stream_state.integer_rate_hz is None:
            raise RuntimeError("Stream rate not converged when computing slot index")
        dt_ns = (msg.timestamp - self._active_batch.start_time).to_ns()
        period_ns = round(1e9 / stream_state.integer_rate_hz)
        dt_ns -= stream_state.phase_offset_ns
        return round(dt_ns / period_ns)

    def _add_to_active_batch(self, msg: Message[Any]) -> None:
        if self._active_batch is None:
            raise RuntimeError("No active batch when adding message")
        self._active_batch.messages.append(msg)

    def _is_batch_complete(self) -> bool:
        if self._active_batch is None:
            return False

        # Timeout fallback
        if self._batch_start_wall is not None:
            elapsed = self._clock() - self._batch_start_wall
            if elapsed >= self._timeout_s:
                return True

        # Check all gated converged streams
        gated_converged = {
            sid: state
            for sid, state in self._streams.items()
            if self._is_gated(sid) and state.converged
        }

        if not gated_converged:
            return False

        for sid, state in gated_converged.items():
            tracker = self._batch_trackers.get(sid)
            if tracker is None or tracker.max_slot < 0:
                return False
            expected = state.expected_count(self.batch_length_s)
            if tracker.max_slot < expected - 1:
                return False

        return True

    def _active_batch_has_no_gated_messages(self) -> bool:
        """True if no gated converged stream has messages in the active batch."""
        for sid, state in self._streams.items():
            if self._is_gated(sid) and state.converged:
                tracker = self._batch_trackers.get(sid)
                if tracker is not None and tracker.count > 0:
                    return False
        return True

    def _advance_past_gap(self) -> None:
        """Skip the batch window forward to where the overflow messages are."""
        if self._active_batch is None or not self._overflow:
            return
        earliest = min(m.timestamp for m in self._overflow)
        # Align to a batch boundary: advance in whole batch lengths
        gap_ns = (earliest - self._active_batch.start_time).to_ns()
        batch_ns = self._batch_length.to_ns()
        steps = gap_ns // batch_ns
        advance = Duration.from_ns(steps * batch_ns)
        new_start = self._active_batch.start_time + advance
        self._active_batch = MessageBatch(
            start_time=new_start,
            end_time=new_start + self._batch_length,
            messages=[],
        )
        self._batch_trackers = {}
        self._batch_start_wall = self._clock()

    def _close_batch(self) -> MessageBatch:
        if self._active_batch is None:
            raise RuntimeError("No active batch when closing batch")
        batch = self._active_batch

        # Update rate estimates, phase offsets, and absence tracking
        present_streams: set[StreamId] = set()
        for sid, tracker in self._batch_trackers.items():
            if self._is_gated(sid) and tracker.messages:
                present_streams.add(sid)
                state = self._streams[sid]
                state.absent_batches = 0
                state.update_rate(tracker.count, self.batch_length_s, self._ema_alpha)
                if state.integer_rate_hz is not None:
                    period_ns = round(1e9 / state.integer_rate_hz)
                    dt_ns = (tracker.messages[0].timestamp - batch.start_time).to_ns()
                    residual = dt_ns % period_ns
                    # Integer truncation in timestamps can make a near-zero
                    # offset appear as period_ns - 1. Unwrap that case.
                    if period_ns - residual <= 2:
                        residual -= period_ns
                    state.phase_offset_ns = residual

        # Track absence and evict streams missing for too long
        to_evict: list[StreamId] = []
        for sid, state in self._streams.items():
            if self._is_gated(sid) and sid not in present_streams:
                state.absent_batches += 1
                if state.absent_batches >= ABSENT_BATCHES_FOR_EVICTION:
                    to_evict.append(sid)
        for sid in to_evict:
            del self._streams[sid]

        # Apply pending batch length change
        if self._pending_batch_length is not None:
            self._batch_length = self._pending_batch_length
            self._pending_batch_length = None

        # Set up next batch
        next_start = batch.end_time
        self._active_batch = MessageBatch(
            start_time=next_start,
            end_time=next_start + self._batch_length,
            messages=[],
        )
        self._batch_trackers = {}
        self._batch_start_wall = self._clock()

        return batch
