# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Rate-aware message batcher.

Batches messages based on per-stream rate estimation and slot-based completion.
A batch is considered complete for a given stream when a message arrives whose
timestamp falls in the last expected "pulse slot" for that stream within the
batch window — not when a fixed message count is reached.
"""

from __future__ import annotations

from collections import defaultdict
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
class StreamRateEstimator:
    """EMA-based rate estimator for a single stream.

    Tracks the message rate via exponential moving average and determines
    when enough observations have been collected to consider the estimate
    converged.
    """

    rate_hz: float | None = None
    observation_count: int = 0

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

    def update(self, message_count: int, batch_length_s: float, alpha: float) -> None:
        observed = message_count / batch_length_s
        if self.rate_hz is None:
            self.rate_hz = observed
        else:
            self.rate_hz = alpha * observed + (1 - alpha) * self.rate_hz
        self.observation_count += 1


@dataclass(frozen=True)
class PulseGrid:
    """Fixed temporal grid for mapping timestamps to pulse indices.

    Created once per stream when the rate estimate converges. The origin
    and period are fixed at creation; jitter tolerance is ``period/2`` by
    construction of the ``round()`` in :meth:`pulse_index`.

    Handles omitted messages (gaps in indices) and split messages (same
    timestamp maps to same index) naturally.
    """

    origin_ns: int
    period_ns: int
    slots_per_batch: int

    def pulse_index(self, timestamp: Timestamp) -> int:
        """Global absolute pulse index for a timestamp."""
        return round((timestamp.to_ns() - self.origin_ns) / self.period_ns)

    def batch_base_index(self, batch_start: Timestamp) -> int:
        """Index of the first pulse that belongs to a batch.

        If ``batch_start`` is close to a pulse (within 1 % of the period),
        that pulse is the base.  Otherwise the next pulse is used.

        The tolerance absorbs the small per-batch drift caused by
        ``round(1e9 / rate) * rate != 1e9``.  Without it, a strict
        ceiling division would overshoot by one pulse after each batch.
        """
        delta = batch_start.to_ns() - self.origin_ns
        quotient, remainder = divmod(delta, self.period_ns)
        if remainder > self.period_ns // 100:
            return quotient + 1
        return quotient

    def slot_in_batch(self, timestamp: Timestamp, batch_start: Timestamp) -> int:
        """Pulse slot relative to the batch start."""
        return self.pulse_index(timestamp) - self.batch_base_index(batch_start)


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
        ema_alpha: float = 0.05,
    ) -> None:
        self._batch_length = Duration.from_seconds(batch_length_s)
        self._timeout = Duration.from_seconds(
            timeout_s if timeout_s is not None else batch_length_s * 0.8
        )
        self._ema_alpha = ema_alpha

        self._estimators: defaultdict[StreamId, StreamRateEstimator] = defaultdict(
            StreamRateEstimator
        )
        self._grids: dict[StreamId, PulseGrid] = {}
        self._absent_batches: dict[StreamId, int] = {}

        self._pending_batch_length: Duration | None = None
        self._active_batch: MessageBatch | None = None
        self._batch_trackers: dict[StreamId, _BatchStreamTracker] = {}
        self._high_water_mark: Timestamp | None = None
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

        if messages:
            latest = max(m.timestamp for m in messages)
            if self._high_water_mark is None or self._high_water_mark < latest:
                self._high_water_mark = latest

        if self._active_batch is None:
            return self._start_first_batch(messages)

        all_messages = self._overflow + messages
        self._overflow = []

        for msg in all_messages:
            self._route_message(msg)

        if self._overflow and self._active_batch_has_no_gated_messages():
            # Collect messages already in the active batch (non-converged
            # streams routed before the gap was detected) so they aren't
            # lost when _advance_past_gap replaces the batch.
            stashed = self._active_batch.messages
            self._advance_past_gap()
            for msg in stashed + self._overflow:
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
        return batch

    def _is_gated(self, stream_id: StreamId) -> bool:
        return stream_id.kind in GATED_STREAM_KINDS

    def _route_message(self, msg: Message[Any]) -> None:
        if self._active_batch is None:
            raise RuntimeError("No active batch when routing message")

        if not self._is_gated(msg.stream):
            self._active_batch.messages.append(msg)
            return

        tracker = self._batch_trackers.setdefault(msg.stream, _BatchStreamTracker())
        grid = self._grids.get(msg.stream)

        if grid is None:
            self._active_batch.messages.append(msg)
            tracker.add(msg, slot=-1)
            return

        slot = grid.slot_in_batch(msg.timestamp, self._active_batch.start_time)
        if slot >= grid.slots_per_batch:
            self._overflow.append(msg)
            tracker.max_slot = max(tracker.max_slot, grid.slots_per_batch - 1)
        else:
            self._active_batch.messages.append(msg)
            tracker.add(msg, slot)

    def _is_batch_complete(self) -> bool:
        if self._active_batch is None:
            return False

        # Logical-clock timeout: high-water mark past batch start + timeout
        if self._high_water_mark is not None:
            threshold = self._active_batch.start_time + self._timeout
            if not self._high_water_mark < threshold:
                return True

        if not self._grids:
            return False

        for sid, grid in self._grids.items():
            tracker = self._batch_trackers.get(sid)
            if tracker is None or tracker.max_slot < 0:
                return False
            if tracker.max_slot < grid.slots_per_batch - 1:
                return False

        return True

    def _active_batch_has_no_gated_messages(self) -> bool:
        """True if no stream with a grid has messages in the active batch."""
        for sid in self._grids:
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

    def _close_batch(self) -> MessageBatch:
        if self._active_batch is None:
            raise RuntimeError("No active batch when closing batch")
        batch = self._active_batch

        # Update rate estimates and build/rebuild grids
        present_streams: set[StreamId] = set()
        for sid, tracker in self._batch_trackers.items():
            if tracker.messages:
                present_streams.add(sid)
                self._absent_batches[sid] = 0
                estimator = self._estimators[sid]
                old_int_rate = estimator.integer_rate_hz
                estimator.update(tracker.count, self.batch_length_s, self._ema_alpha)
                new_int_rate = estimator.integer_rate_hz
                if (
                    estimator.converged
                    and new_int_rate is not None
                    and new_int_rate > 0
                ):
                    if sid not in self._grids or old_int_rate != new_int_rate:
                        period_ns = round(1e9 / new_int_rate)
                        origin = self._grid_origin(tracker, batch.start_time)
                        self._grids[sid] = PulseGrid(
                            origin_ns=origin,
                            period_ns=period_ns,
                            slots_per_batch=round(new_int_rate * self.batch_length_s),
                        )

        # Track absence and evict streams missing for too long
        to_evict: list[StreamId] = []
        for sid in self._estimators:
            if sid not in present_streams:
                absent = self._absent_batches.get(sid, 0) + 1
                self._absent_batches[sid] = absent
                if absent >= ABSENT_BATCHES_FOR_EVICTION:
                    to_evict.append(sid)
        for sid in to_evict:
            del self._estimators[sid]
            self._grids.pop(sid, None)
            self._absent_batches.pop(sid, None)

        # Apply pending batch length change and rebuild grids
        if self._pending_batch_length is not None:
            self._batch_length = self._pending_batch_length
            self._pending_batch_length = None
            for sid, grid in self._grids.items():
                est = self._estimators[sid]
                int_rate = est.integer_rate_hz
                if int_rate is not None and int_rate > 0:
                    self._grids[sid] = PulseGrid(
                        origin_ns=grid.origin_ns,
                        period_ns=grid.period_ns,
                        slots_per_batch=round(int_rate * self.batch_length_s),
                    )

        # Set up next batch
        next_start = batch.end_time
        self._active_batch = MessageBatch(
            start_time=next_start,
            end_time=next_start + self._batch_length,
            messages=[],
        )
        self._batch_trackers = {}

        return batch

    @staticmethod
    def _grid_origin(tracker: _BatchStreamTracker, batch_start: Timestamp) -> int:
        """Pick a grid origin from the first in-window message."""
        for m in tracker.messages:
            if not m.timestamp < batch_start:
                return m.timestamp.to_ns()
        return tracker.messages[0].timestamp.to_ns()
