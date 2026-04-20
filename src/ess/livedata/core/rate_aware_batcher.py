# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Rate-aware message batcher.

Batches messages based on per-stream rate estimation and slot-based completion.
A batch is considered complete for a given stream when a message arrives whose
timestamp falls in the last expected "pulse slot" for that stream within the
batch window — not when a fixed message count is reached.
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
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

MIN_DIFFS_FOR_GATE = 4
DIFF_BUFFER_SIZE = 32
ABSENT_BATCHES_FOR_EVICTION = 5

# Small absolute tolerance for integer-Hz rounding drift.  Per-batch
# drift is at most a few ns (|batch_length_ns - slots_per_batch *
# period_ns|); 1 ms covers many hours of accumulated drift while still
# refusing to absorb true phase offsets at low rates.
_DRIFT_TOLERANCE_NS = 1_000_000


@dataclass
class StreamPeriodEstimator:
    """Infers pulse period from inter-arrival times between messages.

    Accumulates positive timestamp differences across batches in a bounded
    ring buffer. The period is derived in two steps: the median of all
    diffs is used as a seed (unbiased under symmetric jitter as long as
    single-period diffs are a majority), then each diff is snapped to its
    nearest integer multiple of the seed and divided back to a per-pulse
    estimate; the median of these is the final period. This is robust to
    missed pulses (integer-multiple outliers), split messages (zero diffs
    filtered out), out-of-order arrivals, and timestamp jitter. Unlike a
    bare ``min``, jitter bias scales as ``s/√N`` rather than
    ``-s·√(2 ln N)``, which matters at high rates where the integer-Hz
    snap has a sub-percent tolerance. ``integer_rate_hz`` snaps to integer
    Hz, the rate format published by ESS sources.

    Convergence requires ``MIN_DIFFS_FOR_GATE`` positive diffs, which can
    be accumulated within a single batch for high-rate streams or across
    batches for low-rate streams.
    """

    last_ts_ns: int | None = None
    diffs: deque[int] = field(default_factory=lambda: deque(maxlen=DIFF_BUFFER_SIZE))

    def observe(self, ts_ns: int) -> None:
        if self.last_ts_ns is not None:
            diff = ts_ns - self.last_ts_ns
            if diff > 0:
                self.diffs.append(diff)
        if self.last_ts_ns is None or ts_ns > self.last_ts_ns:
            self.last_ts_ns = ts_ns

    @property
    def integer_rate_hz(self) -> int | None:
        if len(self.diffs) < MIN_DIFFS_FOR_GATE:
            return None
        seed = statistics.median(self.diffs)
        per_pulse = [d / k for d in self.diffs if (k := round(d / seed)) >= 1]
        period_ns = statistics.median(per_pulse) if per_pulse else seed
        rate = round(1e9 / period_ns)
        return rate if rate >= 1 else None


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

        Ceiling division of ``(batch_start - origin) / period`` with a
        narrow symmetric tolerance for integer-Hz rounding drift.  Per
        batch, the drift is at most a few ns (difference between
        ``slots_per_batch * period_ns`` and the actual batch length);
        a small absolute tolerance absorbs that while still flagging
        true phase offsets (milliseconds) as out-of-window.

        A wide symmetric tolerance would misclassify a pulse that sits
        a few ms *before* ``batch_start`` (a phase offset, not drift)
        as the batch's first pulse, pushing the real first in-window
        pulse into overflow.  At ``slots_per_batch = 1`` that silently
        drops every batch's only pulse.
        """
        delta = batch_start.to_ns() - self.origin_ns
        quotient, remainder = divmod(delta, self.period_ns)
        tolerance = min(_DRIFT_TOLERANCE_NS, self.period_ns // 2)
        if remainder <= tolerance:
            return quotient
        return quotient + 1

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
    ) -> None:
        self._batch_length = Duration.from_seconds(batch_length_s)
        self._timeout_factor = (
            timeout_s / batch_length_s if timeout_s is not None else 1.2
        )

        self._estimators: defaultdict[StreamId, StreamPeriodEstimator] = defaultdict(
            StreamPeriodEstimator
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

    @property
    def timeout_factor(self) -> float:
        return self._timeout_factor

    @timeout_factor.setter
    def timeout_factor(self, value: float) -> None:
        self._timeout_factor = value

    @property
    def timeout_s(self) -> float:
        return self._timeout_factor * self.batch_length_s

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
            # Detach messages already in the active batch (non-gated and
            # unconverged-gated streams routed before the gap was
            # detected).  _advance_past_gap is a no-op when the pending
            # messages still fit in the current window (steps == 0), so
            # without detaching, re-routing would append onto the same
            # list and duplicate every stashed message.
            stashed = self._active_batch.messages
            self._active_batch.messages = []
            pending = self._overflow
            self._overflow = []
            self._advance_past_gap(pending)
            for msg in stashed + pending:
                self._route_message(msg)

        if self._is_batch_complete():
            return self._close_batch()
        return None

    def _start_first_batch(self, messages: list[Message[Any]]) -> MessageBatch | None:
        if not messages:
            return None
        for msg in messages:
            if self._is_gated(msg.stream):
                self._estimators[msg.stream].observe(msg.timestamp.to_ns())
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
        self._rebuild_grids()
        return batch

    def _is_gated(self, stream_id: StreamId) -> bool:
        return stream_id.kind in GATED_STREAM_KINDS

    def _route_message(self, msg: Message[Any]) -> None:
        if self._active_batch is None:
            raise RuntimeError("No active batch when routing message")

        if not self._is_gated(msg.stream):
            self._active_batch.messages.append(msg)
            return

        self._estimators[msg.stream].observe(msg.timestamp.to_ns())
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
            threshold = self._active_batch.start_time + Duration.from_seconds(
                self.timeout_s
            )
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

    def _advance_past_gap(self, pending: list[Message[Any]]) -> None:
        """Skip the batch window forward to where the pending messages are.

        No-op when ``pending`` fits entirely inside the current window
        (``steps == 0``): in that case the caller is expected to re-route
        the messages against the existing window, and any re-generated
        overflow must be preserved rather than dropped.
        """
        if self._active_batch is None or not pending:
            return
        earliest = min(m.timestamp for m in pending)
        gap_ns = (earliest - self._active_batch.start_time).to_ns()
        batch_ns = self._batch_length.to_ns()
        steps = gap_ns // batch_ns
        if steps == 0:
            return
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

        present_streams: set[StreamId] = set()
        for sid, tracker in self._batch_trackers.items():
            if tracker.messages:
                present_streams.add(sid)
                self._absent_batches[sid] = 0
                self._update_grid(sid, tracker, batch.start_time)

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

        if self._pending_batch_length is not None:
            self._batch_length = self._pending_batch_length
            self._pending_batch_length = None
            for sid in list(self._grids):
                self._update_grid(sid, self._batch_trackers.get(sid), batch.start_time)

        next_start = batch.end_time
        self._active_batch = MessageBatch(
            start_time=next_start,
            end_time=next_start + self._batch_length,
            messages=[],
        )
        self._batch_trackers = {}

        return batch

    def _rebuild_grids(self) -> None:
        """Update grids for all observed streams from current estimator state."""
        if self._active_batch is None:
            return
        for sid in list(self._estimators):
            self._update_grid(
                sid, self._batch_trackers.get(sid), self._active_batch.start_time
            )

    def _update_grid(
        self,
        sid: StreamId,
        tracker: _BatchStreamTracker | None,
        batch_start: Timestamp,
    ) -> None:
        """Build or rebuild the pulse grid for a stream from its estimator.

        No-op if the estimator hasn't converged. For existing grids, only
        rebuilds when period or slots_per_batch changes; the origin is
        preserved across rebuilds to keep slot assignments stable.
        """
        estimator = self._estimators[sid]
        int_rate = estimator.integer_rate_hz
        if int_rate is None or int_rate <= 0:
            return
        period_ns = round(1e9 / int_rate)
        slots_per_batch = round(int_rate * self.batch_length_s)
        existing = self._grids.get(sid)
        if existing is not None:
            if (
                existing.period_ns == period_ns
                and existing.slots_per_batch == slots_per_batch
            ):
                return
            origin = existing.origin_ns
        else:
            origin = self._pick_grid_origin(sid, tracker, batch_start)
            if origin is None:
                return
        self._grids[sid] = PulseGrid(
            origin_ns=origin, period_ns=period_ns, slots_per_batch=slots_per_batch
        )

    def _pick_grid_origin(
        self,
        sid: StreamId,
        tracker: _BatchStreamTracker | None,
        batch_start: Timestamp,
    ) -> int | None:
        """Pick an origin timestamp that lies on the pulse grid."""
        if tracker is not None and tracker.messages:
            for m in tracker.messages:
                if not m.timestamp < batch_start:
                    return m.timestamp.to_ns()
            return tracker.messages[0].timestamp.to_ns()
        return self._estimators[sid].last_ts_ns
