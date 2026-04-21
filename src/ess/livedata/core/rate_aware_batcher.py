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

# Tolerance for snapping the raw rate to its nearest integer Hz.  Uses
# the larger of a relative bound and an absolute floor: relative scales
# the allowed deviation with the rate so high-rate estimator noise
# (~1%-2% residual after median-of-32 with millisecond jitter) doesn't
# veto legitimate snaps, while the absolute floor tightens low-rate
# behaviour where a flat 20% relative bound pathologically accepts
# non-integer neighbours -- e.g. a true 0.85 Hz stream snaps to 1 Hz
# with 15% error, then slot math places pulses in the wrong slots and
# the stream never gates cleanly.  0.1 Hz absolute rejects anything
# more than ~10% off 1 Hz while staying wide enough for 5 ms jitter
# median noise (<0.01 Hz in practice).
_INTEGER_SNAP_RELATIVE_TOLERANCE = 0.1
_INTEGER_SNAP_ABSOLUTE_TOLERANCE_HZ = 0.1

# Small absolute tolerance for integer-Hz rounding drift.  Per-batch
# drift is at most a few ns (|batch_length_ns - slots_per_batch *
# period_ns|); 1 ms covers many hours of accumulated drift while still
# refusing to absorb true phase offsets at low rates.
_DRIFT_TOLERANCE_NS = 1_000_000

# Max allowed distance between a new grid's origin and the active
# batch_start, in multiples of batch_length.  Protects against streams
# whose timestamps live in a disjoint epoch (wrong schema field, unit
# mismatch, run-local clock): such a stream's grid would place every
# slot billions of slots away from batch_start, pinning the bucket's
# max_slot at -1 and vetoing every slot-gate closure for the whole batcher.
# 1000 batches (>=16 min at 1 s batches) is well above any legitimate
# cold-start or post-eviction origin offset but below any real epoch
# mismatch by many orders of magnitude.
_MAX_ORIGIN_OFFSET_BATCHES = 1000

# Max distance ``_high_water_mark`` is allowed to sit past
# ``_active_window.start``, in multiples of batch_length.  Protects
# against a single malformed timestamp (e.g. upstream epoch bug
# producing a value years ahead) permanently pinning the HWM in the
# future, which would otherwise force every subsequent ``batch()`` call
# to close a batch via the timeout path for millions of iterations --
# effectively a DoS until the process restarts.
#
# Bounding HWM relative to the active window (rather than to the prior
# HWM) keeps HWM self-healing: each batch close advances the window by
# one batch_length, so after a bounded number of cascading empty
# closures the HWM is no longer past the timeout threshold and timeout
# firing stops on its own.  An absolute per-call cap doesn't self-heal
# because the window keeps moving away from the clamped HWM.
#
# Must be >= ``timeout_factor`` (default 1.2) for the timeout path to
# ever fire -- and comfortably above that for sub-Hz-only streams whose
# sparse arrivals rely on multi-batch HWM jumps to trigger cascading
# timeout closes of empty batches between pulses.  Three batches allows
# one pulse's worth of HWM advance to cover the preceding empty batch,
# matching the natural cadence of 0.5 Hz-and-below gated streams.
_MAX_HWM_PAST_WINDOW_BATCHES = 3


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
        raw_rate = 1e9 / period_ns
        rate = round(raw_rate)
        if rate < 1:
            return None
        tolerance_hz = max(
            _INTEGER_SNAP_RELATIVE_TOLERANCE * rate,
            _INTEGER_SNAP_ABSOLUTE_TOLERANCE_HZ,
        )
        if abs(raw_rate - rate) > tolerance_hz:
            return None
        return rate


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
class _ActiveWindow:
    """Time range of the batch currently in progress."""

    start: Timestamp
    end: Timestamp


@dataclass
class _GatedStream:
    """Per-stream state for a gated stream.

    Owns both persistent state (rate estimator, pulse grid, absence
    counter) and transient per-batch state (bucketed messages and highest
    slot seen).  The grid's Optional nature is sealed inside this class:
    callers observe arrivals, route messages, and ask whether the gate is
    satisfied -- they never branch on ``None``.  Transient state is reset
    by :meth:`drain` at batch close.
    """

    estimator: StreamPeriodEstimator = field(default_factory=StreamPeriodEstimator)
    grid: PulseGrid | None = None
    absent_batches: int = 0
    messages: list[Message[Any]] = field(default_factory=list)
    max_slot: int = -1

    @property
    def is_gating(self) -> bool:
        return self.grid is not None

    @property
    def has_messages(self) -> bool:
        return bool(self.messages)

    def observe(self, msg: Message[Any]) -> None:
        self.estimator.observe(msg.timestamp.to_ns())

    def route(self, msg: Message[Any], window_start: Timestamp) -> Message[Any] | None:
        """Place ``msg`` in the bucket; return it unchanged if it overflows.

        Overflow still bumps ``max_slot`` to the last grid slot so the
        slot gate observes that the window's final pulse was reached.
        """
        self.observe(msg)
        if self.grid is None:
            self._add(msg)
            return None
        slot = self.grid.slot_in_batch(msg.timestamp, window_start)
        if slot >= self.grid.slots_per_batch:
            last = self.grid.slots_per_batch - 1
            if last > self.max_slot:
                self.max_slot = last
            return msg
        self._add(msg, slot)
        return None

    def _add(self, msg: Message[Any], slot: int = -1) -> None:
        self.messages.append(msg)
        if slot > self.max_slot:
            self.max_slot = slot

    def is_gate_satisfied(self) -> bool:
        """True if this stream does not block a batch close.

        Ungridded streams flow opportunistically and never block.  A
        gridded stream needs ``max_slot`` to have reached the grid's
        final slot.
        """
        if self.grid is None:
            return True
        return self.max_slot >= self.grid.slots_per_batch - 1

    def mark_present(self) -> None:
        self.absent_batches = 0

    def mark_absent(self) -> bool:
        """Increment absence; return True when the stream should be evicted."""
        self.absent_batches += 1
        return self.absent_batches >= ABSENT_BATCHES_FOR_EVICTION

    def drain(self) -> list[Message[Any]]:
        """Remove and return bucketed messages; reset slot tracking."""
        messages = self.messages
        self.messages = []
        self.max_slot = -1
        return messages

    def refresh_grid(self, batch_start: Timestamp, batch_length: Duration) -> None:
        """Build or rebuild the pulse grid from the estimator.

        No-op if the estimator hasn't converged.  Streams whose rate is below one pulse
        per batch (``int_rate * batch_length_s < 1``) cannot reliably fill a slot per
        batch; any prior grid is dropped and the stream reverts to opportunistic
        (non-gated) delivery.

        The origin is preserved across rebuilds while it stays within
        ``_MAX_ORIGIN_OFFSET_BATCHES`` of ``batch_start``.  A fresh candidate is
        otherwise drawn from current bucket state or the estimator's last timestamp;
        when no candidate is plausibly near ``batch_start`` the grid is dropped (streams
        whose timestamps live in a disjoint epoch: schema bug, clock reset, producer
        replaying an old topic).
        """
        int_rate = self.estimator.integer_rate_hz
        if int_rate is None or int_rate <= 0:
            return
        batch_length_s = batch_length.to_seconds()
        if int_rate * batch_length_s < 1.0:
            self.grid = None
            return
        period_ns = round(1e9 / int_rate)
        slots_per_batch = round(int_rate * batch_length_s)
        origin = self._choose_origin(batch_start, batch_length)
        if origin is None:
            self.grid = None
            return
        existing = self.grid
        if (
            existing is not None
            and existing.origin_ns == origin
            and existing.period_ns == period_ns
            and existing.slots_per_batch == slots_per_batch
        ):
            return
        self.grid = PulseGrid(
            origin_ns=origin, period_ns=period_ns, slots_per_batch=slots_per_batch
        )

    def _choose_origin(
        self, batch_start: Timestamp, batch_length: Duration
    ) -> int | None:
        if self.grid is not None and not _origin_too_far(
            self.grid.origin_ns, batch_start, batch_length
        ):
            return self.grid.origin_ns
        candidate = self._pick_origin(batch_start)
        if candidate is None or _origin_too_far(candidate, batch_start, batch_length):
            return None
        return candidate

    def _pick_origin(self, batch_start: Timestamp) -> int | None:
        """Pick an origin timestamp that lies on the pulse grid."""
        if self.messages:
            for m in self.messages:
                if not m.timestamp < batch_start:
                    return m.timestamp.to_ns()
            return self.messages[0].timestamp.to_ns()
        return self.estimator.last_ts_ns


def _origin_too_far(
    origin_ns: int, batch_start: Timestamp, batch_length: Duration
) -> bool:
    """True if ``origin_ns`` is implausibly far from ``batch_start``."""
    max_offset_ns = _MAX_ORIGIN_OFFSET_BATCHES * batch_length.to_ns()
    return abs(origin_ns - batch_start.to_ns()) > max_offset_ns


class RateAwareMessageBatcher(MessageBatcher):
    """A batcher that uses per-stream rate estimation and slot-based completion.

    Completion for each gated stream is determined by whether a message has
    arrived whose timestamp falls in the last expected pulse slot, rather than
    by message count alone. This handles missing pulses (never published) and
    split messages (two messages with the same timestamp) gracefully.

    Streams whose kind is not in ``GATED_STREAM_KINDS`` are included
    opportunistically in whatever batch is active.

    Notes
    -----
    Two behaviours differ from :class:`SimpleMessageBatcher` and are worth
    flagging for downstream consumers:

    - **Long silences emit no placeholder batches.** ``SimpleMessageBatcher``
      emits one empty batch per skipped window; this batcher's gap-advance
      path jumps the active window directly to where the next message is.
      Downstream code that drives UI ticks or heartbeats off batch arrivals
      will see fewer events during gaps.
    - **Empty batch on the pulse-slot boundary edge.** For a low-rate stream
      with ``slots_per_batch == 1``, a message whose pulse slot is ahead of
      the active batch (e.g. a 1 Hz stray at ``t0 + 0.6`` against origin
      ``t0``) closes the current batch as empty; the message is delivered
      in the next window.  Downstream already tolerates empty batches, and
      no message is lost.
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

        self._streams: defaultdict[StreamId, _GatedStream] = defaultdict(_GatedStream)

        self._pending_batch_length: Duration | None = None
        self._active_window: _ActiveWindow | None = None
        self._high_water_mark: Timestamp | None = None
        self._overflow: list[Message[Any]] = []
        self._non_gated: list[Message[Any]] = []
        self._future: list[Message[Any]] = []

    @property
    def batch_length_s(self) -> float:
        return self._batch_length.to_seconds()

    @property
    def tracked_streams(self) -> set[StreamId]:
        """Stream IDs currently tracked by the batcher."""
        return set(self._streams)

    def grid_of(self, stream_id: StreamId) -> PulseGrid | None:
        """Return the pulse grid for a tracked stream, or ``None``."""
        stream = self._streams.get(stream_id)
        return stream.grid if stream is not None else None

    def is_gating(self, stream_id: StreamId) -> bool:
        """True if the stream has a converged grid and gates batch closure."""
        return self.grid_of(stream_id) is not None

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
        if messages:
            latest = max(m.timestamp for m in messages)
            if self._high_water_mark is None:
                self._high_water_mark = latest
            elif self._high_water_mark < latest:
                self._high_water_mark = self._clamped_hwm(latest)

        window = self._active_window
        if window is None:
            if not messages:
                return None
            return self._bootstrap_batch(messages)

        for msg in messages:
            self._route_message(msg, window)

        if self._should_recover_from_gap(window):
            window = self._recover_from_gap(window)

        if self._is_batch_complete(window):
            return self._close_batch(window)
        return None

    def _clamped_hwm(self, latest: Timestamp) -> Timestamp:
        """Clamp an HWM update to a bounded distance past the active window.

        See ``_MAX_HWM_PAST_WINDOW_BATCHES``.  When no active window is
        present (the first-batch path handles its own bootstrap), the
        latest timestamp is accepted as-is.  The clamp is floored at the
        current HWM to preserve monotonicity: a window advance (close or
        gap advance) may briefly leave HWM past ``start + cap``, which
        the next update would otherwise regress.
        """
        if self._active_window is None or self._high_water_mark is None:
            return latest
        cap = Duration.from_ns(
            _MAX_HWM_PAST_WINDOW_BATCHES * self._batch_length.to_ns()
        )
        max_allowed = self._active_window.start + cap
        return min(latest, max(max_allowed, self._high_water_mark))

    def _bootstrap_batch(self, messages: list[Message[Any]]) -> MessageBatch:
        """Flush the startup backlog and open the active window.

        Seeds estimators from gated-stream arrivals, opens the window at
        the max input timestamp (so it starts immediately after the
        flush), builds grids for any streams whose estimators already
        converged, and returns the flushed messages as the first batch.
        """
        start_time = min(m.timestamp for m in messages)
        end_time = max(m.timestamp for m in messages)
        for msg in messages:
            if msg.stream.kind in GATED_STREAM_KINDS:
                self._streams[msg.stream].observe(msg)
        self._active_window = _ActiveWindow(
            start=end_time, end=end_time + self._batch_length
        )
        for stream in self._streams.values():
            stream.refresh_grid(end_time, self._batch_length)
        return MessageBatch(start_time=start_time, end_time=end_time, messages=messages)

    def _route_message(self, msg: Message[Any], window: _ActiveWindow) -> None:
        """Bucket a message by stream kind and timestamp relative to the window.

        Ungridded streams (non-gated kind OR sub-Hz gated without a grid) hold
        messages with ``window.end < ts <= window.end + K * batch_length`` in
        ``_future`` so batch contents stay bounded by the batch's time range.
        ``K`` reuses ``_MAX_HWM_PAST_WINDOW_BATCHES``: beyond that, a timestamp
        is implausibly far and the message falls through to the active batch,
        preventing a pathological timestamp (epoch bug, unit mismatch) from
        caching messages indefinitely.

        Gridded gated streams use the slot-based overflow path instead, which
        drives gap recovery via ``_should_recover_from_gap``.
        """
        is_gated = msg.stream.kind in GATED_STREAM_KINDS
        stream = self._streams[msg.stream] if is_gated else None
        if (stream is None or stream.grid is None) and self._is_future(msg, window):
            self._future.append(msg)
            return
        if stream is None:
            self._non_gated.append(msg)
            return
        overflow = stream.route(msg, window.start)
        if overflow is not None:
            self._overflow.append(overflow)

    def _is_future(self, msg: Message[Any], window: _ActiveWindow) -> bool:
        """True if ``msg`` belongs in a future window within the hold-back cap."""
        if not (msg.timestamp > window.end):
            return False
        cap = _MAX_HWM_PAST_WINDOW_BATCHES * self._batch_length
        return msg.timestamp - window.end <= cap

    def _is_batch_complete(self, window: _ActiveWindow) -> bool:
        if self._high_water_mark is not None:
            threshold = window.start + Duration.from_seconds(self.timeout_s)
            if not self._high_water_mark < threshold:
                return True

        has_gating = False
        for stream in self._streams.values():
            if not stream.is_gating:
                continue
            has_gating = True
            if not stream.is_gate_satisfied():
                return False
        return has_gating

    def _should_recover_from_gap(self, window: _ActiveWindow) -> bool:
        """True if gated overflow exists but no gated stream has contributed.

        This indicates the window is lagging behind live traffic: every
        gridded stream's arrivals landed past the last slot, so they were
        overflowed rather than routed into the window.  Caller advances
        the window past the gap.
        """
        if not self._overflow:
            return False
        for stream in self._streams.values():
            if stream.is_gating and stream.has_messages:
                return False
        return True

    def _recover_from_gap(self, window: _ActiveWindow) -> _ActiveWindow:
        """Advance the window past a detected gap, then re-route stashed traffic.

        Drains non-gated/ungridded messages already bucketed in the window
        and the gated overflow, advances the window to where the pending
        traffic lives, and re-routes everything.  At ``steps == 0`` (pending
        still fits in the current window) the window is kept but draining
        resets per-stream slot placement so re-routing recomputes it from
        scratch against the same ``start_time``.
        """
        stashed = self._drain_window()
        pending = self._overflow
        self._overflow = []
        future = self._future
        self._future = []

        earliest = min(m.timestamp for m in pending)
        gap_ns = (earliest - window.start).to_ns()
        batch_ns = self._batch_length.to_ns()
        steps = max(gap_ns // batch_ns, 0)
        if steps > 0:
            new_start = window.start + Duration.from_ns(steps * batch_ns)
            window = _ActiveWindow(start=new_start, end=new_start + self._batch_length)
        self._active_window = window

        for msg in stashed + pending + future:
            self._route_message(msg, window)
        return window

    def _drain_window(self) -> list[Message[Any]]:
        """Remove and return all messages buffered for the active batch."""
        messages = self._non_gated
        self._non_gated = []
        for stream in self._streams.values():
            messages.extend(stream.drain())
        return messages

    def _close_batch(self, window: _ActiveWindow) -> MessageBatch:
        self._refresh_stream_registry(window)
        batch = MessageBatch(
            start_time=window.start, end_time=window.end, messages=self._drain_window()
        )

        new_start = window.end
        new_window = _ActiveWindow(start=new_start, end=new_start + self._batch_length)
        self._active_window = new_window
        # Drain overflow into the new window.  Timestamps that still fall
        # past the last slot land back in ``_overflow`` and wait for the
        # next close; gap recovery handles jumps larger than one batch.
        overflow = self._overflow
        self._overflow = []
        for msg in overflow:
            self._route_message(msg, new_window)
        future = self._future
        self._future = []
        for msg in future:
            self._route_message(msg, new_window)

        return batch

    def _refresh_stream_registry(self, window: _ActiveWindow) -> None:
        """Update grids, mark absence, evict dead streams, apply batch-length change.

        Runs once per close before draining, so each stream's buckets
        feed fresh origins into ``rebuild_grid``.
        """
        for sid in list(self._streams):
            stream = self._streams[sid]
            if stream.has_messages:
                stream.mark_present()
                stream.refresh_grid(window.start, self._batch_length)
            elif stream.mark_absent():
                del self._streams[sid]

        if self._pending_batch_length is not None:
            self._batch_length = self._pending_batch_length
            self._pending_batch_length = None
            # Iterate all known streams: growing the batch length can
            # promote a previously-demoted sub-rate stream back into the
            # grid, and that stream has ``grid is None``.
            for stream in self._streams.values():
                stream.refresh_grid(window.start, self._batch_length)
