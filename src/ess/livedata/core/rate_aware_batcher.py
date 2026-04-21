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
from collections import deque
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
class _StreamBucket:
    """Per-stream state within the active window.

    Holds the messages routed to this stream for the current batch and the
    highest pulse slot seen (``-1`` for non-gated or ungridded streams --
    those never participate in the slot gate).  Overflow messages are not
    stored here but still update ``max_slot`` so the slot gate can observe
    that the last expected slot was reached.
    """

    messages: list[Message[Any]] = field(default_factory=list)
    max_slot: int = -1

    def add(self, msg: Message[Any], slot: int = -1) -> None:
        self.messages.append(msg)
        if slot > self.max_slot:
            self.max_slot = slot

    @property
    def count(self) -> int:
        return len(self.messages)


@dataclass
class _ActiveWindow:
    """Mutable time window accumulating messages until a batch closes.

    The window owns per-stream ``_StreamBucket`` s for every stream that has
    contributed to it -- gated and non-gated alike.  ``MessageBatch`` is
    built once at close time by flattening buckets; cross-stream order is
    not preserved, but downstream only groups by stream.
    """

    start: Timestamp
    end: Timestamp
    buckets: dict[StreamId, _StreamBucket] = field(default_factory=dict)

    def bucket(self, sid: StreamId) -> _StreamBucket:
        return self.buckets.setdefault(sid, _StreamBucket())

    def flatten(self) -> list[Message[Any]]:
        return [m for bucket in self.buckets.values() for m in bucket.messages]

    def close(self) -> MessageBatch:
        return MessageBatch(
            start_time=self.start, end_time=self.end, messages=self.flatten()
        )


@dataclass
class _GatedStream:
    """Per-stream persistent state for a gated stream.

    Owns the rate estimator, the pulse grid (absent until the estimator
    converges), and the consecutive-absent-batches counter driving
    eviction.  The grid's Optional nature is sealed inside this class:
    callers observe arrivals, route messages into a bucket, and ask
    whether the gate is satisfied -- they never branch on ``None``.
    """

    estimator: StreamPeriodEstimator = field(default_factory=StreamPeriodEstimator)
    grid: PulseGrid | None = None
    absent_batches: int = 0

    @property
    def is_gating(self) -> bool:
        return self.grid is not None

    def observe(self, msg: Message[Any]) -> None:
        self.estimator.observe(msg.timestamp.to_ns())

    def route(
        self, msg: Message[Any], bucket: _StreamBucket, window_start: Timestamp
    ) -> Message[Any] | None:
        """Place ``msg`` in ``bucket``; return it unchanged if it overflows.

        Overflow still bumps ``bucket.max_slot`` to the last grid slot so
        the slot gate observes that the window's final pulse was reached.
        """
        if self.grid is None:
            bucket.add(msg)
            return None
        slot = self.grid.slot_in_batch(msg.timestamp, window_start)
        if slot >= self.grid.slots_per_batch:
            last = self.grid.slots_per_batch - 1
            if last > bucket.max_slot:
                bucket.max_slot = last
            return msg
        bucket.add(msg, slot)
        return None

    def is_gate_satisfied(self, bucket: _StreamBucket | None) -> bool:
        """True if this stream does not block a batch close.

        Ungridded streams flow opportunistically and never block.  A
        gridded stream needs a bucket whose highest observed slot has
        reached the grid's final slot.
        """
        if self.grid is None:
            return True
        if bucket is None:
            return False
        return bucket.max_slot >= self.grid.slots_per_batch - 1

    def mark_present(self) -> None:
        self.absent_batches = 0

    def mark_absent(self) -> bool:
        """Increment absence; return True when the stream should be evicted."""
        self.absent_batches += 1
        return self.absent_batches >= ABSENT_BATCHES_FOR_EVICTION

    def rebuild_grid(
        self,
        bucket: _StreamBucket | None,
        batch_start: Timestamp,
        batch_length: Duration,
    ) -> None:
        """Build or rebuild the pulse grid from the estimator.

        No-op if the estimator hasn't converged.  Streams whose rate is
        below one pulse per batch (``int_rate * batch_length_s < 1``)
        cannot reliably fill a slot per batch; any prior grid is dropped
        and the stream reverts to opportunistic (non-gated) delivery.

        The origin is preserved across rebuilds while it stays within
        ``_MAX_ORIGIN_OFFSET_BATCHES`` of ``batch_start``.  A fresh
        candidate is otherwise drawn from current bucket state or the
        estimator's last timestamp; when no candidate is plausibly near
        ``batch_start`` the grid is dropped (streams whose timestamps
        live in a disjoint epoch: schema bug, clock reset, producer
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
        origin = self._choose_origin(bucket, batch_start, batch_length)
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
        self,
        bucket: _StreamBucket | None,
        batch_start: Timestamp,
        batch_length: Duration,
    ) -> int | None:
        if self.grid is not None and not _origin_too_far(
            self.grid.origin_ns, batch_start, batch_length
        ):
            return self.grid.origin_ns
        candidate = self._pick_origin(bucket, batch_start)
        if candidate is None or _origin_too_far(candidate, batch_start, batch_length):
            return None
        return candidate

    def _pick_origin(
        self, bucket: _StreamBucket | None, batch_start: Timestamp
    ) -> int | None:
        """Pick an origin timestamp that lies on the pulse grid."""
        if bucket is not None and bucket.messages:
            for m in bucket.messages:
                if not m.timestamp < batch_start:
                    return m.timestamp.to_ns()
            return bucket.messages[0].timestamp.to_ns()
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

        self._streams: dict[StreamId, _GatedStream] = {}

        self._pending_batch_length: Duration | None = None
        self._active_window: _ActiveWindow | None = None
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
            start_time = min(m.timestamp for m in messages)
            end_time = max(m.timestamp for m in messages)
            self._active_window = self._start_active_window(
                messages, window_start=end_time
            )
            return MessageBatch(
                start_time=start_time, end_time=end_time, messages=messages
            )

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

    def _start_active_window(
        self, messages: list[Message[Any]], *, window_start: Timestamp
    ) -> _ActiveWindow:
        """Bootstrap the active window from the first batch of messages.

        Seeds estimators from gated-stream arrivals, creates the window
        at ``window_start`` (typically the max input timestamp, so the
        window opens immediately after the startup flush), and builds
        grids for any streams whose estimators already converged.
        """
        for msg in messages:
            if msg.stream.kind in GATED_STREAM_KINDS:
                self._stream(msg.stream).observe(msg)
        window = _ActiveWindow(
            start=window_start, end=window_start + self._batch_length
        )
        for sid, stream in self._streams.items():
            stream.rebuild_grid(
                window.buckets.get(sid), window.start, self._batch_length
            )
        return window

    def _stream(self, sid: StreamId) -> _GatedStream:
        """Return the gated-stream state for ``sid``, creating it on demand."""
        return self._streams.setdefault(sid, _GatedStream())

    def _route_message(self, msg: Message[Any], window: _ActiveWindow) -> None:
        bucket = window.bucket(msg.stream)
        if msg.stream.kind not in GATED_STREAM_KINDS:
            bucket.add(msg)
            return
        stream = self._stream(msg.stream)
        stream.observe(msg)
        overflow = stream.route(msg, bucket, window.start)
        if overflow is not None:
            self._overflow.append(overflow)

    def _is_batch_complete(self, window: _ActiveWindow) -> bool:
        if self._high_water_mark is not None:
            threshold = window.start + Duration.from_seconds(self.timeout_s)
            if not self._high_water_mark < threshold:
                return True

        has_gating = False
        for sid, stream in self._streams.items():
            if not stream.is_gating:
                continue
            has_gating = True
            if not stream.is_gate_satisfied(window.buckets.get(sid)):
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
        for sid, stream in self._streams.items():
            if not stream.is_gating:
                continue
            bucket = window.buckets.get(sid)
            if bucket is not None and bucket.messages:
                return False
        return True

    def _recover_from_gap(self, window: _ActiveWindow) -> _ActiveWindow:
        """Advance the window past a detected gap, then re-route stashed traffic.

        Collects non-gated/ungridded messages already bucketed in the window
        and the gated overflow, advances the window to where the pending
        traffic lives, and re-routes everything.  At ``steps == 0`` (pending
        still fits in the current window) the window is kept but buckets
        are cleared -- re-routing recomputes slot placement from scratch
        against the same ``start_time``.
        """
        stashed = window.flatten()
        pending = self._overflow
        self._overflow = []

        earliest = min(m.timestamp for m in pending)
        gap_ns = (earliest - window.start).to_ns()
        batch_ns = self._batch_length.to_ns()
        steps = max(gap_ns // batch_ns, 0)
        if steps > 0:
            new_start = window.start + Duration.from_ns(steps * batch_ns)
            window = _ActiveWindow(start=new_start, end=new_start + self._batch_length)
        else:
            window.buckets = {}
        self._active_window = window

        for msg in stashed + pending:
            self._route_message(msg, window)
        return window

    def _close_batch(self, window: _ActiveWindow) -> MessageBatch:
        batch = window.close()

        self._refresh_stream_registry(window)

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

        return batch

    def _refresh_stream_registry(self, window: _ActiveWindow) -> None:
        """Update grids, mark absence, evict dead streams, apply batch-length change.

        Runs once per close while the just-closed ``window`` is still
        available -- the window's buckets feed fresh origins into
        ``rebuild_grid``.
        """
        present: set[StreamId] = set()
        for sid, bucket in window.buckets.items():
            if sid.kind not in GATED_STREAM_KINDS or not bucket.messages:
                continue
            stream = self._streams.get(sid)
            if stream is None:
                continue
            present.add(sid)
            stream.mark_present()
            stream.rebuild_grid(bucket, window.start, self._batch_length)

        for sid in list(self._streams):
            if sid in present:
                continue
            if self._streams[sid].mark_absent():
                del self._streams[sid]

        if self._pending_batch_length is not None:
            self._batch_length = self._pending_batch_length
            self._pending_batch_length = None
            # Iterate all known streams: growing the batch length can
            # promote a previously-demoted sub-rate stream back into the
            # grid, and that stream has ``grid is None``.
            for sid, stream in self._streams.items():
                stream.rebuild_grid(
                    window.buckets.get(sid), window.start, self._batch_length
                )
