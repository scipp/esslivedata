# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Rate-aware message batcher.

Batches messages based on per-stream rate estimation and slot-based completion.
A batch is considered complete for a given stream when a message arrives whose
timestamp falls in the last expected "pulse slot" for that stream within the
batch window — not when a fixed message count is reached.

Clock policy
------------
Data-derived timestamps are the batcher's clock for *placement*: where windows
sit, when gates and timeouts close them, how far gap recovery jumps.  Wall
time never influences placement or window sizing — during Kafka backlog
catch-up, hours of data time pass in seconds of wall time, and wall-clock
windows would batch that backlog wrongly.

Wall time is consulted for exactly one thing: *liveness*.  A window placed by
a pathological timestamp can make closure impossible (no slot gate satisfiable,
the timeout threshold unreachable), and no data-derived signal distinguishes
that from a quiet stream — the data clock is the very thing that broke.  So
when no batch has closed for a bounded wall-clock interval while traffic is
buffered, the window is re-placed at the plausible anchor of that traffic
(see ``_recover_from_stall``).  The backstop is self-correcting: a wrong
re-placement is just another stall, corrected the same way, so it needs no
per-pathology analysis of *how* the window got misplaced.

Freshness policy
----------------
The batcher bounds its backlog and, when the service cannot keep up, stays
live rather than complete: the stalest surplus of bulk data is shed, and the
window jumps to the newest retained data instead of replaying a queue it can
never clear.  Operators steer beam, samples and detector commissioning from
this output, where a stale reading is worse than an intermittent hole.  Only
``GATED_STREAM_KINDS`` can be shed -- control, log and context data never
reach the overflow, so the selectivity is structural rather than stated.
Shedding is an exception state to be fixed upstream, not a mode to run in:
drops are counted, logged (``batcher_backlog_shedding``) and reported via
``drain_metrics``.  The bounds and their sizing rationale are documented at
``DEFAULT_MAX_BACKLOG_S``, ``DEFAULT_MAX_BACKLOG_BATCHES`` and
``DEFAULT_MAX_BACKLOG_BYTES``.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import structlog

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.core.message_batcher import (
    MAX_TIMESTAMP_AHEAD_BATCHES,
    BatcherMetrics,
    MessageBatch,
    MessageBatcher,
    plausible_anchor,
)
from ess.livedata.core.timestamp import Duration, Timestamp

logger = structlog.get_logger(__name__)

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

# Bounds on the retained gated overflow, enforcing the module docstring's
# freshness policy.  A window advances by at most one batch length per
# ``batch()`` call, so whenever a service iteration takes longer than the
# batch length the caller hands over more data than the window can release
# and the surplus stays in ``_overflow`` for the life of the process --
# unbounded memory, while the Kafka consumer reports zero lag because the
# backlog is downstream of it.  This is the same bounded-buffer policy the
# consumer queue already applies (``BackgroundMessageSource``,
# ``max_queue_size``): drop rather than grow, and say so.
#
# The primary bound is *data time*, not memory: retained backlog is data the
# service has not caught up with, and under genuine overload -- the loop
# itself too slow -- the batcher only catches up at ``batch_length -
# iteration_time`` per call, so a deep backlog buys late data at the price
# of showing everything late for as long as the replay takes.  A bound in
# data time caps the lag a stall can leave behind regardless of how deep
# the stall was.  Expressing it in data time also makes it uniform across
# instruments -- a byte budget buys ~20 s of ev44 but ~1.5 s of
# area-detector frames.  It is applied per stream, against each stream's
# own frontier: a single horizon for the whole backlog would shed a
# stream's fresh data merely because a peer is stamped further ahead (see
# ``_shed_backlog``).
#
# The bound is the larger of two terms.  ``DEFAULT_MAX_BACKLOG_S`` is the
# burst tolerance: a delivery gap (consumer-group rebalance, broker blip,
# producer restart) that hands over less than this in one poll is replayed
# in full rather than holed -- and cheaply, because an *idle* service
# recovers a retained burst by a gap jump plus timeout closes at poll rate,
# not by the slow crawl of an overloaded loop.  Its price is paid only
# under sustained overload past what escalation absorbs, where the steady
# lag approaches the bound.  No pre-production statistics on gap durations
# exist; 10 s sits above single-rebalance scale, and the ``max_backlog_s``
# metric and drop counters are the instrument for tuning it.
#
# ``DEFAULT_MAX_BACKLOG_BATCHES`` is a floor that scales with the window:
# the overflow legitimately holds up to about one batch length of
# in-transit next-window traffic, so a fixed seconds bound below the
# escalated batch length would shed healthy traffic.  Two batch lengths
# leaves ordinary jitter alone -- an iteration merely overshooting its
# window sheds nothing -- and follows the adaptive wrapper's escalated
# windows (16 s at the 8 s ceiling, where the floor exceeds the seconds
# term and takes over).
DEFAULT_MAX_BACKLOG_S = 10.0
DEFAULT_MAX_BACKLOG_BATCHES = 2

# Secondary hard bound, on memory.  The data-time bound alone does not bound
# bytes: payload sizes span four orders of magnitude, and one ad00
# area-detector frame (4096x4096) is 67 MB as retained -- the adapter widens
# the uint16 wire dtype to int32 -- so two batch lengths of frames is
# gigabytes.  Sizing, against a 32 GB production host shared by all backend
# services and their workflow/job data: 512 MB holds ~13000 ev44 chunks
# while capping area-detector retention at ~7 frames, and every backend
# service shedding at once still costs only ~2 GB of the host.
DEFAULT_MAX_BACKLOG_BYTES = 512 * 1024**2

# Assumed size of a payload whose arrays cannot be measured.  Only the four
# bulk kinds reach the overflow and all of them carry numpy arrays, so this
# is a floor for exotic payloads rather than a value the services rely on.
UNKNOWN_PAYLOAD_BYTES = 1024

# Shedding is intermittent by nature -- each jump clears the backlog, which
# then refills -- so per-event logging would flap once per service iteration.
# Report a cumulative summary at a fixed wall-clock interval instead.
SHED_LOG_INTERVAL_S = 60.0

# Wall-clock stall threshold for the liveness backstop, in multiples of the
# batch length (see the module docstring's clock policy).  It must sit above
# the slowest legitimate close so healthy operation never triggers it: the
# timeout path closes within ``timeout_factor`` batch lengths of data time,
# which under live traffic is roughly wall time.  Scaling with the batch
# length keeps the margin when the adaptive wrapper escalates the window.
_STALL_THRESHOLD_BATCHES = 10

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

# Disjoint-epoch horizon, in multiples of batch_length: the maximum distance
# between a timestamp and the active window at which the timestamp is still
# treated as belonging to the same epoch.  Beyond it, the timestamp is
# assumed pathological (wrong schema field, unit mismatch, run-local clock)
# rather than a legitimate silence gap.  Used to bound the distance between
# a new grid's origin and the active batch_start -- a disjoint-epoch grid
# would place every slot billions of slots away from batch_start, pinning
# the bucket's max_slot at -1 and vetoing every slot-gate closure for the
# whole batcher -- and to divert disjoint-epoch messages out of routing
# (see ``_route_message``).  1000 batches (>=16 min at 1 s batches) is well
# above any legitimate cold-start, post-eviction, or silence-gap offset but
# below any real epoch mismatch by many orders of magnitude.
_MAX_ORIGIN_OFFSET_BATCHES = 1000

# ``MAX_TIMESTAMP_AHEAD_BATCHES`` (imported from message_batcher) is the
# plausibility horizon shared by all uses in this module: the HWM clamp,
# the future-message hold-back cap, and the outlier absorption in
# ``_route_message``.  It must be >= ``timeout_factor`` (default 1.2) for
# the timeout path to ever fire -- the HWM clamp caps how far past the
# window the HWM can reach, so a larger timeout_factor would starve the
# timeout permanently; the constructor and setter enforce the invariant.
# It should also sit comfortably above the default for
# sub-Hz-only streams whose sparse arrivals rely on multi-batch HWM jumps
# to trigger cascading timeout closes of empty batches between pulses.
# Three batches allows one pulse's worth of HWM advance to cover the
# preceding empty batch, matching the natural cadence of 0.5 Hz-and-below
# gated streams.


def _payload_nbytes(value: Any) -> int:
    """Best-effort retained size of a message payload.

    Payloads reaching the overflow are ``DetectorEvents``/``MonitorEvents``
    (a dataclass of numpy arrays viewing the source flatbuffer, which is
    what the retained message actually pins) or the ``sc.DataArray`` that
    the da00/ad00 adapters produce upstream of the batcher (monitor counts,
    area-detector frames).  Scipp exposes ``underlying_size()``, which
    measures the owned buffers a retained slice pins; numpy arrays are
    measured by the allocation they view (:func:`_pinned_nbytes`), and a
    dataclass over its array fields.
    """
    underlying_size = getattr(value, 'underlying_size', None)
    if callable(underlying_size):
        size = underlying_size()
        if isinstance(size, int):
            return size
    if isinstance(value, tuple):
        parts: Iterable[Any] = value
    elif hasattr(value, 'nbytes'):
        parts = (value,)
    else:
        parts = getattr(value, '__dict__', {}).values()
    return _pinned_nbytes(parts) or UNKNOWN_PAYLOAD_BYTES


def _pinned_nbytes(arrays: Iterable[Any]) -> int:
    """Bytes kept alive by numpy arrays: each viewed allocation, counted once.

    A view's own ``nbytes`` understates what it retains -- a ``MonitorEvents``
    keeps only the time-of-arrival half of its ev44 buffer visible but pins
    the whole buffer -- and two views of one buffer (``DetectorEvents``)
    must not count it twice.
    """
    seen: set[int] = set()
    total = 0
    for array in arrays:
        if not isinstance(getattr(array, 'nbytes', None), int):
            continue
        owner = array
        while getattr(owner, 'base', None) is not None:
            owner = owner.base
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        try:
            total += memoryview(owner).nbytes
        except TypeError:
            total += array.nbytes
    return total


@dataclass(slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(slots=True)
class _ActiveWindow:
    """Time range of the batch currently in progress."""

    start: Timestamp
    end: Timestamp


@dataclass(slots=True)
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

        Overflow *near* the window bumps ``max_slot`` to the last grid slot:
        a pulse just past the window is evidence that the window's final
        pulse was reached.  A pulse implausibly far ahead is no such
        evidence, and because overflow is re-routed into every new window at
        close, an unbounded bump would pre-satisfy the gate on every window
        -- the batcher then closes a batch per *call*, racing the window
        toward the outlier at poll rate with time ranges detached from the
        traffic it delivers.
        """
        self.observe(msg)
        if self.grid is None:
            self._add(msg)
            return None
        slot = self.grid.slot_in_batch(msg.timestamp, window_start)
        if slot >= self.grid.slots_per_batch:
            horizon = self.grid.slots_per_batch * (1 + MAX_TIMESTAMP_AHEAD_BATCHES)
            if slot < horizon:
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
        origin = self._choose_origin(batch_start, batch_length)
        if origin is None:
            self.grid = None
            return
        slots_per_batch = round(int_rate * batch_length_s)
        period_ns = round(1e9 / int_rate)
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
                if m.timestamp >= batch_start:
                    return m.timestamp.to_ns()
            return self.messages[0].timestamp.to_ns()
        return self.estimator.last_ts_ns


def _origin_too_far(
    origin_ns: int, batch_start: Timestamp, batch_length: Duration
) -> bool:
    """True if ``origin_ns`` is implausibly far from ``batch_start``."""
    max_offset_ns = _MAX_ORIGIN_OFFSET_BATCHES * batch_length.to_ns()
    return abs(origin_ns - batch_start.to_ns()) > max_offset_ns


def _validate_timeout_factor(timeout_factor: float) -> None:
    """Reject a timeout the HWM clamp makes unreachable.

    The clamp caps the high-water mark at ``MAX_TIMESTAMP_AHEAD_BATCHES``
    past the window start, so a finite timeout threshold beyond that can
    never be reached and the timeout path would silently never fire.  An
    explicit ``inf`` states the intent instead: closure by slot gates only.
    """
    if math.isinf(timeout_factor) and timeout_factor > 0:
        return
    if not 0.0 < timeout_factor <= MAX_TIMESTAMP_AHEAD_BATCHES:
        raise ValueError(
            f"timeout_factor must be in (0, {MAX_TIMESTAMP_AHEAD_BATCHES}] "
            f"(the HWM plausibility horizon) or inf to disable the timeout, "
            f"got {timeout_factor}"
        )


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
        clock: Callable[[], float] = time.monotonic,
        max_backlog_s: float = DEFAULT_MAX_BACKLOG_S,
        max_backlog_batches: int = DEFAULT_MAX_BACKLOG_BATCHES,
        max_backlog_bytes: int = DEFAULT_MAX_BACKLOG_BYTES,
    ) -> None:
        self._batch_length = Duration.from_seconds(batch_length_s)
        self._timeout_factor = (
            timeout_s / batch_length_s if timeout_s is not None else 1.2
        )
        _validate_timeout_factor(self._timeout_factor)
        for name, limit in (
            ('max_backlog_s', max_backlog_s),
            ('max_backlog_batches', max_backlog_batches),
            ('max_backlog_bytes', max_backlog_bytes),
        ):
            if not math.isfinite(limit) or limit < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {limit}")

        self._streams: defaultdict[StreamId, _GatedStream] = defaultdict(_GatedStream)

        self._pending_batch_length: Duration | None = None
        self._active_window: _ActiveWindow | None = None
        self._high_water_mark: Timestamp | None = None
        self._overflow: list[Message[Any]] = []
        self._non_gated: list[Message[Any]] = []
        self._future: list[Message[Any]] = []
        self._clock = clock
        self._last_close_wall = clock()
        self._max_backlog_s = max_backlog_s
        self._max_backlog_batches = max_backlog_batches
        self._max_backlog_bytes = max_backlog_bytes
        self._total_dropped_messages = 0
        self._total_dropped_bytes = 0
        self._dropped_messages_since_drain = 0
        self._dropped_bytes_since_drain = 0
        self._backlog_peak_s = 0.0
        self._last_shed_log = -math.inf

    @property
    def batch_length_s(self) -> float:
        return self._batch_length.to_seconds()

    @property
    def total_dropped_messages(self) -> int:
        """Total gated messages dropped to keep the backlog bounded."""
        return self._total_dropped_messages

    @property
    def total_dropped_bytes(self) -> int:
        """Total payload bytes dropped to keep the backlog bounded."""
        return self._total_dropped_bytes

    def drain_metrics(self) -> BatcherMetrics:
        metrics = BatcherMetrics(
            max_backlog_s=self._backlog_peak_s,
            dropped_messages=self._dropped_messages_since_drain,
            dropped_bytes=self._dropped_bytes_since_drain,
        )
        self._backlog_peak_s = 0.0
        self._dropped_messages_since_drain = 0
        self._dropped_bytes_since_drain = 0
        return metrics

    @property
    def tracked_streams(self) -> set[StreamId]:
        """Stream IDs currently tracked by the batcher."""
        return set(self._streams)

    def is_gating(self, stream_id: StreamId) -> bool:
        """True if the stream has a converged grid and gates batch closure."""
        stream = self._streams.get(stream_id)
        return stream.is_gating if stream is not None else False

    @property
    def timeout_factor(self) -> float:
        return self._timeout_factor

    @timeout_factor.setter
    def timeout_factor(self, value: float) -> None:
        _validate_timeout_factor(value)
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
            self._high_water_mark = self._clamped_hwm(latest)

        window = self._active_window
        if window is None:
            if not messages:
                return None
            return self._bootstrap_batch(messages)

        for msg in messages:
            self._route_message(msg, window)

        # Shed before the gap check so a jump targets the surviving backlog.
        self._shed_backlog()

        if self._should_recover_from_gap(window):
            window = self._recover_from_gap(window)

        # Stall recovery is a last resort, checked only when nothing closes:
        # checking it first would preempt the timeout close of a stream
        # sparser than the stall threshold (e.g. a log value every 15 s),
        # re-placing the window and resetting the HWM on every arrival
        # instead of delivering it.
        if self._is_batch_complete(window):
            return self._close_batch(window)
        if self._is_stalled():
            window = self._recover_from_stall(window)
            if self._is_batch_complete(window):
                return self._close_batch(window)
        return None

    def _clamped_hwm(self, latest: Timestamp) -> Timestamp:
        """Clamp an HWM update to a bounded distance past the active window.

        Protects against a single malformed timestamp (e.g. upstream epoch
        bug producing a value years ahead) permanently pinning the HWM in
        the future, which would otherwise force every subsequent ``batch()``
        call to close a batch via the timeout path for millions of
        iterations -- effectively a DoS until the process restarts.

        Bounding HWM relative to the active window (rather than to the
        prior HWM) keeps HWM self-healing: each batch close advances the
        window by one batch_length, so after a bounded number of cascading
        empty closures the HWM is no longer past the timeout threshold and
        timeout firing stops on its own.  An absolute per-call cap doesn't
        self-heal because the window keeps moving away from the clamped HWM.

        Cold start (no window or no prior HWM) accepts ``latest`` as-is;
        ``_bootstrap_batch`` re-anchors it.  Otherwise the new value is
        capped at ``window.start + cap`` and floored at the current HWM
        so it never regresses -- a window advance (close or gap advance)
        may briefly leave HWM past the cap, and the next update must hold
        that value until the window catches up.
        """
        if self._active_window is None or self._high_water_mark is None:
            return latest
        cap = MAX_TIMESTAMP_AHEAD_BATCHES * self._batch_length
        ceiling = self._active_window.start + cap
        return max(self._high_water_mark, min(latest, ceiling))

    def _bootstrap_batch(self, messages: list[Message[Any]]) -> MessageBatch:
        """Flush the startup backlog and open the active window.

        Seeds estimators from gated-stream arrivals, opens the window at
        the newest plausible input timestamp (so it starts immediately
        after the flush; see :func:`plausible_anchor` for why a bare max
        must not be used), builds grids for any streams whose estimators
        already converged, and returns the flushed messages as the first
        batch.  The high-water mark is re-anchored likewise, since at cold
        start ``_clamped_hwm`` accepted the raw maximum unclamped.
        Outlier timestamps beyond the anchor are excluded from estimator
        seeding so they cannot pin ``last_ts_ns`` in the far future.
        """
        start_time = min(m.timestamp for m in messages)
        end_time = plausible_anchor([m.timestamp for m in messages], self._batch_length)
        for msg in messages:
            if msg.stream.kind in GATED_STREAM_KINDS and msg.timestamp <= end_time:
                self._streams[msg.stream].observe(msg)
        self._active_window = _ActiveWindow(
            start=end_time, end=end_time + self._batch_length
        )
        self._high_water_mark = end_time
        self._last_close_wall = self._clock()
        for stream in self._streams.values():
            stream.refresh_grid(end_time, self._batch_length)
        return MessageBatch(start_time=start_time, end_time=end_time, messages=messages)

    def _route_message(self, msg: Message[Any], window: _ActiveWindow) -> None:
        """Bucket a message by stream kind and timestamp relative to the window.

        Messages beyond the disjoint-epoch horizon
        (``_MAX_ORIGIN_OFFSET_BATCHES`` past ``window.end``) are delivered
        with the active batch, bypassing estimators, slot gates, overflow,
        and the future hold-back: such a timestamp cannot come from a
        legitimate silence gap and must not be cached indefinitely, drive
        gap recovery, or pin a stream's estimator.

        Ungridded streams (non-gated kind OR sub-Hz gated without a grid)
        hold messages with ``window.end < ts <= window.end + K * batch_length``
        in ``_future`` so batch contents stay bounded by the batch's time
        range.  ``K`` is ``MAX_TIMESTAMP_AHEAD_BATCHES``: beyond that, the
        message falls through to the active batch instead of being cached.

        Gridded gated streams use the slot-based overflow path instead, which
        drives gap recovery via ``_should_recover_from_gap``; overflow
        timestamps are therefore bounded by the disjoint-epoch horizon, and
        so is the gap-recovery jump.
        """
        is_gated = msg.stream.kind in GATED_STREAM_KINDS
        stream = self._streams[msg.stream] if is_gated else None
        if self._is_disjoint_epoch(msg, window):
            # Deliberately not placed in the stream's own bucket: that would
            # mark the stream as having contributed to the batch (vetoing gap
            # recovery for every other stream) while never satisfying its slot
            # gate, since the message has no slot. Delivery is unaffected --
            # both buffers drain into the same batch.
            self._non_gated.append(msg)
            return
        if (stream is None or not stream.is_gating) and self._is_future(msg, window):
            self._future.append(msg)
            return
        if stream is None:
            self._non_gated.append(msg)
            return
        overflow = stream.route(msg, window.start)
        if overflow is not None:
            self._overflow.append(overflow)

    def _is_disjoint_epoch(self, msg: Message[Any], window: _ActiveWindow) -> bool:
        """True if ``msg`` is implausibly far ahead of the active window."""
        cap = _MAX_ORIGIN_OFFSET_BATCHES * self._batch_length
        return msg.timestamp - window.end > cap

    def _is_future(self, msg: Message[Any], window: _ActiveWindow) -> bool:
        """True if ``msg`` belongs in a future window within the hold-back cap."""
        if msg.timestamp <= window.end:
            return False
        cap = MAX_TIMESTAMP_AHEAD_BATCHES * self._batch_length
        return msg.timestamp - window.end <= cap

    def _is_batch_complete(self, window: _ActiveWindow) -> bool:
        if self._high_water_mark is not None and not math.isinf(self._timeout_factor):
            threshold = window.start + Duration.from_seconds(self.timeout_s)
            if self._high_water_mark >= threshold:
                return True

        has_gating = False
        for stream in self._streams.values():
            if not stream.is_gating:
                continue
            has_gating = True
            if not stream.is_gate_satisfied():
                return False
        return has_gating

    def _shed_backlog(self) -> None:
        """Drop the stalest overflow until it fits both backlog bounds.

        Each stream retains at most the data-time bound -- the larger of
        ``max_backlog_batches`` batch lengths and ``max_backlog_s`` -- and
        the backlog as a whole at most ``max_backlog_bytes``; see the
        bounds' rationale at the top of this module.

        The data-time horizon is per stream, anchored on that stream's own
        frontier.  A single horizon for the whole backlog would conflate
        depth with the offset between streams: a stream stamped seconds
        ahead of its peers would set the horizon for all of them and shed
        their fresh data as if it were stale.  Each frontier is the stream's
        plausible anchor rather than its bare maximum, so one stray
        far-future timestamp cannot condemn the real backlog behind it
        (:func:`plausible_anchor`).

        The byte bound stays global -- memory is -- and drops the stalest
        survivors regardless of stream, after any message stamped ahead of
        its stream's frontier: such a message is disconnected from the
        stream's traffic by construction, and trimming oldest-first alone
        would shed valid traffic to retain it, then hand the eventual gap
        jump that stray as its only anchor.

        Keeps the newest messages: once the window's own region drains, gap
        detection advances it to the oldest survivor in one jump, so keeping
        the newest is what lets the window catch up to live traffic instead
        of crawling through a backlog it can never clear.  The alternative
        -- dropping the newest -- bounds memory just as well but leaves the
        window falling permanently further behind, delivering ever-staler
        data.  The newest message survives unconditionally, even alone over
        the byte bound: without a survivor there is nothing for that jump
        to anchor the new window on.

        Shedding deliberately does *not* force the jump itself.  Gap
        recovery is vetoed while any gated stream still has messages in the
        window, and that veto must hold: a shed can be a byte-bound trim of
        one oversized stream while a peer gates normally in the current
        window, and jumping then overruns the peer's live traffic --
        leaving its gate permanently unsatisfiable and delivery stalled
        until the wall-clock backstop.  When a shed does strand the window
        behind the survivors, its region is empty by construction (the data
        was just discarded), so the ordinary gap check fires on the next
        call and jumps in one step.  Bounded lag arrives one call late;
        livelock never.
        """
        if not self._overflow:
            return
        sized = sorted(
            ((_payload_nbytes(msg.value), msg) for msg in self._overflow),
            key=lambda item: item[1].timestamp,
        )
        by_stream: defaultdict[StreamId, list[Timestamp]] = defaultdict(list)
        for _, msg in sized:
            by_stream[msg.stream].append(msg.timestamp)
        bound = max(
            self._max_backlog_batches * self._batch_length,
            Duration.from_seconds(self._max_backlog_s),
        )
        frontiers: dict[StreamId, Timestamp] = {}
        for stream, stamps in by_stream.items():
            frontier = plausible_anchor(stamps, self._batch_length)
            self._backlog_peak_s = max(
                self._backlog_peak_s, (frontier - stamps[0]).to_seconds()
            )
            frontiers[stream] = frontier
        strays: list[tuple[int, Message[Any]]] = []
        kept: list[tuple[int, Message[Any]]] = []
        for item in sized:
            frontier = frontiers[item[1].stream]
            stamp = item[1].timestamp
            if stamp > frontier:
                strays.append(item)
            elif stamp >= frontier - bound:
                kept.append(item)
        candidates = strays + kept
        retained = sum(nbytes for nbytes, _ in candidates)
        shed_by_bytes = 0
        while (
            shed_by_bytes < len(candidates) - 1 and retained > self._max_backlog_bytes
        ):
            retained -= candidates[shed_by_bytes][0]
            shed_by_bytes += 1
        survivors = candidates[shed_by_bytes:]
        if len(survivors) == len(sized):
            return
        dropped = len(sized) - len(survivors)
        dropped_bytes = sum(nbytes for nbytes, _ in sized) - retained
        self._overflow = [msg for _, msg in survivors]
        self._total_dropped_messages += dropped
        self._total_dropped_bytes += dropped_bytes
        self._dropped_messages_since_drain += dropped
        self._dropped_bytes_since_drain += dropped_bytes
        now = self._clock()
        if now - self._last_shed_log >= SHED_LOG_INTERVAL_S:
            self._last_shed_log = now
            logger.warning(
                'batcher_backlog_shedding',
                backlog_limit_s=bound.to_seconds(),
                backlog_limit_bytes=self._max_backlog_bytes,
                max_backlog_s=round(self._backlog_peak_s, 3),
                total_dropped_messages=self._total_dropped_messages,
                total_dropped_bytes=self._total_dropped_bytes,
                batch_length_s=self.batch_length_s,
            )

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

        Despite the name, this is not only a silence-gap path: it is also how
        the window catches up whenever a call hands over more data time than
        one batch length.  A batch closes at most once per call, so the close
        path alone advances the window by one batch length per call -- a
        ceiling that binds only while the window sits inside continuous data.
        When a call's gated arrivals all land past the window, no gridded
        stream has contributed and this path advances the window in one jump
        instead.

        Jumping does not skip delivery.  Stashed and pending messages older
        than the new ``window.start`` re-route to negative slots: they ride
        along in the next batch (whose ``start_time`` then understates its
        contents, as in ``SimpleMessageBatcher._split_messages``) rather than
        being dropped.  Only ``_shed_backlog`` discards messages, so the data
        actually lost under overload is well below the surplus the window
        could not release.
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

    def _is_stalled(self) -> bool:
        """True if traffic is buffered but no batch has closed for too long.

        This is the liveness backstop of the module docstring's clock
        policy: it makes no attempt to diagnose *why* nothing closes (a
        window misplaced by a poisoned bootstrap, a poisoned gap jump, an
        upstream clock jump, ...) -- any placement the buffered traffic
        cannot close is corrected the same way, by re-placing the window
        at that traffic.

        The wall-clock threshold is what makes this safe against ordinary
        buffer states: a healthy batcher closes several times per threshold
        interval, so a single poll carrying only a lagging partition's
        backlog -- every message of it legitimately behind the window --
        can never drag the window backwards.  Without buffered traffic
        there is nothing to re-place onto: a quiet stream is not a stall,
        and the data clock must not advance on wall-time evidence alone.
        """
        threshold = _STALL_THRESHOLD_BATCHES * self.batch_length_s
        if self._clock() - self._last_close_wall < threshold:
            return False
        return bool(self._buffered_messages())

    def _recover_from_stall(self, window: _ActiveWindow) -> _ActiveWindow:
        """Re-place the window at the buffered traffic and re-route it.

        The anchor follows the bulk of the buffered traffic, not a bare
        max: the message that misplaced the window may itself be buffered,
        and must not veto the recovery it caused.  The high-water mark is
        reset likewise -- it was derived from the same timestamps that
        misplaced the window, and keeping it would force a long cascade of
        empty timeout closures instead of an immediate recovery.  Grids
        with implausible origins rebuild at the next close via
        ``_refresh_stream_registry``.

        A wrong re-placement (e.g. onto a lone stray while real traffic is
        silent) is not a hazard: it is just another stall, corrected the
        same way once real traffic buffers up again.

        Held-back and overflow messages are drained along with the window
        buckets, and any message still implausibly far ahead of the
        recovered window is diverted to delivery rather than re-cached: a
        stall proves the placement such messages drove was wrong, and
        re-caching the driver would re-trigger the same wrong gap jump --
        and with it a stall per encounter -- until data time reaches it.

        An anchor already inside the active window means placement agrees
        with the buffered traffic and the stall is mere quietness (e.g. the
        trailing partial batch after a stream stops -- deliberately not
        delivered, since the data clock must not advance on wall time
        alone).  Re-placing would only shift the boundaries and log a
        recovery per threshold interval, so the backstop re-arms instead.
        """
        anchor = plausible_anchor(
            [m.timestamp for m in self._buffered_messages()], self._batch_length
        )
        self._last_close_wall = self._clock()
        if window.start <= anchor < window.end:
            return window
        stashed = self._drain_window() + self._overflow + self._future
        self._overflow = []
        self._future = []
        logger.warning(
            'batcher_stall_recovery',
            window_start_ns=window.start.to_ns(),
            anchor_ns=anchor.to_ns(),
            buffered_messages=len(stashed),
        )
        window = _ActiveWindow(start=anchor, end=anchor + self._batch_length)
        self._active_window = window
        self._high_water_mark = anchor
        hold_back = MAX_TIMESTAMP_AHEAD_BATCHES * self._batch_length
        for msg in stashed:
            if msg.timestamp - window.end > hold_back:
                self._non_gated.append(msg)
            else:
                self._route_message(msg, window)
        return window

    def _buffered_messages(self) -> list[Message[Any]]:
        """All messages currently buffered for the active batch."""
        messages = list(self._non_gated)
        for stream in self._streams.values():
            messages.extend(stream.messages)
        return messages

    def _drain_window(self) -> list[Message[Any]]:
        """Remove and return all messages buffered for the active batch."""
        messages = self._non_gated
        self._non_gated = []
        for stream in self._streams.values():
            messages.extend(stream.drain())
        return messages

    def _any_gating(self) -> bool:
        """True if at least one tracked stream currently gates closure."""
        return any(stream.is_gating for stream in self._streams.values())

    def _close_batch(self, window: _ActiveWindow) -> MessageBatch:
        self._last_close_wall = self._clock()
        self._refresh_stream_registry(window)
        messages = self._drain_window()

        if self._any_gating():
            end_time = window.end
        else:
            # No gridded stream gates closure, so the timeout path closed the
            # batch.  Stepping ``end_time`` by one batch_length per call would
            # leave the batch's timestamps behind the data when traffic spans
            # several batch_lengths per call.  Mirror ``SimpleMessageBatcher``:
            # include all held-back traffic and set ``end_time`` to the newest
            # plausible message so the batch covers its real time range -- a
            # bare max would let one absorbed outlier anchor the next window
            # in the far future (see ``plausible_anchor``).
            messages += self._future + self._overflow
            self._future = []
            self._overflow = []
            if messages:
                anchor = plausible_anchor(
                    [m.timestamp for m in messages], self._batch_length
                )
                end_time = max(anchor, window.end)
            else:
                end_time = window.end

        batch = MessageBatch(
            start_time=window.start, end_time=end_time, messages=messages
        )

        new_start = end_time
        new_window = _ActiveWindow(start=new_start, end=new_start + self._batch_length)
        self._active_window = new_window
        # Drain overflow into the new window.  Timestamps that still fall
        # past the last slot land back in ``_overflow`` and wait for the
        # next close; gap recovery handles jumps larger than one batch.  In
        # the non-gating branch both buffers are already drained above, so
        # these loops are no-ops there.
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
