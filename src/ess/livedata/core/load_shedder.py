# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Load shedding for backend message processing.

When the backend can't keep up with the Kafka message stream, the LoadShedder
selectively drops bulk event data while preserving control messages and f144 logs.

Overload detection relies on the ``SimpleMessageBatcher`` producing consecutive
non-empty batches.  Under normal operation, the batcher uses 1-second time windows
aligned to message timestamps.  Because the processing loop runs at ~10 ms intervals,
each cycle fetches only ~10 ms worth of messages — well within the current window —
so ``batch()`` returns None roughly 99 out of 100 calls.  A non-None result means
messages have crossed a window boundary.  Consecutive non-None results mean the
processor could not drain the window before the next boundary arrived, i.e., it is
falling behind real-time.

Empty batches (non-None but with zero messages) are excluded from the overload signal.
The batcher emits these when message timestamps jump forward (e.g., after a pause
between measurement runs) to step through the gap one window at a time.  These do not
indicate overload and must not trigger shedding.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import structlog

from .message import Message, StreamKind

logger = structlog.get_logger(__name__)

DROPPABLE_KINDS = frozenset(
    {
        StreamKind.DETECTOR_EVENTS,
        StreamKind.MONITOR_EVENTS,
        StreamKind.MONITOR_COUNTS,
        StreamKind.AREA_DETECTOR,
    }
)

# Consecutive non-empty batcher results before entering shedding mode.
# With 1-second batch windows this means ~5 seconds of sustained overload.
_ACTIVATION_THRESHOLD = 5
# Consecutive idle cycles (no batch, or empty batch) before de-escalating one level.
_DEACTIVATION_THRESHOLD = 3

_MAX_LEVEL = 3

_N_BUCKETS = 10
_BUCKET_DURATION_S = 6.0  # 10 buckets x 6s = 60s rolling window


@dataclass(frozen=True, slots=True)
class LoadShedderState:
    """Snapshot of load shedder state for status reporting."""

    is_shedding: bool
    shedding_level: int
    messages_dropped: int
    messages_eligible: int


class _RollingCounter:
    """Pair of (dropped, eligible) counters over a fixed rolling time window.

    The window is divided into fixed-size time buckets. Buckets older than the
    window are discarded when the counter is advanced to the current time.
    """

    def __init__(
        self,
        n_buckets: int = _N_BUCKETS,
        bucket_duration_s: float = _BUCKET_DURATION_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._n_buckets = n_buckets
        self._bucket_duration_s = bucket_duration_s
        self._clock = clock
        self._dropped = [0] * n_buckets
        self._eligible = [0] * n_buckets
        self._current_bucket: int = 0
        self._last_time: float = clock()

    def _advance(self) -> None:
        """Advance to the current time, zeroing any expired buckets."""
        now = self._clock()
        elapsed = now - self._last_time
        steps = int(elapsed / self._bucket_duration_s)
        if steps <= 0:
            return
        # Cap: if we've been idle longer than the full window, just clear everything
        steps = min(steps, self._n_buckets)
        for i in range(1, steps + 1):
            bucket = (self._current_bucket + i) % self._n_buckets
            self._dropped[bucket] = 0
            self._eligible[bucket] = 0
        self._current_bucket = (self._current_bucket + steps) % self._n_buckets
        self._last_time += steps * self._bucket_duration_s

    def record(self, *, dropped: int, eligible: int) -> None:
        """Record counts into the current bucket."""
        self._advance()
        self._dropped[self._current_bucket] += dropped
        self._eligible[self._current_bucket] += eligible

    def totals(self) -> tuple[int, int]:
        """Return (dropped, eligible) summed over the rolling window."""
        self._advance()
        return sum(self._dropped), sum(self._eligible)


class LoadShedder:
    """Selectively drops bulk event data when the backend falls behind.

    Overload detection counts consecutive non-empty batches produced by the
    message batcher.  A non-empty batch means messages crossed a time-window
    boundary, which happens approximately once per batch window under normal
    load.  Consecutive non-empty batches mean the processor is not keeping up:
    by the time one batch is processed, enough new messages have arrived to
    immediately complete the next window.

    Empty batches (non-None result with zero messages) are explicitly excluded.
    The ``SimpleMessageBatcher`` emits these to step through timestamp gaps
    (e.g., after a pause between measurements) and they do not indicate load.

    Shedding uses exponential levels: level N keeps every ``2**N``-th droppable
    message.  Each level handles a 2x increase in overload (level 1 = 50% drop,
    level 2 = 75%, level 3 = 87.5%).  The level escalates by 1 after
    ``_ACTIVATION_THRESHOLD`` consecutive non-empty batches, up to
    ``_MAX_LEVEL``, and de-escalates by 1 after ``_DEACTIVATION_THRESHOLD``
    consecutive idle cycles.

    Drop statistics are tracked over a rolling 60-second window.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._consecutive_batches: int = 0
        self._consecutive_idle: int = 0
        self._level: int = 0
        self._subsample_counter: int = 0
        self._rolling = _RollingCounter(clock=clock)

    @property
    def state(self) -> LoadShedderState:
        dropped, eligible = self._rolling.totals()
        return LoadShedderState(
            is_shedding=self._level > 0,
            shedding_level=self._level,
            messages_dropped=dropped,
            messages_eligible=eligible,
        )

    def report_batch_result(self, batch_message_count: int) -> None:
        """Update overload detection counters after a batcher cycle.

        Only batches with at least one message count toward the activation
        threshold.  Empty batches (zero messages) are treated as idle because
        they arise from the batcher stepping through timestamp gaps, not from
        genuine overload.

        Parameters
        ----------
        batch_message_count:
            Number of messages in the batch returned by the batcher, or 0 if
            the batcher returned None (no batch) or an empty batch.
        """
        if batch_message_count > 0:
            self._consecutive_batches += 1
            self._consecutive_idle = 0
            if (
                self._consecutive_batches >= _ACTIVATION_THRESHOLD
                and self._level < _MAX_LEVEL
            ):
                self._level += 1
                self._consecutive_batches = 0
                logger.warning(
                    'shedding_escalated',
                    level=self._level,
                    keeping=f"1/{2**self._level}",
                )
        else:
            self._consecutive_idle += 1
            self._consecutive_batches = 0
            if self._level > 0 and self._consecutive_idle >= _DEACTIVATION_THRESHOLD:
                self._level -= 1
                self._consecutive_idle = 0
                if self._level == 0:
                    self._subsample_counter = 0
                    logger.warning('shedding_stopped')
                else:
                    logger.warning(
                        'shedding_deescalated',
                        level=self._level,
                        keeping=f"1/{2**self._level}",
                    )

    def shed(self, messages: list[Message]) -> list[Message]:
        """Filter messages when shedding is active.

        When inactive, returns all messages unchanged.
        When active, keeps every ``2**level``-th droppable message.
        Non-droppable messages (control, f144 logs) are always preserved.

        Both active and inactive calls record eligible message counts into the
        rolling window so the drop rate reflects what fraction is being shed.
        """
        eligible = sum(1 for m in messages if m.stream.kind in DROPPABLE_KINDS)
        if self._level == 0:
            self._rolling.record(dropped=0, eligible=eligible)
            return messages
        keep_every = 2**self._level
        dropped = 0
        result: list[Message] = []
        for msg in messages:
            if msg.stream.kind not in DROPPABLE_KINDS:
                result.append(msg)
            else:
                self._subsample_counter += 1
                if self._subsample_counter % keep_every == 0:
                    result.append(msg)
                else:
                    dropped += 1
        self._rolling.record(dropped=dropped, eligible=eligible)
        return result
