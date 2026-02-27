# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Load shedding for backend message processing.

When the backend can't keep up with the Kafka message stream, the LoadShedder
selectively drops bulk event data while preserving control messages and f144 logs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .message import Message, StreamKind

DROPPABLE_KINDS = frozenset(
    {
        StreamKind.DETECTOR_EVENTS,
        StreamKind.MONITOR_EVENTS,
        StreamKind.MONITOR_COUNTS,
        StreamKind.AREA_DETECTOR,
    }
)

# Consecutive non-None batcher results before entering shedding mode
_ACTIVATION_THRESHOLD = 5
# Consecutive idle (None) batcher results before exiting shedding mode
_DEACTIVATION_THRESHOLD = 3

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

    Detection uses consecutive non-None batcher results as the overload signal.
    Shedding uses exponential levels: level N keeps every ``2**N``-th droppable
    message.  Each level handles a 2x increase in overload (level 1 = 50% drop,
    level 2 = 75%, level 3 = 87.5%, …).  The level escalates by 1 after
    ``_ACTIVATION_THRESHOLD`` consecutive non-idle batcher cycles and
    de-escalates by 1 after ``_DEACTIVATION_THRESHOLD`` consecutive idle cycles.
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

    def report_batch_result(self, batch_produced: bool) -> None:
        """Update overload detection counters after a batcher cycle.

        Parameters
        ----------
        batch_produced:
            True if the batcher returned a batch (non-None), False if idle (None).
        """
        if batch_produced:
            self._consecutive_batches += 1
            self._consecutive_idle = 0
            if self._consecutive_batches >= _ACTIVATION_THRESHOLD:
                self._level += 1
                self._consecutive_batches = 0
        else:
            self._consecutive_idle += 1
            self._consecutive_batches = 0
            if self._level > 0 and self._consecutive_idle >= _DEACTIVATION_THRESHOLD:
                self._level -= 1
                self._consecutive_idle = 0
                if self._level == 0:
                    self._subsample_counter = 0

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
