# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Deterministic cyclic garbage collection for long-running service loops.

CPython triggers generation-2 collection on object *counts*, not bytes. The
per-chunk task-graph updates in sciline/cyclebane leave a handful of
self-referential ``networkx`` graphs behind for every processed chunk, each
transitively holding that chunk's detector arrays (O(100 MB) for large
detectors). Such graphs are reclaimable only by the cyclic collector, and in
steady state the generation-2 collector never runs because a few huge arrays
do not trip any count threshold. The result is unbounded growth of purely
reclaimable garbage until the service is OOM-killed.

This module makes collection deterministic instead of count-triggered:

- :meth:`PeriodicGarbageCollector.freeze` moves everything alive at loop start
  into the permanent generation, so a full collection only walks objects
  allocated since (measured: ~0.5 ms per collection instead of ~84 ms).
- :meth:`PeriodicGarbageCollector.maybe_collect` runs a full collection at a
  fixed time interval from the service loop, bounding retained garbage to
  roughly one interval's worth.

This is a mitigation, not a fix: the garbage should not be created in the
first place. See https://github.com/scipp/esslivedata/issues/1264 for the
root-cause analysis and the upstream work it depends on.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable

import structlog


class PeriodicGarbageCollector:
    """Runs full cyclic garbage collections on a fixed time interval.

    Collection cost scales with the number of tracked objects allocated after
    :meth:`freeze`, so it can drift upward as jobs accumulate; every
    ``log_interval`` seconds a summary of collection durations is logged so
    that drift is observed rather than assumed.

    Durations and reclaimed counts are logged together because their
    *combination* is what diagnoses. A full collection walks everything
    reachable, so duration tracks the live tracked population, while the
    reclaimed count tracks only the garbage. Rising duration at a flat
    reclaimed count therefore means live objects are accumulating -- a
    retained buffer rather than a cycle leak -- and no amount of collection
    will help. At roughly 0.045 ms per 1000 live tracked objects the logged
    duration converts back to an object count without a heap dump. Do not
    read a rising duration as the benign drift described above without
    checking the reclaimed count beside it: that pair is how the unbounded
    batcher backlog was found after this mitigation was already deployed.

    Parameters
    ----------
    interval:
        Minimum seconds between collections. Chosen to match the ~1 s batch
        cadence of the services: garbage from at most one batch cycle is
        retained, at a per-collection cost that is negligible against it.
    log_interval:
        Seconds between summary log lines reporting collection statistics.
    clock:
        Monotonic time source, injectable for testing.
    """

    def __init__(
        self,
        interval: float = 1.0,
        log_interval: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._interval = interval
        self._log_interval = log_interval
        self._clock = clock
        self._logger = structlog.get_logger()
        self._last_collect = clock()
        self._last_log = clock()
        self._count = 0
        self._collected = 0
        self._total_duration = 0.0
        self._max_duration = 0.0

    def freeze(self) -> None:
        """Move all currently live objects into the permanent generation.

        Call once, after service construction and before the processing loop.
        Frozen objects are exempt from collection forever, which is safe for
        process-lifetime service state but would silently leak anything
        shorter-lived; per-job state is allocated later and stays collectable.
        """
        gc.collect()
        gc.freeze()
        self._logger.info("gc_freeze", frozen_objects=gc.get_freeze_count())

    def maybe_collect(self) -> bool:
        """Run a full collection if ``interval`` has elapsed since the last one.

        Returns
        -------
        :
            True if a collection ran.
        """
        now = self._clock()
        if now - self._last_collect < self._interval:
            return False
        start = time.perf_counter()
        collected = gc.collect()
        duration = time.perf_counter() - start
        self._last_collect = now
        self._count += 1
        self._collected += collected
        self._total_duration += duration
        self._max_duration = max(self._max_duration, duration)
        if now - self._last_log >= self._log_interval:
            self._logger.info(
                "gc_collect_stats",
                collections=self._count,
                collected_objects=self._collected,
                mean_ms=round(1e3 * self._total_duration / self._count, 2),
                max_ms=round(1e3 * self._max_duration, 2),
            )
            self._last_log = now
            self._count = 0
            self._collected = 0
            self._total_duration = 0.0
            self._max_duration = 0.0
        return True
