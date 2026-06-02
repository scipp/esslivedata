# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Self-capturing watchdog for dashboard CPU-spin freezes.

The dashboard has been observed to enter a state where a single thread pins a
CPU core indefinitely while the process emits no log output (the spinning
thread starves every other thread, so ordinary logging never runs). Because it
only recurs after days of uptime and leaves no trace, the stuck stack is hard
to obtain.

This watchdog captures it automatically. A daemon thread samples the process's
own CPU usage; when usage stays above a threshold for a sustained window it
dumps the tracebacks of all threads (via :mod:`faulthandler`) to ``stderr``
(captured by journald), pinpointing the spinning code.

Two detection paths cover both freeze modes:

* **Python-level spin** (e.g. an asyncio busy-loop on a half-open socket): the
  spinning thread releases the GIL periodically, so the daemon thread still
  runs and detects the sustained CPU via :func:`_process_cpu_seconds`.
* **C-level GIL hold** (a native call that never releases the GIL): the daemon
  thread itself is starved and cannot sample. A ``faulthandler`` timer armed in
  C is re-armed on every healthy sample; if the daemon stops re-arming, the
  timer fires and dumps from its own C thread.

Thresholds are tunable via environment variables so production behaviour can be
adjusted without a redeploy.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping

import structlog

logger = structlog.get_logger(__name__)


def _process_cpu_seconds() -> float:
    """Total CPU time (user+system, all threads) of this process, in seconds."""
    t = os.times()
    return t.user + t.system


class FreezeWatchdog:
    """Dump all-thread stacks when the process CPU stays pinned.

    Parameters
    ----------
    cpu_threshold_cores:
        Sustained CPU usage, in cores (1.0 == one full core), above which the
        process is considered spinning. The observed freeze sits at ~1.07.
    stall_seconds:
        How long CPU must stay above the threshold before dumping. Long enough
        that normal bursty load does not trigger it.
    sample_seconds:
        Interval between CPU samples.
    dump_interval_seconds:
        Minimum spacing between successive dumps while a freeze persists.
    max_dumps:
        Stop dumping after this many, to bound journal volume during a
        multi-hour freeze (the first few stacks are what matter).
    cpu_source:
        Callable returning cumulative process CPU seconds. Injectable for
        testing; defaults to :func:`_process_cpu_seconds`.
    metrics_source:
        Optional callable returning a mapping of diagnostic metrics. When
        provided, the metrics are logged every ``metrics_interval_seconds`` as
        ``dashboard_diagnostics``. Used to track slow-growing state (e.g. the
        HoloViews custom-options store) that drives the gradual lag onset, so
        the buildup is visible in the journal long before any hard freeze.
    metrics_interval_seconds:
        Spacing between diagnostic metric log lines.
    census_source:
        Optional callable returning a heavyweight leak census (a single pass
        over the live object graph). When provided it is logged every
        ``census_interval_seconds`` as ``dashboard_leak_census``. The census
        holds the GIL for the duration of its scan, so it runs at a much slower
        cadence than the lightweight metrics. Used to identify *what* retains
        the growing HoloViews custom-options store.
    census_interval_seconds:
        Spacing between leak census log lines.
    """

    def __init__(
        self,
        *,
        cpu_threshold_cores: float | None = None,
        stall_seconds: float | None = None,
        sample_seconds: float | None = None,
        dump_interval_seconds: float | None = None,
        max_dumps: int | None = None,
        cpu_source: Callable[[], float] = _process_cpu_seconds,
        metrics_source: Callable[[], Mapping[str, object]] | None = None,
        metrics_interval_seconds: float | None = None,
        census_source: Callable[[], Mapping[str, object]] | None = None,
        census_interval_seconds: float | None = None,
    ) -> None:
        env = os.environ.get
        self._threshold = cpu_threshold_cores or float(
            env('LIVEDATA_WATCHDOG_CPU_CORES', '0.85')
        )
        self._stall = stall_seconds or float(
            env('LIVEDATA_WATCHDOG_STALL_SECONDS', '180')
        )
        self._sample = sample_seconds or float(
            env('LIVEDATA_WATCHDOG_SAMPLE_SECONDS', '5')
        )
        self._dump_interval = dump_interval_seconds or float(
            env('LIVEDATA_WATCHDOG_DUMP_INTERVAL_SECONDS', '60')
        )
        self._max_dumps = max_dumps or int(env('LIVEDATA_WATCHDOG_MAX_DUMPS', '10'))
        self._cpu_source = cpu_source
        self._metrics_source = metrics_source
        self._metrics_interval = metrics_interval_seconds or float(
            env('LIVEDATA_WATCHDOG_METRICS_SECONDS', '60')
        )
        self._last_metrics = 0.0
        self._census_source = census_source
        self._census_interval = census_interval_seconds or float(
            env('LIVEDATA_WATCHDOG_CENSUS_SECONDS', '600')
        )
        self._last_census = 0.0
        # The C-timer backstop fires if the daemon thread is itself starved for
        # this long (a native GIL hold). Comfortably longer than the sample
        # interval so it never fires during healthy operation.
        self._backstop = max(self._stall, 10 * self._sample)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        # Arm the C-level backstop; the loop re-arms it on every healthy sample.
        faulthandler.dump_traceback_later(self._backstop)
        self._thread = threading.Thread(
            target=self._run, name='freeze-watchdog', daemon=True
        )
        self._thread.start()
        logger.info(
            "freeze_watchdog_started",
            cpu_threshold_cores=self._threshold,
            stall_seconds=self._stall,
        )

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        faulthandler.cancel_dump_traceback_later()
        self._thread.join(timeout=2 * self._sample)
        self._thread = None

    def _run(self) -> None:
        prev_cpu = self._cpu_source()
        prev_t = time.monotonic()
        high_since: float | None = None
        last_dump = 0.0
        dumps = 0
        while not self._stop.wait(self._sample):
            now = time.monotonic()
            cpu = self._cpu_source()
            cores = (cpu - prev_cpu) / (now - prev_t) if now > prev_t else 0.0
            prev_cpu, prev_t = cpu, now

            # Daemon thread is alive: push the C backstop out so it only fires
            # if we (the watchdog) get starved by a native GIL hold.
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(self._backstop)

            self._maybe_log_metrics(now, cores)
            self._maybe_log_census(now)

            if cores < self._threshold:
                high_since = None
                continue
            if high_since is None:
                high_since = now
            stalled = now - high_since
            if stalled < self._stall or now - last_dump < self._dump_interval:
                continue
            if dumps >= self._max_dumps:
                continue
            dumps += 1
            last_dump = now
            self._dump(cores=cores, stalled=stalled, dump=dumps)

    def _maybe_log_metrics(self, now: float, cores: float) -> None:
        """Emit injected diagnostic metrics at the configured cadence."""
        if self._metrics_source is None:
            return
        if now - self._last_metrics < self._metrics_interval:
            return
        self._last_metrics = now
        try:
            metrics = dict(self._metrics_source())
        except Exception:
            logger.exception("dashboard_diagnostics_failed")
            return
        logger.info("dashboard_diagnostics", cpu_cores=round(cores, 2), **metrics)

    def _maybe_log_census(self, now: float) -> None:
        """Emit the heavyweight leak census at the configured (slow) cadence."""
        if self._census_source is None:
            return
        if now - self._last_census < self._census_interval:
            return
        self._last_census = now
        try:
            census = dict(self._census_source())
        except Exception:
            logger.exception("dashboard_leak_census_failed")
            return
        logger.info("dashboard_leak_census", **census)

    def _dump(self, *, cores: float, stalled: float, dump: int) -> None:
        """Record the freeze and dump all thread stacks for diagnosis."""
        logger.warning(
            "freeze_watchdog_triggered",
            cpu_cores=round(cores, 2),
            stalled_seconds=round(stalled, 1),
            dump=dump,
            max_dumps=self._max_dumps,
        )
        # All-thread tracebacks -> stderr -> journald. Done after the log line
        # so the marker precedes the dump in the journal.
        faulthandler.dump_traceback(all_threads=True, file=sys.stderr)
