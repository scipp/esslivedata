# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Measurement of how much of the shared IOLoop is left for other sessions.

Every browser session on a dashboard process is served by one Tornado IOLoop.
Work done for one session -- materializing a plot grid, pushing a frame into
every visible layer -- runs on that loop, so while it runs no other session is
served. Nothing in the process reports this: Tornado's ``request_time`` starts
its clock when the request is *read*, and a loop that cannot read the socket has
not started it, so a blocked loop under-reports as a fast one.

This module measures the loop from the loop itself. A probe reschedules itself
at a fixed interval and compares when it actually ran against when it was due;
that lateness is time the loop spent unable to serve anyone.
"""

from __future__ import annotations

import asyncio

import structlog
from tornado.ioloop import IOLoop

logger = structlog.get_logger(__name__)

_PROBE_INTERVAL_S = 0.2
"""How often the probe runs. Bounds the resolution of a single block."""

_BLOCK_THRESHOLD_S = 1.0
"""Lateness reported on its own, rather than only in the periodic summary."""

_SUMMARY_INTERVAL_S = 60.0
"""How often the accumulated view of loop availability is logged."""


class LogThrottle:
    """Passes the first event, then at most one per cooldown.

    A loop kept saturated by one large grid crosses the reporting thresholds on
    every data frame, so an unthrottled warning would arrive once a second per
    session for as long as the grid stays open -- loudest exactly when the
    journal most needs to stay readable. Counting what is suppressed keeps the
    frequency in the record: the next event through carries how many it stands
    for, and the periodic summary carries the severity.
    """

    def __init__(self, cooldown: float = _SUMMARY_INTERVAL_S) -> None:
        self._cooldown = cooldown
        self._last: float | None = None
        self._suppressed = 0

    def take(self, now: float) -> int | None:
        """Report this event, or suppress it.

        Returns
        -------
        :
            How many events were suppressed since the last one reported, or
            ``None`` if this event is itself suppressed.
        """
        if self._last is not None and now - self._last < self._cooldown:
            self._suppressed += 1
            return None
        self._last = now
        suppressed, self._suppressed = self._suppressed, 0
        return suppressed


class LoopMonitor:
    """Reports how long the IOLoop is unavailable to serve sessions.

    Emits ``dashboard_loop_blocked`` at WARNING for any single block over
    ``block_threshold``, throttled to one per cooldown, and
    ``dashboard_loop_metrics`` at INFO once per summary interval. A loop that
    stays blocked therefore reads as one warning and a summary per minute
    rather than one warning per probe.

    ``unavailable_fraction`` is a lower bound: work that starts and finishes
    between two probes delays neither of them and is not seen. It measures harm
    to other sessions rather than CPU use -- a loop that is busy but always
    yields before the next probe is due is, correctly, not reported.
    """

    def __init__(
        self,
        *,
        interval: float = _PROBE_INTERVAL_S,
        block_threshold: float = _BLOCK_THRESHOLD_S,
        summary_interval: float = _SUMMARY_INTERVAL_S,
        warn_cooldown: float = _SUMMARY_INTERVAL_S,
    ) -> None:
        self._interval = interval
        self._block_threshold = block_threshold
        self._summary_interval = summary_interval
        self._blocked_throttle = LogThrottle(warn_cooldown)
        self._due = 0.0
        self._window_start = 0.0
        self._late_total = 0.0
        self._late_max = 0.0
        self._running = False

    def start(self) -> None:
        """Begin probing the loop that is current on the calling thread."""
        if self._running:
            return
        loop = IOLoop.current()
        self._running = True
        now = loop.time()
        self._window_start = now
        self._schedule(loop, now)

    def stop(self) -> None:
        """Stop probing once the pending probe has fired."""
        self._running = False

    def _schedule(self, loop: IOLoop, now: float) -> None:
        # Relative to now rather than to the missed deadline: after a long block
        # several probes would otherwise come due at once and report the same
        # block repeatedly, with decreasing lateness.
        self._due = now + self._interval
        loop.call_at(self._due, self._probe, loop)

    def _probe(self, loop: IOLoop) -> None:
        now = loop.time()
        late = max(0.0, now - self._due)
        self._late_total += late
        self._late_max = max(self._late_max, late)
        if late >= self._block_threshold:
            suppressed = self._blocked_throttle.take(now)
            if suppressed is not None:
                logger.warning(
                    'dashboard_loop_blocked',
                    blocked_seconds=round(late, 3),
                    suppressed=suppressed,
                )
        if now - self._window_start >= self._summary_interval:
            self._log_summary(now)
        if self._running:
            self._schedule(loop, now)

    def _log_summary(self, now: float) -> None:
        window = now - self._window_start
        logger.info(
            'dashboard_loop_metrics',
            unavailable_fraction=round(self._late_total / window, 3),
            unavailable_seconds=round(self._late_total, 3),
            max_block_seconds=round(self._late_max, 3),
            interval_seconds=round(window, 1),
        )
        self._window_start = now
        self._late_total = 0.0
        self._late_max = 0.0


_monitor: LoopMonitor | None = None


def start_loop_monitor() -> None:
    """Start the process-wide loop monitor, unless it is already running.

    Call from the loop that serves sessions. Idempotent, so the first session
    built starts it and later ones are no-ops. Does nothing when there is no
    running loop, which is how layouts get built outside a server -- in tests
    and screenshot runs -- where there is no shared loop to contend for.
    """
    global _monitor
    if _monitor is not None:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    _monitor = LoopMonitor()
    _monitor.start()
