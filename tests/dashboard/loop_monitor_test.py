# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the IOLoop availability monitor.

These run a real Tornado IOLoop and really block it, since what is under test
is how the monitor behaves when the loop it probes stops running.
"""

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import Any

from structlog.testing import capture_logs

from ess.livedata.core.log_throttle import LogThrottle
from ess.livedata.dashboard.loop_monitor import LoopMonitor

_INTERVAL = 0.02


def drive(
    monitor: LoopMonitor, body: Callable[[], Coroutine[Any, Any, None]]
) -> list[dict[str, Any]]:
    """Run ``body`` on a loop watched by ``monitor``, returning the log events."""

    async def main() -> None:
        # Inside the running loop, so the monitor watches the one under test.
        monitor.start()
        await body()
        monitor.stop()

    with capture_logs() as captured:
        asyncio.run(main())
    return captured


def events(captured: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [event for event in captured if event['event'] == name]


def test_block_is_reported_once_however_many_probes_it_spans() -> None:
    monitor = LoopMonitor(interval=_INTERVAL, block_threshold=0.1, summary_interval=1e9)

    async def body() -> None:
        await asyncio.sleep(_INTERVAL * 2)
        time.sleep(_INTERVAL * 15)
        await asyncio.sleep(_INTERVAL * 2)

    blocked = events(drive(monitor, body), 'dashboard_loop_blocked')

    # The probe reschedules from when it ran, so the ~15 deadlines the block
    # covered collapse into the one report a reader should see.
    assert len(blocked) == 1
    assert blocked[0]['blocked_seconds'] >= 0.1


def test_block_shorter_than_the_threshold_is_not_reported() -> None:
    monitor = LoopMonitor(interval=_INTERVAL, block_threshold=1.0, summary_interval=1e9)

    async def body() -> None:
        await asyncio.sleep(_INTERVAL * 2)
        time.sleep(_INTERVAL * 5)
        await asyncio.sleep(_INTERVAL * 2)

    assert events(drive(monitor, body), 'dashboard_loop_blocked') == []


def test_summary_reports_an_idle_loop_as_available() -> None:
    monitor = LoopMonitor(interval=_INTERVAL, block_threshold=1e9, summary_interval=0.1)

    async def body() -> None:
        await asyncio.sleep(0.3)

    metrics = events(drive(monitor, body), 'dashboard_loop_metrics')

    assert metrics
    assert metrics[0]['unavailable_fraction'] < 0.5
    assert metrics[0]['max_block_seconds'] < 0.1


def test_summary_accounts_for_a_block_that_went_unreported() -> None:
    monitor = LoopMonitor(interval=_INTERVAL, block_threshold=1e9, summary_interval=0.1)

    async def body() -> None:
        time.sleep(0.2)
        await asyncio.sleep(0.2)

    metrics = events(drive(monitor, body), 'dashboard_loop_metrics')

    assert metrics
    assert metrics[0]['max_block_seconds'] >= 0.1
    assert metrics[0]['unavailable_fraction'] > 0.1


def test_repeated_blocks_are_reported_once_per_cooldown() -> None:
    monitor = LoopMonitor(
        interval=_INTERVAL,
        block_threshold=0.1,
        summary_interval=1e9,
        warn_cooldown=1e9,
    )

    async def body() -> None:
        for _ in range(3):
            await asyncio.sleep(_INTERVAL * 2)
            time.sleep(0.15)
        await asyncio.sleep(_INTERVAL * 2)

    blocked = events(drive(monitor, body), 'dashboard_loop_blocked')

    assert len(blocked) == 1
    assert blocked[0]['suppressed'] == 0


def test_blocks_past_the_cooldown_are_all_reported() -> None:
    monitor = LoopMonitor(
        interval=_INTERVAL,
        block_threshold=0.1,
        summary_interval=1e9,
        warn_cooldown=0.0,
    )

    async def body() -> None:
        for _ in range(2):
            await asyncio.sleep(_INTERVAL * 2)
            time.sleep(0.15)
        await asyncio.sleep(_INTERVAL * 2)

    assert len(events(drive(monitor, body), 'dashboard_loop_blocked')) == 2


class TestLogThrottle:
    def test_first_event_is_reported(self) -> None:
        assert LogThrottle(cooldown=10.0).take(100.0) == 0

    def test_events_within_the_cooldown_are_suppressed(self) -> None:
        throttle = LogThrottle(cooldown=10.0)
        throttle.take(100.0)
        assert throttle.take(105.0) is None
        assert throttle.take(109.9) is None

    def test_report_after_the_cooldown_carries_the_suppressed_count(self) -> None:
        throttle = LogThrottle(cooldown=10.0)
        throttle.take(100.0)
        throttle.take(101.0)
        throttle.take(102.0)
        assert throttle.take(110.0) == 2

    def test_the_count_restarts_from_the_last_report(self) -> None:
        throttle = LogThrottle(cooldown=10.0)
        throttle.take(100.0)
        throttle.take(101.0)
        assert throttle.take(110.0) == 1
        assert throttle.take(120.0) == 0
