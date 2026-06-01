# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the dashboard freeze watchdog."""

from __future__ import annotations

import time

from ess.livedata.dashboard.freeze_watchdog import FreezeWatchdog


class RecordingWatchdog(FreezeWatchdog):
    """Watchdog that records dump events instead of dumping stacks."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dumps: list[float] = []

    def _dump(self, *, cores: float, stalled: float, dump: int) -> None:
        self.dumps.append(cores)


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)


def test_dumps_when_cpu_stays_pinned() -> None:
    # cpu_source == wall clock => exactly 1.0 cores of CPU per second of elapsed
    # time, i.e. a perfectly pinned core, independent of machine speed.
    wd = RecordingWatchdog(
        cpu_threshold_cores=0.5,
        stall_seconds=0.1,
        sample_seconds=0.02,
        dump_interval_seconds=0.0,
        max_dumps=3,
        cpu_source=time.monotonic,
    )
    wd.start()
    try:
        _wait_until(lambda: len(wd.dumps) >= 1)
    finally:
        wd.stop()
    assert wd.dumps  # the pinned core was detected
    assert all(c >= 0.5 for c in wd.dumps)
    # Dumping is bounded to avoid flooding the journal during a long freeze.
    assert len(wd.dumps) <= 3


def test_no_dump_while_idle() -> None:
    # Constant CPU reading => zero cores consumed => never triggers.
    wd = RecordingWatchdog(
        cpu_threshold_cores=0.5,
        stall_seconds=0.1,
        sample_seconds=0.02,
        dump_interval_seconds=0.0,
        cpu_source=lambda: 0.0,
    )
    wd.start()
    try:
        time.sleep(0.4)
    finally:
        wd.stop()
    assert wd.dumps == []


def test_brief_spike_below_stall_does_not_dump() -> None:
    # CPU pinned, but the stall window is longer than we observe -> no dump.
    wd = RecordingWatchdog(
        cpu_threshold_cores=0.5,
        stall_seconds=10.0,
        sample_seconds=0.02,
        dump_interval_seconds=0.0,
        cpu_source=time.monotonic,
    )
    wd.start()
    try:
        time.sleep(0.3)
    finally:
        wd.stop()
    assert wd.dumps == []


def test_stop_is_idempotent_and_joins() -> None:
    wd = RecordingWatchdog(cpu_source=lambda: 0.0)
    wd.start()
    wd.stop()
    wd.stop()  # second stop must not raise
