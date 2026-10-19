# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the ingestion loop monitor."""

from typing import Any

import pytest
from structlog.testing import capture_logs

from ess.livedata.dashboard.ingest_monitor import IngestMonitor


class FakeClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds


class FakeConsumerLag:
    def __init__(self) -> None:
        self.lag: int | None = 0

    def __call__(self) -> int | None:
        return self.lag


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def consumer_lag() -> FakeConsumerLag:
    return FakeConsumerLag()


@pytest.fixture
def monitor(clock: FakeClock, consumer_lag: FakeConsumerLag) -> IngestMonitor:
    return IngestMonitor(consumer_lag=consumer_lag, summary_interval=60.0, clock=clock)


def summaries(captured: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in captured if e['event'] == 'dashboard_ingest_metrics']


def test_no_summary_before_interval_elapses(
    monitor: IngestMonitor, clock: FakeClock
) -> None:
    with capture_logs() as captured:
        clock.seconds = 59.0
        monitor.record_pass(100, pump_seconds=0.1, flush_seconds=0.2)

    assert summaries(captured) == []


def test_summary_accumulates_passes_of_the_interval(
    monitor: IngestMonitor, clock: FakeClock, consumer_lag: FakeConsumerLag
) -> None:
    with capture_logs() as captured:
        clock.seconds = 10.0
        monitor.record_pass(2, pump_seconds=0.5, flush_seconds=1.0)
        clock.seconds = 20.0
        monitor.record_pass(0, pump_seconds=0.05, flush_seconds=0.0)
        consumer_lag.lag = 18855
        clock.seconds = 60.0
        monitor.record_pass(3, pump_seconds=0.25, flush_seconds=1.5)

    (summary,) = summaries(captured)
    assert summary['passes'] == 3
    assert summary['messages'] == 5
    assert summary['max_messages_per_pass'] == 3
    assert summary['pump_seconds'] == pytest.approx(0.8)
    assert summary['flush_seconds'] == pytest.approx(2.5)
    assert summary['consumer_lag'] == 18855
    assert summary['interval_seconds'] == 60.0


def test_each_summary_covers_only_its_own_interval(
    monitor: IngestMonitor, clock: FakeClock
) -> None:
    with capture_logs() as captured:
        clock.seconds = 60.0
        monitor.record_pass(100, pump_seconds=1.0, flush_seconds=1.0)
        clock.seconds = 120.0
        monitor.record_pass(3, pump_seconds=0.1, flush_seconds=0.2)

    first, second = summaries(captured)
    assert first['max_messages_per_pass'] == 100
    assert second['passes'] == 1
    assert second['messages'] == 3
    assert second['max_messages_per_pass'] == 3
    assert second['pump_seconds'] == pytest.approx(0.1)
    assert second['interval_seconds'] == 60.0


def test_unknown_consumer_lag_is_reported_as_none(
    monitor: IngestMonitor, clock: FakeClock, consumer_lag: FakeConsumerLag
) -> None:
    consumer_lag.lag = None
    with capture_logs() as captured:
        clock.seconds = 60.0
        monitor.record_pass(0, pump_seconds=0.05, flush_seconds=0.0)

    (summary,) = summaries(captured)
    assert summary['consumer_lag'] is None
