# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the ServiceRegistry dashboard component."""

import time

from ess.livedata.core.job import ServiceState, ServiceStatus
from ess.livedata.dashboard.service_registry import ServiceRegistry, make_worker_key


class TestMakeWorkerKey:
    def test_creates_key_from_status(self) -> None:
        status = ServiceStatus(
            instrument="dream",
            namespace="test_namespace",
            worker_id="abc123",
            state=ServiceState.running,
            started_at=1000,
            active_job_count=2,
            messages_processed=100,
        )
        key = make_worker_key(status)
        assert key == "dream:test_namespace:abc123"


class TestServiceRegistry:
    def test_status_updated_stores_status(self) -> None:
        registry = ServiceRegistry()
        status = ServiceStatus(
            instrument="dream",
            namespace="test_namespace",
            worker_id="abc123",
            state=ServiceState.running,
            started_at=1000,
            active_job_count=2,
            messages_processed=100,
        )

        registry.status_updated(status)

        worker_key = make_worker_key(status)
        assert worker_key in registry.worker_statuses
        assert registry.worker_statuses[worker_key] == status

    def test_multiple_workers_tracked_separately(self) -> None:
        registry = ServiceRegistry()
        status1 = ServiceStatus(
            instrument="dream",
            namespace="ns1",
            worker_id="worker1",
            state=ServiceState.running,
            started_at=1000,
            active_job_count=2,
            messages_processed=100,
        )
        status2 = ServiceStatus(
            instrument="dream",
            namespace="ns2",
            worker_id="worker2",
            state=ServiceState.starting,
            started_at=2000,
            active_job_count=0,
            messages_processed=0,
        )

        registry.status_updated(status1)
        registry.status_updated(status2)

        assert len(registry.worker_statuses) == 2
        assert registry.worker_statuses[make_worker_key(status1)] == status1
        assert registry.worker_statuses[make_worker_key(status2)] == status2

    def test_status_updated_replaces_existing(self) -> None:
        registry = ServiceRegistry()
        status_old = ServiceStatus(
            instrument="dream",
            namespace="ns1",
            worker_id="worker1",
            state=ServiceState.running,
            started_at=1000,
            active_job_count=2,
            messages_processed=100,
        )
        status_new = ServiceStatus(
            instrument="dream",
            namespace="ns1",
            worker_id="worker1",
            state=ServiceState.running,
            started_at=1000,
            active_job_count=3,
            messages_processed=200,
        )

        registry.status_updated(status_old)
        registry.status_updated(status_new)

        assert len(registry.worker_statuses) == 1
        stored = registry.worker_statuses[make_worker_key(status_new)]
        assert stored.messages_processed == 200
        assert stored.active_job_count == 3

    def test_is_status_stale_returns_false_for_recent_status(self) -> None:
        registry = ServiceRegistry()
        status = ServiceStatus(
            instrument="dream",
            namespace="ns1",
            worker_id="worker1",
            state=ServiceState.running,
            started_at=1000,
            active_job_count=2,
            messages_processed=100,
        )

        registry.status_updated(status)

        worker_key = make_worker_key(status)
        assert not registry.is_status_stale(worker_key)

    def test_is_status_stale_returns_true_for_expired_status(self) -> None:
        # Use a very short timeout for testing
        registry = ServiceRegistry(heartbeat_timeout_ns=1_000_000)  # 1 ms
        status = ServiceStatus(
            instrument="dream",
            namespace="ns1",
            worker_id="worker1",
            state=ServiceState.running,
            started_at=1000,
            active_job_count=2,
            messages_processed=100,
        )

        registry.status_updated(status)

        # Wait for the timeout to expire
        time.sleep(0.01)  # 10ms should be enough

        worker_key = make_worker_key(status)
        assert registry.is_status_stale(worker_key)

    def test_is_status_stale_returns_false_for_stopped_worker(self) -> None:
        """Stopped workers should never be considered stale."""
        # Use a very short timeout for testing
        registry = ServiceRegistry(heartbeat_timeout_ns=1_000_000)  # 1 ms
        status = ServiceStatus(
            instrument="dream",
            namespace="ns1",
            worker_id="worker1",
            state=ServiceState.stopped,  # Terminal state
            started_at=1000,
            active_job_count=0,
            messages_processed=100,
        )

        registry.status_updated(status)

        # Wait for the timeout to expire
        time.sleep(0.01)  # 10ms should be enough

        worker_key = make_worker_key(status)
        # Stopped workers are never stale - they shut down intentionally
        assert not registry.is_status_stale(worker_key)

    def test_get_stale_workers_returns_only_stale(self) -> None:
        # Use a very short timeout for testing
        registry = ServiceRegistry(heartbeat_timeout_ns=1_000_000)  # 1 ms

        # Add a worker that will become stale
        status_stale = ServiceStatus(
            instrument="dream",
            namespace="ns1",
            worker_id="stale_worker",
            state=ServiceState.running,
            started_at=1000,
            active_job_count=2,
            messages_processed=100,
        )
        registry.status_updated(status_stale)

        # Wait for timeout
        time.sleep(0.01)

        # Add a fresh worker
        status_fresh = ServiceStatus(
            instrument="dream",
            namespace="ns2",
            worker_id="fresh_worker",
            state=ServiceState.running,
            started_at=2000,
            active_job_count=1,
            messages_processed=50,
        )
        registry.status_updated(status_fresh)

        stale = registry.get_stale_workers()
        assert len(stale) == 1
        assert make_worker_key(status_stale) in stale

    def test_get_worker_uptime_seconds(self) -> None:
        registry = ServiceRegistry()
        # started_at is in nanoseconds
        started_at_ns = time.time_ns() - 60_000_000_000  # 60 seconds ago
        status = ServiceStatus(
            instrument="dream",
            namespace="ns1",
            worker_id="worker1",
            state=ServiceState.running,
            started_at=started_at_ns,
            active_job_count=2,
            messages_processed=100,
        )

        registry.status_updated(status)

        worker_key = make_worker_key(status)
        uptime = registry.get_worker_uptime_seconds(worker_key)
        assert uptime is not None
        assert 59 <= uptime <= 61  # Allow for timing variance

    def test_get_worker_uptime_seconds_returns_none_for_unknown(self) -> None:
        registry = ServiceRegistry()
        uptime = registry.get_worker_uptime_seconds("unknown:worker:key")
        assert uptime is None

    def test_get_last_seen_seconds_ago_returns_recent_value(self) -> None:
        registry = ServiceRegistry()
        status = ServiceStatus(
            instrument="dream",
            namespace="ns1",
            worker_id="worker1",
            state=ServiceState.running,
            started_at=1000,
            active_job_count=2,
            messages_processed=100,
        )

        registry.status_updated(status)

        worker_key = make_worker_key(status)
        last_seen = registry.get_last_seen_seconds_ago(worker_key)
        assert last_seen is not None
        # Should be very recent (less than 1 second)
        assert last_seen < 1.0

    def test_get_last_seen_seconds_ago_returns_none_for_unknown(self) -> None:
        registry = ServiceRegistry()
        last_seen = registry.get_last_seen_seconds_ago("unknown:worker:key")
        assert last_seen is None

    def test_get_last_seen_seconds_ago_increases_over_time(self) -> None:
        registry = ServiceRegistry()
        status = ServiceStatus(
            instrument="dream",
            namespace="ns1",
            worker_id="worker1",
            state=ServiceState.running,
            started_at=1000,
            active_job_count=2,
            messages_processed=100,
        )

        registry.status_updated(status)
        worker_key = make_worker_key(status)

        first_check = registry.get_last_seen_seconds_ago(worker_key)
        time.sleep(0.05)  # 50ms
        second_check = registry.get_last_seen_seconds_ago(worker_key)

        assert first_check is not None
        assert second_check is not None
        assert second_check > first_check
