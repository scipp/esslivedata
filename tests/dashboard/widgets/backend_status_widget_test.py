# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the backend status widget."""

import time

from ess.livedata.core.job import ServiceState, ServiceStatus
from ess.livedata.dashboard.service_registry import ServiceRegistry
from ess.livedata.dashboard.widgets.backend_status_widget import BackendStatusWidget


def _make_status(
    *,
    worker_id: str = "worker1",
    namespace: str = "ns1",
    state: ServiceState = ServiceState.running,
) -> ServiceStatus:
    return ServiceStatus(
        instrument="dream",
        namespace=namespace,
        worker_id=worker_id,
        state=state,
        started_at=time.time_ns(),
        active_job_count=1,
        messages_processed=10,
    )


class TestBackendStatusWidgetClearButton:
    def test_clear_button_disabled_when_no_workers(self) -> None:
        registry = ServiceRegistry()
        widget = BackendStatusWidget(service_registry=registry)
        widget.refresh()
        assert widget._clear_button.disabled

    def test_clear_button_disabled_when_only_running_workers(self) -> None:
        registry = ServiceRegistry()
        registry.status_updated(_make_status(state=ServiceState.running))
        widget = BackendStatusWidget(service_registry=registry)
        widget.refresh()
        assert widget._clear_button.disabled

    def test_clear_button_enabled_when_stopped_worker_exists(self) -> None:
        registry = ServiceRegistry()
        registry.status_updated(_make_status(state=ServiceState.stopped))
        widget = BackendStatusWidget(service_registry=registry)
        widget.refresh()
        assert not widget._clear_button.disabled

    def test_clear_button_enabled_when_stale_worker_exists(self) -> None:
        registry = ServiceRegistry(heartbeat_timeout_ns=1_000_000)  # 1ms
        registry.status_updated(_make_status(state=ServiceState.running))
        time.sleep(0.01)

        widget = BackendStatusWidget(service_registry=registry)
        widget.refresh()
        assert not widget._clear_button.disabled

    def test_clicking_clear_removes_stopped_workers(self) -> None:
        registry = ServiceRegistry()
        registry.status_updated(
            _make_status(worker_id="running1", state=ServiceState.running)
        )
        registry.status_updated(
            _make_status(worker_id="stopped1", state=ServiceState.stopped)
        )
        widget = BackendStatusWidget(service_registry=registry)
        widget.refresh()

        widget._on_clear_stopped()
        widget.refresh()

        assert len(registry.worker_statuses) == 1
        remaining_key = next(iter(registry.worker_statuses))
        assert "running1" in remaining_key

    def test_clicking_clear_removes_stale_workers(self) -> None:
        registry = ServiceRegistry(heartbeat_timeout_ns=1_000_000)  # 1ms
        registry.status_updated(
            _make_status(worker_id="stale1", state=ServiceState.running)
        )
        time.sleep(0.01)

        widget = BackendStatusWidget(service_registry=registry)
        assert len(registry.worker_statuses) == 1

        widget._on_clear_stopped()
        widget.refresh()

        assert len(registry.worker_statuses) == 0

    def test_clear_button_becomes_disabled_after_clearing(self) -> None:
        registry = ServiceRegistry()
        registry.status_updated(_make_status(state=ServiceState.stopped))
        widget = BackendStatusWidget(service_registry=registry)
        widget.refresh()
        assert not widget._clear_button.disabled

        widget._on_clear_stopped()
        widget.refresh()
        assert widget._clear_button.disabled

    def test_rows_removed_after_clearing(self) -> None:
        registry = ServiceRegistry()
        registry.status_updated(_make_status(state=ServiceState.stopped))
        widget = BackendStatusWidget(service_registry=registry)
        widget.refresh()
        assert len(widget._worker_rows) == 1

        widget._on_clear_stopped()
        widget.refresh()
        assert len(widget._worker_rows) == 0
