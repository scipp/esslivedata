# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for pending-command expiry when no backend responds."""

import time

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.job_orchestrator import PENDING_COMMAND_TIMEOUT_SECONDS
from ess.livedata.dashboard.notification_queue import NotificationType
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.backend import DashboardBackend
from tests.integration.helpers import wait_for_backend_condition


@pytest.mark.integration
def test_unacknowledged_start_command_expires(
    dashboard_backend: DashboardBackend,
) -> None:
    """
    A start command nobody consumes expires with an error notification.

    No services run in this test, so the start command written to the
    commands topic is never acknowledged. After
    PENDING_COMMAND_TIMEOUT_SECONDS the pending command must expire and
    surface as an error notification (the toast the UI would show); the
    workflow state itself is left untouched (no status, job stays pending).
    """
    backend = dashboard_backend
    workflow_id = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)

    sent_at = time.monotonic()
    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=['monitor1'],
        config=MonitorDataParams(),
    )

    def expiry_notified() -> bool:
        return any(
            event.notification_type == NotificationType.ERROR
            and 'no response from backend' in event.message
            for event in backend.notification_queue.get_all_events()
        )

    # The expiry cannot fire before the timeout, so this wait is dominated by
    # PENDING_COMMAND_TIMEOUT_SECONDS (30 s); the margin covers slow CI.
    wait_for_backend_condition(
        backend,
        expiry_notified,
        timeout=PENDING_COMMAND_TIMEOUT_SECONDS + 30.0,
        poll_interval=1.0,
    )
    elapsed = time.monotonic() - sent_at
    assert elapsed > PENDING_COMMAND_TIMEOUT_SECONDS - 1.0, (
        f"Expiry fired after {elapsed:.1f}s, before the "
        f"{PENDING_COMMAND_TIMEOUT_SECONDS:.0f}s timeout"
    )

    # No service consumed the command: the job never reported any status.
    job_statuses = backend.job_service.job_statuses
    assert all(job_id not in job_statuses for job_id in job_ids)
