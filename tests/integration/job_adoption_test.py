# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for job adoption from heartbeat observation (ADR 0008)."""

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.job import JobState
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.backend import DashboardBackend
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import (
    topic_high_watermark,
    wait_for_backend_condition,
    wait_for_job_data,
)

COMMANDS_TOPIC = 'dummy_livedata_commands'


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_new_dashboard_adopts_running_job_without_restart(
    integration_env: IntegrationEnv,
) -> None:
    """
    A fresh dashboard adopts a running job from heartbeats (ADR 0008).

    Stopping the dashboard leaves the backend service running the job and
    publishing heartbeats. A new dashboard instance must adopt the observed
    job as the workflow's current generation: same job_number, ACTIVE status,
    and data flowing again -- all without sending any command (no spurious
    restart).
    """
    backend_a = integration_env.backend
    workflow_id = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)

    job_ids = backend_a.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=['monitor1'],
        config=MonitorDataParams(),
    )
    job_id = job_ids[0]

    def job_active_on(backend: DashboardBackend) -> bool:
        status = backend.job_service.job_statuses.get(job_id)
        return status is not None and status.state == JobState.active

    wait_for_backend_condition(
        backend_a, lambda: job_active_on(backend_a), timeout=30.0
    )
    wait_for_job_data(backend_a, workflow_id, job_ids, timeout=30.0)

    # Stop the dashboard. The service keeps running the job and heartbeating;
    # no stop command is sent on dashboard shutdown.
    backend_a.stop()
    commands_watermark = topic_high_watermark(COMMANDS_TOPIC)

    # Backends are single-use, so a dashboard restart is a fresh instance.
    # Its config store is empty (in-memory), so adoption can only come from
    # heartbeat observation, not from a persisted record.
    with DashboardBackend(instrument='dummy', dev=True) as backend_b:
        wait_for_backend_condition(
            backend_b,
            lambda: (
                backend_b.job_orchestrator.get_active_job_number(workflow_id)
                == job_id.job_number
            ),
            timeout=30.0,
        )
        wait_for_backend_condition(
            backend_b, lambda: job_active_on(backend_b), timeout=30.0
        )

        # Data is admitted again under the adopted generation.
        wait_for_job_data(backend_b, workflow_id, job_ids, timeout=30.0)

        # No spurious restart: the adopted job_number is unchanged and no
        # command (start or stop) was written to the commands topic.
        assert (
            backend_b.job_orchestrator.get_active_job_number(workflow_id)
            == job_id.job_number
        )
        assert topic_high_watermark(COMMANDS_TOPIC) == commands_watermark
