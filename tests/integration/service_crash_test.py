# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for a backend service crashing mid-run."""

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.job import JobState, ServiceState
from ess.livedata.dashboard.service_registry import SERVICE_HEARTBEAT_TIMEOUT_NS
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import (
    topic_high_watermark,
    wait_for_backend_condition,
    wait_for_job_data,
    wait_for_watermark_stall,
)

DATA_TOPIC = 'dummy_livedata_data'

WORKFLOW_ID = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_dashboard_flags_crashed_service_worker_stale(
    integration_env: IntegrationEnv,
) -> None:
    """
    A service killed mid-run surfaces as a stale worker in the dashboard.

    SIGKILL gives the worker no chance to publish its terminal heartbeat, so
    unlike a clean shutdown (state ``stopped``, never stale) the crash
    signature is a worker whose last state is still ``running`` while its
    heartbeat has aged past SERVICE_HEARTBEAT_TIMEOUT (30 s). The job-status
    tier has a longer staleness window (60 s), so at that point the dashboard
    still shows the job's last observed state — worker staleness is what
    flags the loss first. Results cease immediately: the data-topic watermark
    stalls and stays put through the staleness window.
    """
    backend = integration_env.backend
    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=WORKFLOW_ID,
        source_names=['monitor1'],
        config=MonitorDataParams(),
    )
    job_id = job_ids[0]
    wait_for_job_data(backend, WORKFLOW_ID, job_ids, timeout=30.0)

    def running_monitor_worker_keys() -> list[str]:
        return [
            key
            for key, status in backend.service_registry.worker_statuses.items()
            if status.service_name == 'monitor_data'
            and status.state == ServiceState.running
        ]

    wait_for_backend_condition(
        backend, lambda: len(running_monitor_worker_keys()) > 0, timeout=30.0
    )
    worker_key = running_monitor_worker_keys()[0]
    assert not backend.service_registry.is_status_stale(worker_key)

    # SIGKILL: no terminal heartbeat, no graceful teardown.
    integration_env.services['monitor_data'].process.kill()

    # Results cease: no further messages on the data topic.
    data_watermark = wait_for_watermark_stall(DATA_TOPIC)

    # Heartbeats stopped, so the worker's status ages into staleness.
    wait_for_backend_condition(
        backend,
        lambda: backend.service_registry.is_status_stale(worker_key),
        timeout=SERVICE_HEARTBEAT_TIMEOUT_NS / 1e9 + 20.0,
        poll_interval=1.0,
    )
    # Stale while still nominally running — the crash signature, as opposed
    # to a clean shutdown whose terminal heartbeat is never considered stale.
    assert (
        backend.service_registry.worker_statuses[worker_key].state
        == ServiceState.running
    )

    # The job-status tier still shows the last observed state: its 60 s
    # staleness window has not elapsed, so the loss is (intentionally) not
    # yet reflected there.
    status = backend.job_service.job_statuses.get(job_id)
    assert status is not None
    assert status.state == JobState.active
    assert not backend.job_service.is_status_stale(job_id)

    # Nothing was published during the staleness window either.
    assert topic_high_watermark(DATA_TOPIC) == data_watermark
