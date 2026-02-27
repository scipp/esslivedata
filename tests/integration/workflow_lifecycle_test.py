# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration tests for workflow lifecycle (start, data updates, stop)."""

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.handlers.monitor_workflow_specs import MonitorDataParams
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import (
    wait_for_job_data,
    wait_for_job_statuses,
)


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_workflow_can_start_and_receive_data(integration_env: IntegrationEnv) -> None:
    """
    Test that a workflow can be started and receives data from backend services.

    This test demonstrates the basic integration test pattern:
    1. Start a workflow via WorkflowController
    2. Process messages by calling backend.update()
    3. Wait for workflow response using helpers
    4. Wait for data to arrive
    """
    backend = integration_env.backend

    # Define workflow parameters for monitor histogram workflow
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )
    source_names = ['monitor1']

    # Start the workflow with default monitor parameters
    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=source_names,
        config=MonitorDataParams(),
    )

    # We started one source, expect exactly one job
    assert len(job_ids) == 1, f"Expected 1 job, got {len(job_ids)}"
    job_id = job_ids[0]
    assert job_id.source_name == 'monitor1'

    # Wait for job data to arrive for the specific jobs we created
    all_job_data = wait_for_job_data(backend, job_ids, timeout=10.0)

    # Verify that we received data with expected outputs
    job_data = all_job_data[job_id]
    output_names = {key.output_name for key in job_data}
    assert 'cumulative' in output_names, (
        f"Expected 'cumulative' output, got outputs: {output_names}"
    )


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_workflow_status_updates(integration_env: IntegrationEnv) -> None:
    """
    Test that workflow status updates are received properly.

    This test verifies that we can wait for specific job statuses and retrieve
    job data as expected.
    """
    backend = integration_env.backend

    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )
    source_names = ['monitor1']

    # Start workflow
    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id, source_names=source_names, config=MonitorDataParams()
    )

    # We started one source, expect exactly one job
    assert len(job_ids) == 1, f"Expected 1 job, got {len(job_ids)}"
    job_id = job_ids[0]

    # Wait for job status updates for the specific jobs we created
    all_statuses = wait_for_job_statuses(backend, job_ids, timeout=10.0)

    # Verify we received status updates for the specific job
    assert job_id in all_statuses, (
        f"Expected status update for job {job_id}, "
        f"got statuses for: {list(all_statuses.keys())}"
    )
