# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration tests for workflow lifecycle (start, data updates, stop)."""

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.handlers.monitor_workflow_specs import MonitorDataParams

from .conftest import IntegrationEnv
from .helpers import wait_for_condition


@pytest.mark.integration
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
    backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=source_names,
        config=MonitorDataParams(),
    )

    # Process messages continuously until we receive job data
    def check_for_job_data():
        backend.update()
        return len(backend.job_service.job_data) > 0

    wait_for_condition(check_for_job_data, timeout=10.0, poll_interval=0.5)

    # Verify that we received job data
    assert len(backend.job_service.job_data) > 0, "Expected to receive job data"


@pytest.mark.integration
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
    backend.workflow_controller.start_workflow(
        workflow_id=workflow_id, source_names=source_names, config=MonitorDataParams()
    )

    # Process messages and wait for status updates
    def check_for_status_updates():
        backend.update()
        return len(backend.job_service.job_statuses) > 0

    wait_for_condition(check_for_status_updates, timeout=10.0, poll_interval=0.5)

    # Verify we received status updates
    assert len(backend.job_service.job_statuses) > 0, "Expected job status updates"
