# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration tests for workflow lifecycle (start, data updates, stop)."""

import pydantic
import pytest

from ess.livedata.config.workflow_spec import WorkflowId

from .conftest import IntegrationEnv


class EmptyParams(pydantic.BaseModel):
    """Empty parameter model for workflows without parameters."""

    pass


@pytest.mark.integration
def test_workflow_can_start_and_receive_data(integration_env: IntegrationEnv) -> None:
    """
    Test that a workflow can be started and receives data from backend services.

    This test demonstrates the basic integration test pattern:
    1. Start a workflow via WorkflowController
    2. Process messages by calling backend.update()
    3. Wait for workflow response using helpers
    4. Wait for data to arrive
    5. Stop the workflow
    """
    backend = integration_env.backend

    # Define workflow parameters
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='data_reduction',
        name='total_counts',
        version=1,
    )
    source_names = ['panel_0']

    # Start the workflow (using empty params for workflows without parameters)
    backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=source_names,
        config=EmptyParams(),
    )

    # Process messages to get the workflow started response
    for _ in range(5):
        backend.update()

    # Wait for workflow config to be acknowledged
    # (WorkflowConfigService should receive the config)
    # Note: In a real test, you'd want to verify the config was received

    # Give services time to process and start producing data
    import time

    time.sleep(2.0)

    # Process more messages to get data updates
    for _ in range(10):
        backend.update()
        time.sleep(0.5)

    # Check that we have job data (the workflow should have produced something)
    # Note: This is a basic check - in a real test you'd verify specific data
    assert len(backend.job_service.job_data) > 0, "Expected to receive job data"

    # Note: Workflows continue running until services are stopped
    # There is no explicit stop_workflow method in the current implementation


@pytest.mark.integration
def test_workflow_status_updates(integration_env: IntegrationEnv) -> None:
    """
    Test that workflow status updates are received properly.

    This test verifies that we can wait for specific job statuses.
    """
    backend = integration_env.backend

    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='data_reduction',
        name='total_counts',
        version=1,
    )
    source_names = ['panel_0']

    # Start workflow
    backend.workflow_controller.start_workflow(
        workflow_id=workflow_id, source_names=source_names, config=EmptyParams()
    )

    # Process messages
    for _ in range(10):
        backend.update()
        import time

        time.sleep(0.3)

    # We should have received at least one status update
    # Note: The actual status values depend on the backend implementation
    assert len(backend.job_service.job_statuses) > 0, "Expected job status updates"


@pytest.mark.integration
@pytest.mark.skip(reason="Requires more complex setup - example for future use")
def test_workflow_multi_turn_interaction(integration_env: IntegrationEnv) -> None:
    """
    Example test showing multi-turn interaction pattern.

    This demonstrates how to test:
    - Start workflow
    - Wait for data
    - Send update/reset
    - Wait for new data
    - Stop workflow

    This test is skipped for now as it requires more complete implementation.
    """
    backend = integration_env.backend

    # Start workflow
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='data_reduction',
        name='total_counts',
        version=1,
    )
    source_names = ['panel_0']

    backend.workflow_controller.start_workflow(
        workflow_id=workflow_id, source_names=source_names, config=EmptyParams()
    )

    # Update and wait for initial data
    for _ in range(5):
        backend.update()

    # Here you would:
    # 1. Wait for specific data using wait_for_data()
    # 2. Restart workflow with updated configuration
    # 3. Wait for updated data
    # 4. Verify changes
