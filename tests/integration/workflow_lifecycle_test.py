# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for the start path: a started workflow produces data."""

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import wait_for_job_data


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_workflow_can_start_and_receive_data(integration_env: IntegrationEnv) -> None:
    """
    A workflow started by the dashboard yields results for the job it created.

    The wire content is the round trip: the start command reaches a real
    backend service, which runs the workflow on real ev44 data and publishes
    results the dashboard admits under the job it just created. Which outputs
    the workflow declares is a workflow-spec fact, pinned by the workflow and
    service unit tests, and deliberately not re-asserted here.
    """
    backend = integration_env.backend

    workflow_id = WorkflowId(
        instrument='dummy',
        name='monitor_histogram',
        version=1,
    )

    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=['monitor1'],
        config=MonitorDataParams(),
    )

    # We started one source, expect exactly one job
    assert len(job_ids) == 1, f"Expected 1 job, got {len(job_ids)}"
    assert job_ids[0].source_name == 'monitor1'

    wait_for_job_data(backend, workflow_id, job_ids, timeout=30.0)
