# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for the data_reduction service's wire path.

``tests/services/data_reduction_test.py`` runs the same workflow in-process,
handing detector events straight to the service loop, so the science and the
service's own bookkeeping are already pinned there. What only the wire can
show is addressing: ``data_reduction`` and ``detector_data`` share both the
commands topic and the raw detector topic, so a start command addressed to a
reduction workflow has to reach the one service that hosts it, and that
service's results have to come back to the dashboard under the job the
dashboard just created.
"""

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.job import JobState
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import (
    NoParams,
    wait_for_backend_condition,
    wait_for_job_data,
)

WORKFLOW_ID = WorkflowId(instrument='dummy', name='total_counts', version=1)
SOURCE_NAME = 'panel_0'


@pytest.mark.integration
@pytest.mark.services('reduction')
def test_reduction_workflow_delivers_results_to_dashboard(
    integration_env: IntegrationEnv,
) -> None:
    """A reduction workflow started by the dashboard yields results for its job."""
    backend = integration_env.backend

    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=WORKFLOW_ID,
        source_names=[SOURCE_NAME],
        config=NoParams(),
    )
    assert len(job_ids) == 1
    job_id = job_ids[0]
    assert job_id.source_name == SOURCE_NAME

    def job_active() -> bool:
        status = backend.job_service.job_statuses.get(job_id)
        return status is not None and status.state == JobState.active

    # Status is keyed by the full JobId, so this is where the round trip is
    # tied back to the job_number the dashboard minted; the data plane's keys
    # carry only the source name.
    wait_for_backend_condition(backend, job_active, timeout=60.0)
    wait_for_job_data(backend, WORKFLOW_ID, job_ids, timeout=60.0)
