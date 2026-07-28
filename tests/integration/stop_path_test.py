# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for the stop path: a commanded stop halts backend work."""

import time

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import (
    topic_high_watermark,
    wait_for_job_data,
    wait_for_watermark_advance,
    wait_for_watermark_stall,
)

# Gap between detecting the stall and reconfirming it, so the reconfirmation
# is not just re-reading the sample that triggered the stall detection.
STALL_RECHECK_GAP = 8.0

# fake_monitors keeps publishing raw ev44 data on this topic regardless of
# whether any workflow is running, so it must not be used for the stall
# assertions -- only the results topic (livedata_data) goes quiet on stop.
DATA_TOPIC = 'dummy_livedata_data'

WORKFLOW_ID = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_stop_workflow_halts_backend_publishing(
    integration_env: IntegrationEnv,
) -> None:
    """
    A commanded stop actually stops backend work, not just the dashboard status.

    JobOrchestrator.stop_workflow also deactivates the workflow dashboard-side,
    so DataService would filter out late results even if the backend kept
    churning -- a dashboard-only assertion (e.g. on job status) would pass
    vacuously whether or not the worker process actually stopped consuming
    and producing. The only assertion that proves the backend itself halted
    is on Kafka: the results topic's high watermark must stop advancing, and
    stay stopped, after the stop command is issued.

    Uses ``job_orchestrator.stop_workflow`` directly, the same call the
    dashboard's stop button makes (``WorkflowStatusWidget._on_stop_click``).
    """
    backend = integration_env.backend
    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=WORKFLOW_ID,
        source_names=['monitor1'],
        config=MonitorDataParams(),
    )
    wait_for_job_data(backend, WORKFLOW_ID, job_ids, timeout=30.0)

    # Prove the watermark is actually advancing before stopping -- otherwise
    # a later stall assertion would be vacuous (e.g. if the topic name were
    # wrong, or the pipeline had already stalled for an unrelated reason).
    first = topic_high_watermark(DATA_TOPIC)
    wait_for_watermark_advance(DATA_TOPIC, since=first, timeout=30.0)

    assert backend.job_orchestrator.stop_workflow(WORKFLOW_ID)

    # Watermark stalls once the worker consumes the stop command...
    stalled_watermark = wait_for_watermark_stall(DATA_TOPIC)
    # ...and stays stalled: a real gap separates detection from
    # reconfirmation, so this rules out a late trickle rather than
    # re-reading the sample that triggered the stall detection.
    time.sleep(STALL_RECHECK_GAP)
    assert topic_high_watermark(DATA_TOPIC) == stalled_watermark
