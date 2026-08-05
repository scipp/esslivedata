# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for a start command that no backend service consumes."""

import time

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.backend import DashboardBackend
from tests.integration.helpers import topic_high_watermark, wait_for_watermark_advance

COMMANDS_TOPIC = 'dummy_livedata_commands'
RESPONSES_TOPIC = 'dummy_livedata_responses'

# Window granted to a hypothetical responder. No service runs in this test, so
# any response would have to come from a stray consumer; a short window is
# enough to distinguish "nobody answered" from "answer still in flight".
RESPONSE_WINDOW = 3.0


@pytest.mark.integration
def test_start_command_is_published_but_goes_unanswered(
    dashboard_backend: DashboardBackend,
) -> None:
    """
    A start command reaches the commands topic even with no service running.

    Only the wire can show that the dashboard's start actually serialized into
    a durable Kafka record rather than staying dashboard-internal, and that
    the record went unanswered. What the dashboard then does with the silence
    -- expire the pending command after PENDING_COMMAND_TIMEOUT_SECONDS and
    raise one error notification -- is driven by an injected clock in the
    JobOrchestrator and PendingCommandTracker unit tests, so this test does
    not pay 30 s of wall clock to observe it.
    """
    backend = dashboard_backend
    workflow_id = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)

    commands_before = topic_high_watermark(COMMANDS_TOPIC)
    responses_before = topic_high_watermark(RESPONSES_TOPIC)

    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=workflow_id,
        source_names=['monitor1'],
        config=MonitorDataParams(),
    )

    wait_for_watermark_advance(COMMANDS_TOPIC, since=commands_before, timeout=30.0)

    deadline = time.time() + RESPONSE_WINDOW
    while time.time() < deadline:
        backend.update()
        time.sleep(0.2)

    assert topic_high_watermark(RESPONSES_TOPIC) == responses_before
    assert all(job_id not in backend.job_service.job_statuses for job_id in job_ids)
