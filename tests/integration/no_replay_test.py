# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for the no-replay contract of a starting service.

``tests/kafka/consumer_test.py`` pins ``assign_all_partitions`` against a fake
consumer: the assignment carries each partition's current high watermark. Only
a broker can show the consequence that follows from it, because it is a fact
about durable records the service never reads -- a worker coming up finds the
commands topic's history behind its assigned offsets and acts on none of it,
however long those commands have been sitting there.

The proof is ordering, not timing. Both commands go to the same
single-partition topic, the stale one first. Results for the later command
establish that the worker consumed past the earlier one's offset, so the stale
job's absence cannot be read as "not reached yet" -- the reading that would
make a timing-based version of this test vacuous.

Note that this is the mechanism behind the deliberate asymmetry documented on
``JobOrchestrator.reconcile_observed_jobs``: a start command is never
re-issued, so a start that a worker was down for is lost rather than delayed,
and recommit is the explicit recovery.
"""

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.conftest import IntegrationEnv
from tests.integration.helpers import (
    topic_high_watermark,
    wait_for_job_data,
    wait_for_watermark_advance,
)

COMMANDS_TOPIC = 'dummy_livedata_commands'

WORKFLOW_ID = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)


@pytest.mark.integration
@pytest.mark.services('monitor')
def test_restarted_service_ignores_commands_issued_while_down(
    integration_env: IntegrationEnv,
) -> None:
    """A restarted worker acts on commands issued after it came back, only.

    ``fake_monitors`` keeps producing throughout, so the raw monitor topic
    accumulates a backlog across the downtime as well; the restarted worker
    skips past that for the same reason, and the counts it goes on to publish
    are of live data only.
    """
    backend = integration_env.backend
    service = integration_env.services['monitor_data']

    service.stop()

    commands_before = topic_high_watermark(COMMANDS_TOPIC)
    stale_job_ids = backend.workflow_controller.start_workflow(
        workflow_id=WORKFLOW_ID,
        source_names=['monitor1'],
        config=MonitorDataParams(),
    )
    # The command is a durable record on the topic before the worker returns,
    # which is what makes its later absence a statement about offsets rather
    # than about a lost send.
    wait_for_watermark_advance(COMMANDS_TOPIC, since=commands_before, timeout=30.0)

    service.start()

    fresh_job_ids = backend.workflow_controller.start_workflow(
        workflow_id=WORKFLOW_ID,
        source_names=['monitor1'],
        config=MonitorDataParams(),
    )
    assert fresh_job_ids[0].job_number != stale_job_ids[0].job_number
    wait_for_job_data(backend, WORKFLOW_ID, fresh_job_ids, timeout=60.0)

    # The worker is demonstrably past the stale command's offset by now, and
    # never scheduled the job it asked for: no status was ever published for
    # it. Job numbers are unique per commit, so this cannot be confused with
    # the job the second command started.
    assert all(
        job_id not in backend.job_service.job_statuses for job_id in stale_job_ids
    )
