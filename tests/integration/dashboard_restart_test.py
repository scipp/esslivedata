# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Integration test for a dashboard restart against running workers.

A dashboard derives each workflow's current generation from the desired-state
topic, which it reads to its end when it starts. Its config store holds only
staged configs, so a restarted dashboard with an empty store must still know
the running job, with its params, and must not write anything to do so.
"""

import pytest

from ess.livedata.config.streams import stream_kind_to_topic
from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.message import StreamKind
from ess.livedata.kafka.desired_state import KafkaDesiredStateWriter
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.backend import DashboardBackend
from tests.integration.helpers import (
    topic_high_watermark,
    wait_for_job_data,
    wait_for_watermark_stall,
)
from tests.integration.service_process import ServiceGroup

COMMANDS_TOPIC = stream_kind_to_topic('dummy', StreamKind.LIVEDATA_COMMANDS)
DESIRED_TOPIC = stream_kind_to_topic('dummy', StreamKind.LIVEDATA_DESIRED)
DATA_TOPIC = stream_kind_to_topic('dummy', StreamKind.LIVEDATA_DATA)

WORKFLOW_ID = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)


@pytest.mark.integration
def test_restarted_dashboard_derives_current_generation_from_topic(
    desired_state: KafkaDesiredStateWriter, monitor_services: ServiceGroup
) -> None:
    """
    A restarted dashboard knows the running job from the topic alone.

    The first dashboard commits and shuts down; shutdown writes nothing, so
    the worker keeps running the job. A fresh dashboard with an empty
    in-memory config store has the job as its current generation as soon as
    it is constructed, labelled with the committed params, and admits the
    job's data. It writes no record and no command. A stop through it is read
    by the worker, and a dashboard started after the stop has no current
    generation.
    """
    with DashboardBackend(instrument='dummy', dev=True) as backend_a:
        job_ids = backend_a.workflow_controller.start_workflow(
            workflow_id=WORKFLOW_ID,
            source_names=['monitor1'],
            config=MonitorDataParams(),
        )
        job_id = job_ids[0]
        committed_config = backend_a.job_orchestrator.get_active_config(WORKFLOW_ID)
        wait_for_job_data(backend_a, WORKFLOW_ID, job_ids, timeout=30.0)

    desired_watermark = topic_high_watermark(DESIRED_TOPIC)
    commands_watermark = topic_high_watermark(COMMANDS_TOPIC)
    with DashboardBackend(instrument='dummy', dev=True) as backend_b:
        orchestrator = backend_b.job_orchestrator
        assert orchestrator.get_active_job_number(WORKFLOW_ID) == job_id.job_number
        assert orchestrator.get_active_config(WORKFLOW_ID) == committed_config
        wait_for_job_data(backend_b, WORKFLOW_ID, job_ids, timeout=30.0)
        assert topic_high_watermark(DESIRED_TOPIC) == desired_watermark
        assert topic_high_watermark(COMMANDS_TOPIC) == commands_watermark

        assert orchestrator.stop_workflow(WORKFLOW_ID)
        wait_for_watermark_stall(DATA_TOPIC)

    with DashboardBackend(instrument='dummy', dev=True) as backend_c:
        assert backend_c.job_orchestrator.get_active_job_number(WORKFLOW_ID) is None
