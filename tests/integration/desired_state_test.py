# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Integration tests for workers reconciling against the desired-state topic.

A worker reads the compacted desired-state topic from offset 0 when it starts,
so the jobs it runs follow the table across restarts. Only a broker can show
this, because it is a fact about records written before the worker's consumer
existed. The tests observe the worker through its heartbeats, not through the
dashboard.
"""

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.job import JobState, JobStatus, ServiceState, ServiceStatus
from ess.livedata.kafka.desired_state import KafkaDesiredStateWriter
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.helpers import (
    Status,
    TopicTail,
    commit_workflow,
    job_in_state,
)
from tests.integration.service_process import ServiceGroup

WORKFLOW_ID = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)
SOURCE_NAME = 'monitor1'


def running_monitor_worker(status: Status) -> bool:
    return (
        isinstance(status, ServiceStatus)
        and status.service_name == 'monitor_data'
        and status.state == ServiceState.running
    )


@pytest.mark.integration
def test_restarted_worker_runs_committed_job(
    desired_state: KafkaDesiredStateWriter,
    monitor_services: ServiceGroup,
    heartbeats: TopicTail[Status],
) -> None:
    """A job committed before a restart runs again after it, with no new commit.

    The final heartbeat of a gracefully stopped worker reports the job as
    ``stopped``; an ``active`` status after it can only come from the restarted
    worker, which knows the job from the topic alone.
    """
    service = monitor_services['monitor_data']
    job_id = commit_workflow(
        desired_state, WORKFLOW_ID, SOURCE_NAME, MonitorDataParams()
    )
    heartbeats.wait_for(job_in_state(job_id, JobState.active))

    service.stop()
    heartbeats.wait_for(job_in_state(job_id, JobState.stopped))

    service.start()
    heartbeats.wait_for(job_in_state(job_id, JobState.active), timeout=60.0)


@pytest.mark.integration
def test_tombstone_written_while_worker_is_down_is_honoured(
    desired_state: KafkaDesiredStateWriter,
    monitor_services: ServiceGroup,
    heartbeats: TopicTail[Status],
) -> None:
    """A workflow stopped while its worker is down does not come back with it.

    Every heartbeat of a worker lists all its jobs. The restarted worker reads
    the whole table and reconciles before its first heartbeat, so two of its
    heartbeats without the job prove that it never ran it.
    """
    service = monitor_services['monitor_data']
    job_id = commit_workflow(
        desired_state, WORKFLOW_ID, SOURCE_NAME, MonitorDataParams()
    )
    heartbeats.wait_for(job_in_state(job_id, JobState.active))

    service.stop()
    heartbeats.wait_for(job_in_state(job_id, JobState.stopped))
    after_stop = heartbeats.cursor
    desired_state.write(WORKFLOW_ID, None)

    service.start()
    for _ in range(2):
        heartbeats.wait_for(running_monitor_worker, timeout=60.0)

    assert not any(
        isinstance(s, JobStatus) and s.job_id == job_id
        for s in heartbeats.received[after_stop:]
    )
