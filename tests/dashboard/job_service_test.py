# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for JobService."""

from ess.livedata.config.workflow_spec import JobId, WorkflowId
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.dashboard.job_service import JobService


def make_status(source_name: str, job_number: int = 1) -> JobStatus:
    return JobStatus(
        job_id=JobId(source_name=source_name, job_number=job_number),
        workflow_id=WorkflowId(instrument='test', name='wf', version=1),
        state=JobState.active,
    )


def test_job_statuses_returns_snapshot_decoupled_from_updates() -> None:
    service = JobService()
    first = make_status('det_1')
    service.status_updated(first)

    snapshot = service.job_statuses
    service.status_updated(make_status('det_2'))

    assert snapshot == {first.job_id: first}
    assert len(service.job_statuses) == 2
