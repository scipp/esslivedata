# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for JobService."""

from uuid import uuid4

from ess.livedata.config.workflow_spec import JobId, JobNumber, WorkflowId
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.dashboard.job_service import JobService


def make_status(source_name: str, job_number: JobNumber | None = None) -> JobStatus:
    return JobStatus(
        job_id=JobId(source_name=source_name, job_number=job_number or uuid4()),
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


class TestVersion:
    """Counter widgets gate their refresh on.

    It must advance whenever the stored statuses changed, and stay put
    otherwise: a counter moving on every no-op call would make every widget
    re-render on every housekeeping tick.
    """

    def test_status_updated_advances_version(self) -> None:
        service = JobService()
        before = service.version

        service.status_updated(make_status('det_1'))

        assert service.version == before + 1

    def test_prune_stale_advances_version_when_something_was_pruned(self) -> None:
        # Negative timeout makes every status stale regardless of elapsed time.
        service = JobService(heartbeat_timeout_ns=-1)
        service.status_updated(make_status('det_1'))
        before = service.version

        service.prune_stale()

        assert service.job_statuses == {}
        assert service.version == before + 1

    def test_prune_stale_does_not_advance_version_when_nothing_stale(self) -> None:
        service = JobService()
        service.status_updated(make_status('det_1'))
        before = service.version

        service.prune_stale()

        assert len(service.job_statuses) == 1
        assert service.version == before

    def test_remove_jobs_by_number_advances_version_on_match(self) -> None:
        service = JobService()
        job_number = uuid4()
        service.status_updated(make_status('det_1', job_number=job_number))
        before = service.version

        service.remove_jobs_by_number(job_number)

        assert service.job_statuses == {}
        assert service.version == before + 1

    def test_remove_jobs_by_number_does_not_advance_version_without_match(self) -> None:
        service = JobService()
        service.status_updated(make_status('det_1'))
        before = service.version

        service.remove_jobs_by_number(uuid4())

        assert len(service.job_statuses) == 1
        assert service.version == before
