# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.dashboard.active_job_registry import ActiveJobRegistry
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService

_workflow_id = WorkflowId(instrument="test", namespace="ns", name="wf", version=1)


def _make_result_key(job_number, source_name="det1", output_name="result"):
    return ResultKey(
        workflow_id=_workflow_id,
        job_id=JobId(source_name=source_name, job_number=job_number),
        output_name=output_name,
    )


def _make_registry():
    ds = DataService()
    js = JobService()
    return ActiveJobRegistry(data_service=ds, job_service=js), ds, js


class TestDeactivate:
    def test_removes_matching_data_service_entries(self):
        registry, ds, _ = _make_registry()
        job_number = uuid.uuid4()

        key_a = _make_result_key(job_number, source_name="det1")
        key_b = _make_result_key(job_number, source_name="det2")
        ds[key_a] = sc.scalar(1.0)
        ds[key_b] = sc.scalar(2.0)

        registry.restore(job_number)
        registry.deactivate(job_number)

        assert len(ds) == 0

    def test_removes_matching_job_service_entries(self):
        registry, _, js = _make_registry()
        job_number = uuid.uuid4()

        job_id = JobId(source_name="det1", job_number=job_number)
        js.status_updated(
            JobStatus(job_id=job_id, workflow_id=_workflow_id, state=JobState.active)
        )
        assert len(js.job_statuses) == 1

        registry.restore(job_number)
        registry.deactivate(job_number)

        assert len(js.job_statuses) == 0

    def test_leaves_other_job_numbers_untouched(self):
        registry, ds, js = _make_registry()
        keep_number = uuid.uuid4()
        remove_number = uuid.uuid4()

        # Data for both jobs
        keep_key = _make_result_key(keep_number)
        remove_key = _make_result_key(remove_number)
        ds[keep_key] = sc.scalar(1.0)
        ds[remove_key] = sc.scalar(2.0)

        # Status for both jobs
        for jn in (keep_number, remove_number):
            job_id = JobId(source_name="det1", job_number=jn)
            js.status_updated(
                JobStatus(
                    job_id=job_id, workflow_id=_workflow_id, state=JobState.active
                )
            )

        registry.restore(keep_number)
        registry.restore(remove_number)
        registry.deactivate(remove_number)

        assert keep_key in ds
        assert remove_key not in ds
        assert len(js.job_statuses) == 1
        assert registry.is_active(keep_number)
        assert not registry.is_active(remove_number)

    def test_unknown_job_number_is_noop(self):
        registry, ds, _js = _make_registry()
        existing = uuid.uuid4()

        key = _make_result_key(existing)
        ds[key] = sc.scalar(1.0)

        # Deactivate a job number that was never activated
        registry.deactivate(uuid.uuid4())

        assert key in ds
        assert len(ds) == 1
