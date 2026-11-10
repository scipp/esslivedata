# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import uuid

import pydantic
import pytest
import scipp as sc
from structlog.testing import capture_logs

from ess.livedata.config.instrument import Instrument
from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    WorkflowId,
    WorkflowOutputsBase,
)
from ess.livedata.core.desired_state import DesiredJob, DesiredTable, DesiredWorkflow
from ess.livedata.core.job import JobState
from ess.livedata.core.job_manager import JobFactory, JobManager
from ess.livedata.core.job_reconciler import JobReconciler

from .job_test import FakeProcessor

SERVICE = 'timeseries'
OTHER_INSTRUMENT = WorkflowId(instrument='dream', name='owned', version=1)
UNREGISTERED = WorkflowId(instrument='tbl', name='gone', version=1)


class _Outputs(WorkflowOutputsBase):
    result: sc.DataArray = pydantic.Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0.0)), title='Result'
    )


class Setup:
    """A worker of service ``SERVICE`` with a real job manager and factory."""

    def __init__(self) -> None:
        instrument = Instrument(name='tbl')
        self.owned = self._register(
            instrument, SERVICE, 'owned', lambda: FakeProcessor()
        )
        self.other_service = self._register(
            instrument, 'detector_data', 'other', lambda: FakeProcessor()
        )
        self.broken_builds = 0

        def broken() -> FakeProcessor:
            self.broken_builds += 1
            raise RuntimeError('build failed')

        self.broken = self._register(instrument, SERVICE, 'broken', broken)
        job_factory = JobFactory(instrument=instrument, service_name=SERVICE)
        self.manager = JobManager(job_factory=job_factory, context_reader=dict)
        self.reconciler = JobReconciler(
            job_manager=self.manager, job_factory=job_factory
        )

    @staticmethod
    def _register(
        instrument: Instrument, service: str, name: str, factory
    ) -> WorkflowId:
        handle = instrument.register_spec(
            service=service,
            name=name,
            version=1,
            title=name,
            source_names=['a', 'b'],
            outputs=_Outputs,
        )
        handle.attach_factory()(factory)
        return handle.workflow_id

    @property
    def job_ids(self) -> set[JobId]:
        return {job.job_id for job in self.manager.all_jobs}

    def reconcile_fully(self, table: DesiredTable) -> None:
        while self.reconciler.reconcile(table):
            pass


@pytest.fixture
def setup() -> Setup:
    return Setup()


def desired(
    workflow_id: WorkflowId,
    *sources: str,
    job_number: JobNumber | None = None,
    params: dict | None = None,
) -> DesiredWorkflow:
    return DesiredWorkflow(
        workflow_id=workflow_id,
        job_number=job_number or uuid.uuid4(),
        jobs={
            source: DesiredJob(params=params or {}, aux_source_names={})
            for source in sources
        },
    )


def table(*records: DesiredWorkflow) -> DesiredTable:
    return {record.workflow_id: record for record in records}


def job_ids(record: DesiredWorkflow) -> set[JobId]:
    return {
        JobId(source_name=source, job_number=record.job_number)
        for source in record.jobs
    }


def test_builds_wanted_job(setup: Setup) -> None:
    record = desired(setup.owned, 'a')
    assert not setup.reconciler.reconcile(table(record))
    assert setup.job_ids == job_ids(record)
    assert setup.reconciler.rejected == []


def test_builds_at_most_one_job_per_call(setup: Setup) -> None:
    record = desired(setup.owned, 'a', 'b')
    assert setup.reconciler.reconcile(table(record))
    assert len(setup.job_ids) == 1
    assert not setup.reconciler.reconcile(table(record))
    assert setup.job_ids == job_ids(record)


def test_reconciling_twice_changes_nothing(setup: Setup) -> None:
    current = table(desired(setup.owned, 'a', 'b'))
    setup.reconcile_fully(current)
    jobs = setup.manager.all_jobs
    assert not setup.reconciler.reconcile(current)
    assert setup.manager.all_jobs == jobs


def test_tombstone_stops_jobs(setup: Setup) -> None:
    setup.reconcile_fully(table(desired(setup.owned, 'a', 'b')))
    assert not setup.reconciler.reconcile({})
    assert setup.job_ids == set()


def test_new_job_number_supersedes_previous_jobs(setup: Setup) -> None:
    setup.reconcile_fully(table(desired(setup.owned, 'a')))
    record = desired(setup.owned, 'a')
    setup.reconciler.reconcile(table(record))
    assert setup.job_ids == job_ids(record)


def test_source_dropped_from_workflow_stops_only_its_job(setup: Setup) -> None:
    record = desired(setup.owned, 'a', 'b')
    setup.reconcile_fully(table(record))
    kept = next(job for job in setup.manager.all_jobs if job.job_id.source_name == 'a')
    shrunk = desired(setup.owned, 'a', job_number=record.job_number)
    setup.reconciler.reconcile(table(shrunk))
    assert setup.manager.all_jobs == [kept]


@pytest.mark.parametrize('owner', ['other_service', 'other_instrument'])
def test_record_owned_by_another_worker_is_ignored(setup: Setup, owner: str) -> None:
    workflow_id = setup.other_service if owner == 'other_service' else OTHER_INSTRUMENT
    with capture_logs() as logs:
        assert not setup.reconciler.reconcile(table(desired(workflow_id, 'a')))
    assert setup.job_ids == set()
    assert setup.reconciler.rejected == []
    assert logs == []


def test_unregistered_workflow_is_logged_once_and_ignored(setup: Setup) -> None:
    current = table(desired(UNREGISTERED, 'a'))
    with capture_logs() as logs:
        assert not setup.reconciler.reconcile(current)
        setup.reconciler.reconcile(current)
    assert [log['event'] for log in logs] == ['desired_workflow_unregistered']
    assert setup.job_ids == set()
    assert setup.reconciler.rejected == []


def test_unregistered_workflow_is_logged_again_for_a_new_record(setup: Setup) -> None:
    with capture_logs() as logs:
        setup.reconciler.reconcile(table(desired(UNREGISTERED, 'a')))
        setup.reconciler.reconcile(table(desired(UNREGISTERED, 'a')))
    assert len(logs) == 2


def test_build_failure_is_reported_as_error_status(setup: Setup) -> None:
    record = desired(setup.broken, 'a')
    setup.reconciler.reconcile(table(record))
    assert setup.job_ids == set()
    [status] = setup.reconciler.rejected
    assert status.job_id in job_ids(record)
    assert status.workflow_id == setup.broken
    assert status.state == JobState.error
    assert status.error_message == 'build failed'


def test_invalid_params_are_reported_as_error_status(setup: Setup) -> None:
    setup.reconciler.reconcile(table(desired(setup.owned, 'a', params={'x': 1})))
    [status] = setup.reconciler.rejected
    assert 'x' in status.error_message


def test_rejected_job_is_not_retried(setup: Setup) -> None:
    current = table(desired(setup.broken, 'a'))
    setup.reconciler.reconcile(current)
    assert not setup.reconciler.reconcile(current)
    assert setup.broken_builds == 1
    assert len(setup.reconciler.rejected) == 1


def test_rejected_job_is_retried_after_record_change(setup: Setup) -> None:
    setup.reconciler.reconcile(table(desired(setup.broken, 'a')))
    record = desired(setup.broken, 'a')
    setup.reconciler.reconcile(table(record))
    assert setup.broken_builds == 2
    assert [status.job_id for status in setup.reconciler.rejected] == list(
        job_ids(record)
    )


def test_rejected_job_is_pruned_when_record_is_tombstoned(setup: Setup) -> None:
    setup.reconciler.reconcile(table(desired(setup.broken, 'a')))
    setup.reconciler.reconcile({})
    assert setup.reconciler.rejected == []


def test_rejected_job_does_not_block_other_jobs(setup: Setup) -> None:
    good = desired(setup.owned, 'a')
    setup.reconcile_fully(table(desired(setup.broken, 'a'), good))
    assert setup.job_ids == job_ids(good)
    assert len(setup.reconciler.rejected) == 1
