# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for deferred run transition (start/stop) resets in JobManager."""

import uuid

import pytest

from ess.livedata.config.workflow_spec import (
    JobSchedule,
    WorkflowConfig,
    WorkflowId,
)
from ess.livedata.core.job import Job, JobId, JobState
from ess.livedata.core.job_manager import JobManager, WorkflowData
from ess.livedata.core.message import RunStart, RunStop

from .job_manager_test import FakeJobFactory
from .job_test import FakeProcessor


def _workflow_id(name: str = 'test') -> WorkflowId:
    return WorkflowId(instrument="test", name=name, version=1)


def _make_config(name: str = 'test') -> WorkflowConfig:
    return WorkflowConfig(identifier=_workflow_id(name))


def _activate_jobs(manager: JobManager) -> None:
    """Push minimal data to activate scheduled jobs."""
    manager.push_data(WorkflowData(start_time=0, end_time=1, data={}))


def _push_data_to(manager: JobManager, end_time: int) -> None:
    """Push data with given end_time to trigger time advancement."""
    manager.push_data(WorkflowData(start_time=0, end_time=end_time, data={}))


class TestDeferredRunTransitionReset:
    def test_reset_does_not_fire_before_scheduled_time(self, fake_job_factory):
        manager = JobManager(job_factory=fake_job_factory)
        job_id = manager.schedule_job('det1', _make_config())
        _activate_jobs(manager)
        processor = fake_job_factory.processors[job_id]

        manager.on_run_start(RunStart(run_name='run_1', start_time=1000))
        # Push data that doesn't reach the scheduled time
        _push_data_to(manager, end_time=999)
        assert processor.clear_calls == 0

    def test_reset_fires_when_data_reaches_scheduled_time(self, fake_job_factory):
        manager = JobManager(job_factory=fake_job_factory)
        job_id = manager.schedule_job('det1', _make_config())
        _activate_jobs(manager)
        processor = fake_job_factory.processors[job_id]

        manager.on_run_start(RunStart(run_name='run_1', start_time=1000))
        _push_data_to(manager, end_time=1000)
        assert processor.clear_calls == 1

    def test_reset_fires_on_run_stop(self, fake_job_factory):
        manager = JobManager(job_factory=fake_job_factory)
        job_id = manager.schedule_job('det1', _make_config())
        _activate_jobs(manager)
        processor = fake_job_factory.processors[job_id]

        manager.on_run_stop(RunStop(run_name='run_1', stop_time=500))
        _push_data_to(manager, end_time=500)
        assert processor.clear_calls == 1

    def test_past_reset_time_fires_on_next_data(self, fake_job_factory):
        manager = JobManager(job_factory=fake_job_factory)
        job_id = manager.schedule_job('det1', _make_config())
        _activate_jobs(manager)
        processor = fake_job_factory.processors[job_id]

        manager.on_run_start(RunStart(run_name='run_1', start_time=100))
        _push_data_to(manager, end_time=500)
        assert processor.clear_calls == 1

    def test_reset_time_in_past_relative_to_data_already_seen(self, fake_job_factory):
        """If data has advanced to T=5000 and a RunStart arrives with T=3000,
        the reset fires on the next data push regardless of its end_time."""
        manager = JobManager(job_factory=fake_job_factory)
        job_id = manager.schedule_job('det1', _make_config())
        _activate_jobs(manager)
        processor = fake_job_factory.processors[job_id]

        _push_data_to(manager, end_time=5000)
        assert processor.clear_calls == 0

        manager.on_run_start(RunStart(run_name='run_1', start_time=3000))
        # Next data push has end_time=5001, but reset time 3000 is already past
        _push_data_to(manager, end_time=5001)
        assert processor.clear_calls == 1

    def test_run_start_with_stop_time_schedules_two_resets(self, fake_job_factory):
        manager = JobManager(job_factory=fake_job_factory)
        job_id = manager.schedule_job('det1', _make_config())
        _activate_jobs(manager)
        processor = fake_job_factory.processors[job_id]

        manager.on_run_start(RunStart(run_name='run_1', start_time=100, stop_time=500))
        # First reset at start_time
        _push_data_to(manager, end_time=100)
        assert processor.clear_calls == 1

        # Second reset at stop_time
        _push_data_to(manager, end_time=500)
        assert processor.clear_calls == 2

    def test_multiple_pending_resets_collapse_within_batch(self, fake_job_factory):
        manager = JobManager(job_factory=fake_job_factory)
        job_id = manager.schedule_job('det1', _make_config())
        _activate_jobs(manager)
        processor = fake_job_factory.processors[job_id]

        manager.on_run_start(RunStart(run_name='run_1', start_time=100))
        manager.on_run_stop(RunStop(run_name='run_1', stop_time=200))
        # Both fire in one batch → single _reset_eligible_jobs call
        _push_data_to(manager, end_time=300)
        assert processor.clear_calls == 1

    def test_skips_jobs_with_flag_disabled(self):
        """Jobs with reset_on_run_transition=False are not reset."""
        processor = FakeProcessor()
        job = Job(
            job_id=JobId(source_name='log1', job_number=uuid.uuid4()),
            workflow_id=_workflow_id('timeseries'),
            processor=processor,
            source_names=['log1'],
            reset_on_run_transition=False,
        )
        factory = FakeJobFactory()
        manager = JobManager(job_factory=factory)
        manager._active_jobs[job.job_id] = job
        manager._job_states[job.job_id] = JobState.active
        manager._job_schedules[job.job_id] = JobSchedule()

        manager.on_run_start(RunStart(run_name='run_1', start_time=100))
        _push_data_to(manager, end_time=200)
        assert processor.clear_calls == 0

    def test_mixed_jobs_selective_reset(self):
        """Mixed jobs: only those with reset_on_run_transition=True are reset."""
        factory = FakeJobFactory()
        manager = JobManager(job_factory=factory)

        proc_reset = FakeProcessor()
        proc_keep = FakeProcessor()

        job_reset = Job(
            job_id=JobId(source_name='det1', job_number=uuid.uuid4()),
            workflow_id=_workflow_id('reduction'),
            processor=proc_reset,
            source_names=['det1'],
            reset_on_run_transition=True,
        )
        job_keep = Job(
            job_id=JobId(source_name='log1', job_number=uuid.uuid4()),
            workflow_id=_workflow_id('timeseries'),
            processor=proc_keep,
            source_names=['log1'],
            reset_on_run_transition=False,
        )

        manager._active_jobs[job_reset.job_id] = job_reset
        manager._job_states[job_reset.job_id] = JobState.active
        manager._job_schedules[job_reset.job_id] = JobSchedule()
        manager._active_jobs[job_keep.job_id] = job_keep
        manager._job_states[job_keep.job_id] = JobState.active
        manager._job_schedules[job_keep.job_id] = JobSchedule()

        manager.on_run_start(RunStart(run_name='run_2', start_time=300))
        _push_data_to(manager, end_time=300)
        assert proc_reset.clear_calls == 1
        assert proc_keep.clear_calls == 0

    def test_no_active_jobs_consumes_pending_reset(self, fake_job_factory):
        """Pending resets are consumed even when no active jobs exist."""
        manager = JobManager(job_factory=fake_job_factory)
        manager.on_run_start(RunStart(run_name='run_1', start_time=100))
        assert manager._pending_reset_times == [100]
        # No jobs scheduled/active, but push data past the reset time
        _push_data_to(manager, end_time=200)
        # Should not error; pending list should be cleared
        assert manager._pending_reset_times == []

    def test_pending_resets_persist_without_data(self, fake_job_factory):
        """Pending resets accumulate and fire when data finally arrives."""
        manager = JobManager(job_factory=fake_job_factory)
        job_id = manager.schedule_job('det1', _make_config())
        _activate_jobs(manager)
        processor = fake_job_factory.processors[job_id]

        manager.on_run_start(RunStart(run_name='run_1', start_time=100))
        manager.on_run_stop(RunStop(run_name='run_1', stop_time=200))
        manager.on_run_start(RunStart(run_name='run_2', start_time=300))
        assert len(manager._pending_reset_times) == 3

        # All fire when data catches up (collapsed to single call)
        _push_data_to(manager, end_time=400)
        assert processor.clear_calls == 1
        assert manager._pending_reset_times == []

    def test_no_pending_resets_is_noop(self, fake_job_factory):
        """Pushing data without any scheduled resets does nothing."""
        manager = JobManager(job_factory=fake_job_factory)
        job_id = manager.schedule_job('det1', _make_config())
        _activate_jobs(manager)
        processor = fake_job_factory.processors[job_id]

        _push_data_to(manager, end_time=1000)
        assert processor.clear_calls == 0


@pytest.fixture
def fake_job_factory():
    return FakeJobFactory()
