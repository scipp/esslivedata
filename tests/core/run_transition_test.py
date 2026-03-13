# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for run transition (start/stop) handling in JobManager."""

import uuid

import pytest

from ess.livedata.config.workflow_spec import (
    WorkflowConfig,
    WorkflowId,
)
from ess.livedata.core.job import Job, JobId, JobState
from ess.livedata.core.job_manager import JobManager, WorkflowData
from ess.livedata.core.message import RunStart, RunStop

from .job_manager_test import FakeJobFactory
from .job_test import FakeProcessor


def _workflow_id(name: str = 'test') -> WorkflowId:
    return WorkflowId(
        instrument="test", namespace="data_reduction", name=name, version=1
    )


def _make_config(name: str = 'test') -> WorkflowConfig:
    return WorkflowConfig(identifier=_workflow_id(name))


def _activate_jobs(manager: JobManager) -> None:
    """Push minimal data to activate scheduled jobs."""
    manager.push_data(WorkflowData(start_time=0, end_time=1, data={}))


class TestRunTransitionReset:
    def test_on_run_start_resets_active_jobs(self, fake_job_factory):
        manager = JobManager(job_factory=fake_job_factory)
        job_id = manager.schedule_job('det1', _make_config())
        _activate_jobs(manager)
        processor = fake_job_factory.processors[job_id]

        manager.on_run_start(RunStart(run_name='run_1', start_time=100))
        assert processor.clear_calls > 0

    def test_on_run_stop_resets_active_jobs(self, fake_job_factory):
        manager = JobManager(job_factory=fake_job_factory)
        job_id = manager.schedule_job('det1', _make_config())
        _activate_jobs(manager)
        processor = fake_job_factory.processors[job_id]

        manager.on_run_stop(RunStop(run_name='run_1', stop_time=200))
        assert processor.clear_calls > 0

    def test_run_transition_skips_jobs_with_flag_disabled(self):
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

        manager.on_run_start(RunStart(run_name='run_1', start_time=100))
        assert processor.clear_calls == 0

    def test_run_transition_resets_some_but_not_all(self):
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
        manager._active_jobs[job_keep.job_id] = job_keep
        manager._job_states[job_keep.job_id] = JobState.active

        manager.on_run_start(RunStart(run_name='run_2', start_time=300))
        assert proc_reset.clear_calls > 0
        assert proc_keep.clear_calls == 0

    def test_run_transition_with_no_active_jobs_is_noop(self, fake_job_factory):
        manager = JobManager(job_factory=fake_job_factory)
        manager.on_run_start(RunStart(run_name='run_1', start_time=100))
        manager.on_run_stop(RunStop(run_name='run_1', stop_time=200))


@pytest.fixture
def fake_job_factory():
    return FakeJobFactory()
