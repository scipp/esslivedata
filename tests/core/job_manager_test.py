# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pydantic
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import (
    JobSchedule,
    WorkflowConfig,
    WorkflowId,
    WorkflowOutputsBase,
)
from ess.livedata.core.job import Job, JobId, JobReply, JobResult, JobState
from ess.livedata.core.job_manager import JobFactory, JobManager, WorkflowData
from ess.livedata.core.message import StreamId

from .job_test import FakeProcessor


class SimpleTestOutputs(WorkflowOutputsBase):
    """Simple outputs model for testing."""

    result: sc.DataArray = pydantic.Field(title='Result')


class FakeJobFactory(JobFactory):
    """Fake implementation of JobFactory for testing."""

    def __init__(self):
        self.created_jobs = []
        self.processors: dict[JobId, FakeProcessor] = {}

    def get_workflow_spec(self, workflow_id):
        """Return None for tests - no enrichment needed in test scenarios."""
        return None

    def enrich_result(self, result):
        """Pass-through implementation for tests."""
        return result

    def create(self, *, job_id: JobId, config: WorkflowConfig) -> Job:
        processor = FakeProcessor()
        self.processors[job_id] = processor

        job = Job(
            job_id=job_id,
            workflow_id=config.identifier,
            processor=processor,
            source_names=[job_id.source_name],
            aux_source_names=config.aux_source_names,
        )

        self.created_jobs.append((job_id, config))
        return job


@pytest.fixture
def fake_job_factory():
    return FakeJobFactory()


@pytest.fixture
def base_workflow_config():
    """Base workflow config with immediate start and no end time."""
    return WorkflowConfig(
        identifier=WorkflowId(
            instrument="test",
            namespace="data_reduction",
            name="test_workflow",
            version=1,
        )
    )


@pytest.fixture
def scheduled_workflow_config():
    """Workflow config with scheduled start and end times."""
    return WorkflowConfig(
        identifier=WorkflowId(
            instrument="test",
            namespace="data_reduction",
            name="scheduled_workflow",
            version=1,
        ),
        schedule=JobSchedule(start_time=50, end_time=250),
    )


@pytest.fixture
def delayed_start_config():
    """Workflow config with delayed start time."""
    return WorkflowConfig(
        identifier=WorkflowId(
            instrument="test",
            namespace="data_reduction",
            name="delayed_workflow",
            version=1,
        ),
        schedule=JobSchedule(start_time=200),
    )


class TestJobManager:
    def test_initial_state(self, fake_job_factory):
        """Test initial state of JobManager."""
        manager = JobManager(fake_job_factory)

        assert manager.service_name == 'data_reduction'
        assert len(manager.active_jobs) == 0

    def test_schedule_job_creates_job(self, fake_job_factory, base_workflow_config):
        """Test that scheduling a job creates it using the factory."""
        manager = JobManager(fake_job_factory)

        job_id = manager.schedule_job("test_source", base_workflow_config)

        assert job_id.job_number is not None
        assert job_id.source_name == "test_source"
        assert len(fake_job_factory.created_jobs) == 1
        assert fake_job_factory.created_jobs[0] == (job_id, base_workflow_config)

    def test_schedule_multiple_jobs_gives_unique_ids(
        self, fake_job_factory, base_workflow_config
    ):
        """Test that scheduling multiple jobs increments job IDs."""
        manager = JobManager(fake_job_factory)

        job_id1 = manager.schedule_job("source1", base_workflow_config)
        job_id2 = manager.schedule_job("source2", base_workflow_config)

        assert job_id1.job_number != job_id2.job_number
        assert job_id1.source_name == "source1"
        assert job_id2.source_name == "source2"

    def test_push_data_activates_scheduled_jobs_with_immediate_start(
        self, fake_job_factory, base_workflow_config
    ):
        """Test that pushing data activates jobs scheduled to start immediately."""
        manager = JobManager(fake_job_factory)

        _ = manager.schedule_job("test_source", base_workflow_config)
        assert len(manager.active_jobs) == 0

        # Push data that should activate the job (since start_time=-1)
        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        statuses = manager.push_data(data)

        assert len(manager.active_jobs) == 1
        assert len(statuses) == 1
        assert not statuses[0].has_error

    def test_push_data_returns_job_statuses(
        self, fake_job_factory, base_workflow_config
    ):
        """Test that push_data returns status for each active job."""
        manager = JobManager(fake_job_factory)

        _ = manager.schedule_job("source1", base_workflow_config)
        _ = manager.schedule_job("source2", base_workflow_config)

        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={
                StreamId(name="source1"): sc.scalar(42.0),
                StreamId(name="source2"): sc.scalar(24.0),
            },
        )
        statuses = manager.push_data(data)

        assert len(statuses) == 2
        assert all(isinstance(status, JobReply) for status in statuses)
        assert all(not status.has_error for status in statuses)

    def test_push_data_handles_job_errors(self, fake_job_factory, base_workflow_config):
        """Test that push_data handles and reports job errors."""
        manager = JobManager(fake_job_factory)

        job_id = manager.schedule_job("test_source", base_workflow_config)

        # Make the processor fail
        fake_job_factory.processors[job_id].should_fail_accumulate = True

        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        statuses = manager.push_data(data)

        assert len(statuses) == 1
        assert statuses[0].has_error
        assert statuses[0].job_id == job_id
        assert "Job failed to process latest data" in statuses[0].error_message

    def test_push_data_activates_jobs_based_on_schedule(self, fake_job_factory):
        """Test that jobs are activated based on their scheduled start time."""
        manager = JobManager(fake_job_factory)
        config1 = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="early_workflow",
                version=1,
            ),
            schedule=JobSchedule(start_time=50),
        )
        config2 = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="late_workflow",
                version=1,
            ),
            schedule=JobSchedule(start_time=150),
        )

        _ = manager.schedule_job("source1", config1)
        _ = manager.schedule_job("source2", config2)
        assert len(manager.active_jobs) == 0

        # Push early data - should only activate job1
        early_data = WorkflowData(
            start_time=100,
            end_time=120,
            data={StreamId(name="source1"): sc.scalar(42.0)},
        )
        statuses = manager.push_data(early_data)
        assert len(manager.active_jobs) == 1
        assert len(statuses) == 1

        # Push later data - should activate job2
        later_data = WorkflowData(
            start_time=160,
            end_time=180,
            data={StreamId(name="source2"): sc.scalar(42.0)},
        )
        statuses = manager.push_data(later_data)
        assert len(manager.active_jobs) == 2
        assert len(statuses) == 1  # Only job2 gets data this time

    def test_push_data_feeds_active_jobs(self, fake_job_factory, base_workflow_config):
        """Test that pushing data feeds all active jobs."""
        manager = JobManager(fake_job_factory)

        _ = manager.schedule_job("source1", base_workflow_config)
        _ = manager.schedule_job("source1", base_workflow_config)

        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="source1"): sc.scalar(42.0)},
        )
        statuses = manager.push_data(data)

        # Both jobs should be active and receive data
        assert len(manager.active_jobs) == 2
        assert len(statuses) == 2
        for job in manager.active_jobs:
            assert job.start_time == 100  # Data start time
            assert job.end_time == 200  # Data end time

    def test_stop_job_scheduled_removes_from_scheduled(
        self, fake_job_factory, delayed_start_config
    ):
        """Test stopping a scheduled job removes it completely."""
        manager = JobManager(fake_job_factory)

        job_id = manager.schedule_job("test_source", delayed_start_config)
        # Before data reaches start time, job should be scheduled but not active
        assert len(manager.active_jobs) == 0

        manager.stop_job(job_id)

        # After stopping, even when data reaches start time, job should not activate
        data = WorkflowData(
            start_time=250,
            end_time=300,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)
        assert len(manager.active_jobs) == 0

    def test_stop_job_stops_active_immediately(
        self, fake_job_factory, base_workflow_config
    ):
        """Test stopping an active job."""
        manager = JobManager(fake_job_factory)

        job_id = manager.schedule_job("test_source", base_workflow_config)

        # Activate the job
        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)
        assert len(manager.active_jobs) == 1

        manager.stop_job(job_id)
        # Job stopped, even before compute_results is called
        assert len(manager.active_jobs) == 0

    def test_stop_job_nonexistent_raises_error(self, fake_job_factory):
        """Test stopping a non-existent job raises KeyError."""
        manager = JobManager(fake_job_factory)

        with pytest.raises(KeyError, match="Job 999 not found"):
            manager.stop_job(999)

    def test_reset_job_active(self, fake_job_factory, base_workflow_config):
        """Test resetting an active job calls its reset method."""
        manager = JobManager(fake_job_factory)

        job_id = manager.schedule_job("test_source", base_workflow_config)

        # Activate and feed data to the job
        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)

        # Verify job has data
        assert manager.active_jobs[0].start_time == 100
        assert manager.active_jobs[0].end_time == 200

        manager.reset_job(job_id)

        # Verify job was reset
        assert manager.active_jobs[0].start_time is None
        assert manager.active_jobs[0].end_time is None
        assert fake_job_factory.processors[job_id].clear_calls == 1

    def test_reset_job_scheduled(self, fake_job_factory, delayed_start_config):
        """Test resetting a scheduled job calls its reset method."""
        manager = JobManager(fake_job_factory)

        job_id = manager.schedule_job("test_source", delayed_start_config)

        manager.reset_job(job_id)
        assert fake_job_factory.processors[job_id].clear_calls == 1

    def test_reset_job_nonexistent_raises_error(self, fake_job_factory):
        """Test resetting a non-existent job raises KeyError."""
        manager = JobManager(fake_job_factory)

        with pytest.raises(KeyError, match="Job 999 not found"):
            manager.reset_job(999)

    def test_compute_results_returns_job_results(
        self, fake_job_factory, base_workflow_config
    ):
        """Test that compute_results returns results from all active jobs."""
        manager = JobManager(fake_job_factory)

        _ = manager.schedule_job("source1", base_workflow_config)
        _ = manager.schedule_job("source2", base_workflow_config)

        # Activate jobs
        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={
                StreamId(name="source1"): sc.scalar(42.0),
                StreamId(name="source2"): sc.scalar(24.0),
            },
        )
        manager.push_data(data)

        results = manager.compute_results()

        assert len(results) == 2
        assert all(isinstance(result, JobResult) for result in results)

    def test_compute_results_ignores_stopped_jobs(
        self, fake_job_factory, base_workflow_config
    ):
        """Test that compute_results removes jobs that were stopped."""
        manager = JobManager(fake_job_factory)

        job_id = manager.schedule_job("test_source", base_workflow_config)

        # Activate job
        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)
        assert len(manager.active_jobs) == 1

        # Stop the job
        manager.stop_job(job_id)
        assert len(manager.active_jobs) == 0  # Not active

        # Compute results should not give a result for the stopped job
        results = manager.compute_results()
        assert len(results) == 0  # Should not return result

    def test_job_lifecycle_with_schedule_based_activation(
        self, fake_job_factory, scheduled_workflow_config
    ):
        """Test complete job lifecycle with schedule-based activation."""
        manager = JobManager(fake_job_factory)
        config2 = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="workflow2",
                version=1,
            ),
            schedule=JobSchedule(start_time=150, end_time=350),
        )

        # Schedule two jobs with different start times
        _ = manager.schedule_job("source1", scheduled_workflow_config)
        _ = manager.schedule_job("source2", config2)

        # Initially no active jobs
        assert len(manager.active_jobs) == 0

        # Push early data - should activate job1 only (start_time=50)
        early_data = WorkflowData(
            start_time=100,
            end_time=150,
            data={StreamId(name="source1"): sc.scalar(10.0)},
        )
        manager.push_data(early_data)
        assert len(manager.active_jobs) == 1

        # Push later data - should activate job2 (start_time=150)
        later_data = WorkflowData(
            start_time=200,
            end_time=250,
            data={
                StreamId(name="source1"): sc.scalar(20.0),
                StreamId(name="source2"): sc.scalar(15.0),
            },
        )
        manager.push_data(later_data)
        assert len(manager.active_jobs) == 2

        # Push data that should finish job1 (end_time=250)
        finishing_data = WorkflowData(
            start_time=251,
            end_time=300,
            data={
                StreamId(name="source1"): sc.scalar(30.0),
                StreamId(name="source2"): sc.scalar(25.0),
            },
        )
        manager.push_data(finishing_data)

        # Get results - job1 should be finished and removed
        results = manager.compute_results()
        assert len(results) == 2  # Both jobs return results
        assert len(manager.active_jobs) == 1  # Only job2 remains

    def test_multiple_data_accumulation(self, fake_job_factory, base_workflow_config):
        """Test that multiple data pushes accumulate correctly in jobs."""
        manager = JobManager(fake_job_factory)

        job_id = manager.schedule_job("test_source", base_workflow_config)

        # Push multiple data batches
        data1 = WorkflowData(
            start_time=100,
            end_time=150,
            data={StreamId(name="test_source"): sc.scalar(10.0)},
        )
        data2 = WorkflowData(
            start_time=151,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(20.0)},
        )

        manager.push_data(data1)
        manager.push_data(data2)

        # Verify processor received both accumulate calls
        processor = fake_job_factory.processors[job_id]
        assert len(processor.accumulate_calls) == 2
        assert sc.identical(
            processor.accumulate_calls[0]["test_source"], sc.scalar(10.0)
        )
        assert sc.identical(
            processor.accumulate_calls[1]["test_source"], sc.scalar(20.0)
        )

        # Verify accumulated data (our fake processor sums numeric values)
        results = manager.compute_results()
        assert len(results) == 1
        # The accumulated value should be 30.0 (10.0 + 20.0)
        assert processor.data["test_source"] == 30.0

    def test_jobs_finish_based_on_schedule_end_time(self, fake_job_factory):
        """Test that jobs are marked for finishing based on schedule end_time."""
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            ),
            schedule=JobSchedule(end_time=175),
        )

        _ = manager.schedule_job("test_source", config)

        # Activate job with initial data
        initial_data = WorkflowData(
            start_time=100,
            end_time=150,
            data={StreamId(name="test_source"): sc.scalar(10.0)},
        )
        manager.push_data(initial_data)
        assert len(manager.active_jobs) == 1

        # Push data with start_time that goes beyond the job's scheduled end_time
        finishing_data = WorkflowData(
            start_time=180,  # Beyond job's scheduled end_time of 175
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(20.0)},
        )
        manager.push_data(finishing_data)

        # Job should be marked for finishing
        results = manager.compute_results()
        assert len(results) == 1
        assert len(manager.active_jobs) == 0  # Job should be finished and removed

    def test_multiple_jobs_different_schedule_end_times(self, fake_job_factory):
        """Test handling multiple jobs with different scheduled end times."""
        manager = JobManager(fake_job_factory)
        config1 = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="workflow1",
                version=1,
            ),
            schedule=JobSchedule(end_time=150),
        )
        config2 = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="workflow2",
                version=1,
            ),
            schedule=JobSchedule(end_time=200),
        )
        config3 = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="workflow3",
                version=1,
            ),
            schedule=JobSchedule(end_time=300),
        )

        # Schedule three jobs with different end times
        _ = manager.schedule_job("source1", config1)
        _ = manager.schedule_job("source2", config2)
        _ = manager.schedule_job("source3", config3)

        # Activate all jobs
        initial_data = WorkflowData(
            start_time=100,
            end_time=120,
            data={
                StreamId(name="source1"): sc.scalar(10.0),
                StreamId(name="source2"): sc.scalar(20.0),
                StreamId(name="source3"): sc.scalar(30.0),
            },
        )
        manager.push_data(initial_data)
        assert len(manager.active_jobs) == 3

        # Push data with start_time=175, should finish job1 (end_time=150), not others
        intermediate_data = WorkflowData(
            start_time=175,
            end_time=180,
            data={
                StreamId(name="source2"): sc.scalar(5.0),
                StreamId(name="source3"): sc.scalar(5.0),
            },
        )
        manager.push_data(intermediate_data)

        # Compute results should finish job1
        results = manager.compute_results()
        assert len(results) == 3
        assert len(manager.active_jobs) == 2  # Job1 should be removed

        # Push data with start_time=250, should finish job2 (end_time=200) but not job3
        later_data = WorkflowData(
            start_time=250,
            end_time=260,
            data={StreamId(name="source3"): sc.scalar(15.0)},
        )
        manager.push_data(later_data)

        # Compute results should finish job2
        results = manager.compute_results()
        assert len(results) == 1  # Only job3 returns result (got recent data)
        assert len(manager.active_jobs) == 1  # Only job3 should remain

    def test_job_finishing_edge_case_exact_schedule_end_time(self, fake_job_factory):
        """Test job finishing when data start_time is exactly scheduled end_time."""
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            ),
            schedule=JobSchedule(end_time=200),
        )

        _ = manager.schedule_job("test_source", config)

        # Activate job
        initial_data = WorkflowData(
            start_time=100,
            end_time=150,
            data={StreamId(name="test_source"): sc.scalar(10.0)},
        )
        manager.push_data(initial_data)
        assert len(manager.active_jobs) == 1

        # Push data with start_time exactly matching job's scheduled start_time
        exact_data = WorkflowData(
            start_time=200,  # Exactly matches job's end_time
            end_time=250,
            data={StreamId(name="test_source"): sc.scalar(20.0)},
        )
        manager.push_data(exact_data)

        # Job should be marked for finishing
        results = manager.compute_results()
        assert len(results) == 1
        assert len(manager.active_jobs) == 0  # Job should be finished

    def test_no_premature_job_finishing(self, fake_job_factory):
        """Test jobs don't finish prematurely when data is before scheduled end_time."""
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            ),
            schedule=JobSchedule(end_time=300),
        )

        _ = manager.schedule_job("test_source", config)

        # Activate job
        initial_data = WorkflowData(
            start_time=100,
            end_time=150,
            data={StreamId(name="test_source"): sc.scalar(10.0)},
        )
        manager.push_data(initial_data)
        assert len(manager.active_jobs) == 1

        # Push data with start_time < job's scheduled end_time
        early_data = WorkflowData(
            start_time=200,  # Before job's scheduled end_time of 300
            end_time=250,
            data={StreamId(name="test_source"): sc.scalar(20.0)},
        )
        manager.push_data(early_data)

        # Job should NOT be marked for finishing
        results = manager.compute_results()
        assert len(results) == 1
        assert len(manager.active_jobs) == 1  # Job should still be active

    def test_job_finishing_with_mixed_scenarios(self, fake_job_factory):
        """
        Test complex scenario with job stopping, schedule-based finishing,
        and continuation.
        """
        manager = JobManager(fake_job_factory)
        config1 = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="workflow1",
                version=1,
            ),
            schedule=JobSchedule(end_time=160),
        )
        config2 = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="workflow2",
                version=1,
            ),
            schedule=JobSchedule(end_time=250),
        )
        config3 = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="workflow3",
                version=1,
            ),
            schedule=JobSchedule(end_time=None),
        )  # No end time

        # Schedule three jobs
        _ = manager.schedule_job("source1", config1)
        job_id2 = manager.schedule_job("source2", config2)
        _ = manager.schedule_job("source3", config3)

        # Activate all jobs
        initial_data = WorkflowData(
            start_time=100,
            end_time=150,
            data={
                StreamId(name="source1"): sc.scalar(10.0),
                StreamId(name="source2"): sc.scalar(20.0),
                StreamId(name="source3"): sc.scalar(30.0),
            },
        )
        manager.push_data(initial_data)
        assert len(manager.active_jobs) == 3

        # Manually stop job2
        manager.stop_job(job_id2)

        # Push data that would finish job1 due to schedule end_time, keep job3 active
        finishing_data = WorkflowData(
            start_time=170,  # Beyond job1's scheduled end_time of 160
            end_time=180,
            data={
                StreamId(name="source1"): sc.scalar(5.0),
                StreamId(name="source3"): sc.scalar(15.0),
            },
        )
        manager.push_data(finishing_data)

        # Compute results should:
        # - Return results from all 2 jobs (not job2 since it was stopped)
        # - Remove job1 (finished by schedule)
        # - Keep job3 active (no end time)
        results = manager.compute_results()
        assert len(results) == 2
        assert len(manager.active_jobs) == 1  # Only job3 should remain

    def test_jobs_without_end_time_never_finish_automatically(self, fake_job_factory):
        """Test that jobs without scheduled end_time never finish automatically."""
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            ),
            schedule=JobSchedule(end_time=None),
        )

        _ = manager.schedule_job("test_source", config)

        # Activate job and push lots of data
        for i in range(5):
            data = WorkflowData(
                start_time=100 + i * 100,
                end_time=150 + i * 100,
                data={StreamId(name="test_source"): sc.scalar(10.0 * i)},
            )
            manager.push_data(data)

        # Job should never be marked for finishing
        results = manager.compute_results()
        assert len(results) == 1
        assert len(manager.active_jobs) == 1  # Job should still be active

    def test_schedule_start_time_edge_cases(
        self, fake_job_factory, base_workflow_config
    ):
        """Test edge cases for schedule start times."""
        manager = JobManager(fake_job_factory)

        # Test future start
        config_future = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test", namespace="data_reduction", name="future", version=1
            ),
            schedule=JobSchedule(start_time=200),
        )
        # Test past start (should activate immediately when data arrives)
        config_past = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test", namespace="data_reduction", name="past", version=1
            ),
            schedule=JobSchedule(start_time=50),
        )

        _ = manager.schedule_job("source1", base_workflow_config)
        _ = manager.schedule_job("source2", config_future)
        _ = manager.schedule_job("source3", config_past)

        # Push data at time 100
        data = WorkflowData(
            start_time=100,
            end_time=150,
            data={
                StreamId(name="source1"): sc.scalar(10.0),
                StreamId(name="source2"): sc.scalar(20.0),
                StreamId(name="source3"): sc.scalar(30.0),
            },
        )
        manager.push_data(data)

        # Should activate immediate and past, but not future
        assert len(manager.active_jobs) == 2

        # Push later data that should activate future job
        later_data = WorkflowData(
            start_time=250,
            end_time=300,
            data={StreamId(name="source2"): sc.scalar(40.0)},
        )
        manager.push_data(later_data)

        # All jobs should now be active
        assert len(manager.active_jobs) == 3

    def test_schedule_job_with_immediate_start_and_end_time_allowed(
        self, fake_job_factory
    ):
        """Test that immediate start (-1) with any end_time is allowed."""
        manager = JobManager(fake_job_factory)

        # This should be allowed: immediate start with specific end time
        config_valid = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="valid_workflow",
                version=1,
            ),
            schedule=JobSchedule(end_time=100),
        )
        job_id = manager.schedule_job("test_source", config_valid)
        assert job_id.source_name == "test_source"

    def test_job_with_zero_duration_after_immediate_start(self, fake_job_factory):
        """Test behavior of job with immediate start and very early end time."""
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            ),
            schedule=JobSchedule(end_time=50),
        )

        _ = manager.schedule_job("test_source", config)

        data = WorkflowData(
            start_time=30,  # Before job's end_time
            end_time=100,  # Beyond job's end_time of 50 - so job should finish
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)

        # Job should activate and immediately finish
        assert len(manager.active_jobs) == 1  # Activated

        results = manager.compute_results()
        assert len(results) == 1  # Returns result
        assert len(manager.active_jobs) == 0  # Finished and removed

    def test_job_schedule_edge_case_start_equals_data_time(self, fake_job_factory):
        """
        Test job activation when data start_time exactly matches scheduled start_time.
        """
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            ),
            schedule=JobSchedule(start_time=100, end_time=200),
        )

        _ = manager.schedule_job("test_source", config)

        # Push data with start_time exactly matching job's scheduled start_time
        data = WorkflowData(
            start_time=100,  # Exactly matches job's start_time
            end_time=150,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)

        # Job should be activated
        assert len(manager.active_jobs) == 1

    def test_job_schedule_edge_case_end_equals_data_time(self, fake_job_factory):
        """Test job finishing when data end_time exactly matches scheduled end_time."""
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            ),
            schedule=JobSchedule(end_time=200),
        )

        _ = manager.schedule_job("test_source", config)

        # Activate job
        initial_data = WorkflowData(
            start_time=100,
            end_time=150,
            data={StreamId(name="test_source"): sc.scalar(10.0)},
        )
        manager.push_data(initial_data)
        assert len(manager.active_jobs) == 1

        # Push data with end_time exactly matching job's scheduled end_time
        final_data = WorkflowData(
            start_time=180,
            end_time=200,  # Exactly matches job's end_time
            data={StreamId(name="test_source"): sc.scalar(20.0)},
        )
        manager.push_data(final_data)

        # Job should NOT be marked for finishing yet (end_time <= end_time, not <)
        assert len(manager.active_jobs) == 1

        # But the next data batch should finish it
        beyond_data = WorkflowData(
            start_time=201,  # Beyond job's end_time
            end_time=250,
            data={StreamId(name="test_source"): sc.scalar(5.0)},
        )
        manager.push_data(beyond_data)

        results = manager.compute_results()
        assert len(results) == 1
        assert len(manager.active_jobs) == 0  # Job should be finished

    def test_multiple_jobs_same_schedule_times(self, fake_job_factory):
        """Test multiple jobs with identical start and end times."""
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            ),
            schedule=JobSchedule(start_time=100, end_time=200),
        )

        # Schedule multiple jobs with same timing
        _ = manager.schedule_job("source1", config)
        _ = manager.schedule_job("source2", config)
        _ = manager.schedule_job("source3", config)

        # All should activate together
        data = WorkflowData(
            start_time=150,
            end_time=170,
            data={
                StreamId(name="source1"): sc.scalar(10.0),
                StreamId(name="source2"): sc.scalar(20.0),
                StreamId(name="source3"): sc.scalar(30.0),
            },
        )
        manager.push_data(data)
        assert len(manager.active_jobs) == 3

        # All should finish together
        finishing_data = WorkflowData(
            start_time=250,  # Beyond all jobs' end_time
            end_time=270,
            data={StreamId(name="source1"): sc.scalar(5.0)},
        )
        manager.push_data(finishing_data)

        results = manager.compute_results()
        assert len(results) == 3
        assert len(manager.active_jobs) == 0

    def test_negative_start_times_other_than_minus_one(self, fake_job_factory):
        """Test behavior with negative start times other than -1."""
        manager = JobManager(fake_job_factory)

        # Negative start times other than -1 should be treated as regular timestamps
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            ),
            schedule=JobSchedule(start_time=-100, end_time=200),
        )
        _ = manager.schedule_job("test_source", config)

        # Data with positive time should activate job (since -100 < any positive time)
        data = WorkflowData(
            start_time=100,
            end_time=150,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)
        assert len(manager.active_jobs) == 1

    def test_accumulate_failure_handled_gracefully(self, fake_job_factory):
        """Test that an accumulate failure in the processor is handled gracefully."""
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            )
        )
        job_id = manager.schedule_job("test_source", config)

        # Activate the job
        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)
        assert len(manager.active_jobs) == 1

        # Induce an accumulate failure
        processor = fake_job_factory.processors[job_id]
        processor.should_fail_accumulate = True

        # Push new data - should be handled gracefully, not crash
        new_data = WorkflowData(
            start_time=201,
            end_time=250,
            data={StreamId(name="test_source"): sc.scalar(84.0)},
        )
        manager.push_data(new_data)

        # Job should still be active (time window doesn't update on error)
        assert len(manager.active_jobs) == 1
        assert manager.active_jobs[0].start_time == 100
        assert manager.active_jobs[0].end_time == 200  # Not updated due to error

        results = manager.compute_results()
        assert len(results) == 1
        # Finalize succeeds if accumulate failed (no new data was actually added)
        assert results[0].error_message is None

    def test_finalize_failure_handled_gracefully(self, fake_job_factory):
        """Test that a finalize failure in the processor is handled gracefully."""
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            )
        )
        job_id = manager.schedule_job("test_source", config)

        # Activate the job
        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)
        assert len(manager.active_jobs) == 1

        # Induce a finalize failure
        processor = fake_job_factory.processors[job_id]

        processor.should_fail_finalize = True
        results = manager.compute_results()
        assert len(results) == 1
        assert results[0].error_message is not None
        assert len(manager.active_jobs) == 1

        # Push more data so job gets included in next compute_results
        more_data = WorkflowData(
            start_time=201,
            end_time=250,
            data={StreamId(name="test_source"): sc.scalar(24.0)},
        )
        manager.push_data(more_data)

        processor.should_fail_finalize = False
        results = manager.compute_results()
        assert len(results) == 1
        assert results[0].error_message is None

    def test_jobs_with_finalize_errors_will_compute_again_without_new_primary_data(
        self, fake_job_factory
    ):
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            )
        )
        job_id = manager.schedule_job("test_source", config)

        # Activate job and push data
        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)

        # Induce finalize failure
        processor = fake_job_factory.processors[job_id]
        processor.should_fail_finalize = True

        results = manager.compute_results()
        assert len(results) == 1
        assert results[0].error_message is not None

        # Fix the processor and call compute_results again without new data
        processor.should_fail_finalize = False
        results = manager.compute_results()
        assert len(results) == 1
        assert results[0].error_message is None

    def test_warning_from_none_values_propagates_to_job_status(self, fake_job_factory):
        """Test that warnings from None values in results are tracked in job status."""
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            )
        )
        job_id = manager.schedule_job("test_source", config)

        # Activate job and push data
        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)

        # Make processor return a dict with None value
        processor = fake_job_factory.processors[job_id]
        processor.data = {
            "valid_output": sc.DataArray(sc.scalar(1.0)),
            "invalid_output": None,
        }

        results = manager.compute_results()
        assert len(results) == 1
        # Result should have valid data and a warning
        assert results[0].error_message is None
        assert results[0].warning_message is not None
        assert "invalid_output" in results[0].warning_message
        assert results[0].data is not None
        assert "valid_output" in results[0].data

        # Job status should reflect warning state
        statuses = manager.get_all_job_statuses()
        assert len(statuses) == 1
        assert statuses[0].state == JobState.warning
        assert statuses[0].warning_message is not None
        assert "invalid_output" in statuses[0].warning_message

        # Now fix the processor to return valid data (no None values)
        processor.data = {
            "valid_output": sc.DataArray(sc.scalar(2.0)),
            "other_output": sc.DataArray(sc.scalar(3.0)),
        }

        # Push more data to trigger recomputation
        more_data = WorkflowData(
            start_time=201,
            end_time=300,
            data={StreamId(name="test_source"): sc.scalar(24.0)},
        )
        manager.push_data(more_data)

        results = manager.compute_results()
        assert len(results) == 1
        assert results[0].error_message is None
        assert results[0].warning_message is None  # Warning should be cleared

        # Job status should be back to active
        statuses = manager.get_all_job_statuses()
        assert len(statuses) == 1
        assert statuses[0].state == JobState.active
        assert statuses[0].warning_message is None

    def test_successful_jobs_will_not_compute_again_without_new_primary_data(
        self, fake_job_factory
    ):
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            )
        )
        job_id = manager.schedule_job("test_source", config)

        # Activate job and push data
        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)
        assert len(manager.active_jobs) == 1
        assert manager.active_jobs[0].job_id == job_id

        # Compute results successfully
        results = manager.compute_results()
        assert len(results) == 1
        assert results[0].error_message is None

        # Calling compute_results again should not include this job
        results = manager.compute_results()
        assert len(results) == 0

    def test_jobs_without_primary_data_not_included_in_compute_results(
        self, fake_job_factory
    ):
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            )
        )
        job_id1 = manager.schedule_job("source1", config)
        job_id2 = manager.schedule_job("source2", config)

        # Activate both jobs with initial data
        initial_data = WorkflowData(
            start_time=100,
            end_time=150,
            data={
                StreamId(name="source1"): sc.scalar(10.0),
                StreamId(name="source2"): sc.scalar(20.0),
            },
        )
        manager.push_data(initial_data)
        assert len(manager.active_jobs) == 2
        assert {job.job_id for job in manager.active_jobs} == {job_id1, job_id2}

        # Compute results - both jobs should be processed
        results = manager.compute_results()
        assert len(results) == 2

        # Push data for only one job
        new_data = WorkflowData(
            start_time=151,
            end_time=200,
            data={StreamId(name="source1"): sc.scalar(15.0)},
        )
        manager.push_data(new_data)

        # Compute results - only job1 should be processed
        results = manager.compute_results()
        assert len(results) == 1
        assert results[0].job_id == job_id1

    def test_auxiliary_data_only_does_not_trigger_compute_results(
        self, fake_job_factory
    ):
        manager = JobManager(fake_job_factory)
        config = WorkflowConfig(
            identifier=WorkflowId(
                instrument="test",
                namespace="data_reduction",
                name="test_workflow",
                version=1,
            )
        )
        job_id = manager.schedule_job("test_source", config)

        # Activate job with initial primary data
        initial_data = WorkflowData(
            start_time=100,
            end_time=150,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(initial_data)
        assert len(manager.active_jobs) == 1
        assert manager.active_jobs[0].job_id == job_id

        # Compute results
        results = manager.compute_results()
        assert len(results) == 1

        # Push only auxiliary data
        aux_data = WorkflowData(
            start_time=151,
            end_time=200,
            data={StreamId(name="aux_source"): sc.scalar(99.0)},
        )
        manager.push_data(aux_data)

        # Compute results should not include this job
        results = manager.compute_results()
        assert len(results) == 0


class TestPushFailureCascade:
    """Regression tests: a push failure must not cascade into a finalize error."""

    def test_failed_push_does_not_trigger_finalize(
        self, fake_job_factory, base_workflow_config
    ):
        """When push fails on empty accumulators, compute_results must not finalize."""
        manager = JobManager(fake_job_factory)
        job_id = manager.schedule_job("test_source", base_workflow_config)

        processor = fake_job_factory.processors[job_id]
        processor.fail_finalize_when_empty = True
        processor.should_fail_accumulate = True

        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)

        # Job should be in warning state from the push failure
        status = manager.get_job_status(job_id)
        assert status.state == JobState.warning
        assert status.warning_message is not None
        assert "Accumulate failure" in status.warning_message

        # compute_results() must not attempt to finalize the empty job
        results = manager.compute_results()
        assert len(results) == 0

        # Job should still be in warning state, not error
        status = manager.get_job_status(job_id)
        assert status.state == JobState.warning
        assert "Accumulate failure" in status.warning_message
        assert status.error_message is None

    def test_recovery_after_push_failure(self, fake_job_factory, base_workflow_config):
        """Job recovers to active state when push succeeds after a failure."""
        manager = JobManager(fake_job_factory)
        job_id = manager.schedule_job("test_source", base_workflow_config)

        processor = fake_job_factory.processors[job_id]
        processor.fail_finalize_when_empty = True
        processor.should_fail_accumulate = True

        data = WorkflowData(
            start_time=100,
            end_time=200,
            data={StreamId(name="test_source"): sc.scalar(42.0)},
        )
        manager.push_data(data)
        assert manager.get_job_status(job_id).state == JobState.warning

        # Fix the processor and push again
        processor.should_fail_accumulate = False
        data2 = WorkflowData(
            start_time=200,
            end_time=300,
            data={StreamId(name="test_source"): sc.scalar(43.0)},
        )
        manager.push_data(data2)

        # Job should recover to active state
        status = manager.get_job_status(job_id)
        assert status.state == JobState.active
        assert status.warning_message is None

        # compute_results should now work
        results = manager.compute_results()
        assert len(results) == 1
        assert results[0].error_message is None


class TestJobFactoryRender:
    """Tests for JobFactory calling render() on aux sources."""

    def test_job_factory_calls_render_on_aux_sources(self) -> None:
        """Test that JobFactory calls render() when creating a job."""
        from typing import Literal

        from ess.livedata.config.instrument import Instrument
        from ess.livedata.config.workflow_spec import AuxSourcesBase

        # Create a custom aux sources model with render spy
        class TestAuxSources(AuxSourcesBase):
            monitor: Literal['monitor1'] = 'monitor1'

            def render(self, job_id: JobId) -> dict[str, str]:
                # Append job_number to stream name
                base = self.model_dump(mode='json')
                return {
                    field: f"{job_id.job_number}/{stream}"
                    for field, stream in base.items()
                }

        # Setup instrument and workflow
        instrument = Instrument(name='test')
        instrument.active_namespace = 'data_reduction'

        handle = instrument.register_spec(
            name='test_workflow',
            version=1,
            title='Test',
            description='Test',
            source_names=['detector1'],
            aux_sources=TestAuxSources,
            outputs=SimpleTestOutputs,
        )

        @handle.attach_factory()
        def test_workflow():
            return FakeProcessor()

        # Create JobFactory and job config
        factory = JobFactory(instrument)
        workflow_id = WorkflowId(
            instrument='test',
            namespace='data_reduction',
            name='test_workflow',
            version=1,
        )
        config = WorkflowConfig(
            identifier=workflow_id,
            aux_source_names={'monitor': 'monitor1'},
        )
        job_id = JobId(source_name='detector1', job_number='test-uuid-123')

        # Create job
        job = factory.create(job_id=job_id, config=config)

        # Verify that render() was called and result was used
        # The job should have the rendered aux source names
        assert job.aux_source_names == ['test-uuid-123/monitor1']

    def test_job_factory_default_render_preserves_names(self) -> None:
        """Test JobFactory with default render() behavior."""
        from typing import Literal

        from ess.livedata.config.instrument import Instrument
        from ess.livedata.config.workflow_spec import AuxSourcesBase

        class DefaultAuxSources(AuxSourcesBase):
            incident_monitor: Literal['monitor1'] = 'monitor1'
            transmission_monitor: Literal['monitor2'] = 'monitor2'

        # Setup
        instrument = Instrument(name='test')
        instrument.active_namespace = 'data_reduction'

        handle = instrument.register_spec(
            name='default_workflow',
            version=1,
            title='Default',
            description='Default',
            source_names=['detector1'],
            aux_sources=DefaultAuxSources,
            outputs=SimpleTestOutputs,
        )

        @handle.attach_factory()
        def default_workflow():
            return FakeProcessor()

        factory = JobFactory(instrument)
        spec = instrument.workflow_factory[
            WorkflowId(
                instrument='test',
                namespace='data_reduction',
                name='default_workflow',
                version=1,
            )
        ]
        config = WorkflowConfig(
            identifier=spec.get_id(),
            aux_source_names={
                'incident_monitor': 'monitor1',
                'transmission_monitor': 'monitor2',
            },
        )
        job_id = JobId(source_name='detector1', job_number='uuid-456')

        # Create job
        job = factory.create(job_id=job_id, config=config)

        # Default render should preserve the original names
        assert set(job.aux_source_names) == {'monitor1', 'monitor2'}

    def test_job_factory_with_no_aux_sources(self) -> None:
        """Test JobFactory with workflow that has no aux sources."""
        from ess.livedata.config.instrument import Instrument

        instrument = Instrument(name='test')
        instrument.active_namespace = 'data_reduction'

        handle = instrument.register_spec(
            name='no_aux_workflow',
            version=1,
            title='No Aux',
            description='No aux sources',
            source_names=['detector1'],
            outputs=SimpleTestOutputs,
        )

        @handle.attach_factory()
        def no_aux_workflow():
            return FakeProcessor()

        factory = JobFactory(instrument)
        spec = instrument.workflow_factory[
            WorkflowId(
                instrument='test',
                namespace='data_reduction',
                name='no_aux_workflow',
                version=1,
            )
        ]
        config = WorkflowConfig(identifier=spec.get_id())
        job_id = JobId(source_name='detector1', job_number='uuid-789')

        # Should not raise, should create job with empty aux sources
        job = factory.create(job_id=job_id, config=config)
        assert job.aux_source_names == []

    def test_job_factory_with_empty_aux_source_names(self) -> None:
        """Test JobFactory when config has empty aux_source_names dict with defaults."""
        from typing import Literal

        from ess.livedata.config.instrument import Instrument
        from ess.livedata.config.workflow_spec import AuxSourcesBase

        class OptionalAuxSources(AuxSourcesBase):
            monitor: Literal['monitor1'] = 'monitor1'

        instrument = Instrument(name='test')
        instrument.active_namespace = 'data_reduction'

        handle = instrument.register_spec(
            name='optional_aux_workflow',
            version=1,
            title='Optional Aux',
            description='Optional',
            source_names=['detector1'],
            aux_sources=OptionalAuxSources,
            outputs=SimpleTestOutputs,
        )

        @handle.attach_factory()
        def optional_aux_workflow():
            return FakeProcessor()

        factory = JobFactory(instrument)
        spec = instrument.workflow_factory[
            WorkflowId(
                instrument='test',
                namespace='data_reduction',
                name='optional_aux_workflow',
                version=1,
            )
        ]
        # Config with empty aux_source_names
        config = WorkflowConfig(identifier=spec.get_id(), aux_source_names={})
        job_id = JobId(source_name='detector1', job_number='uuid-abc')

        job = factory.create(job_id=job_id, config=config)
        # Empty dict triggers model defaults, so should use 'monitor1'
        assert job.aux_source_names == ['monitor1']

    def test_job_factory_render_uses_source_name(self) -> None:
        """Test that render() can use source_name from JobId."""
        from typing import Literal

        from ess.livedata.config.instrument import Instrument
        from ess.livedata.config.workflow_spec import AuxSourcesBase

        class SourcePrefixedAuxSources(AuxSourcesBase):
            roi: Literal['roi_rectangle'] = 'roi_rectangle'

            def render(self, job_id: JobId) -> dict[str, str]:
                base = self.model_dump(mode='json')
                return {
                    field: f"{job_id.source_name}/{stream}"
                    for field, stream in base.items()
                }

        instrument = Instrument(name='test')
        instrument.active_namespace = 'data_reduction'

        handle = instrument.register_spec(
            name='source_prefix_workflow',
            version=1,
            title='Source Prefix',
            description='Source prefix',
            source_names=['detector1', 'detector2'],
            aux_sources=SourcePrefixedAuxSources,
            outputs=SimpleTestOutputs,
        )

        @handle.attach_factory()
        def source_prefix_workflow():
            return FakeProcessor()

        factory = JobFactory(instrument)
        spec = instrument.workflow_factory[
            WorkflowId(
                instrument='test',
                namespace='data_reduction',
                name='source_prefix_workflow',
                version=1,
            )
        ]

        # Create jobs for different sources
        config = WorkflowConfig(
            identifier=spec.get_id(),
            aux_source_names={'roi': 'roi_rectangle'},
        )
        job_id_1 = JobId(source_name='detector1', job_number='uuid-1')
        job_id_2 = JobId(source_name='detector2', job_number='uuid-2')

        job1 = factory.create(job_id=job_id_1, config=config)
        job2 = factory.create(job_id=job_id_2, config=config)

        # Each job should have source-specific stream name
        assert job1.aux_source_names == ['detector1/roi_rectangle']
        assert job2.aux_source_names == ['detector2/roi_rectangle']

    def test_job_factory_render_with_empty_dict_uses_defaults(self) -> None:
        """Test that empty aux_source_names dict triggers model defaults and render."""
        from typing import Literal

        from ess.livedata.config.instrument import Instrument
        from ess.livedata.config.workflow_spec import AuxSourcesBase

        class DefaultedAuxSources(AuxSourcesBase):
            monitor: Literal['monitor1', 'monitor2'] = 'monitor1'  # Has default

            def render(self, job_id: JobId) -> dict[str, str]:
                base = self.model_dump(mode='json')
                return {
                    field: f"{job_id.source_name}/{stream}"
                    for field, stream in base.items()
                }

        instrument = Instrument(name='test')
        instrument.active_namespace = 'data_reduction'

        handle = instrument.register_spec(
            name='defaulted_aux_workflow',
            version=1,
            title='Defaulted Aux',
            description='Defaulted',
            source_names=['detector1'],
            aux_sources=DefaultedAuxSources,
            outputs=SimpleTestOutputs,
        )

        @handle.attach_factory()
        def defaulted_aux_workflow():
            return FakeProcessor()

        factory = JobFactory(instrument)
        spec = instrument.workflow_factory[
            WorkflowId(
                instrument='test',
                namespace='data_reduction',
                name='defaulted_aux_workflow',
                version=1,
            )
        ]

        # Config with empty aux_source_names should use model defaults
        config = WorkflowConfig(identifier=spec.get_id(), aux_source_names={})
        job_id = JobId(source_name='detector1', job_number='uuid-default')

        job = factory.create(job_id=job_id, config=config)

        # Should use default 'monitor1' and render it with source prefix
        assert job.aux_source_names == ['detector1/monitor1']
