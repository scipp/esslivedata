# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid
from copy import deepcopy
from typing import Any

import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobSchedule, WorkflowId
from ess.livedata.core.job import Job, JobData, JobId, JobResult
from ess.livedata.handlers.workflow_factory import Workflow


class TestJobResult:
    def test_stream_name_uses_default_output_name(self):
        """stream_name uses default output_name; UnrollingSinkAdapter replaces it."""
        workflow_id = WorkflowId(
            instrument="TEST",
            namespace="data_reduction",
            name="test_workflow",
            version=1,
        )
        job_number = uuid.uuid4()
        job_id = JobId(source_name="test_source", job_number=job_number)
        result = JobResult(
            job_id=job_id,
            workflow_id=workflow_id,
            start_time=100,
            end_time=200,
            data=sc.DataGroup({'out': sc.DataArray(sc.scalar(3.14))}),
            error_message=None,
        )
        assert result.stream_name == (
            '{"workflow_id":{"instrument":"TEST","namespace":"data_reduction",'
            '"name":"test_workflow","version":1},"job_id":{"source_name":"test_source",'
            '"job_number":"' + str(job_number) + '"},"output_name":"result"}'
        )


class FakeProcessor(Workflow):
    """Fake implementation of Workflow for testing."""

    def __init__(self):
        self.data: dict[str, Any] = {}
        self.accumulate_calls = []
        self.finalize_calls = 0
        self.clear_calls = 0
        self.should_fail_accumulate = False
        self.should_fail_finalize = False
        self.fail_finalize_when_empty = False

    def accumulate(
        self, data: dict[str, Any], *, start_time: int, end_time: int
    ) -> None:
        if self.should_fail_accumulate:
            raise RuntimeError("Accumulate failure")
        self.accumulate_calls.append(data.copy())
        for key, value in data.items():
            if key in self.data:
                self.data[key] += value
            else:
                self.data[key] = deepcopy(value)

    def finalize(self) -> dict[str, Any]:
        if self.should_fail_finalize:
            raise RuntimeError("Finalize failure")
        if self.fail_finalize_when_empty and not self.data:
            raise RuntimeError("No data has been added")
        self.finalize_calls += 1
        return self.data.copy()

    def clear(self) -> None:
        self.clear_calls += 1
        self.data.clear()
        self.accumulate_calls.clear()


@pytest.fixture
def fake_processor():
    return FakeProcessor()


@pytest.fixture
def sample_workflow_id():
    return WorkflowId(
        instrument="TEST",
        namespace="data_reduction",
        name="test_workflow",
        version=1,
    )


@pytest.fixture
def sample_job(fake_processor: FakeProcessor, sample_workflow_id: WorkflowId):
    job_id = JobId(source_name="test_source", job_number=1)
    return Job(
        job_id=job_id,
        workflow_id=sample_workflow_id,
        processor=fake_processor,
        source_names=["test_source"],
        aux_source_names={"aux_source": "aux_source"},
    )


@pytest.fixture
def sample_job_data():
    return JobData(
        start_time=100,
        end_time=200,
        primary_data={"test_source": sc.scalar(42.0)},
        aux_data={"aux_source": sc.scalar(10.0)},
    )


class TestJobSchedule:
    def test_valid_schedule_with_start_and_end(self):
        """Test creating a valid schedule with start and end times."""
        schedule = JobSchedule(start_time=100, end_time=200)
        assert schedule.start_time == 100
        assert schedule.end_time == 200

    def test_valid_schedule_with_immediate_start_and_end(self):
        """Test creating a valid schedule with immediate start (-1) and end time."""
        schedule = JobSchedule(start_time=-1, end_time=100)
        assert schedule.start_time == -1
        assert schedule.end_time == 100

    def test_valid_schedule_with_no_end_time(self):
        """Test creating a valid schedule with no end time (None)."""
        schedule = JobSchedule(start_time=100, end_time=None)
        assert schedule.start_time == 100
        assert schedule.end_time is None

    def test_valid_schedule_defaults(self):
        """Test creating a schedule with default values."""
        schedule = JobSchedule()
        assert schedule.start_time is None
        assert schedule.end_time is None

    def test_invalid_schedule_end_before_start(self):
        """Test that end_time < start_time raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Job end_time=100 must be greater than start_time=200",
        ):
            JobSchedule(start_time=200, end_time=100)

    def test_invalid_schedule_end_equals_start(self):
        """Test that end_time == start_time raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Job end_time=100 must be greater than start_time=100",
        ):
            JobSchedule(start_time=100, end_time=100)

    def test_valid_schedule_negative_start_times_other_than_minus_one(self):
        """Test that negative start times other than -1 are treated as regular times."""
        schedule = JobSchedule(start_time=-100, end_time=200)
        assert schedule.start_time == -100
        assert schedule.end_time == 200

    def test_invalid_schedule_negative_start_with_equal_end(self):
        """Test that negative start time (not -1) with equal end time still raises."""
        with pytest.raises(
            ValueError,
            match="Job end_time=-50 must be greater than start_time=-50",
        ):
            JobSchedule(start_time=-50, end_time=-50)


class TestJob:
    def test_initial_state(self, sample_job):
        """Test initial state of a Job."""
        assert sample_job.start_time is None
        assert sample_job.end_time is None

    def test_add_data_sets_times(self, sample_job, sample_job_data):
        """Test that adding data sets start and end times."""
        status = sample_job.add(sample_job_data)

        assert not status.has_error
        assert status.error_message is None
        assert sample_job.start_time == 100
        assert sample_job.end_time == 200

    def test_add_data_multiple_times_updates_end_time(self, sample_job):
        """Test that adding data multiple times only updates end time."""
        data1 = JobData(
            start_time=100,
            end_time=150,
            primary_data={"test_source": sc.scalar(10.0)},
            aux_data={},
        )
        data2 = JobData(
            start_time=120,
            end_time=250,
            primary_data={"test_source": sc.scalar(20.0)},
            aux_data={},
        )

        status1 = sample_job.add(data1)
        assert not status1.has_error
        assert sample_job.start_time == 100
        assert sample_job.end_time == 150

        status2 = sample_job.add(data2)
        assert not status2.has_error
        assert sample_job.start_time == 100  # Should not change
        assert sample_job.end_time == 250  # Should update

    def test_add_data_processes_all_provided_data(self, sample_job, fake_processor):
        """Test that add() processes all provided data."""
        data = JobData(
            start_time=100,
            end_time=200,
            primary_data={"test_source": sc.scalar(42.0)},
            aux_data={"aux_source": sc.scalar(10.0)},
        )

        status = sample_job.add(data)
        assert not status.has_error

        # Check that processor received all data
        assert len(fake_processor.accumulate_calls) == 1
        accumulated = fake_processor.accumulate_calls[0]
        assert "test_source" in accumulated
        assert "aux_source" in accumulated
        assert accumulated["test_source"] == sc.scalar(42.0)
        assert accumulated["aux_source"] == sc.scalar(10.0)
        assert len(accumulated) == 2

    def test_add_data_with_no_primary_data_does_not_set_times(self, sample_job):
        """Test that adding data with no primary data doesn't set start/end times."""
        data = JobData(
            start_time=100,
            end_time=200,
            primary_data={},
            aux_data={"aux_source": sc.scalar(10.0)},
        )

        status = sample_job.add(data)
        assert not status.has_error
        # Times should remain None since no primary data was provided
        assert sample_job.start_time is None
        assert sample_job.end_time is None

    def test_add_data_error_handling(self, fake_processor, sample_workflow_id):
        """Test error handling during data processing."""
        job_id = JobId(source_name="test_source", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["test_source"],
        )

        # Make processor fail
        fake_processor.should_fail_accumulate = True

        data = JobData(
            start_time=100,
            end_time=200,
            primary_data={"test_source": sc.scalar(42.0)},
            aux_data={},
        )

        status = job.add(data)
        assert status.has_error
        assert status.job_id == job_id
        assert "Job failed to process latest data" in status.error_message
        assert "Accumulate failure" in status.error_message

    def test_add_data_error_recovery(self, fake_processor, sample_workflow_id):
        """Test that job can recover after an error."""
        job_id = JobId(source_name="test_source", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["test_source"],
        )

        data = JobData(
            start_time=100,
            end_time=200,
            primary_data={"test_source": sc.scalar(42.0)},
            aux_data={},
        )

        # First, cause an error
        fake_processor.should_fail_accumulate = True
        status1 = job.add(data)
        assert status1.has_error

        # Then fix the processor and try again
        fake_processor.should_fail_accumulate = False
        status2 = job.add(data)
        assert not status2.has_error
        assert status2.error_message is None

        # Job should now return successful result
        result = job.get()
        assert result.error_message is None
        assert result.data is not None

    def test_get_returns_job_result(self, sample_job, sample_job_data):
        """Test that get() returns a proper JobResult."""
        sample_job.add(sample_job_data)
        result = sample_job.get()

        assert isinstance(result, JobResult)
        assert result.job_id.source_name == "test_source"
        assert result.job_id.job_number == 1
        assert result.workflow_id.name == "test_workflow"
        assert result.start_time == 100
        assert result.end_time == 200
        assert isinstance(result.data, sc.DataGroup)
        assert result.error_message is None

    def test_get_adds_time_coords_to_data_arrays(
        self, fake_processor, sample_workflow_id
    ):
        """Test that get() adds start_time and end_time coords to DataArrays."""
        job_id = JobId(source_name="test_source", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["test_source"],
        )

        # Set up processor to return a DataArray
        fake_processor.data = {
            "output": sc.DataArray(data=sc.array(dims=["x"], values=[1, 2, 3]))
        }

        # Add data to set job times
        data = JobData(
            start_time=1000,
            end_time=2000,
            primary_data={"test_source": sc.scalar(42.0)},
            aux_data={},
        )
        job.add(data)

        result = job.get()
        output = result.data["output"]

        assert "start_time" in output.coords
        assert "end_time" in output.coords
        assert output.coords["start_time"].value == 1000
        assert output.coords["end_time"].value == 2000
        assert output.coords["start_time"].unit == "ns"
        assert output.coords["end_time"].unit == "ns"

    def test_get_preserves_existing_time_coords_on_data_arrays(
        self, fake_processor, sample_workflow_id
    ):
        """Test that get() skips DataArrays that already have time coords.

        This allows workflows to set their own time coords for outputs that
        represent different time ranges (e.g., delta outputs that only cover
        the period since the last finalize, not the entire job duration).
        """
        job_id = JobId(source_name="test_source", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["test_source"],
        )

        # Set up processor to return DataArrays: one with existing time coords,
        # one without
        workflow_start = sc.scalar(500, unit='ns')
        workflow_end = sc.scalar(1500, unit='ns')
        fake_processor.data = {
            "delta_output": sc.DataArray(
                data=sc.scalar(1.0),
                coords={'start_time': workflow_start, 'end_time': workflow_end},
            ),
            "cumulative_output": sc.DataArray(data=sc.scalar(2.0)),
        }

        # Add data to set job times (different from workflow times)
        data = JobData(
            start_time=1000,
            end_time=2000,
            primary_data={"test_source": sc.scalar(42.0)},
            aux_data={},
        )
        job.add(data)

        result = job.get()

        # Delta output should preserve workflow-set times
        delta = result.data["delta_output"]
        assert delta.coords["start_time"].value == 500
        assert delta.coords["end_time"].value == 1500

        # Cumulative output should get job-level times
        cumulative = result.data["cumulative_output"]
        assert cumulative.coords["start_time"].value == 1000
        assert cumulative.coords["end_time"].value == 2000

    def test_get_does_not_add_time_coords_when_no_data_added(
        self, fake_processor, sample_workflow_id
    ):
        """Test that get() does not add time coords when job has no timing info."""
        job_id = JobId(source_name="test_source", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["test_source"],
        )

        # Set up processor to return a DataArray
        fake_processor.data = {
            "output": sc.DataArray(data=sc.array(dims=["x"], values=[1, 2, 3]))
        }

        # Don't add any data, so start_time and end_time remain None
        result = job.get()
        output = result.data["output"]

        assert "start_time" not in output.coords
        assert "end_time" not in output.coords

    def test_get_calls_processor_finalize(self, sample_job, fake_processor):
        """Test that get() calls processor.finalize()."""
        sample_job.get()
        assert fake_processor.finalize_calls == 1

    def test_get_with_finalize_error(self, fake_processor, sample_workflow_id):
        """Test get() handles finalize errors."""
        job_id = JobId(source_name="test_source", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["test_source"],
        )

        # Add some data successfully
        data = JobData(
            start_time=100,
            end_time=200,
            primary_data={"test_source": sc.scalar(42.0)},
            aux_data={},
        )
        job.add(data)

        # Make finalize fail
        fake_processor.should_fail_finalize = True

        result = job.get()
        assert result.error_message is not None
        assert "Job failed to compute result" in result.error_message
        assert "Finalize failure" in result.error_message
        assert result.data is None

    def test_get_with_none_value_in_result_reports_warning(
        self, fake_processor, sample_workflow_id
    ):
        """Test that a None value in workflow result is reported as a warning.

        Workflows should not return None values, but if they do, the job should
        filter them out, return the valid outputs, and report a warning rather
        than crashing the backend when serializing.
        """
        job_id = JobId(source_name="test_source", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["test_source"],
        )

        # Add some data successfully
        data = JobData(
            start_time=100,
            end_time=200,
            primary_data={"test_source": sc.scalar(42.0)},
            aux_data={},
        )
        job.add(data)

        # Make processor return a dict with None value
        fake_processor.data = {
            "valid_output": sc.DataArray(sc.scalar(1.0)),
            "invalid_output": None,  # This should cause a warning
        }

        result = job.get()
        # Should not be an error
        assert result.error_message is None
        # Should have a warning
        assert result.warning_message is not None
        assert "invalid_output" in result.warning_message
        # Valid data should be returned
        assert result.data is not None
        assert "valid_output" in result.data
        assert "invalid_output" not in result.data

    def test_reset_clears_processor_and_times(
        self, sample_job, sample_job_data, fake_processor
    ):
        """Test that reset() clears processor and resets times."""
        sample_job.add(sample_job_data)
        assert sample_job.start_time == 100
        assert sample_job.end_time == 200

        sample_job.reset()

        assert fake_processor.clear_calls == 1
        assert sample_job.start_time is None
        assert sample_job.end_time is None

    def test_can_get_after_error_in_add(self, fake_processor, sample_workflow_id):
        """Adding bad data should not directly affect get()."""
        job_id = JobId(source_name="test_source", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["test_source"],
        )

        # Cause an error
        fake_processor.should_fail_accumulate = True
        data = JobData(
            start_time=100,
            end_time=200,
            primary_data={"test_source": sc.scalar(42.0)},
            aux_data={},
        )
        job.add(data)

        result = job.get()
        # No error, provided that the processor does not fail finalize
        assert result.error_message is None


class TestJobAuxSourceMapping:
    """Tests for auxiliary source name mapping (field names vs stream names)."""

    def test_aux_sources_dict_remaps_keys_to_field_names(
        self, fake_processor: FakeProcessor, sample_workflow_id: WorkflowId
    ):
        """Test that aux_data keys are remapped from stream names to field names."""
        job_id = JobId(source_name="detector1", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["detector1"],
            aux_source_names={
                "incident_monitor": "monitor1",
                "transmission_monitor": "monitor2",
            },
        )

        # Send data with stream names (monitor1, monitor2)
        data = JobData(
            start_time=100,
            end_time=200,
            primary_data={"detector1": sc.scalar(100.0)},
            aux_data={
                "monitor1": sc.scalar(10.0),  # Stream name
                "monitor2": sc.scalar(20.0),  # Stream name
            },
        )
        job.add(data)

        # Verify workflow received data with field names
        # (incident_monitor, transmission_monitor)
        assert len(fake_processor.accumulate_calls) == 1
        accumulated = fake_processor.accumulate_calls[0]
        assert "detector1" in accumulated
        assert "incident_monitor" in accumulated  # Field name, not monitor1
        assert "transmission_monitor" in accumulated  # Field name, not monitor2
        assert "monitor1" not in accumulated  # Stream name should not appear
        assert "monitor2" not in accumulated  # Stream name should not appear
        assert accumulated["incident_monitor"] == sc.scalar(10.0)
        assert accumulated["transmission_monitor"] == sc.scalar(20.0)

    def test_aux_sources_when_field_names_equal_stream_names(
        self, fake_processor: FakeProcessor, sample_workflow_id: WorkflowId
    ):
        """Test case where field names and stream names are identical."""
        job_id = JobId(source_name="detector1", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["detector1"],
            aux_source_names={"monitor1": "monitor1", "monitor2": "monitor2"},
        )

        # Send data with stream names
        data = JobData(
            start_time=100,
            end_time=200,
            primary_data={"detector1": sc.scalar(100.0)},
            aux_data={
                "monitor1": sc.scalar(10.0),
                "monitor2": sc.scalar(20.0),
            },
        )
        job.add(data)

        # Verify workflow received data with same keys (field names == stream names)
        assert len(fake_processor.accumulate_calls) == 1
        accumulated = fake_processor.accumulate_calls[0]
        assert "monitor1" in accumulated
        assert "monitor2" in accumulated
        assert accumulated["monitor1"] == sc.scalar(10.0)
        assert accumulated["monitor2"] == sc.scalar(20.0)

    def test_aux_source_names_property_returns_stream_names(
        self, fake_processor: FakeProcessor, sample_workflow_id: WorkflowId
    ):
        """Test that aux_source_names property returns stream names for routing."""
        job_id = JobId(source_name="detector1", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["detector1"],
            aux_source_names={
                "incident_monitor": "monitor1",
                "transmission_monitor": "monitor2",
            },
        )

        # The property should return stream names (values) for JobManager routing
        aux_names = job.aux_source_names
        assert set(aux_names) == {"monitor1", "monitor2"}

    def test_aux_sources_empty_dict(
        self, fake_processor: FakeProcessor, sample_workflow_id: WorkflowId
    ):
        """Test handling of empty aux_sources dict."""
        job_id = JobId(source_name="detector1", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["detector1"],
            aux_source_names={},  # Empty dict
        )

        assert job.aux_source_names == []

        data = JobData(
            start_time=100,
            end_time=200,
            primary_data={"detector1": sc.scalar(100.0)},
            aux_data={},
        )
        job.add(data)

        # Should work fine with no aux data
        assert len(fake_processor.accumulate_calls) == 1

    def test_aux_sources_none(
        self, fake_processor: FakeProcessor, sample_workflow_id: WorkflowId
    ):
        """Test handling of None aux_sources."""
        job_id = JobId(source_name="detector1", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["detector1"],
            aux_source_names=None,
        )

        assert job.aux_source_names == []

    def test_partial_aux_data_with_dict_mapping(
        self, fake_processor: FakeProcessor, sample_workflow_id: WorkflowId
    ):
        """Test that only available aux data is remapped and passed."""
        job_id = JobId(source_name="detector1", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["detector1"],
            aux_source_names={
                "incident_monitor": "monitor1",
                "transmission_monitor": "monitor2",
            },
        )

        # Send data with only one of the two aux sources
        data = JobData(
            start_time=100,
            end_time=200,
            primary_data={"detector1": sc.scalar(100.0)},
            aux_data={
                "monitor1": sc.scalar(10.0),  # Only monitor1, not monitor2
            },
        )
        job.add(data)

        # Verify only the available aux data is passed with correct field name
        assert len(fake_processor.accumulate_calls) == 1
        accumulated = fake_processor.accumulate_calls[0]
        assert "incident_monitor" in accumulated
        assert "transmission_monitor" not in accumulated
        assert accumulated["incident_monitor"] == sc.scalar(10.0)

    def test_stream_multiplexing_to_multiple_fields(
        self, fake_processor: FakeProcessor, sample_workflow_id: WorkflowId
    ):
        """Test that one stream can be multiplexed to multiple field names."""
        job_id = JobId(source_name="detector1", job_number=1)
        job = Job(
            job_id=job_id,
            workflow_id=sample_workflow_id,
            processor=fake_processor,
            source_names=["detector1"],
            aux_source_names={
                "incident_monitor": "monitor1",  # Both fields map to same stream
                "normalization_monitor": "monitor1",
            },
        )

        # Send data with the multiplexed stream
        data = JobData(
            start_time=100,
            end_time=200,
            primary_data={"detector1": sc.scalar(100.0)},
            aux_data={
                "monitor1": sc.scalar(10.0),  # One stream
            },
        )
        job.add(data)

        # Verify workflow received data with both field names pointing to same value
        assert len(fake_processor.accumulate_calls) == 1
        accumulated = fake_processor.accumulate_calls[0]
        assert "detector1" in accumulated
        assert "incident_monitor" in accumulated
        assert "normalization_monitor" in accumulated
        assert "monitor1" not in accumulated  # Stream name should not appear
        assert accumulated["incident_monitor"] == sc.scalar(10.0)
        assert accumulated["normalization_monitor"] == sc.scalar(10.0)
        assert len(accumulated) == 3  # detector1 + 2 field names
