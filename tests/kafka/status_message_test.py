# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import json
import uuid

import pytest
from streaming_data_types import deserialise_x5f2, serialise_x5f2

from ess.livedata.config.workflow_spec import JobId, WorkflowId
from ess.livedata.core.job import JobState, JobStatus, ServiceState, ServiceStatus
from ess.livedata.kafka.x5f2_compat import (
    JobStatusJSON,
    JobStatusMessage,
    JobStatusPayload,
    NicosStatus,
    ServiceId,
    ServiceServiceId,
    ServiceStatusMessage,
    ServiceStatusPayload,
    job_state_to_nicos_status_constant,
    job_status_to_x5f2,
    service_state_to_nicos_status_constant,
    service_status_to_x5f2,
    x5f2_to_job_status,
    x5f2_to_service_status,
    x5f2_to_status,
)


def make_job_status(**overrides) -> JobStatus:
    """Create a JobStatus with defaults that can be overridden."""
    defaults = {
        "job_id": JobId(source_name="detector_1", job_number=uuid.uuid4()),
        "workflow_id": WorkflowId(
            instrument="test_inst",
            namespace="data_reduction",
            name="test_workflow",
            version=1,
        ),
        "state": JobState.active,
        "error_message": None,
        "warning_message": None,
        "start_time": None,
        "end_time": None,
    }
    defaults.update(overrides)
    return JobStatus(**defaults)


def make_service_status(**overrides) -> ServiceStatus:
    """Create a ServiceStatus with defaults that can be overridden."""
    defaults = {
        "instrument": "dream",
        "namespace": "data_reduction",
        "worker_id": str(uuid.uuid4()),
        "state": ServiceState.running,
        "started_at": 1000000000,
        "active_job_count": 3,
        "messages_processed": 5000,
        "error": None,
    }
    defaults.update(overrides)
    return ServiceStatus(**defaults)


class TestServiceId:
    def test_from_string_valid_format(self):
        """Test parsing valid service_id string."""
        job_number = uuid.uuid4()
        service_id_str = f"detector_1:{job_number}"
        service_id = ServiceId.from_string(service_id_str)

        assert service_id.job_id.source_name == "detector_1"
        assert service_id.job_id.job_number == job_number

    def test_from_string_invalid_format_no_colon(self):
        """Test error handling for service_id without colon."""
        with pytest.raises(ValueError, match="Invalid service_id format"):
            ServiceId.from_string("detector_1")

    def test_from_string_invalid_format_multiple_colons(self):
        """Test parsing service_id with multiple colons (should work)."""
        job_number = uuid.uuid4()
        service_id_str = f"detector:group:1:{job_number}"
        service_id = ServiceId.from_string(service_id_str)

        assert service_id.job_id.source_name == "detector:group:1"
        assert service_id.job_id.job_number == job_number

    def test_from_string_invalid_uuid(self):
        """Test error handling for invalid UUID in job_number."""
        with pytest.raises(ValueError, match="Invalid service_id format"):
            ServiceId.from_string("detector_1:not-a-uuid")

    def test_from_job_id(self):
        """Test creating ServiceId from JobId."""
        job_id = JobId(source_name="detector_2", job_number=uuid.uuid4())
        service_id = ServiceId.from_job_id(job_id)

        assert service_id.job_id == job_id

    def test_to_string(self):
        """Test converting ServiceId to string format."""
        job_number = uuid.uuid4()
        job_id = JobId(source_name="detector_3", job_number=job_number)
        service_id = ServiceId.from_job_id(job_id)

        expected = f"detector_3:{job_number}"
        assert service_id.to_string() == expected
        assert str(service_id) == expected

    def test_round_trip_string_conversion(self):
        """Test that string conversion is reversible."""
        original_job_id = JobId(source_name="test_detector", job_number=uuid.uuid4())
        service_id = ServiceId.from_job_id(original_job_id)

        # Convert to string and back
        service_id_str = service_id.to_string()
        parsed_service_id = ServiceId.from_string(service_id_str)

        assert parsed_service_id.job_id == original_job_id


class TestJobStateToNicosStatus:
    def test_all_job_states_mapped(self):
        """Test that all JobState values have corresponding NicosStatus."""
        for state in JobState:
            status = job_state_to_nicos_status_constant(state)
            assert isinstance(status, NicosStatus)

    def test_specific_mappings(self):
        """Test specific state mappings."""
        assert job_state_to_nicos_status_constant(JobState.active) == NicosStatus.OK
        assert job_state_to_nicos_status_constant(JobState.error) == NicosStatus.ERROR
        assert job_state_to_nicos_status_constant(JobState.finishing) == NicosStatus.OK
        assert (
            job_state_to_nicos_status_constant(JobState.paused) == NicosStatus.DISABLED
        )
        assert (
            job_state_to_nicos_status_constant(JobState.scheduled)
            == NicosStatus.DISABLED
        )
        assert (
            job_state_to_nicos_status_constant(JobState.warning) == NicosStatus.WARNING
        )


class TestJobStatusPayload:
    def test_payload_creation_minimal(self):
        """Test creating JobStatusPayload with minimal required fields."""
        job_id = JobId(source_name="test", job_number=uuid.uuid4())
        message = JobStatusPayload(
            state=JobState.active,
            job_id=job_id,
            workflow_id="instrument/namespace/workflow/1",
        )

        assert message.state == JobState.active
        assert message.job_id == job_id
        assert message.workflow_id == "instrument/namespace/workflow/1"
        assert message.warning is None
        assert message.error is None
        assert message.start_time is None
        assert message.end_time is None

    def test_payload_creation_complete(self):
        """Test creating JobStatusPayload with all fields."""
        job_id = JobId(source_name="test", job_number=uuid.uuid4())
        message = JobStatusPayload(
            state=JobState.warning,
            warning="Test warning",
            error="Test error",
            job_id=job_id,
            workflow_id="instrument/namespace/workflow/1",
            start_time=1000000000,
            end_time=2000000000,
        )

        assert message.state == JobState.warning
        assert message.warning == "Test warning"
        assert message.error == "Test error"
        assert message.start_time == 1000000000
        assert message.end_time == 2000000000


class TestJobStatusMessage:
    def test_from_job_status_minimal(self):
        """Test converting minimal JobStatus to JobStatusMessage."""
        job_status = make_job_status()
        status_msg = JobStatusMessage.from_job_status(job_status)

        assert status_msg.software_name == "livedata"
        assert status_msg.software_version == "0.0.0"
        assert status_msg.service_id.job_id == job_status.job_id
        assert status_msg.host_name == ""
        assert status_msg.process_id == 0
        assert status_msg.update_interval == 1000

        assert status_msg.status_json.status == NicosStatus.OK
        assert status_msg.status_json.message.state == JobState.active
        assert status_msg.status_json.message.job_id == job_status.job_id
        assert status_msg.status_json.message.workflow_id == str(job_status.workflow_id)

    def test_from_job_status_with_error(self):
        """Test converting JobStatus with error to JobStatusMessage."""
        job_status = make_job_status(
            state=JobState.error, error_message="Test error message"
        )
        status_msg = JobStatusMessage.from_job_status(job_status)

        assert status_msg.status_json.status == NicosStatus.ERROR
        assert status_msg.status_json.message.state == JobState.error
        assert status_msg.status_json.message.error == "Test error message"

    def test_from_job_status_with_warning(self):
        """Test converting JobStatus with warning to JobStatusMessage."""
        job_status = make_job_status(
            state=JobState.warning, warning_message="Test warning message"
        )
        status_msg = JobStatusMessage.from_job_status(job_status)

        assert status_msg.status_json.status == NicosStatus.WARNING
        assert status_msg.status_json.message.state == JobState.warning
        assert status_msg.status_json.message.warning == "Test warning message"

    def test_from_job_status_with_times(self):
        """Test converting JobStatus with start/end times to JobStatusMessage."""
        job_status = make_job_status(start_time=1000000000, end_time=2000000000)
        status_msg = JobStatusMessage.from_job_status(job_status)

        assert status_msg.status_json.message.start_time == 1000000000
        assert status_msg.status_json.message.end_time == 2000000000

    def test_to_job_status_minimal(self):
        """Test converting minimal JobStatusMessage to JobStatus."""
        job_id = JobId(source_name="test", job_number=uuid.uuid4())
        workflow_id = WorkflowId(
            instrument="test", namespace="ns", name="wf", version=1
        )

        status_msg = JobStatusMessage(
            service_id=ServiceId.from_job_id(job_id),
            status_json=JobStatusJSON(
                status=NicosStatus.OK,
                message=JobStatusPayload(
                    state=JobState.active, job_id=job_id, workflow_id=str(workflow_id)
                ),
            ),
        )

        job_status = status_msg.to_job_status()

        assert job_status.job_id == job_id
        assert job_status.workflow_id == workflow_id
        assert job_status.state == JobState.active
        assert job_status.error_message is None
        assert job_status.warning_message is None
        assert job_status.start_time is None
        assert job_status.end_time is None

    def test_to_job_status_complete(self):
        """Test converting complete JobStatusMessage to JobStatus."""
        job_id = JobId(source_name="test", job_number=uuid.uuid4())
        workflow_id = WorkflowId(
            instrument="test", namespace="ns", name="wf", version=1
        )

        status_msg = JobStatusMessage(
            service_id=ServiceId.from_job_id(job_id),
            status_json=JobStatusJSON(
                status=NicosStatus.WARNING,
                message=JobStatusPayload(
                    state=JobState.warning,
                    warning="Test warning",
                    error="Test error",
                    job_id=job_id,
                    workflow_id=str(workflow_id),
                    start_time=1000000000,
                    end_time=2000000000,
                ),
            ),
        )

        job_status = status_msg.to_job_status()

        assert job_status.job_id == job_id
        assert job_status.workflow_id == workflow_id
        assert job_status.state == JobState.warning
        assert job_status.error_message == "Test error"
        assert job_status.warning_message == "Test warning"
        assert job_status.start_time == 1000000000
        assert job_status.end_time == 2000000000

    def test_service_id_validation_from_string(self):
        """Test that service_id field accepts string input and converts to ServiceId."""
        job_number = uuid.uuid4()
        service_id_str = f"detector_1:{job_number}"

        # Create JobStatusMessage with string service_id
        data = {
            "service_id": service_id_str,
            "status_json": {
                "status": NicosStatus.OK,
                "message": {
                    "state": JobState.active,
                    "job_id": {"source_name": "detector_1", "job_number": job_number},
                    "workflow_id": "inst/ns/wf/1",
                },
            },
        }

        status_msg = JobStatusMessage.model_validate(data)
        assert isinstance(status_msg.service_id, ServiceId)
        assert status_msg.service_id.job_id.source_name == "detector_1"
        assert status_msg.service_id.job_id.job_number == job_number

    def test_service_id_serialization(self):
        """Test that service_id is serialized to string format."""
        job_id = JobId(source_name="test", job_number=uuid.uuid4())
        status_msg = JobStatusMessage(
            service_id=ServiceId.from_job_id(job_id),
            status_json=JobStatusJSON(
                status=NicosStatus.OK,
                message=JobStatusPayload(
                    state=JobState.active, job_id=job_id, workflow_id="inst/ns/wf/1"
                ),
            ),
        )

        # Serialize to dict and check service_id format
        data = status_msg.model_dump()
        expected_service_id = f"{job_id.source_name}:{job_id.job_number}"
        assert data["service_id"] == expected_service_id


class TestRoundTripConversion:
    """Test round-trip conversion between JobStatus and JobStatusMessage."""

    def test_round_trip_minimal_job_status(self):
        """Test round-trip conversion of minimal JobStatus."""
        original = make_job_status()

        # Convert to JobStatusMessage and back
        status_msg = JobStatusMessage.from_job_status(original)
        converted = status_msg.to_job_status()

        assert converted.job_id == original.job_id
        assert converted.workflow_id == original.workflow_id
        assert converted.state == original.state
        assert converted.error_message == original.error_message
        assert converted.warning_message == original.warning_message
        assert converted.start_time == original.start_time
        assert converted.end_time == original.end_time

    def test_round_trip_all_job_states(self):
        """Test round-trip conversion for all JobState values."""
        for state in JobState:
            original = make_job_status(state=state)

            status_msg = JobStatusMessage.from_job_status(original)
            converted = status_msg.to_job_status()

            assert converted.state == original.state, f"Failed for state {state}"

    def test_round_trip_with_error_message(self):
        """Test round-trip conversion with error message."""
        original = make_job_status(
            state=JobState.error, error_message="Critical error occurred"
        )

        status_msg = JobStatusMessage.from_job_status(original)
        converted = status_msg.to_job_status()

        assert converted.error_message == original.error_message
        assert converted.state == original.state

    def test_round_trip_with_warning_message(self):
        """Test round-trip conversion with warning message."""
        original = make_job_status(
            state=JobState.warning, warning_message="Warning: potential issue detected"
        )

        status_msg = JobStatusMessage.from_job_status(original)
        converted = status_msg.to_job_status()

        assert converted.warning_message == original.warning_message
        assert converted.state == original.state

    def test_round_trip_with_both_messages(self):
        """Test round-trip conversion with both error and warning messages."""
        original = make_job_status(
            state=JobState.error,
            error_message="Error occurred",
            warning_message="Warning was issued earlier",
        )

        status_msg = JobStatusMessage.from_job_status(original)
        converted = status_msg.to_job_status()

        assert converted.error_message == original.error_message
        assert converted.warning_message == original.warning_message
        assert converted.state == original.state

    def test_round_trip_with_timestamps(self):
        """Test round-trip conversion with start and end times."""
        original = make_job_status(start_time=1000000000, end_time=2000000000)

        status_msg = JobStatusMessage.from_job_status(original)
        converted = status_msg.to_job_status()

        assert converted.start_time == original.start_time
        assert converted.end_time == original.end_time

    def test_round_trip_complex_workflow_id(self):
        """Test round-trip conversion with complex WorkflowId."""
        workflow_id = WorkflowId(
            instrument="complex-instrument-name",
            namespace="special_namespace",
            name="workflow_with_underscores",
            version=42,
        )
        original = make_job_status(workflow_id=workflow_id)

        status_msg = JobStatusMessage.from_job_status(original)
        converted = status_msg.to_job_status()

        assert converted.workflow_id == original.workflow_id

    def test_round_trip_complex_job_id(self):
        """Test round-trip conversion with complex JobId."""
        job_id = JobId(source_name="complex:source:name", job_number=uuid.uuid4())
        original = make_job_status(job_id=job_id)

        status_msg = JobStatusMessage.from_job_status(original)
        converted = status_msg.to_job_status()

        assert converted.job_id == original.job_id


class TestX5F2Integration:
    """Test integration with x5f2 serialization/deserialization."""

    def test_x5f2_round_trip_minimal(self):
        """Test round-trip conversion through x5f2 with minimal JobStatus."""
        original = make_job_status()

        # Convert to x5f2 and back
        x5f2_data = job_status_to_x5f2(original)
        converted = x5f2_to_job_status(x5f2_data)

        assert converted.job_id == original.job_id
        assert converted.workflow_id == original.workflow_id
        assert converted.state == original.state
        assert converted.error_message == original.error_message
        assert converted.warning_message == original.warning_message
        assert converted.start_time == original.start_time
        assert converted.end_time == original.end_time

    def test_x5f2_round_trip_all_states(self):
        """Test x5f2 round-trip for all JobState values."""
        for state in JobState:
            original = make_job_status(state=state)

            x5f2_data = job_status_to_x5f2(original)
            converted = x5f2_to_job_status(x5f2_data)

            assert converted.state == original.state, f"Failed for state {state}"

    def test_x5f2_round_trip_with_messages(self):
        """Test x5f2 round-trip with error and warning messages."""
        original = make_job_status(
            state=JobState.warning,
            error_message="Previous error",
            warning_message="Current warning",
        )

        x5f2_data = job_status_to_x5f2(original)
        converted = x5f2_to_job_status(x5f2_data)

        assert converted.error_message == original.error_message
        assert converted.warning_message == original.warning_message
        assert converted.state == original.state

    def test_x5f2_round_trip_with_timestamps(self):
        """Test x5f2 round-trip with timestamps."""
        original = make_job_status(
            start_time=1640995200000000000,  # 2022-01-01 00:00:00 UTC in nanoseconds
            end_time=1640995800000000000,  # 2022-01-01 00:10:00 UTC in nanoseconds
        )

        x5f2_data = job_status_to_x5f2(original)
        converted = x5f2_to_job_status(x5f2_data)

        assert converted.start_time == original.start_time
        assert converted.end_time == original.end_time

    def test_x5f2_serialization_is_bytes(self):
        """Test that x5f2 serialization produces bytes."""
        job_status = make_job_status()
        x5f2_data = job_status_to_x5f2(job_status)

        assert isinstance(x5f2_data, bytes)
        assert len(x5f2_data) > 0

    def test_x5f2_deserialization_from_bytes(self):
        """Test that x5f2 deserialization works with raw bytes."""
        job_status = make_job_status()

        # First serialize to get reference data
        status_msg = JobStatusMessage.from_job_status(job_status)
        expected_data = status_msg.model_dump()

        # Serialize manually using streaming_data_types
        x5f2_data = serialise_x5f2(**expected_data)

        # Deserialize using our function
        converted = x5f2_to_job_status(x5f2_data)

        assert converted.job_id == job_status.job_id
        assert converted.workflow_id == job_status.workflow_id
        assert converted.state == job_status.state

    def test_x5f2_with_unicode_messages(self):
        """Test x5f2 handling of unicode characters in messages."""
        original = make_job_status(
            state=JobState.error,
            error_message="Error with unicode: åäö 中文 🚀",
            warning_message="Warning with unicode: ñáéíóú",
        )

        x5f2_data = job_status_to_x5f2(original)
        converted = x5f2_to_job_status(x5f2_data)

        assert converted.error_message == original.error_message
        assert converted.warning_message == original.warning_message

    def test_x5f2_data_compatibility(self):
        """Test that x5f2 data is compatible with direct streaming_data_types usage."""
        job_status = make_job_status(
            state=JobState.active, start_time=1000000000, end_time=2000000000
        )

        # Serialize using our function
        x5f2_data = job_status_to_x5f2(job_status)

        # Deserialize using streaming_data_types directly
        raw_data = deserialise_x5f2(x5f2_data)

        # Validate the structure - raw_data is a namedtuple with attributes
        assert hasattr(raw_data, 'service_id')
        assert hasattr(raw_data, 'status_json')
        assert hasattr(raw_data, 'software_name')
        assert raw_data.software_name == "livedata"

        # Validate nested structure - status_json is a string that needs parsing
        status_json = json.loads(raw_data.status_json)
        assert "status" in status_json
        assert "message" in status_json

        message = status_json["message"]
        assert "state" in message
        assert "job_id" in message
        assert "workflow_id" in message


class TestMessageTypeField:
    """Test that message_type field is correctly included in serialized messages."""

    def test_job_status_payload_has_message_type_field(self):
        """Test that JobStatusPayload includes message_type field."""
        job_id = JobId(source_name="test", job_number=uuid.uuid4())
        payload = JobStatusPayload(
            state=JobState.active,
            job_id=job_id,
            workflow_id="inst/ns/wf/1",
        )

        assert payload.message_type == "job"

        # Verify it's in the serialized JSON
        data = json.loads(payload.model_dump_json())
        assert data["message_type"] == "job"

    def test_job_status_x5f2_includes_message_type(self):
        """Test that x5f2 serialized job status includes message_type in status_json."""
        job_status = JobStatus(
            job_id=JobId(source_name="test", job_number=uuid.uuid4()),
            workflow_id=WorkflowId(
                instrument="test", namespace="ns", name="wf", version=1
            ),
            state=JobState.active,
        )

        x5f2_data = job_status_to_x5f2(job_status)
        raw_data = deserialise_x5f2(x5f2_data)
        status_json = json.loads(raw_data.status_json)

        assert status_json["message"]["message_type"] == "job"


class TestServiceStateToNicosStatus:
    """Test ServiceState to NICOS status mapping."""

    def test_all_service_states_mapped(self):
        """Test that all ServiceState values have corresponding NicosStatus."""
        for state in ServiceState:
            status = service_state_to_nicos_status_constant(state)
            assert isinstance(status, NicosStatus)

    def test_specific_mappings(self):
        """Test specific ServiceState mappings."""
        assert (
            service_state_to_nicos_status_constant(ServiceState.starting)
            == NicosStatus.DISABLED
        )
        assert (
            service_state_to_nicos_status_constant(ServiceState.running)
            == NicosStatus.OK
        )
        assert (
            service_state_to_nicos_status_constant(ServiceState.stopping)
            == NicosStatus.DISABLED
        )
        assert (
            service_state_to_nicos_status_constant(ServiceState.error)
            == NicosStatus.ERROR
        )


class TestServiceServiceId:
    """Test ServiceServiceId parsing and serialization."""

    def test_from_string_valid_format(self):
        """Test parsing valid service_id string."""
        worker_id = str(uuid.uuid4())
        service_id_str = f"dream:data_reduction:{worker_id}"
        service_id = ServiceServiceId.from_string(service_id_str)

        assert service_id.instrument == "dream"
        assert service_id.namespace == "data_reduction"
        assert service_id.worker_id == worker_id

    def test_from_string_invalid_format_no_colons(self):
        """Test error handling for service_id without colons."""
        with pytest.raises(ValueError, match="Invalid service_id format"):
            ServiceServiceId.from_string("invalid")

    def test_from_string_invalid_uuid(self):
        """Test error handling for invalid UUID in worker_id."""
        with pytest.raises(ValueError, match="Invalid service_id format"):
            ServiceServiceId.from_string("dream:data_reduction:not-a-uuid")

    def test_from_service_status(self):
        """Test creating ServiceServiceId from ServiceStatus."""
        worker_id = str(uuid.uuid4())
        status = ServiceStatus(
            instrument="dream",
            namespace="data_reduction",
            worker_id=worker_id,
            state=ServiceState.running,
            started_at=1000000000,
            active_job_count=3,
            messages_processed=1000,
        )
        service_id = ServiceServiceId.from_service_status(status)

        assert service_id.instrument == "dream"
        assert service_id.namespace == "data_reduction"
        assert service_id.worker_id == worker_id

    def test_to_string(self):
        """Test converting ServiceServiceId to string format."""
        worker_id = str(uuid.uuid4())
        service_id = ServiceServiceId(
            instrument="dream", namespace="data_reduction", worker_id=worker_id
        )

        expected = f"dream:data_reduction:{worker_id}"
        assert service_id.to_string() == expected
        assert str(service_id) == expected

    def test_round_trip_string_conversion(self):
        """Test that string conversion is reversible."""
        original = ServiceServiceId(
            instrument="bifrost",
            namespace="monitor_data",
            worker_id=str(uuid.uuid4()),
        )

        # Convert to string and back
        service_id_str = original.to_string()
        parsed = ServiceServiceId.from_string(service_id_str)

        assert parsed.instrument == original.instrument
        assert parsed.namespace == original.namespace
        assert parsed.worker_id == original.worker_id


class TestServiceStatusPayload:
    """Test ServiceStatusPayload model."""

    def test_payload_creation(self):
        """Test creating ServiceStatusPayload with all fields."""
        worker_id = str(uuid.uuid4())
        payload = ServiceStatusPayload(
            instrument="dream",
            namespace="data_reduction",
            worker_id=worker_id,
            state=ServiceState.running,
            started_at=1000000000,
            active_job_count=5,
            messages_processed=10000,
            error=None,
        )

        assert payload.message_type == "service"
        assert payload.instrument == "dream"
        assert payload.namespace == "data_reduction"
        assert payload.worker_id == worker_id
        assert payload.state == ServiceState.running
        assert payload.active_job_count == 5

    def test_payload_with_error(self):
        """Test creating ServiceStatusPayload with error."""
        payload = ServiceStatusPayload(
            instrument="dream",
            namespace="data_reduction",
            worker_id=str(uuid.uuid4()),
            state=ServiceState.error,
            started_at=1000000000,
            active_job_count=0,
            messages_processed=5000,
            error="Connection lost to Kafka",
        )

        assert payload.state == ServiceState.error
        assert payload.error == "Connection lost to Kafka"


class TestServiceStatusMessage:
    """Test ServiceStatusMessage model."""

    def test_from_service_status(self):
        """Test converting ServiceStatus to ServiceStatusMessage."""
        status = make_service_status()
        msg = ServiceStatusMessage.from_service_status(status)

        assert msg.software_name == "livedata"
        assert msg.service_id.instrument == status.instrument
        assert msg.service_id.namespace == status.namespace
        assert msg.service_id.worker_id == status.worker_id
        assert msg.status_json.message.state == status.state
        assert msg.status_json.message.active_job_count == status.active_job_count

    def test_from_service_status_with_metadata(self):
        """Test converting ServiceStatus with custom metadata."""
        status = make_service_status()
        msg = ServiceStatusMessage.from_service_status(
            status,
            software_version="1.2.3",
            host_name="worker-01",
            process_id=12345,
        )

        assert msg.software_version == "1.2.3"
        assert msg.host_name == "worker-01"
        assert msg.process_id == 12345

    def test_to_service_status(self):
        """Test converting ServiceStatusMessage to ServiceStatus."""
        original = make_service_status()
        msg = ServiceStatusMessage.from_service_status(original)
        converted = msg.to_service_status()

        assert converted.instrument == original.instrument
        assert converted.namespace == original.namespace
        assert converted.worker_id == original.worker_id
        assert converted.state == original.state
        assert converted.started_at == original.started_at
        assert converted.active_job_count == original.active_job_count
        assert converted.messages_processed == original.messages_processed

    def test_round_trip_with_shedding_fields(self):
        """Test that shedding fields survive model round-trip."""
        original = make_service_status(
            is_shedding=True, messages_dropped=42, messages_eligible=100
        )
        msg = ServiceStatusMessage.from_service_status(original)
        converted = msg.to_service_status()

        assert converted.is_shedding is True
        assert converted.messages_dropped == 42
        assert converted.messages_eligible == 100

    def test_round_trip_defaults_shedding_fields(self):
        """Test that shedding fields default gracefully."""
        original = make_service_status()
        msg = ServiceStatusMessage.from_service_status(original)
        converted = msg.to_service_status()

        assert converted.is_shedding is False
        assert converted.messages_dropped == 0
        assert converted.messages_eligible == 0


class TestServiceStatusX5F2Integration:
    """Test service status x5f2 serialization/deserialization."""

    def test_service_status_to_x5f2_produces_bytes(self):
        """Test that service_status_to_x5f2 produces bytes."""
        status = make_service_status()
        x5f2_data = service_status_to_x5f2(status)

        assert isinstance(x5f2_data, bytes)
        assert len(x5f2_data) > 0

    def test_service_status_x5f2_round_trip(self):
        """Test round-trip conversion through x5f2."""
        original = make_service_status()
        x5f2_data = service_status_to_x5f2(original)
        converted = x5f2_to_service_status(x5f2_data)

        assert converted.instrument == original.instrument
        assert converted.namespace == original.namespace
        assert converted.worker_id == original.worker_id
        assert converted.state == original.state
        assert converted.started_at == original.started_at
        assert converted.active_job_count == original.active_job_count
        assert converted.messages_processed == original.messages_processed
        assert converted.error == original.error

    def test_service_status_x5f2_round_trip_with_shedding(self):
        """Test x5f2 round-trip includes load shedding fields."""
        original = make_service_status(
            is_shedding=True, messages_dropped=1234, messages_eligible=3000
        )
        x5f2_data = service_status_to_x5f2(original)
        converted = x5f2_to_service_status(x5f2_data)

        assert converted.is_shedding is True
        assert converted.messages_dropped == 1234
        assert converted.messages_eligible == 3000

    def test_service_status_x5f2_with_error(self):
        """Test x5f2 round-trip with error message."""
        original = make_service_status(
            state=ServiceState.error,
            error="Kafka connection lost",
        )
        x5f2_data = service_status_to_x5f2(original)
        converted = x5f2_to_service_status(x5f2_data)

        assert converted.state == ServiceState.error
        assert converted.error == "Kafka connection lost"

    def test_service_status_x5f2_with_metadata(self):
        """Test that metadata is included in x5f2 message."""
        status = make_service_status()
        x5f2_data = service_status_to_x5f2(
            status,
            software_version="1.2.3",
            host_name="worker-node-01.esss.se",
            process_id=12345,
        )

        raw_data = deserialise_x5f2(x5f2_data)
        assert raw_data.software_version == "1.2.3"
        assert raw_data.host_name == "worker-node-01.esss.se"
        assert raw_data.process_id == 12345

    def test_service_status_x5f2_message_type_field(self):
        """Test that message_type='service' is in the status_json."""
        status = make_service_status()
        x5f2_data = service_status_to_x5f2(status)

        raw_data = deserialise_x5f2(x5f2_data)
        status_json = json.loads(raw_data.status_json)

        assert status_json["message"]["message_type"] == "service"

    def test_service_status_x5f2_service_id_format(self):
        """Test that service_id has the correct format."""
        status = make_service_status(
            instrument="dream",
            namespace="data_reduction",
            worker_id="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        )
        x5f2_data = service_status_to_x5f2(status)

        raw_data = deserialise_x5f2(x5f2_data)
        assert (
            raw_data.service_id
            == "dream:data_reduction:7c9e6679-7425-40de-944b-e07fc1f90ae7"
        )

    def test_service_status_x5f2_all_states(self):
        """Test x5f2 round-trip for all ServiceState values."""
        for state in ServiceState:
            original = make_service_status(state=state)
            x5f2_data = service_status_to_x5f2(original)
            converted = x5f2_to_service_status(x5f2_data)
            assert converted.state == state, f"Failed for state {state}"


class TestX5f2ToStatusDiscriminator:
    """Test x5f2_to_status function.

    Tests the discriminator between job and service status messages.
    """

    def test_returns_job_status_for_job_message_with_message_type(self):
        """Test that job status with message_type='job' returns JobStatus."""
        job_status = JobStatus(
            job_id=JobId(source_name="detector1", job_number=uuid.uuid4()),
            workflow_id=WorkflowId(
                instrument="test", namespace="ns", name="wf", version=1
            ),
            state=JobState.active,
        )

        x5f2_data = job_status_to_x5f2(job_status)
        result = x5f2_to_status(x5f2_data)

        assert isinstance(result, JobStatus)
        assert result.job_id == job_status.job_id
        assert result.state == job_status.state

    def test_returns_job_status_for_legacy_message_without_message_type(self):
        """Test backward compatibility of messages without message_type.

        Legacy messages without message_type field should return JobStatus.
        """
        job_id = JobId(source_name="detector1", job_number=uuid.uuid4())
        workflow_id = WorkflowId(
            instrument="test", namespace="ns", name="wf", version=1
        )

        # Manually create x5f2 data WITHOUT message_type field (legacy format)
        status_json = json.dumps(
            {
                "status": 200,
                "message": {
                    # No message_type field - legacy format
                    "state": "active",
                    "warning": None,
                    "error": None,
                    "job_id": {
                        "source_name": job_id.source_name,
                        "job_number": str(job_id.job_number),
                    },
                    "workflow_id": str(workflow_id),
                    "start_time": None,
                    "end_time": None,
                },
            }
        )

        x5f2_data = serialise_x5f2(
            software_name="livedata",
            software_version="0.0.0",
            service_id=f"{job_id.source_name}:{job_id.job_number}",
            host_name="",
            process_id=0,
            update_interval=1000,
            status_json=status_json,
        )

        result = x5f2_to_status(x5f2_data)

        assert isinstance(result, JobStatus)
        assert result.job_id == job_id
        assert result.state == JobState.active

    def test_returns_service_status_for_service_message(self):
        """Test service status with message_type='service'.

        Service messages with message_type='service' should return ServiceStatus.
        """
        worker_id = str(uuid.uuid4())
        service_status = ServiceStatus(
            instrument="dream",
            namespace="data_reduction",
            worker_id=worker_id,
            state=ServiceState.running,
            started_at=1000000000,
            active_job_count=3,
            messages_processed=5000,
        )

        x5f2_data = service_status_to_x5f2(service_status)
        result = x5f2_to_status(x5f2_data)

        assert isinstance(result, ServiceStatus)
        assert result.instrument == service_status.instrument
        assert result.namespace == service_status.namespace
        assert result.worker_id == service_status.worker_id
        assert result.state == service_status.state
