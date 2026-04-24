# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid
from contextlib import contextmanager

import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.core.message import (
    RESPONSES_STREAM_ID,
    STATUS_STREAM_ID,
    Message,
    StreamId,
    StreamKind,
)
from ess.livedata.dashboard.active_job_registry import ActiveJobRegistry
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.orchestrator import Orchestrator
from ess.livedata.dashboard.service_registry import ServiceRegistry


def make_job_number() -> uuid.UUID:
    """Generate a random UUID for job number."""
    return uuid.uuid4()


def make_service_registry() -> ServiceRegistry:
    """Create a service registry for testing."""
    return ServiceRegistry()


class FakeJobOrchestrator:
    """Fake job orchestrator that records processed acknowledgements."""

    def __init__(self):
        self.acknowledgements: list[tuple[str, str, str | None]] = []

    def process_acknowledgement(
        self, message_id: str, response: str, error_message: str | None = None
    ) -> None:
        """Record the processed acknowledgement."""
        self.acknowledgements.append((message_id, response, error_message))


class PermissiveJobRegistry:
    """Accepts every job number; used where filtering is orthogonal to the test."""

    def is_active(self, job_number: uuid.UUID) -> bool:
        return True

    @contextmanager
    def ingestion_guard(self):
        yield


def _make_orchestrator(
    *,
    message_source,
    data_service,
    job_service,
    service_registry=None,
    job_orchestrator=None,
    active_job_registry=None,
) -> Orchestrator:
    return Orchestrator(
        message_source=message_source,
        data_service=data_service,
        job_service=job_service,
        service_registry=service_registry or ServiceRegistry(),
        job_orchestrator=job_orchestrator or FakeJobOrchestrator(),
        active_job_registry=active_job_registry or PermissiveJobRegistry(),
    )


class FakeMessageSource:
    """Simple fake message source for testing."""

    def __init__(self):
        self._messages = []

    def add_message(self, stream_name: str, data: sc.DataArray, timestamp: int = 1000):
        """Add a message to be returned by get_messages."""
        message = Message(
            timestamp=timestamp,
            stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name=stream_name),
            value=data,
        )
        self._messages.append(message)

    def add_config_message(self, config_data: dict, timestamp: int = 1000):
        """Add a config message to be returned by get_messages."""
        message = Message(
            timestamp=timestamp,
            stream=RESPONSES_STREAM_ID,
            value=config_data,
        )
        self._messages.append(message)

    def add_status_message(self, status_data: dict, timestamp: int = 1000):
        """Add a status message to be returned by get_messages."""
        message = Message(
            timestamp=timestamp,
            stream=STATUS_STREAM_ID,
            value=status_data,
        )
        self._messages.append(message)

    def get_messages(self):
        """Return all messages and clear the internal list."""
        messages = self._messages.copy()
        self._messages.clear()
        return messages


class TestOrchestrator:
    def test_update_with_no_messages(self) -> None:
        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
        )

        orchestrator.update()

        assert len(data_service) == 0

    def test_update_with_single_message(self) -> None:
        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
        )

        workflow_id = WorkflowId(
            instrument="test_instrument",
            name="test_workflow",
            version=1,
        )
        job_id = JobId(source_name="detector1", job_number=make_job_number())
        result_key = ResultKey(
            workflow_id=workflow_id, job_id=job_id, output_name='result'
        )

        data = sc.DataArray(sc.array(dims=['x'], values=[1, 2, 3]))
        source.add_message(result_key.model_dump_json(), data)

        orchestrator.update()

        assert result_key in data_service
        assert sc.identical(data_service[result_key], data)

    def test_update_with_multiple_messages(self) -> None:
        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
        )

        workflow_id1 = WorkflowId(
            instrument="test_instrument",
            name="workflow1",
            version=1,
        )
        workflow_id2 = WorkflowId(
            instrument="test_instrument",
            name="workflow2",
            version=1,
        )

        job_id1 = JobId(source_name="detector1", job_number=make_job_number())
        job_id2 = JobId(source_name="detector2", job_number=make_job_number())

        result_key1 = ResultKey(
            workflow_id=workflow_id1, job_id=job_id1, output_name='result'
        )
        result_key2 = ResultKey(
            workflow_id=workflow_id2, job_id=job_id2, output_name='result'
        )

        data1 = sc.DataArray(sc.array(dims=['x'], values=[1, 2]))
        data2 = sc.DataArray(sc.array(dims=['y'], values=[3, 4, 5]))

        source.add_message(result_key1.model_dump_json(), data1)
        source.add_message(result_key2.model_dump_json(), data2)

        orchestrator.update()

        assert result_key1 in data_service
        assert result_key2 in data_service
        assert sc.identical(data_service[result_key1], data1)
        assert sc.identical(data_service[result_key2], data2)

    def test_update_overwrites_existing_data(self) -> None:
        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
        )

        workflow_id = WorkflowId(
            instrument="test_instrument",
            name="test_workflow",
            version=1,
        )
        job_id = JobId(source_name="detector1", job_number=make_job_number())
        result_key = ResultKey(
            workflow_id=workflow_id, job_id=job_id, output_name='result'
        )

        original_data = sc.DataArray(sc.array(dims=['x'], values=[1, 2]))
        new_data = sc.DataArray(sc.array(dims=['y'], values=[3, 4, 5]))

        # Add initial data
        source.add_message(result_key.model_dump_json(), original_data)
        orchestrator.update()
        assert sc.identical(data_service[result_key], original_data)

        # Overwrite with new data
        source.add_message(result_key.model_dump_json(), new_data)
        orchestrator.update()
        assert sc.identical(data_service[result_key], new_data)

    def test_update_with_output_name(self) -> None:
        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
        )

        workflow_id = WorkflowId(
            instrument="test_instrument",
            name="test_workflow",
            version=1,
        )
        job_id = JobId(source_name="detector1", job_number=make_job_number())
        result_key = ResultKey(
            workflow_id=workflow_id, job_id=job_id, output_name="processed_data"
        )

        data = sc.DataArray(sc.array(dims=['x'], values=[1, 2, 3]))
        source.add_message(result_key.model_dump_json(), data)

        orchestrator.update()

        assert result_key in data_service
        assert sc.identical(data_service[result_key], data)

    def test_forward_with_valid_result_key(self) -> None:
        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
        )

        workflow_id = WorkflowId(
            instrument="test_instrument",
            name="test_workflow",
            version=1,
        )
        job_id = JobId(source_name="detector1", job_number=make_job_number())
        result_key = ResultKey(
            workflow_id=workflow_id, job_id=job_id, output_name='result'
        )

        data = sc.DataArray(sc.array(dims=['x'], values=[1, 2, 3]))

        orchestrator.forward(_data_stream_id(result_key), data)

        assert result_key in data_service
        assert sc.identical(data_service[result_key], data)

    def test_forward_with_invalid_json(self) -> None:
        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
        )

        data = sc.DataArray(sc.array(dims=['x'], values=[1, 2, 3]))

        # JSON parsing or Pydantic validation error
        with pytest.raises(ValueError, match="Invalid JSON"):
            orchestrator.forward(
                StreamId(kind=StreamKind.LIVEDATA_DATA, name="invalid_json"), data
            )

    def test_forward_with_different_data_types(self) -> None:
        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
        )

        workflow_id = WorkflowId(
            instrument="test_instrument",
            name="test_workflow",
            version=1,
        )

        job_id1 = JobId(source_name="detector1", job_number=make_job_number())
        job_id2 = JobId(source_name="detector1", job_number=make_job_number())
        job_id3 = JobId(source_name="detector1", job_number=make_job_number())

        result_key1 = ResultKey(
            workflow_id=workflow_id, job_id=job_id1, output_name="int_data"
        )
        result_key2 = ResultKey(
            workflow_id=workflow_id, job_id=job_id2, output_name="float_data"
        )
        result_key3 = ResultKey(
            workflow_id=workflow_id, job_id=job_id3, output_name="string_data"
        )

        int_data = sc.DataArray(sc.array(dims=['x'], values=[1, 2, 3]))
        float_data = sc.DataArray(sc.array(dims=['y'], values=[1.5, 2.5]))
        string_data = sc.DataArray(sc.array(dims=['z'], values=['a', 'b']))

        orchestrator.forward(_data_stream_id(result_key1), int_data)
        orchestrator.forward(_data_stream_id(result_key2), float_data)
        orchestrator.forward(_data_stream_id(result_key3), string_data)

        assert sc.identical(data_service[result_key1], int_data)
        assert sc.identical(data_service[result_key2], float_data)
        assert sc.identical(data_service[result_key3], string_data)

    def test_transaction_mechanism(self) -> None:
        """Test that updates are batched in transactions."""
        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
        )

        # Track transaction calls
        transaction_started = False
        original_transaction = data_service.transaction

        def mock_transaction():
            nonlocal transaction_started
            transaction_started = True
            return original_transaction()

        data_service.transaction = mock_transaction

        workflow_id = WorkflowId(
            instrument="test_instrument",
            name="test_workflow",
            version=1,
        )
        job_id = JobId(source_name="detector1", job_number=make_job_number())
        result_key = ResultKey(
            workflow_id=workflow_id, job_id=job_id, output_name='result'
        )

        data = sc.DataArray(sc.array(dims=['x'], values=[1, 2, 3]))
        source.add_message(result_key.model_dump_json(), data)

        orchestrator.update()

        assert transaction_started
        assert result_key in data_service


class TestOrchestratorAcknowledgementProcessing:
    def test_forward_with_ack_message(self) -> None:
        """Test that acknowledgement messages are forwarded to job orchestrator."""
        from ess.livedata.config.acknowledgement import (
            AcknowledgementResponse,
            CommandAcknowledgement,
        )

        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        job_orchestrator = FakeJobOrchestrator()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
            job_orchestrator=job_orchestrator,
        )

        ack = CommandAcknowledgement(
            message_id="test-uuid",
            device="detector1",
            response=AcknowledgementResponse.ACK,
        )

        orchestrator.forward(RESPONSES_STREAM_ID, ack)

        assert len(job_orchestrator.acknowledgements) == 1
        assert job_orchestrator.acknowledgements[0] == ("test-uuid", "ACK", None)

    def test_forward_with_error_ack_message(self) -> None:
        """Test that error acknowledgement messages include error message."""
        from ess.livedata.config.acknowledgement import (
            AcknowledgementResponse,
            CommandAcknowledgement,
        )

        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        job_orchestrator = FakeJobOrchestrator()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
            job_orchestrator=job_orchestrator,
        )

        ack = CommandAcknowledgement(
            message_id="test-uuid",
            device="detector1",
            response=AcknowledgementResponse.ERR,
            message="Workflow not found",
        )

        orchestrator.forward(RESPONSES_STREAM_ID, ack)

        assert len(job_orchestrator.acknowledgements) == 1
        assert job_orchestrator.acknowledgements[0] == (
            "test-uuid",
            "ERR",
            "Workflow not found",
        )


class TestOrchestratorServiceStatusRouting:
    """Test that ServiceStatus messages are routed to the ServiceRegistry."""

    def test_forward_routes_service_status_to_registry(self) -> None:
        """Test that ServiceStatus messages are routed to service registry."""
        from ess.livedata.core.job import ServiceState, ServiceStatus

        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        service_registry = make_service_registry()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
            service_registry=service_registry,
        )

        status = ServiceStatus(
            instrument="dream",
            service_name="test_namespace",
            worker_id="worker123",
            state=ServiceState.running,
            started_at=1000000000,
            active_job_count=2,
        )

        orchestrator.forward(STATUS_STREAM_ID, status)

        # Verify the service registry received the status
        assert len(service_registry.worker_statuses) == 1
        worker_key = "dream:test_namespace:worker123"
        assert worker_key in service_registry.worker_statuses
        assert service_registry.worker_statuses[worker_key] == status

    def test_forward_routes_job_status_to_job_service(self) -> None:
        """Test that JobStatus messages are routed to job service."""
        from ess.livedata.config.workflow_spec import JobId, WorkflowId
        from ess.livedata.core.job import JobState, JobStatus

        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        service_registry = make_service_registry()
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
            service_registry=service_registry,
        )

        workflow_id = WorkflowId(
            instrument="dream",
            name="test_workflow",
            version=1,
        )
        job_id = JobId(source_name="detector1", job_number=make_job_number())
        job_status = JobStatus(
            job_id=job_id,
            workflow_id=workflow_id,
            state=JobState.active,
        )

        orchestrator.forward(STATUS_STREAM_ID, job_status)

        # Verify job service received the status (not service registry)
        assert len(service_registry.worker_statuses) == 0
        assert len(job_service.job_statuses) == 1


class TestOrchestratorJobFiltering:
    """Test that data from inactive or unknown jobs is filtered out."""

    def _make_orchestrator(self, active_job_numbers):
        source = FakeMessageSource()
        data_service = DataService()
        job_service = JobService()
        registry = ActiveJobRegistry(data_service=data_service, job_service=job_service)
        for jn in active_job_numbers:
            registry.restore(jn)
        orchestrator = _make_orchestrator(
            message_source=source,
            data_service=data_service,
            job_service=job_service,
            active_job_registry=registry,
        )
        return orchestrator, data_service, job_service

    def test_accepts_data_for_active_job(self) -> None:
        job_number = make_job_number()
        orchestrator, data_service, _ = self._make_orchestrator([job_number])

        workflow_id = WorkflowId(instrument="test", name="wf", version=1)
        job_id = JobId(source_name="det1", job_number=job_number)
        result_key = ResultKey(
            workflow_id=workflow_id, job_id=job_id, output_name="result"
        )
        data = sc.DataArray(sc.array(dims=['x'], values=[1, 2]))
        orchestrator.forward(_data_stream_id(result_key), data)

        assert result_key in data_service

    def test_discards_data_for_inactive_job(self) -> None:
        active_number = make_job_number()
        inactive_number = make_job_number()
        orchestrator, data_service, _ = self._make_orchestrator([active_number])

        workflow_id = WorkflowId(instrument="test", name="wf", version=1)
        job_id = JobId(source_name="det1", job_number=inactive_number)
        result_key = ResultKey(
            workflow_id=workflow_id, job_id=job_id, output_name="result"
        )
        data = sc.DataArray(sc.array(dims=['x'], values=[1, 2]))
        orchestrator.forward(_data_stream_id(result_key), data)

        assert result_key not in data_service

    def test_accepts_job_status_for_active_job(self) -> None:
        from ess.livedata.core.job import JobState, JobStatus

        job_number = make_job_number()
        orchestrator, _, job_service = self._make_orchestrator([job_number])

        workflow_id = WorkflowId(instrument="test", name="wf", version=1)
        job_id = JobId(source_name="det1", job_number=job_number)
        status = JobStatus(
            job_id=job_id, workflow_id=workflow_id, state=JobState.active
        )
        orchestrator.forward(STATUS_STREAM_ID, status)

        assert len(job_service.job_statuses) == 1

    def test_discards_job_status_for_inactive_job(self) -> None:
        from ess.livedata.core.job import JobState, JobStatus

        active_number = make_job_number()
        inactive_number = make_job_number()
        orchestrator, _, job_service = self._make_orchestrator([active_number])

        workflow_id = WorkflowId(instrument="test", name="wf", version=1)
        job_id = JobId(source_name="det1", job_number=inactive_number)
        status = JobStatus(
            job_id=job_id, workflow_id=workflow_id, state=JobState.active
        )
        orchestrator.forward(STATUS_STREAM_ID, status)

        assert len(job_service.job_statuses) == 0


def _data_stream_id(key: ResultKey) -> StreamId:
    return StreamId(kind=StreamKind.LIVEDATA_DATA, name=key.model_dump_json())
