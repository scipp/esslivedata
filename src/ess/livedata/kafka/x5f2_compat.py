# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Compatibility with x5f2 status messages used by ECDC and interop with NICOS."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Literal

import pydantic
from pydantic import field_serializer, field_validator
from streaming_data_types import deserialise_x5f2, serialise_x5f2

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.job import (
    JobId,
    JobState,
    JobStatus,
    ServiceState,
    ServiceStatus,
)


class ServiceId(pydantic.BaseModel):
    """
    Helper class for handling service_id in source_name:job_number format.

    NICOS expects a format of device_name:signal_name. In our case the device is our
    internal name for the stream (often the source_name, but sometimes derived also from
    the Kafka topic name). We use the job_number (UUID) as the signal name to report one
    status per job. Note that each job may produce multiple outputs.
    """

    job_id: JobId = pydantic.Field(description="Job identifier")

    @classmethod
    def from_string(cls, service_id: str) -> ServiceId:
        """Parse service_id string in format 'source_name:job_number'."""
        try:
            # Split on the last colon to handle source names with colons
            parts = service_id.rsplit(':', 1)
            if len(parts) != 2:
                raise ValueError("No colon separator found")
            source_name, job_number_str = parts
            job_id = JobId(
                source_name=source_name, job_number=uuid.UUID(job_number_str)
            )
            return cls(job_id=job_id)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid service_id format '{service_id}'. "
                "Expected 'source_name:job_number'"
            ) from e

    @classmethod
    def from_job_id(cls, job_id: JobId) -> ServiceId:
        """Create ServiceId from JobId."""
        return cls(job_id=job_id)

    def to_string(self) -> str:
        """Convert to service_id string in format 'source_name:job_number'."""
        return f"{self.job_id.source_name}:{self.job_id.job_number}"

    def __str__(self) -> str:
        return self.to_string()


class JobStatusPayload(pydantic.BaseModel):
    """The 'message' field within status_json for job heartbeats."""

    message_type: Literal["job"] = pydantic.Field(
        default="job", description="Message type for explicit typing"
    )
    state: JobState = pydantic.Field(description="Current state of the job")
    warning: str | None = pydantic.Field(
        default=None, description="Warning message if any"
    )
    error: str | None = pydantic.Field(default=None, description="Error message if any")
    job_id: JobId = pydantic.Field(description="Job identifier")
    workflow_id: str = pydantic.Field(description="Workflow identifier as string")
    start_time: int | None = pydantic.Field(
        default=None, description="Job start time in nanoseconds since epoch"
    )
    end_time: int | None = pydantic.Field(
        default=None, description="Job end time in nanoseconds since epoch"
    )


class NicosStatus(int, Enum):
    OK = 200
    WARNING = 210
    BUSY = 220  # not used by us
    NOTREACHED = 230  # not used by us
    DISABLED = 235
    ERROR = 240
    UNKNOWN = 999


def job_state_to_nicos_status_constant(state: JobState) -> NicosStatus:
    match state:
        case JobState.active:
            return NicosStatus.OK
        case JobState.error:
            return NicosStatus.ERROR
        case JobState.finishing:
            return NicosStatus.OK
        case JobState.paused:
            return NicosStatus.DISABLED
        case JobState.scheduled:
            return NicosStatus.DISABLED
        case JobState.stopped:
            return NicosStatus.DISABLED
        case JobState.warning:
            return NicosStatus.WARNING
        case _:
            return NicosStatus.UNKNOWN


def service_state_to_nicos_status_constant(state: ServiceState) -> NicosStatus:
    """Map ServiceState to NICOS status code."""
    match state:
        case ServiceState.starting:
            return NicosStatus.DISABLED
        case ServiceState.running:
            return NicosStatus.OK
        case ServiceState.stopping:
            return NicosStatus.DISABLED
        case ServiceState.stopped:
            return NicosStatus.DISABLED
        case ServiceState.error:
            return NicosStatus.ERROR
        case _:
            return NicosStatus.UNKNOWN


class ServiceServiceId(pydantic.BaseModel):
    """
    Helper class for handling service_id in instrument:namespace:worker_id format.

    NICOS expects a format of device_name:signal_name. For service heartbeats, we use
    instrument:namespace:worker_uuid to uniquely identify each worker.
    """

    instrument: str = pydantic.Field(description="Instrument name")
    namespace: str = pydantic.Field(description="Service namespace")
    worker_id: str = pydantic.Field(description="Worker UUID as string")

    @classmethod
    def from_string(cls, service_id: str) -> ServiceServiceId:
        """Parse service_id string in format 'instrument:namespace:worker_id'."""
        try:
            # Split on colons - expect exactly 3 parts
            parts = service_id.rsplit(':', 2)
            if len(parts) != 3:
                raise ValueError("Expected 3 colon-separated parts")
            instrument, namespace, worker_id = parts
            # Validate worker_id is a valid UUID
            uuid.UUID(worker_id)
            return cls(instrument=instrument, namespace=namespace, worker_id=worker_id)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid service_id format '{service_id}'. "
                "Expected 'instrument:namespace:worker_uuid'"
            ) from e

    @classmethod
    def from_service_status(cls, status: ServiceStatus) -> ServiceServiceId:
        """Create ServiceServiceId from ServiceStatus."""
        return cls(
            instrument=status.instrument,
            namespace=status.namespace,
            worker_id=status.worker_id,
        )

    def to_string(self) -> str:
        """Convert to service_id string in format 'instrument:namespace:worker_id'."""
        return f"{self.instrument}:{self.namespace}:{self.worker_id}"

    def __str__(self) -> str:
        return self.to_string()


class ServiceStatusPayload(pydantic.BaseModel):
    """The 'message' field within status_json for service heartbeats."""

    message_type: Literal["service"] = pydantic.Field(
        default="service", description="Message type for explicit typing"
    )
    instrument: str = pydantic.Field(description="Instrument name")
    namespace: str = pydantic.Field(description="Service namespace")
    worker_id: str = pydantic.Field(description="Worker UUID as string")
    state: ServiceState = pydantic.Field(description="Current state of the service")
    started_at: int = pydantic.Field(description="Service start time in nanoseconds")
    active_job_count: int = pydantic.Field(description="Number of active jobs")
    messages_processed: int = pydantic.Field(
        description="Total messages processed since startup"
    )
    error: str | None = pydantic.Field(default=None, description="Error message if any")


class ServiceStatusJSON(pydantic.BaseModel):
    """Status JSON for service heartbeats, corresponding to x5f2 'status_json'."""

    status: NicosStatus = pydantic.Field(description="Status code")
    message: ServiceStatusPayload = pydantic.Field(description="Status message")


class ServiceStatusMessage(pydantic.BaseModel):
    """
    Service status message model corresponding to x5f2 named tuple structure.

    Used for service-level heartbeats independent of job status.
    """

    software_name: str = pydantic.Field(
        default='livedata', description="Name of the software"
    )
    software_version: str = pydantic.Field(
        default='0.0.0', description="Version of the software"
    )
    service_id: ServiceServiceId = pydantic.Field(
        description="Service identifier as instrument:namespace:worker_id"
    )
    host_name: str = pydantic.Field(default='', description="Host name")
    process_id: int = pydantic.Field(default=0, description="Process ID")
    update_interval: int = pydantic.Field(
        default=2000, description="Update interval in milliseconds"
    )
    status_json: ServiceStatusJSON = pydantic.Field(
        description="Status information in JSON format"
    )

    @field_validator('service_id', mode='before')
    @classmethod
    def validate_service_id(cls, v):
        """Convert string service_id to ServiceServiceId object during validation."""
        if isinstance(v, str):
            return ServiceServiceId.from_string(v)
        return v

    @field_validator('status_json', mode='before')
    @classmethod
    def validate_status_json(cls, v):
        """Convert JSON string to ServiceStatusJSON object during validation."""
        if isinstance(v, str):
            return ServiceStatusJSON.model_validate_json(v)
        return v

    @field_serializer('service_id')
    def serialize_service_id(self, service_id: ServiceServiceId) -> str:
        """Serialize ServiceServiceId to string format."""
        return service_id.to_string()

    @field_serializer('status_json')
    def serialize_status_json(self, status_json: ServiceStatusJSON) -> str:
        """Serialize ServiceStatusJSON to JSON string format."""
        return status_json.model_dump_json()

    @staticmethod
    def from_service_status(
        status: ServiceStatus,
        *,
        software_version: str = '0.0.0',
        host_name: str = '',
        process_id: int = 0,
    ) -> ServiceStatusMessage:
        """Create ServiceStatusMessage from ServiceStatus."""
        return ServiceStatusMessage(
            software_version=software_version,
            host_name=host_name,
            process_id=process_id,
            service_id=ServiceServiceId.from_service_status(status),
            status_json=ServiceStatusJSON(
                status=service_state_to_nicos_status_constant(status.state),
                message=ServiceStatusPayload(
                    instrument=status.instrument,
                    namespace=status.namespace,
                    worker_id=status.worker_id,
                    state=status.state,
                    started_at=status.started_at,
                    active_job_count=status.active_job_count,
                    messages_processed=status.messages_processed,
                    error=status.error,
                ),
            ),
        )

    def to_service_status(self) -> ServiceStatus:
        """Convert ServiceStatusMessage to ServiceStatus."""
        message = self.status_json.message
        return ServiceStatus(
            instrument=message.instrument,
            namespace=message.namespace,
            worker_id=message.worker_id,
            state=message.state,
            started_at=message.started_at,
            active_job_count=message.active_job_count,
            messages_processed=message.messages_processed,
            error=message.error,
        )


class JobStatusJSON(pydantic.BaseModel):
    """Status JSON model for job heartbeats, corresponding to 'status_json' in x5f2."""

    status: NicosStatus = pydantic.Field(description="Status code")
    message: JobStatusPayload = pydantic.Field(description="Status message")


class JobStatusMessage(pydantic.BaseModel):
    """
    Job status message model corresponding to x5f2 named tuple structure.

    Field types are specific to encode our job status correctly in the generic x5f2
    format used by the streaming_data_types library.
    """

    software_name: str = pydantic.Field(
        default='livedata', description="Name of the software"
    )
    software_version: str = pydantic.Field(
        default='0.0.0', description="Version of the software"
    )
    service_id: ServiceId = pydantic.Field(
        description="Service identifier defined as source_name:job_number"
    )
    host_name: str = pydantic.Field(default='', description="Host name")
    process_id: int = pydantic.Field(default=0, description="Process ID")
    update_interval: int = pydantic.Field(
        default=1000, description="Update interval in milliseconds"
    )
    status_json: JobStatusJSON = pydantic.Field(
        description="Status information in JSON format"
    )

    @field_validator('service_id', mode='before')
    @classmethod
    def validate_service_id(cls, v):
        """Convert string service_id to ServiceId object during validation."""
        if isinstance(v, str):
            return ServiceId.from_string(v)
        return v

    @field_validator('status_json', mode='before')
    @classmethod
    def validate_status_json(cls, v):
        """Convert JSON string to JobStatusJSON object during validation."""
        if isinstance(v, str):
            return JobStatusJSON.model_validate_json(v)
        return v

    @field_serializer('service_id')
    def serialize_service_id(self, service_id: ServiceId) -> str:
        """Serialize ServiceId to string format."""
        return service_id.to_string()

    @field_serializer('status_json')
    def serialize_status_json(self, status_json: JobStatusJSON) -> str:
        """Serialize JobStatusJSON to JSON string format."""
        return status_json.model_dump_json()

    @staticmethod
    def from_job_status(status: JobStatus) -> JobStatusMessage:
        return JobStatusMessage(
            service_id=ServiceId.from_job_id(status.job_id),
            status_json=JobStatusJSON(
                status=job_state_to_nicos_status_constant(status.state),
                message=JobStatusPayload(
                    state=status.state,
                    warning=status.warning_message,
                    error=status.error_message,
                    job_id=status.job_id,
                    workflow_id=str(status.workflow_id),
                    start_time=status.start_time,
                    end_time=status.end_time,
                ),
            ),
        )

    def to_job_status(self) -> JobStatus:
        """Convert JobStatusMessage to JobStatus."""
        message = self.status_json.message
        return JobStatus(
            job_id=message.job_id,
            workflow_id=WorkflowId.from_string(message.workflow_id),
            state=message.state,
            error_message=message.error,
            warning_message=message.warning,
            start_time=message.start_time,
            end_time=message.end_time,
        )


def x5f2_to_job_status(x5f2_status: bytes) -> JobStatus:
    """Deserialize x5f2 status message to JobStatus."""
    status_msg = deserialise_x5f2(x5f2_status)
    # Manually map namedtuple fields
    status_message = JobStatusMessage(
        software_name=status_msg.software_name,
        software_version=status_msg.software_version,
        service_id=status_msg.service_id,  # Will be converted by validator
        host_name=status_msg.host_name,
        process_id=status_msg.process_id,
        update_interval=status_msg.update_interval,
        status_json=status_msg.status_json,  # Will be converted by validator
    )
    return status_message.to_job_status()


def job_status_to_x5f2(
    status: JobStatus,
    *,
    software_version: str = '0.0.0',
    host_name: str = '',
    process_id: int = 0,
) -> bytes:
    """Serialize JobStatus to x5f2 status message."""
    status_message = JobStatusMessage.from_job_status(status)
    data = status_message.model_dump(mode='json')
    data['software_version'] = software_version
    data['host_name'] = host_name
    data['process_id'] = process_id
    return serialise_x5f2(**data)


def service_status_to_x5f2(
    status: ServiceStatus,
    *,
    software_version: str = '0.0.0',
    host_name: str = '',
    process_id: int = 0,
) -> bytes:
    """Serialize ServiceStatus to x5f2 status message."""
    status_message = ServiceStatusMessage.from_service_status(
        status,
        software_version=software_version,
        host_name=host_name,
        process_id=process_id,
    )
    return serialise_x5f2(**status_message.model_dump(mode='json'))


def x5f2_to_service_status(x5f2_status: bytes) -> ServiceStatus:
    """Deserialize x5f2 status message to ServiceStatus."""
    status_msg = deserialise_x5f2(x5f2_status)
    # Manually map namedtuple fields
    status_message = ServiceStatusMessage(
        software_name=status_msg.software_name,
        software_version=status_msg.software_version,
        service_id=status_msg.service_id,  # Will be converted by validator
        host_name=status_msg.host_name,
        process_id=status_msg.process_id,
        update_interval=status_msg.update_interval,
        status_json=status_msg.status_json,  # Will be converted by validator
    )
    return status_message.to_service_status()


def x5f2_to_status(x5f2_status: bytes) -> JobStatus | ServiceStatus:
    """
    Deserialize x5f2 status message to JobStatus or ServiceStatus.

    Uses the `message_type` field in the status_json to determine the message type.
    Messages without a `message_type` field are assumed to be job status messages
    for backward compatibility.
    """
    import json

    status_msg = deserialise_x5f2(x5f2_status)

    # Parse status_json to check message_type
    status_json = json.loads(status_msg.status_json)
    message = status_json.get("message", {})
    message_type = message.get(
        "message_type", "job"
    )  # Default to "job" for backwards compatibility

    if message_type == "service":
        status_message = ServiceStatusMessage(
            software_name=status_msg.software_name,
            software_version=status_msg.software_version,
            service_id=status_msg.service_id,
            host_name=status_msg.host_name,
            process_id=status_msg.process_id,
            update_interval=status_msg.update_interval,
            status_json=status_msg.status_json,
        )
        return status_message.to_service_status()
    else:
        return x5f2_to_job_status(x5f2_status)
