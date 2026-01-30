# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import traceback
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Any

import scipp as sc

from ess.livedata.handlers.workflow_factory import Workflow

from ..config.workflow_spec import JobId, ResultKey, WorkflowId


@dataclass(slots=True, kw_only=True)
class JobData:
    start_time: int
    end_time: int
    primary_data: dict[str, Any]
    aux_data: dict[str, Any]

    def is_empty(self) -> bool:
        """Check if there is no data in both primary and auxiliary data."""
        return not (self.primary_data or self.aux_data)

    def is_active(self) -> bool:
        """Check if there is any primary data, meaning the job is truly "active"."""
        return bool(self.primary_data)


@dataclass(slots=True, kw_only=True)
class JobResult:
    job_id: JobId
    workflow_id: WorkflowId
    # Should this be included in the data instead?
    start_time: int | None
    end_time: int | None
    data: sc.DataGroup | None = None
    error_message: str | None = None
    warning_message: str | None = None

    @property
    def stream_name(self) -> str:
        """Get the stream name associated with this job result.

        The output_name in the ResultKey is a placeholder. UnrollingSinkAdapter
        unpacks the DataGroup and replaces it with the actual output names.
        """
        return ResultKey(
            workflow_id=self.workflow_id,
            job_id=self.job_id,
        ).model_dump_json()


@dataclass
class JobStatus:
    """Complete status information for a job."""

    job_id: JobId
    workflow_id: WorkflowId
    state: JobState
    error_message: str | None = None
    warning_message: str | None = None
    start_time: int | None = None
    end_time: int | None = None

    @property
    def has_error(self) -> bool:
        """Check if the job status indicates an error."""
        return self.state == JobState.error

    @property
    def has_warning(self) -> bool:
        """Check if the job status indicates a warning."""
        return self.state == JobState.warning or self.warning_message is not None


@dataclass
class JobReply:
    """Reply of a job to adding new data, in particular an error message."""

    job_id: JobId
    error_message: str | None = None

    @property
    def has_error(self) -> bool:
        """Check if the job reply indicates an error."""
        return self.error_message is not None


class JobState(str, Enum):
    scheduled = "scheduled"
    active = "active"
    paused = "paused"
    finishing = "finishing"
    stopped = "stopped"
    error = "error"
    warning = "warning"


class ServiceState(StrEnum):
    """State of a backend service worker."""

    starting = auto()  # Service initializing
    running = auto()  # Normal operation
    stopping = auto()  # Graceful shutdown in progress
    stopped = auto()  # Graceful shutdown completed
    error = auto()  # Service encountered fatal error


@dataclass
class ServiceStatus:
    """Complete status information for a backend service worker."""

    instrument: str
    namespace: str
    worker_id: str  # UUID as string
    state: ServiceState
    started_at: int  # Nanoseconds since epoch
    active_job_count: int
    messages_processed: int
    error: str | None = None


def _add_time_coords(
    data: sc.DataGroup, start_time: int | None, end_time: int | None
) -> sc.DataGroup:
    """
    Add start_time and end_time as 0-D coordinates to all DataArrays in a DataGroup.

    These coordinates provide temporal provenance for each output, enabling lag
    calculation in the dashboard (lag = current_time - end_time).

    DataArrays that already have start_time or end_time coordinates are skipped.
    This allows workflows to set their own time coords for outputs that represent
    different time ranges (e.g., "current" outputs that only cover the period
    since the last finalize, not the entire job duration).
    """
    if start_time is None or end_time is None:
        return data
    start_coord = sc.scalar(start_time, unit='ns')
    end_coord = sc.scalar(end_time, unit='ns')

    def maybe_add_coords(val: sc.DataArray) -> sc.DataArray:
        # Skip if workflow already set time coords - we have no idea what they
        # mean, and adding our own would create an inconsistent pair.
        if 'start_time' in val.coords or 'end_time' in val.coords:
            return val
        return val.assign_coords(start_time=start_coord, end_time=end_coord)

    return sc.DataGroup(
        {
            key: (maybe_add_coords(val) if isinstance(val, sc.DataArray) else val)
            for key, val in data.items()
        }
    )


class Job:
    def __init__(
        self,
        *,
        job_id: JobId,
        workflow_id: WorkflowId,
        processor: Workflow,
        source_names: list[str],
        aux_source_names: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize a Job with the given parameters.

        Parameters
        ----------
        job_id:
            The unique identifier for this job.
        workflow_id:
            The identifier of the workflow this job is running.
        processor:
            The Workflow instance that will process data for this job.
        source_names:
            The names of the primary data sources for this job.
        aux_source_names:
            Mapping from field names to stream names for auxiliary data sources.
            None if no auxiliary sources are needed.
        """
        self._job_id = job_id
        self._workflow_id = workflow_id
        self._processor = processor
        self._start_time: int | None = None
        self._end_time: int | None = None
        self._source_names = source_names
        self._aux_source_mapping: dict[str, str] = aux_source_names or {}

        # Create reverse mapping: stream_name -> list of field_names
        # This supports multiplexing where one stream maps to multiple fields. In most
        # cases this is not desirable, but the pydantic model for the aux sources should
        # perform such validation. Here is not the place to prevent this, since there
        # may be valid use cases.
        self._stream_to_fields: dict[str, list[str]] = {}
        for field_name, stream_name in self._aux_source_mapping.items():
            if stream_name not in self._stream_to_fields:
                self._stream_to_fields[stream_name] = []
            self._stream_to_fields[stream_name].append(field_name)

    @property
    def job_id(self) -> JobId:
        return self._job_id

    @property
    def workflow_id(self) -> WorkflowId:
        return self._workflow_id

    @property
    def start_time(self) -> int | None:
        return self._start_time

    @property
    def end_time(self) -> int | None:
        return self._end_time

    @property
    def source_names(self) -> list[str]:
        return self._source_names

    @property
    def aux_source_names(self) -> list[str]:
        """
        Get the list of auxiliary stream names for routing purposes.

        Returns the stream names (values from the mapping) that JobManager
        should use to route incoming data to this job.
        """
        return list(self._aux_source_mapping.values())

    def add(self, data: JobData) -> JobReply:
        try:
            # Remap aux_data keys from stream names to field names for the workflow
            # Handle multiplexing: one stream may map to multiple fields
            remapped_aux_data = {}
            for stream_name, value in data.aux_data.items():
                field_names = self._stream_to_fields.get(stream_name, [stream_name])
                for field_name in field_names:
                    remapped_aux_data[field_name] = value

            # Pass data to workflow with field names (not stream names)
            self._processor.accumulate(
                {**data.primary_data, **remapped_aux_data},
                start_time=data.start_time,
                end_time=data.end_time,
            )
            if data.is_active():
                if self._start_time is None:
                    self._start_time = data.start_time
                self._end_time = data.end_time
            return JobReply(job_id=self._job_id)
        except Exception:
            tb = traceback.format_exc()
            message = f"Job failed to process latest data.\n\n{tb}"
            return JobReply(job_id=self._job_id, error_message=message)

    def get(self) -> JobResult:
        try:
            raw_result = self._processor.finalize()
            none_keys = [str(key) for key, val in raw_result.items() if val is None]
            valid_items = {
                str(key): val for key, val in raw_result.items() if val is not None
            }
            warning_message = None
            if none_keys:
                warning_message = (
                    f"Workflow returned None for output(s): {', '.join(none_keys)}. "
                    "These outputs were excluded from the result."
                )
            data = sc.DataGroup(valid_items)
            data = _add_time_coords(data, self.start_time, self.end_time)
            return JobResult(
                job_id=self._job_id,
                workflow_id=self._workflow_id,
                start_time=self.start_time,
                end_time=self.end_time,
                data=data,
                warning_message=warning_message,
            )
        except Exception:
            tb = traceback.format_exc()
            message = f"Job failed to compute result.\n\n{tb}"
            return JobResult(
                job_id=self._job_id,
                workflow_id=self._workflow_id,
                start_time=self.start_time,
                end_time=self.end_time,
                error_message=message,
            )

    def reset(self) -> None:
        """Reset the processor for this job."""
        self._processor.clear()
        self._start_time = None
        self._end_time = None
