# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Protocol, runtime_checkable

import scipp as sc

from ess.livedata.workflows.workflow_factory import Workflow

from ..config.workflow_spec import JobId, ResultKey, WorkflowId
from .timestamp import Timestamp


@dataclass(slots=True, kw_only=True)
class JobData:
    start_time: Timestamp
    end_time: Timestamp
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
    start_time: Timestamp | None
    end_time: Timestamp | None
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
    start_time: Timestamp | None = None
    end_time: Timestamp | None = None

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


class JobState(StrEnum):
    scheduled = "scheduled"
    active = "active"
    finishing = "finishing"
    error = "error"
    warning = "warning"
    pending_context = "pending_context"
    stopped = "stopped"


class ServiceState(StrEnum):
    """State of a backend service worker."""

    starting = auto()  # Service initializing
    running = auto()  # Normal operation
    stopped = auto()  # Service shut down
    error = auto()  # Service encountered fatal error


@dataclass(frozen=True, slots=True)
class StreamStat:
    """Message count for a single (topic, source_name) combination."""

    topic: str
    source_name: str
    stream: str | None  # Resolved stream name, or None if unmapped
    count: int


@dataclass(frozen=True, slots=True)
class StreamStats:
    """Per-stream message counts collected over a time window."""

    window_seconds: float
    streams: tuple[StreamStat, ...]


# Defaults chosen against realistic ESS conditions: inter-host NTP/chrony skew
# stays in the single-digit-millisecond range and Kafka CreateTime has only
# millisecond resolution, so 100 ms comfortably clears clock noise while a real
# "payload from the future" anomaly (seconds) trips the ERROR band. The WARNING
# band flags data already seconds stale at publish time.
LAG_WARN_THRESHOLD_S = 2.0
LAG_FUTURE_TOLERANCE_S = 0.1


@dataclass(frozen=True, slots=True)
class StreamLag:
    """Lag statistics for one (topic, source, schema) over a time window.

    Lag is ``kafka_create_time - payload_timestamp`` in seconds: how stale a
    payload already was when the broker recorded it, isolating upstream producer
    staleness from this service's own consumption delay.
    """

    topic: str
    source: str
    schema: str
    min_s: float
    max_s: float
    count: int


@dataclass(frozen=True, slots=True)
class StreamLagReport:
    """Per-stream lag for one window, ordered stably by (topic, source, schema)."""

    streams: tuple[StreamLag, ...]
    warn_threshold_s: float
    future_tolerance_s: float

    def level(self, lag: StreamLag) -> int:
        """Python logging level reflecting a single stream's lag.

        ERROR if the payload is meaningfully in the future (clock/logic fault),
        WARNING if it is more than ``warn_threshold_s`` stale at publish time,
        otherwise INFO.
        """
        if lag.min_s < -self.future_tolerance_s:
            return logging.ERROR
        if lag.max_s > self.warn_threshold_s:
            return logging.WARNING
        return logging.INFO


@runtime_checkable
class StreamStatsProvider(Protocol):
    """Provides accumulated per-stream message counts and lag."""

    def drain(self, window_seconds: float) -> StreamStats:
        """Return accumulated counts and reset."""
        ...

    def drain_lag(self) -> StreamLagReport | None:
        """Return accumulated per-stream lag and reset, or None if empty."""
        ...


@dataclass
class ServiceStatus:
    """Complete status information for a backend service worker."""

    instrument: str
    service_name: str
    worker_id: str  # UUID as string
    state: ServiceState
    started_at: Timestamp
    active_job_count: int
    version: str = '0.0.0'
    error: str | None = None
    batch_interval_s: float = 1.0
    stream_stats: StreamStats | None = None


def _add_time_coords(
    data: sc.DataGroup, start_time: Timestamp | None, end_time: Timestamp | None
) -> sc.DataGroup:
    """
    Add start_time and end_time as 0-D coordinates to all DataArrays in a DataGroup.

    These coordinates provide temporal provenance for each output, enabling lag
    calculation in the dashboard (lag = current_time - end_time).

    DataArrays are skipped in two cases:

    - Already have start_time or end_time coordinates. This allows workflows to
      set their own time coords for outputs that represent different time ranges
      (e.g., "current" outputs that only cover the period since the last finalize,
      not the entire job duration).
    - Have a 'time' coordinate. A 'time' coordinate means the data carries its
      own timestamps (e.g., timeseries log data), making start_time/end_time
      redundant. Adding scalar start_time/end_time to such data would also cause
      a dimension mismatch in TemporalBuffer, which accumulates data along the
      'time' dimension.
    """
    if start_time is None or end_time is None:
        return data
    start_coord = start_time.to_scipp()
    end_coord = end_time.to_scipp()

    def maybe_add_coords(val: sc.DataArray) -> sc.DataArray:
        # Skip if workflow already set time coords - we have no idea what they
        # mean, and adding our own would create an inconsistent pair.
        if 'start_time' in val.coords or 'end_time' in val.coords:
            return val
        # Skip if data carries a 'time' coordinate. A 'time' coordinate means
        # the data has its own timestamps (e.g., timeseries log data), making
        # start_time/end_time redundant.
        if 'time' in val.coords:
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
        workflow: Workflow,
        source_names: list[str],
        input_streams: set[str],
        gating_streams: set[str],
        reset_on_run_transition: bool = True,
    ) -> None:
        """
        Initialize a Job with the given parameters.

        Parameters
        ----------
        job_id:
            The unique identifier for this job.
        workflow_id:
            The identifier of the workflow this job is running.
        workflow:
            The Workflow instance that will process data for this job.
        source_names:
            The names of the primary data sources for this job.
        input_streams:
            Canonical stream names of every non-primary input the job subscribes
            to (see :attr:`input_stream_names`) — user-selected ``AuxSources``
            and framework-routed ``ContextBinding`` wire streams alike. Incoming
            data already arrives keyed by stream name — the same key the
            workflow expects, so :meth:`add` needs no remapping. The role→stream
            mapping is consumed at workflow construction; only the names matter
            here.
        gating_streams:
            Subset of ``input_streams`` whose value must be available before the
            workflow runs. The :class:`JobManager` gates the job on these per
            ADR 0002.
        reset_on_run_transition:
            Whether this job should be reset when a run transition occurs.
        """
        self._job_id = job_id
        self._workflow_id = workflow_id
        self._workflow = workflow
        self._start_time: Timestamp | None = None
        self._end_time: Timestamp | None = None
        self._source_names = source_names
        self._reset_on_run_transition = reset_on_run_transition
        self._input_streams: set[str] = input_streams
        self._gating_streams: set[str] = gating_streams

    @property
    def job_id(self) -> JobId:
        return self._job_id

    @property
    def workflow_id(self) -> WorkflowId:
        return self._workflow_id

    @property
    def start_time(self) -> Timestamp | None:
        return self._start_time

    @property
    def end_time(self) -> Timestamp | None:
        return self._end_time

    @property
    def source_names(self) -> list[str]:
        return self._source_names

    @property
    def input_stream_names(self) -> set[str]:
        """Every non-primary stream name this job consumes."""
        return self._input_streams

    @property
    def gating_streams(self) -> set[str]:
        """Context stream names whose value must be available before running.

        See :meth:`missing_context` and
        :doc:`/developer/adr/0002-context-stream-gating-at-jobmanager`.
        """
        return self._gating_streams

    def missing_context(self, available: set[str]) -> set[str]:
        """
        Gating-stream names that have no value available yet.

        The :class:`JobManager` consults this each tick (against the stream
        names present after refilling the batch from the context cache) to
        decide whether to gate the job: any missing gating stream indicates a
        parametric input
        whose accumulator has not produced a value, so running the workflow
        would either crash or silently produce data attributed to an
        uninitialised context. Dynamic aux (e.g. monitor streams that
        accumulate over time) is not in this set and does not gate.
        See :doc:`/developer/adr/0002-context-stream-gating-at-jobmanager`.
        """
        return self._gating_streams - available

    @property
    def reset_on_run_transition(self) -> bool:
        return self._reset_on_run_transition

    def add(self, data: JobData) -> JobReply:
        try:
            # Primary and aux data are both keyed by canonical stream name, which
            # is exactly how the workflow's dynamic/context keys are keyed (aux
            # roles are resolved at workflow construction). No remapping needed.
            self._workflow.accumulate(
                {**data.primary_data, **data.aux_data},
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

    def process(
        self, data: JobData, *, finalize: bool = False
    ) -> tuple[JobReply, JobResult | None]:
        """Accumulate data and optionally finalize.

        Parameters
        ----------
        data:
            The data to accumulate. If empty, accumulation is skipped.
        finalize:
            If True, finalize even if this push did not deliver primary data
            (e.g., to retry after a previous finalization failure).
        """
        if data.is_empty():
            reply = JobReply(job_id=self._job_id)
        else:
            reply = self.add(data)
            if not reply.has_error and data.is_active():
                finalize = True
        result = self.get() if finalize else None
        return reply, result

    def get(self) -> JobResult:
        try:
            raw_result = self._workflow.finalize()
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
        """Reset the workflow for this job."""
        self._workflow.clear()
        self._start_time = None
        self._end_time = None
