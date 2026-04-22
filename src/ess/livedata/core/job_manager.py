# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import bisect
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

import pydantic
import scipp as sc
import structlog

from ess.livedata.config.instrument import Instrument
from ess.livedata.config.workflow_spec import (
    JobId,
    JobSchedule,
    WorkflowConfig,
    WorkflowId,
    WorkflowSpec,
)

from .job import Job, JobData, JobReply, JobResult, JobState, JobStatus
from .message import RunStart, RunStop, StreamId
from .timestamp import Timestamp

logger = structlog.get_logger(__name__)


@dataclass(slots=True, kw_only=True)
class WorkflowData:
    """
    Data to be processed by a workflow.

    All timestamps are in nanoseconds since the epoch (UTC) and reference the timestamps
    of the raw data being processed (as opposed to when it was processed).
    """

    start_time: Timestamp
    end_time: Timestamp
    data: dict[StreamId, Any]


class DifferentInstrument(Exception):
    """
    Raised when a workflow id does not match the instrument of this worker.

    This is not considered an error, but rather a signal that the workflow should be
    handled by a different worker.
    """


class WorkflowNotFoundError(Exception):
    """Raised when a workflow specification is not found in this worker."""


class JobAction(StrEnum):
    pause = "pause"
    resume = "resume"
    reset = "reset"
    stop = "stop"


class JobCommand(pydantic.BaseModel):
    key: ClassVar[str] = "job_command"
    message_id: str | None = pydantic.Field(
        default=None,
        description=(
            "Unique identifier for command correlation. Frontend generates this UUID "
            "and backend echoes it in CommandAcknowledgement responses."
        ),
    )
    job_id: JobId | None = pydantic.Field(
        default=None, description="ID of the job to control."
    )
    workflow_id: WorkflowId | None = pydantic.Field(
        default=None, description="Workflow ID to cancel jobs for."
    )
    action: JobAction = pydantic.Field(description="Action to perform on the job.")


class JobFactory:
    def __init__(self, instrument: Instrument, *, service_name: str) -> None:
        self._instrument = instrument
        self._service_name = service_name

    def get_workflow_spec(self, workflow_id: WorkflowId) -> WorkflowSpec | None:
        """Get the workflow specification for a given workflow ID."""
        return self._instrument.workflow_factory.get(workflow_id)

    def enrich_result(self, result: JobResult) -> JobResult:
        """
        Enrich a job result by setting human-readable names on output DataArrays.

        This method looks up the workflow specification and uses the output metadata
        to set the `.name` attribute on DataArrays based on their title field.

        Parameters
        ----------
        result:
            The job result to enrich with human-readable names.

        Returns
        -------
        :
            The enriched job result (modified in-place, but also returned).
        """
        # Skip enrichment if result has an error or no data
        if result.error_message is not None or result.data is None:
            return result

        # Get workflow spec to access output metadata
        workflow_spec = self.get_workflow_spec(result.workflow_id)
        if workflow_spec is None:
            return result

        for output_name, value in result.data.items():
            if isinstance(value, sc.DataArray):
                field_info = workflow_spec.outputs.model_fields.get(output_name)
                if field_info is not None and field_info.title is not None:
                    value.name = field_info.title

        return result

    def create(self, *, job_id: JobId, config: WorkflowConfig) -> Job:
        workflow_id = config.identifier
        if workflow_id is None:
            raise ValueError("WorkflowConfig must have an identifier to create a Job")
        factory = self._instrument.workflow_factory
        if (workflow_id.instrument != self._instrument.name) or (
            factory.get_service(workflow_id) != self._service_name
        ):
            raise DifferentInstrument()

        workflow_spec = factory.get(workflow_id)
        if workflow_spec is None:
            raise WorkflowNotFoundError(f"WorkflowSpec with Id {workflow_id} not found")

        # Render aux source names using the spec's render() method if applicable
        if workflow_spec.aux_sources is not None:
            rendered_aux_names = workflow_spec.aux_sources.render(
                job_id=job_id,
                selections=config.aux_source_names if config.aux_source_names else None,
            )
        else:
            rendered_aux_names = config.aux_source_names

        # Note that this initializes the job immediately, i.e., we pay startup cost now.
        stream_processor = factory.create(
            source_name=job_id.source_name,
            config=config,
            aux_source_names=rendered_aux_names,
        )
        return Job(
            job_id=job_id,
            workflow_id=workflow_id,
            processor=stream_processor,
            source_names=[job_id.source_name],
            # Pass rendered aux source names (field name -> stream name mapping)
            # Job will use values for routing and remap keys for workflow
            aux_source_names=rendered_aux_names,
            reset_on_run_transition=workflow_spec.reset_on_run_transition,
        )


class JobManager:
    def __init__(self, job_factory: JobFactory, job_threads: int = 1) -> None:
        self.service_name = 'data_reduction'
        self._last_update: int = 0
        self._job_factory = job_factory
        self._active_jobs: dict[JobId, Job] = {}
        self._scheduled_jobs: dict[JobId, Job] = {}
        self._finishing_jobs: list[JobId] = []
        self._job_schedules: dict[JobId, JobSchedule] = {}
        # Track job states and messages in the manager
        self._job_states: dict[JobId, JobState] = {}
        self._job_error_messages: dict[JobId, str] = {}
        self._job_warning_messages: dict[JobId, str] = {}
        # Track which jobs received primary data since last compute_results
        self._jobs_with_primary_data: set[JobId] = set()
        # Pending reset times, kept sorted via bisect.insort
        self._pending_reset_times: list[Timestamp] = []
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=job_threads) if job_threads > 1 else None
        )

    @property
    def all_jobs(self) -> list[Job]:
        """Get a list of all jobs, both active and scheduled."""
        return [
            *self._active_jobs.values(),
            *self._scheduled_jobs.values(),
        ]

    @property
    def active_jobs(self) -> list[Job]:
        """Get the list of active jobs."""
        return list(self._active_jobs.values())

    def _start_job(self, job_id: JobId) -> None:
        """Start a new job with the given workflow."""
        job = self._scheduled_jobs.pop(job_id, None)
        if job is None:
            raise KeyError(f"Job {job_id} not found in scheduled jobs.")
        self._job_states[job_id] = JobState.active
        self._active_jobs[job_id] = job
        logger.info(
            "job_activated", job_id=str(job_id), workflow_id=str(job.workflow_id)
        )

    def _advance_to_time(self, start_time: Timestamp, end_time: Timestamp) -> None:
        """Fire pending resets, activate jobs, and mark jobs that should finish."""
        self._fire_pending_resets(end_time)
        to_activate = [
            job_id
            for job_id in self._scheduled_jobs.keys()
            if self._job_schedules[job_id].should_start(start_time)
        ]

        # Activate jobs first
        for job_id in to_activate:
            self._start_job(job_id)

        # Now check for jobs to finish (including newly activated ones)
        to_finish = [
            job_id
            for job_id in self._active_jobs.keys()
            if (schedule := self._job_schedules[job_id]).end_time is not None
            and schedule.end_time <= end_time
        ]

        # Do not remove from active jobs yet, we need to compute results.
        self._finishing_jobs.extend(to_finish)

    def schedule_job(self, source_name: str, config: WorkflowConfig) -> JobId:
        """
        Schedule a new job based on the provided configuration.
        """
        job_id = JobId(
            job_number=config.job_number or uuid.uuid4(), source_name=source_name
        )
        job = self._job_factory.create(job_id=job_id, config=config)
        self._job_schedules[job_id] = config.schedule
        self._job_states[job_id] = JobState.scheduled
        self._scheduled_jobs[job_id] = job
        logger.info(
            "job_scheduled",
            job_id=str(job_id),
            workflow_id=str(config.identifier),
            source=source_name,
            schedule=str(config.schedule),
        )
        return job_id

    def stop_job(self, job_id: JobId) -> None:
        """Stop a job and remove it from the system."""
        was_active = self._active_jobs.pop(job_id, None)
        was_scheduled = self._scheduled_jobs.pop(job_id, None)

        if was_active is None and was_scheduled is None:
            raise KeyError(f"Job {job_id} not found in active or scheduled jobs.")

        self._remove_job_tracking(job_id)
        if job_id in self._finishing_jobs:
            self._finishing_jobs.remove(job_id)
        logger.info("job_stopped", job_id=str(job_id))

    def job_command(self, command: JobCommand) -> None:
        logger.info(
            "job_command_received",
            action=command.action.value,
            job_id=str(command.job_id) if command.job_id else None,
            workflow_id=str(command.workflow_id) if command.workflow_id else None,
        )
        if command.job_id is not None:
            self._perform_job_action(job_id=command.job_id, action=command.action)
        elif command.workflow_id is not None:
            self._perform_action(
                action=command.action,
                sel=lambda job: job.workflow_id == command.workflow_id,
            )
        else:
            self._perform_action(action=command.action, sel=lambda job: True)

    def _perform_action(self, action: JobAction, sel: Callable[[Job], bool]) -> None:
        jobs_to_control = [job.job_id for job in self.active_jobs if sel(job)]
        for job_id in jobs_to_control:
            self._perform_job_action(job_id=job_id, action=action)

    def _perform_job_action(self, job_id: JobId, action: JobAction) -> None:
        match action:
            case JobAction.reset:
                self.reset_job(job_id)
            case JobAction.stop:
                self.stop_job(job_id)
            case JobAction.pause:
                raise NotImplementedError("Pause action not implemented yet")
            case JobAction.resume:
                raise NotImplementedError("Resume action not implemented yet")
            case _:
                raise ValueError(f"Unknown job action: {action}")

    def reset_job(self, job_id: JobId) -> None:
        """
        Reset a job with the given ID.
        This will clear the processor and reset the start and end times.
        """
        if (job := self._active_jobs.get(job_id)) is not None:
            job.reset()
        elif (job := self._scheduled_jobs.get(job_id)) is not None:
            job.reset()
        else:
            raise KeyError(f"Job {job_id} not found in active or scheduled jobs.")

        # Clear error/warning state when resetting
        self._job_error_messages.pop(job_id, None)
        self._job_warning_messages.pop(job_id, None)
        # Reset state to scheduled unless it's currently active
        if job_id in self._active_jobs:
            self._job_states[job_id] = JobState.active
        else:
            self._job_states[job_id] = JobState.scheduled

    def on_run_start(self, info: RunStart) -> None:
        """Handle a run-start event by scheduling deferred resets."""
        logger.info("run_start", run_name=info.run_name, start_time=info.start_time)
        self._schedule_reset(info.start_time)
        if info.stop_time is not None:
            self._schedule_reset(info.stop_time)

    def on_run_stop(self, info: RunStop) -> None:
        """Handle a run-stop event by scheduling a deferred reset."""
        logger.info("run_stop", run_name=info.run_name, stop_time=info.stop_time)
        self._schedule_reset(info.stop_time)

    def _schedule_reset(self, time_ns: Timestamp) -> None:
        bisect.insort(self._pending_reset_times, time_ns)

    def _fire_pending_resets(self, end_time: Timestamp) -> None:
        """Fire pending resets whose scheduled time has been reached by data."""
        if not self._pending_reset_times:
            return
        triggered = bisect.bisect_right(self._pending_reset_times, end_time)
        if triggered:
            self._pending_reset_times = self._pending_reset_times[triggered:]
            self._reset_eligible_jobs()

    def _reset_eligible_jobs(self) -> None:
        for job in list(self.active_jobs):
            if job.reset_on_run_transition:
                self.reset_job(job.job_id)

    def peek_pending_streams(self, start_time: int) -> set[str]:
        """Return all stream names needed by jobs that would activate at start_time.

        Includes both primary and auxiliary stream names. This is a read-only
        query with no side effects. It does not activate jobs or mutate any state.
        """
        names: set[str] = set()
        for job_id, job in self._scheduled_jobs.items():
            if self._job_schedules[job_id].should_start(start_time):
                names.update(job.source_names)
                names.update(job.aux_source_names)
        return names

    def push_data(self, data: WorkflowData) -> list[JobReply]:
        """Push data into the active jobs and return status for each job.

        Unlike ``process_jobs`` called via ``OrchestratingProcessor.process``,
        this method does not seed context data for newly activated jobs.
        Callers are responsible for providing complete ``WorkflowData``.
        """
        self._advance_to_time(data.start_time, data.end_time)
        replies = []
        for job in self.active_jobs:
            job_data = _filter_data_for_job(job, data)
            if not job_data.is_empty():
                reply = job.add(job_data)
                self._record_push_result(job, job_data, reply)
                replies.append(reply)
        return replies

    def compute_results(self) -> list[JobResult]:
        """
        Compute results from jobs that received primary data since last successful call.
        """
        results = []
        for job in self._active_jobs.values():
            if job.job_id not in self._jobs_with_primary_data:
                continue
            result = job.get()
            result = self._job_factory.enrich_result(result)
            results.append(result)
            self._record_compute_result(job, result)
        self._finish_jobs()
        return results

    def process_jobs(
        self, data: WorkflowData
    ) -> tuple[list[JobReply], list[JobResult]]:
        """Push data and compute results in a single pass over active jobs.

        When a thread pool is configured, each job's accumulation and
        finalization run as a single task, avoiding two fan-out/fan-in cycles.
        """
        self._advance_to_time(data.start_time, data.end_time)

        # Build work items (cheap, sequential)
        work_items: list[tuple[Job, JobData, bool]] = []
        for job in self.active_jobs:
            job_data = _filter_data_for_job(job, data)
            has_data = not job_data.is_empty()
            has_pending = job.job_id in self._jobs_with_primary_data
            if has_data or has_pending:
                work_items.append((job, job_data, has_pending))

        # Run push+finalize per job — parallelized when executor is available
        outcomes = self._map(_process_job, work_items)

        # Bookkeeping (cheap, sequential)
        all_replies: list[JobReply] = []
        all_results: list[JobResult] = []
        for (job, job_data, _), (reply, result) in zip(
            work_items, outcomes, strict=True
        ):
            self._record_push_result(job, job_data, reply)
            all_replies.append(reply)
            if result is not None:
                result = self._job_factory.enrich_result(result)
                all_results.append(result)
                self._record_compute_result(job, result)

        self._finish_jobs()
        return all_replies, all_results

    def _record_push_result(self, job: Job, job_data: JobData, reply: JobReply) -> None:
        # Track primary data updates only after successful add, so that a
        # failed push does not trigger a finalize attempt on empty accumulators.
        if not reply.has_error and job_data.is_active():
            self._jobs_with_primary_data.add(job.job_id)

        # Track warnings from job operations, or clear them on success
        if reply.has_error and reply.error_message is not None:
            # Pushing new data puts the job into "warning" state: Processing the latest
            # data failed, but the job may still be able to finalize previous data.
            self._job_warning_messages[job.job_id] = reply.error_message
            self._job_states[job.job_id] = JobState.warning
            logger.warning(
                "job_warning",
                job_id=str(job.job_id),
                error_message=reply.error_message,
            )
        else:
            # Clear warning state on successful data processing
            self._job_warning_messages.pop(job.job_id, None)
            # Only update state if it was warning (preserve error state)
            if self._job_states.get(job.job_id) == JobState.warning:
                self._job_states[job.job_id] = JobState.active

    def _record_compute_result(self, job: Job, result: JobResult) -> None:
        # Track errors from job finalization, or clear them on success
        if result.error_message is not None:
            # Finalizing failed, put job into error state, cannot compute results.
            self._job_error_messages[job.job_id] = result.error_message
            self._job_states[job.job_id] = JobState.error
            logger.error(
                "job_error",
                job_id=str(job.job_id),
                error_message=result.error_message,
            )
        else:
            # Clear error state on successful finalization
            self._job_error_messages.pop(job.job_id, None)
            # Track warnings from job finalization (e.g., None values in result)
            if result.warning_message is not None:
                self._job_warning_messages[job.job_id] = result.warning_message
                self._job_states[job.job_id] = JobState.warning
            elif job.job_id in self._job_warning_messages:
                # Preserve existing warnings from data processing
                self._job_states[job.job_id] = JobState.warning
            else:
                self._job_states[job.job_id] = JobState.active
            # Remove from the tracking set only if we successfully computed results.
            # If there was an error we keep it in the set to retry next time, which
            # can be important if a job has not yet initialized itself with the
            # first auxiliary data.
            self._jobs_with_primary_data.remove(job.job_id)

    def shutdown(self) -> None:
        """Shut down the thread pool executor, if one was created."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _map(self, fn: Callable, items: list) -> list:
        if self._executor is not None:
            return list(self._executor.map(fn, items))
        return [fn(item) for item in items]

    def _remove_job_tracking(self, job_id: JobId) -> None:
        """Remove all tracking state for a job."""
        self._job_schedules.pop(job_id, None)
        self._job_states.pop(job_id, None)
        self._job_error_messages.pop(job_id, None)
        self._job_warning_messages.pop(job_id, None)
        self._jobs_with_primary_data.discard(job_id)

    def _finish_jobs(self):
        for job_id in self._finishing_jobs:
            self._active_jobs.pop(job_id, None)
            self._remove_job_tracking(job_id)
        self._finishing_jobs.clear()

    def get_job_status(self, job_id: JobId) -> JobStatus | None:
        """Get the status of a specific job by its ID."""
        job = self._active_jobs.get(job_id) or self._scheduled_jobs.get(job_id)
        if job is None:
            return None

        # Determine current state based on job's location in manager
        if job_id in self._active_jobs:
            if job_id in self._finishing_jobs:
                current_state = JobState.finishing
            else:
                # Use tracked state (may be warning/error from operations)
                current_state = self._job_states.get(job_id, JobState.active)
        else:
            # Use tracked state (may be error/warning from previous operations)
            current_state = self._job_states.get(job_id, JobState.scheduled)

        return JobStatus(
            job_id=job_id,
            workflow_id=job.workflow_id,
            state=current_state,
            error_message=self._job_error_messages.get(job_id),
            warning_message=self._job_warning_messages.get(job_id),
            start_time=job.start_time,
            end_time=job.end_time,
        )

    def get_all_job_statuses(self) -> list[JobStatus]:
        """Get the status of all jobs in the manager."""
        all_job_ids = list(self._active_jobs.keys()) + list(self._scheduled_jobs.keys())
        statuses = []
        for job_id in all_job_ids:
            status = self.get_job_status(job_id)
            if status is not None:
                statuses.append(status)
        return statuses

    def get_jobs_by_workflow(self, workflow_id: WorkflowId) -> list[JobStatus]:
        """Get all jobs for a specific workflow ID."""
        return [
            status
            for status in self.get_all_job_statuses()
            if status.workflow_id == workflow_id
        ]

    def get_jobs_by_state(self, state: JobState) -> list[JobStatus]:
        """Get all jobs in a specific state."""
        return [
            status for status in self.get_all_job_statuses() if status.state == state
        ]

    def format_job_error(self, status: JobReply) -> str:
        """Format a job error message with meaningful job information."""
        job = self._active_jobs.get(status.job_id) or self._scheduled_jobs.get(
            status.job_id
        )
        if job is None:
            return f"Job {status.job_id} error: {status.error_message}"

        return (
            f"Job {job._workflow_id}/{status.job_id.source_name} "
            f"error: {status.error_message}"
        )


def _filter_data_for_job(job: Job, data: WorkflowData) -> JobData:
    """Filter workflow data for a specific job."""
    job_data = JobData(
        start_time=data.start_time,
        end_time=data.end_time,
        primary_data={},
        aux_data={},
    )
    for stream, value in data.data.items():
        if stream.name in job.source_names:
            job_data.primary_data[stream.name] = value
        elif stream.name in job.aux_source_names:
            job_data.aux_data[stream.name] = value
    return job_data


def _process_job(
    item: tuple[Job, JobData, bool],
) -> tuple[JobReply, JobResult | None]:
    """Push data and optionally finalize a single job.

    Used as a map target for thread pool execution.
    """
    job, job_data, has_pending_primary = item
    return job.process(job_data, finalize=has_pending_primary)
