# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import uuid
from collections.abc import Callable
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
from .message import StreamId

logger = structlog.get_logger(__name__)


@dataclass(slots=True, kw_only=True)
class WorkflowData:
    """
    Data to be processed by a workflow.

    All timestamps are in nanoseconds since the epoch (UTC) and reference the timestamps
    of the raw data being processed (as opposed to when it was processed).
    """

    start_time: int
    end_time: int
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
    remove = "remove"


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
    def __init__(self, instrument: Instrument) -> None:
        self._instrument = instrument

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
        if (workflow_id.instrument != self._instrument.name) or (
            workflow_id.namespace != self._instrument.active_namespace
        ):
            raise DifferentInstrument()

        factory = self._instrument.workflow_factory
        workflow_spec = factory.get(workflow_id)
        if workflow_spec is None:
            raise WorkflowNotFoundError(f"WorkflowSpec with Id {workflow_id} not found")

        # Render aux source names using the model's render() method if applicable
        if workflow_spec.aux_sources is not None:
            # Reconstruct the Pydantic model from the dict to call render()
            # Empty dict will trigger Pydantic defaults if defined in the model
            aux_model = workflow_spec.aux_sources.model_validate(
                config.aux_source_names
            )
            rendered_aux_names = aux_model.render(job_id=job_id)
        else:
            rendered_aux_names = config.aux_source_names

        # Note that this initializes the job immediately, i.e., we pay startup cost now.
        stream_processor = factory.create(source_name=job_id.source_name, config=config)
        return Job(
            job_id=job_id,
            workflow_id=workflow_id,
            processor=stream_processor,
            source_names=[job_id.source_name],
            # Pass rendered aux source names (field name -> stream name mapping)
            # Job will use values for routing and remap keys for workflow
            aux_source_names=rendered_aux_names,
        )


class JobManager:
    def __init__(self, job_factory: JobFactory) -> None:
        self.service_name = 'data_reduction'
        self._last_update: int = 0
        self._job_factory = job_factory
        self._active_jobs: dict[JobId, Job] = {}
        self._scheduled_jobs: dict[JobId, Job] = {}
        self._stopped_jobs: dict[JobId, Job] = {}
        self._finishing_jobs: list[JobId] = []
        self._job_schedules: dict[JobId, JobSchedule] = {}
        # Track job states and messages in the manager
        self._job_states: dict[JobId, JobState] = {}
        self._job_error_messages: dict[JobId, str] = {}
        self._job_warning_messages: dict[JobId, str] = {}
        # Track which jobs received primary data since last compute_results
        self._jobs_with_primary_data: set[JobId] = set()

    @property
    def all_jobs(self) -> list[Job]:
        """Get a list of all jobs, both active and scheduled."""
        return [
            *self._active_jobs.values(),
            *self._scheduled_jobs.values(),
            *self._stopped_jobs.values(),
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

    def _advance_to_time(self, start_time: int, end_time: int) -> None:
        """Activate jobs that should start and mark jobs that should finish."""
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
        """Stop a job with the given ID immediately and move it to stopped state."""
        was_active = self._active_jobs.pop(job_id, None)
        was_schedule = self._scheduled_jobs.pop(job_id, None)

        job = was_active or was_schedule
        if job is None:
            raise KeyError(f"Job {job_id} not found in active or scheduled jobs.")

        # Move to stopped jobs and update state
        self._stopped_jobs[job_id] = job
        self._job_states[job_id] = JobState.stopped
        logger.info("job_stopped", job_id=str(job_id))

    def remove_job(self, job_id: JobId) -> None:
        """Remove a job with the given ID completely from the system."""
        was_active = self._active_jobs.pop(job_id, None)
        was_scheduled = self._scheduled_jobs.pop(job_id, None)
        was_stopped = self._stopped_jobs.pop(job_id, None)

        job = was_active or was_scheduled or was_stopped
        if job is None:
            raise KeyError(f"Job {job_id} not found.")

        # Clean up all tracking
        self._job_schedules.pop(job_id, None)
        self._job_states.pop(job_id, None)
        self._job_error_messages.pop(job_id, None)
        self._job_warning_messages.pop(job_id, None)

        # Remove from finishing jobs if present
        if job_id in self._finishing_jobs:
            self._finishing_jobs.remove(job_id)
        logger.info("job_removed", job_id=str(job_id))

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
            case JobAction.remove:
                self.remove_job(job_id)
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
        elif (job := self._stopped_jobs.get(job_id)) is not None:
            job.reset()
        else:
            raise KeyError(
                f"Job {job_id} not found in active, scheduled, or stopped jobs."
            )

        # Clear error/warning state when resetting
        self._job_error_messages.pop(job_id, None)
        self._job_warning_messages.pop(job_id, None)
        # Reset state to scheduled unless it's currently active
        if job_id in self._active_jobs:
            self._job_states[job_id] = JobState.active
        else:
            self._job_states[job_id] = JobState.scheduled

    def push_data(self, data: WorkflowData) -> list[JobReply]:
        """Push data into the active jobs and return status for each job."""
        self._advance_to_time(data.start_time, data.end_time)
        replies = [self._push_data_to_job(job, data) for job in self.active_jobs]
        return [reply for reply in replies if reply is not None]

    def _push_data_to_job(self, job: Job, data: WorkflowData) -> JobReply | None:
        # Filter data for this specific job
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

        # Only process if we have relevant data
        if job_data.is_empty():
            return None

        # Track primary data updates
        if job_data.is_active():
            self._jobs_with_primary_data.add(job.job_id)
        reply = job.add(job_data)

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
        return reply

    def compute_results(self) -> list[JobResult]:
        """
        Compute results from jobs that received primary data since last successful call.
        """
        results = []
        # Only compute results for jobs that received primary data
        for job in self._active_jobs.values():
            if job.job_id not in self._jobs_with_primary_data:
                continue
            result = job.get()
            # Enrich result with human-readable names from workflow spec
            result = self._job_factory.enrich_result(result)
            results.append(result)
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

        self._finish_jobs()
        return results

    def _finish_jobs(self):
        for job_id in self._finishing_jobs:
            job = self._active_jobs.pop(job_id, None)
            if job is not None:
                self._stopped_jobs[job_id] = job
                self._job_states[job_id] = JobState.stopped
            _ = self._job_schedules.pop(job_id, None)  # Clean up schedule
        self._finishing_jobs.clear()

    def get_job_status(self, job_id: JobId) -> JobStatus | None:
        """Get the status of a specific job by its ID."""
        job = (
            self._active_jobs.get(job_id)
            or self._scheduled_jobs.get(job_id)
            or self._stopped_jobs.get(job_id)
        )
        if job is None:
            return None

        # Determine current state based on job's location in manager
        if job_id in self._active_jobs:
            if job_id in self._finishing_jobs:
                current_state = JobState.finishing
            else:
                # Use tracked state (may be warning/error from operations)
                current_state = self._job_states.get(job_id, JobState.active)
        elif job_id in self._scheduled_jobs:
            # Use tracked state (may be error/warning from previous operations)
            current_state = self._job_states.get(job_id, JobState.scheduled)
        elif job_id in self._stopped_jobs:
            current_state = JobState.stopped
        else:
            return None

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
        all_job_ids = (
            list(self._active_jobs.keys())
            + list(self._scheduled_jobs.keys())
            + list(self._stopped_jobs.keys())
        )
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
        job = (
            self._active_jobs.get(status.job_id)
            or self._scheduled_jobs.get(status.job_id)
            or self._stopped_jobs.get(status.job_id)
        )
        if job is None:
            return f"Job {status.job_id} error: {status.error_message}"

        return (
            f"Job {job._workflow_id}/{status.job_id.source_name} "
            f"error: {status.error_message}"
        )
