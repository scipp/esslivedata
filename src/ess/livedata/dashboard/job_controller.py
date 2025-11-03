# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable

from ess.livedata.config.models import ConfigKey
from ess.livedata.config.workflow_spec import JobId, JobNumber, WorkflowId
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.job_service import JobService


class JobController:
    def __init__(
        self, command_service: CommandService, job_service: JobService
    ) -> None:
        self._command_service = command_service
        self._job_service = job_service
        self._update_subscribers: list[Callable[[], None]] = []

        # Register for job updates from the service
        self._job_service.register_job_update_subscriber(self._on_jobs_updated)

    def register_update_subscriber(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when job data is updated."""
        self._update_subscribers.append(callback)

    def _on_jobs_updated(self) -> None:
        """Handle job updates from the job service."""
        # Notify all UI subscribers that job data has changed
        for callback in self._update_subscribers:
            callback()

    def _config_key(self, key: str, source_name: str) -> ConfigKey:
        return ConfigKey(key=key, source_name=source_name)

    def send_global_action(self, action: JobAction) -> None:
        command = JobCommand(action=action)
        self._command_service.send(self._config_key(JobCommand.key), command)

    def get_workflow_ids(self) -> list[WorkflowId]:
        """Get list of all workflow IDs from active jobs."""
        return list(set(self._job_service.job_info.values()))

    def get_job_ids(self, workflow_id: WorkflowId | None = None) -> list[JobId]:
        """Get list of job IDs, optionally filtered by workflow ID."""
        job_ids = []
        for job_number, source_data in self._job_service.job_data.items():
            # Check if this job matches the workflow filter
            if workflow_id is not None:
                job_workflow_id = self._job_service.job_info.get(job_number)
                if job_workflow_id != workflow_id:
                    continue

            # Add JobId for each source in this job
            job_ids.extend(
                JobId(job_number=job_number, source_name=source_name)
                for source_name in source_data.keys()
            )

        return job_ids

    def get_job_numbers(self, workflow_id: WorkflowId | None = None) -> list[JobNumber]:
        """Get list of job numbers, optionally filtered by workflow ID."""
        if workflow_id is None:
            return list(self._job_service.job_info.keys())

        return [
            job_number
            for job_number, wf_id in self._job_service.job_info.items()
            if wf_id == workflow_id
        ]

    def send_job_action(self, job_id: JobId, action: JobAction) -> None:
        """Send action for a specific job ID."""
        # Using full JobId as source_name to work around current limitation of compacted
        # Kafka topic. ConfigKey needs to be overhauled.
        command = JobCommand(job_id=job_id, action=action)
        self._command_service.send(
            self._config_key(JobCommand.key, source_name=str(job_id)), command
        )

        # If this is a remove action, immediately remove from job service for UI
        # responsiveness
        if action == JobAction.remove:
            self._job_service.remove_job(job_id)

    def send_job_actions_batch(self, job_ids: list[JobId], action: JobAction) -> None:
        """Send the same action for multiple job IDs in a batch."""
        commands = [
            (
                self._config_key(JobCommand.key, source_name=str(job_id)),
                JobCommand(job_id=job_id, action=action),
            )
            for job_id in job_ids
        ]
        self._command_service.send_batch(commands)

        # If this is a remove action, immediately remove from job service for UI
        # responsiveness
        if action == JobAction.remove:
            for job_id in job_ids:
                self._job_service.remove_job(job_id)
