# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from ess.livedata.config.models import ConfigKey
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.job_service import JobService


class JobController:
    def __init__(
        self, command_service: CommandService, job_service: JobService
    ) -> None:
        self._command_service = command_service
        self._job_service = job_service

    def send_job_action(self, job_id: JobId, action: JobAction) -> None:
        """Send action for a specific job ID."""
        self.send_job_actions_batch([job_id], action)

    def send_job_actions_batch(self, job_ids: list[JobId], action: JobAction) -> None:
        """Send the same action for multiple job IDs in a batch."""
        commands = [
            (
                ConfigKey(key=JobCommand.key, source_name=str(job_id)),
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
