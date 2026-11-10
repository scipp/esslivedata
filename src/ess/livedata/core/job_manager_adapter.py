# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import structlog

from ..config.acknowledgement import AcknowledgementResponse, CommandAcknowledgement
from .job_manager import JobCommand, JobManager

logger = structlog.get_logger(__name__)


class JobManagerAdapter:
    """
    Applies a JobCommand dispatched by CommandDispatcher to the JobManager.

    The result is wrapped in a CommandAcknowledgement for the initiator, or None
    if this worker has no job the command selects.
    """

    def __init__(self, *, job_manager: JobManager) -> None:
        self._job_manager = job_manager

    def job_command(self, command: JobCommand) -> CommandAcknowledgement | None:
        try:
            affected = self._job_manager.job_command(command)
        except KeyError:
            # Job not found: multiple backend services receive the same commands,
            # but only the one owning the job should respond.
            logger.debug(
                "job_not_found",
                job_id=str(command.job_id),
                message="Assuming handled by another worker",
            )
            return None
        except Exception as e:
            logger.exception("job_command_failed", action=command.action)
            return CommandAcknowledgement(
                message_id=command.message_id,
                device=str(command.job_id) if command.job_id else "all",
                response=AcknowledgementResponse.ERR,
                message=str(e),
            )
        else:
            # A selector-keyed command (workflow_id / broadcast) matches no jobs on
            # workers that do not own it; those stay silent, just as the job_id path
            # does via KeyError. Only acknowledge when this worker actually acted.
            if not affected:
                return None
            return CommandAcknowledgement(
                message_id=command.message_id,
                device=str(command.job_id) if command.job_id else "all",
                response=AcknowledgementResponse.ACK,
            )
