# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import structlog

from ..config.acknowledgement import AcknowledgementResponse, CommandAcknowledgement
from ..config.workflow_spec import WorkflowConfig
from .job_manager import DifferentInstrument, JobCommand, JobManager

logger = structlog.get_logger(__name__)


class JobManagerAdapter:
    """
    Translates Command messages dispatched by ConfigProcessor into JobManager calls.

    A WorkflowConfig is translated into a new scheduled job and a JobCommand into
    a job action, each wrapped in a CommandAcknowledgement for the frontend.
    """

    def __init__(self, *, job_manager: JobManager) -> None:
        self._job_manager = job_manager

    def job_command(self, command: JobCommand) -> CommandAcknowledgement | None:
        try:
            affected = self._job_manager.job_command(command)
        except KeyError:
            # Job not found. Similar to DifferentInstrument for workflows: multiple
            # backend services receive the same commands, but only the one owning the
            # job should respond. Future work may add device-based filtering (#445).
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

    def set_workflow_with_config(
        self, config: WorkflowConfig
    ) -> CommandAcknowledgement | None:
        try:
            _ = self._job_manager.schedule_job(config=config)
        except DifferentInstrument:
            # We have multiple backend services that handle jobs, e.g., data_reduction
            # and monitor_data. The frontend simply sends a WorkflowConfig message and
            # does not make assumptions which service will handle it. The workflows
            # for each backend are part of a different instrument, e.g., 'dream' for
            # data_reduction and 'dream_beam_monitors' for monitor_data, which is
            # included in the identifier. This should thus work safely, but the question
            # is whether it should be filtered out earlier.
            logger.debug(
                "workflow_not_found",
                workflow_id=str(config.identifier),
                message="Assuming handled by another worker",
            )
            return None
        except Exception as e:
            logger.exception(
                "workflow_start_failed", workflow_id=str(config.identifier)
            )
            return CommandAcknowledgement(
                message_id=config.message_id,
                device=config.job_id.source_name,
                response=AcknowledgementResponse.ERR,
                message=str(e),
            )
        else:
            return CommandAcknowledgement(
                message_id=config.message_id,
                device=config.job_id.source_name,
                response=AcknowledgementResponse.ACK,
            )
