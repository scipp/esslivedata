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
    Adapter to convert calls to JobManager into ConfigHandler actions.

    This has two purposes:

    1. We can keep using ConfigHandler until we have fully refactored everything.
    2. We keep the legacy one-source-one-job behavior, replacing old jobs if a new one
       is started. The long-term goal is to change this to a more flexible mechanism,
       but this, too, would require frontend changes.
    """

    def __init__(self, *, job_manager: JobManager) -> None:
        self._job_manager = job_manager

    def job_command(
        self, source_name: str, value: dict
    ) -> CommandAcknowledgement | None:
        _ = source_name  # Legacy, not used.
        command = JobCommand.model_validate(value)

        try:
            self._job_manager.job_command(command)
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
            if command.message_id is not None:
                return CommandAcknowledgement(
                    message_id=command.message_id,
                    device=str(command.job_id) if command.job_id else "all",
                    response=AcknowledgementResponse.ERR,
                    message=str(e),
                )
            return None
        else:
            if command.message_id is not None:
                return CommandAcknowledgement(
                    message_id=command.message_id,
                    device=str(command.job_id) if command.job_id else "all",
                    response=AcknowledgementResponse.ACK,
                )
            return None

    def set_workflow_with_config(
        self, source_name: str | None, value: dict | None
    ) -> CommandAcknowledgement | None:
        if source_name is None:
            raise ValueError("source_name cannot be None for set_workflow_with_config")

        config = WorkflowConfig.model_validate(value)

        try:
            _ = self._job_manager.schedule_job(source_name=source_name, config=config)
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
            if config.message_id is not None:
                return CommandAcknowledgement(
                    message_id=config.message_id,
                    device=source_name,
                    response=AcknowledgementResponse.ERR,
                    message=str(e),
                )
            return None
        else:
            if config.message_id is not None:
                return CommandAcknowledgement(
                    message_id=config.message_id,
                    device=source_name,
                    response=AcknowledgementResponse.ACK,
                )
            return None
