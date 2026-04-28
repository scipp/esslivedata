# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import structlog

from ..config.acknowledgement import AcknowledgementResponse, CommandAcknowledgement
from ..config.workflow_spec import WorkflowConfig
from .job_manager import DifferentInstrument, JobCommand, JobManager
from .message import RESPONSES_STREAM_ID, Message

logger = structlog.get_logger(__name__)


class JobManagerAdapter:
    """
    Adapter to convert calls to JobManager into ConfigHandler actions.

    This has two purposes:

    1. We can keep using ConfigHandler until we have fully refactored everything.
    2. We keep the legacy one-source-one-job behavior, replacing old jobs if a new one
       is started. The long-term goal is to change this to a more flexible mechanism,
       but this, too, would require frontend changes.

    Both action methods return ``list[Message]``: an acknowledgement message (when
    a ``message_id`` was supplied) plus any data-stream messages produced
    synchronously by the action (e.g. results from one-shot jobs that finalize
    inline during ``schedule_job``). The list is empty when there is nothing to
    publish.
    """

    def __init__(self, *, job_manager: JobManager) -> None:
        self._job_manager = job_manager

    def job_command(self, source_name: str, value: dict) -> list[Message]:
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
            return []
        except Exception as e:
            logger.exception("job_command_failed", action=command.action)
            return _ack_messages(
                message_id=command.message_id,
                device=str(command.job_id) if command.job_id else "all",
                response=AcknowledgementResponse.ERR,
                message=str(e),
            )
        return _ack_messages(
            message_id=command.message_id,
            device=str(command.job_id) if command.job_id else "all",
            response=AcknowledgementResponse.ACK,
        )

    def set_workflow_with_config(
        self, source_name: str, value: dict | None
    ) -> list[Message]:
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
            return []
        except Exception as e:
            logger.exception(
                "workflow_start_failed", workflow_id=str(config.identifier)
            )
            return _ack_messages(
                message_id=config.message_id,
                device=source_name,
                response=AcknowledgementResponse.ERR,
                message=str(e),
            )

        # Drain any messages produced inline by one-shot jobs (data result +
        # JobStatus(stopped)) so they reach the sink on the same loop tick as
        # the ack, with no dependency on data traffic to drive emission.
        return (
            _ack_messages(
                message_id=config.message_id,
                device=source_name,
                response=AcknowledgementResponse.ACK,
            )
            + self._job_manager.drain_pending_messages()
        )


def _ack_messages(
    *,
    message_id: str | None,
    device: str,
    response: AcknowledgementResponse,
    message: str | None = None,
) -> list[Message]:
    """Return the ack message wrapped in a list, or an empty list if not requested."""
    if message_id is None:
        return []
    ack = CommandAcknowledgement(
        message_id=message_id,
        device=device,
        response=response,
        message=message,
    )
    return [Message(stream=RESPONSES_STREAM_ID, value=ack)]
