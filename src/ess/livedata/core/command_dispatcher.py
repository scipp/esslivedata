# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import structlog

from ..config.acknowledgement import CommandAcknowledgement
from .job_manager import Command
from .job_manager_adapter import JobManagerAdapter
from .message import RESPONSES_STREAM_ID, Message

logger = structlog.get_logger(__name__)

__all__ = ['CommandDispatcher']


class CommandDispatcher:
    """
    Dispatches :class:`Command` messages from the livedata commands topic to
    :class:`JobManagerAdapter`.
    """

    def __init__(self, *, job_manager_adapter: JobManagerAdapter) -> None:
        self._job_manager_adapter = job_manager_adapter

    def process_messages(
        self, messages: list[Message[Command]]
    ) -> list[Message[CommandAcknowledgement]]:
        response_messages: list[Message[CommandAcknowledgement]] = []
        for message in messages:
            command = message.value
            logger.info('processing_command', value=command.model_dump())
            result = self._job_manager_adapter.job_command(command)
            if result is not None:
                response_messages.append(
                    Message(stream=RESPONSES_STREAM_ID, value=result)
                )
        return response_messages
