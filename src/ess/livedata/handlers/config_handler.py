# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import structlog

from ..config.acknowledgement import CommandAcknowledgement
from ..config.workflow_spec import WorkflowConfig
from ..core.job_manager import Command, JobCommand
from ..core.job_manager_adapter import JobManagerAdapter
from ..core.message import RESPONSES_STREAM_ID, Message
from ..core.stream_alias import BindStreamAlias, UnbindStreamAlias
from ..core.stream_alias_adapter import StreamAliasAdapter

logger = structlog.get_logger(__name__)

__all__ = ['ConfigProcessor']


class ConfigProcessor:
    """
    Dispatches :class:`Command` messages from the livedata commands topic to
    :class:`JobManagerAdapter` and :class:`StreamAliasAdapter`.
    """

    def __init__(
        self,
        *,
        job_manager_adapter: JobManagerAdapter,
        stream_alias_adapter: StreamAliasAdapter | None = None,
    ) -> None:
        self._job_manager_adapter = job_manager_adapter
        self._stream_alias_adapter = stream_alias_adapter

    def process_messages(
        self, messages: list[Message[Command]]
    ) -> list[Message[CommandAcknowledgement]]:
        response_messages: list[Message[CommandAcknowledgement]] = []
        for message in messages:
            command = message.value
            logger.info(
                'processing_command', kind=command.kind, value=command.model_dump()
            )
            try:
                result = self._dispatch(command)
            except Exception:
                logger.exception('error_processing_command', kind=command.kind)
                continue
            if result is not None:
                response_messages.append(
                    Message(stream=RESPONSES_STREAM_ID, value=result)
                )
        return response_messages

    def _dispatch(self, command: Command) -> CommandAcknowledgement | None:
        if isinstance(command, WorkflowConfig):
            return self._job_manager_adapter.set_workflow_with_config(command)
        if isinstance(command, JobCommand):
            return self._job_manager_adapter.job_command(command)
        if isinstance(command, BindStreamAlias | UnbindStreamAlias):
            if self._stream_alias_adapter is None:
                logger.debug('stream_alias_not_configured', kind=command.kind)
                return None
            if isinstance(command, BindStreamAlias):
                return self._stream_alias_adapter.bind(command)
            return self._stream_alias_adapter.unbind(command)
        logger.error('unknown_command_kind', kind=command.kind)
        return None
