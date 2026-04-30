# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Adapter wiring :class:`BindStreamAlias` / :class:`UnbindStreamAlias` commands
into the per-service :class:`StreamAliasRegistry`.

Mirrors the actor / silent-non-actor protocol used by
:class:`JobManagerAdapter`: the service holding the target job (for ``Bind``)
or the alias (for ``Unbind``) ACKs; other services stay silent and the
dashboard times out if no service answers.
"""

from __future__ import annotations

import structlog

from ..config.acknowledgement import AcknowledgementResponse, CommandAcknowledgement
from .job_manager import JobManager
from .stream_alias import (
    AliasAlreadyBoundError,
    BindStreamAlias,
    StreamAliasRegistry,
    UnbindStreamAlias,
)

logger = structlog.get_logger(__name__)


class StreamAliasAdapter:
    def __init__(
        self,
        *,
        registry: StreamAliasRegistry,
        job_manager: JobManager,
    ) -> None:
        self._registry = registry
        self._job_manager = job_manager

    def bind(self, source_name: str, value: dict) -> CommandAcknowledgement | None:
        _ = source_name  # Not used: aliases are global per-service.
        command = BindStreamAlias.model_validate(value)

        if not self._job_manager.has_job(command.job_id):
            # Not the actor: another service hosts this job (or no service does).
            logger.debug(
                "bind_stream_alias_not_actor",
                alias=command.alias,
                job_id=str(command.job_id),
            )
            return None

        try:
            self._registry.bind(command.alias, command.job_id, command.output_name)
        except AliasAlreadyBoundError as e:
            logger.warning(
                "bind_stream_alias_rejected",
                alias=command.alias,
                reason=str(e),
            )
            return self._ack(command.message_id, command.alias, error=str(e))
        except Exception as e:
            logger.exception("bind_stream_alias_failed", alias=command.alias)
            return self._ack(command.message_id, command.alias, error=str(e))

        return self._ack(command.message_id, command.alias)

    def unbind(self, source_name: str, value: dict) -> CommandAcknowledgement | None:
        _ = source_name  # Not used: aliases are global per-service.
        command = UnbindStreamAlias.model_validate(value)

        if not self._registry.has(command.alias):
            # Not the actor: another service holds this alias (or none does).
            logger.debug("unbind_stream_alias_not_actor", alias=command.alias)
            return None

        try:
            self._registry.unbind(command.alias)
        except Exception as e:
            logger.exception("unbind_stream_alias_failed", alias=command.alias)
            return self._ack(command.message_id, command.alias, error=str(e))

        return self._ack(command.message_id, command.alias)

    @staticmethod
    def _ack(
        message_id: str | None, alias: str, *, error: str | None = None
    ) -> CommandAcknowledgement | None:
        if message_id is None:
            return None
        if error is not None:
            return CommandAcknowledgement(
                message_id=message_id,
                device=alias,
                response=AcknowledgementResponse.ERR,
                message=error,
            )
        return CommandAcknowledgement(
            message_id=message_id,
            device=alias,
            response=AcknowledgementResponse.ACK,
        )
