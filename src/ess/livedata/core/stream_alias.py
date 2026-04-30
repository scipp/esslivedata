# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Stream-alias binding mechanism.

A stream alias is a stable, externally meaningful name (e.g., ``"fom-0"``) bound
to a particular ``(job_id, output_name)`` pair. The first use case is the FOM
mechanism, where NICOS consumes a stable stream regardless of which workflow or
job is currently producing it. The mechanism itself is generic and has nothing
FOM-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import pydantic

from ess.livedata.config.workflow_spec import JobId

T = TypeVar('T')


class BindStreamAlias(pydantic.BaseModel):
    """Bind a stable stream alias to a specific (job, output) pair."""

    key: ClassVar[str] = "bind_stream_alias"
    message_id: str | None = pydantic.Field(
        default=None,
        description=(
            "Unique identifier for command correlation. Frontend generates "
            "this UUID and backend echoes it in CommandAcknowledgement."
        ),
    )
    alias: str = pydantic.Field(
        description="Stable stream alias to bind, e.g. 'fom-0'."
    )
    job_id: JobId = pydantic.Field(description="Target job.")
    output_name: str = pydantic.Field(
        description="Target output name within the job's result."
    )


class UnbindStreamAlias(pydantic.BaseModel):
    """Release a previously bound stream alias."""

    key: ClassVar[str] = "unbind_stream_alias"
    message_id: str | None = pydantic.Field(
        default=None,
        description=(
            "Unique identifier for command correlation. Frontend generates "
            "this UUID and backend echoes it in CommandAcknowledgement."
        ),
    )
    alias: str = pydantic.Field(description="Stream alias to release.")


class AliasAlreadyBoundError(Exception):
    """Raised when binding an alias that is already in use locally."""


class StreamAliasRegistry:
    """In-memory registry mapping aliases to ``(JobId, output_name)``.

    Per-service: each backend service has its own registry holding only the
    aliases bound to jobs that this service hosts.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, tuple[JobId, str]] = {}
        self._reverse: dict[tuple[JobId, str], str] = {}

    def bind(self, alias: str, job_id: JobId, output_name: str) -> None:
        """Register a binding. Raises if the alias is already bound."""
        if alias in self._bindings:
            raise AliasAlreadyBoundError(
                f"Alias {alias!r} is already bound; unbind it first."
            )
        self._bindings[alias] = (job_id, output_name)
        self._reverse[(job_id, output_name)] = alias

    def unbind(self, alias: str) -> None:
        """Remove a binding. No-op if the alias is not bound."""
        target = self._bindings.pop(alias, None)
        if target is not None:
            self._reverse.pop(target, None)

    def has(self, alias: str) -> bool:
        return alias in self._bindings

    def lookup(self, job_id: JobId, output_name: str) -> str | None:
        """Return the alias bound to ``(job_id, output_name)``, or ``None``."""
        return self._reverse.get((job_id, output_name))


@dataclass(frozen=True, slots=True, kw_only=True)
class AliasedResult(Generic[T]):
    """Wrapper carrying the underlying data and the bound alias.

    Used as the value type of mirror messages with stream kind
    ``LIVEDATA_FOM`` so the serializer can set the Kafka message key to the
    alias bytes without any field on :class:`Message` itself.
    """

    data: T
    alias: str
