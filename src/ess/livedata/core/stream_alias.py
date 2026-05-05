# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Stream-alias binding mechanism.

A stream alias is a stable, externally meaningful name (e.g., ``"fom-0"``) that
maps to one or more ``(job_id, output_name)`` pairs. The first use case is the
FOM mechanism, where NICOS consumes a stable Kafka stream regardless of which
workflow or jobs are currently producing it. When an alias has multiple
bindings (one workflow running across N source streams), all matching outputs
are mirrored under the same alias and the consumer is expected to aggregate
them (typically by summing the substreams). The mechanism itself is generic
and has nothing FOM-specific.
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
    """Release a previously bound stream alias.

    Removes all ``(job_id, output_name)`` bindings registered under the alias.
    """

    key: ClassVar[str] = "unbind_stream_alias"
    message_id: str | None = pydantic.Field(
        default=None,
        description=(
            "Unique identifier for command correlation. Frontend generates "
            "this UUID and backend echoes it in CommandAcknowledgement."
        ),
    )
    alias: str = pydantic.Field(description="Stream alias to release.")


class BindingConflictError(Exception):
    """Raised when a ``(job_id, output_name)`` is already bound to a different alias."""


class StreamAliasRegistry:
    """In-memory registry mapping aliases to ``(JobId, output_name)`` pairs.

    A single alias may have multiple bindings: when one workflow runs across
    N source streams, all N ``(job_id, output_name)`` pairs share the alias
    and consumers see the substreams interleaved on a single Kafka stream.

    A given ``(job_id, output_name)`` belongs to at most one alias; attempting
    to bind it under a different alias raises :class:`BindingConflictError`.

    Per-service: each backend service has its own registry holding only the
    bindings whose ``job_id`` this service hosts.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, set[tuple[JobId, str]]] = {}
        self._reverse: dict[tuple[JobId, str], str] = {}

    def bind(self, alias: str, job_id: JobId, output_name: str) -> None:
        """Register a binding.

        Idempotent if the same ``(alias, job_id, output_name)`` triple is
        re-registered. Raises :class:`BindingConflictError` if the
        ``(job_id, output_name)`` is already bound under a *different* alias.
        """
        existing = self._reverse.get((job_id, output_name))
        if existing is not None and existing != alias:
            raise BindingConflictError(
                f"Output ({job_id!s}, {output_name!r}) is already bound to "
                f"alias {existing!r}; cannot bind under {alias!r}."
            )
        self._bindings.setdefault(alias, set()).add((job_id, output_name))
        self._reverse[(job_id, output_name)] = alias

    def unbind(self, alias: str) -> None:
        """Remove all bindings under the alias. No-op if the alias is unknown."""
        bindings = self._bindings.pop(alias, None)
        if bindings is None:
            return
        for pair in bindings:
            self._reverse.pop(pair, None)

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
