# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Routing serializers that dispatch to per-kind :class:`MessageSerializer` instances.

Mirrors the source-side routers in :mod:`ess.livedata.kafka.message_adapter`
(``RouteBySchemaAdapter``, ``RouteByTopicAdapter``).
"""

from __future__ import annotations

from typing import Generic, TypeVar

from ..core.job import JobStatus, ServiceStatus
from ..core.message import Message, StreamKind
from .sink import MessageSerializer, SerializationError, SerializedMessage

T = TypeVar('T')


class RouteByStreamKindSerializer(Generic[T]):
    """
    Routes a message to one of several serializers based on ``message.stream.kind``.
    """

    def __init__(self, routes: dict[StreamKind, MessageSerializer]) -> None:
        self._routes = routes

    def serialize(self, message: Message[T]) -> SerializedMessage:
        kind = message.stream.kind
        try:
            serializer = self._routes[kind]
        except KeyError:
            raise SerializationError(
                f"No serializer configured for stream kind {kind!r}. "
                f"Configured kinds: {list(self._routes.keys())}"
            ) from None
        return serializer.serialize(message)


class RouteByStatusTypeSerializer:
    """
    Dispatches status messages to serializers based on the payload type.

    ``LIVEDATA_STATUS`` is unusual in that a single stream carries two value types
    (:class:`ServiceStatus` for service heartbeats and :class:`JobStatus` for job
    heartbeats). This tiny router keeps that dispatch isolated rather than muddying
    :class:`RouteByStreamKindSerializer`.
    """

    def __init__(
        self,
        *,
        service: MessageSerializer[ServiceStatus],
        job: MessageSerializer[JobStatus],
    ) -> None:
        self._service = service
        self._job = job

    def serialize(
        self, message: Message[ServiceStatus | JobStatus]
    ) -> SerializedMessage:
        if isinstance(message.value, ServiceStatus):
            return self._service.serialize(message)  # type: ignore[arg-type]
        return self._job.serialize(message)  # type: ignore[arg-type]
