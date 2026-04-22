# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Generic, Protocol

import structlog

from .message import Message, MessageSink, MessageSource, Tin, Tout

logger = structlog.get_logger(__name__)


class Processor(Protocol):
    """
    Protocol for a processor that processes messages. Used by :py:class:`Service`.
    """

    def process(self) -> None:
        pass

    def finalize(self, *, error: str | None = None) -> None:
        """Finalize after the worker thread has joined.

        Called once by :class:`Service` before shutdown, with ``error`` set to
        the stringified exception if the loop exited unexpectedly.
        """


class IdentityProcessor(Generic[Tin, Tout]):
    """
    Simple processor that passes messages directly from source to sink.

    Used by fake data producers where no actual processing is needed.
    """

    def __init__(
        self,
        *,
        source: MessageSource[Message[Tin]],
        sink: MessageSink[Tout],
    ) -> None:
        self._source = source
        self._sink = sink

    def process(self) -> None:
        messages = self._source.get_messages()
        logger.debug('processing_messages', count=len(messages))
        self._sink.publish_messages(messages)

    def finalize(self, *, error: str | None = None) -> None:
        pass
