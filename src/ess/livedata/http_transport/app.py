# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""FastAPI application factory for exposing message queues via HTTP."""

import logging
from typing import Generic

from fastapi import FastAPI, Response

from ..core.message import T
from .serialization import MessageSerializer
from .sink import QueueBasedMessageSink


def create_message_api(
    sink: QueueBasedMessageSink[T],
    serializer: MessageSerializer[T],
    logger: logging.Logger | None = None,
) -> FastAPI:
    """
    Create a FastAPI application that exposes a message queue.

    The created application provides a GET /messages endpoint that returns
    all available messages from the sink's queue.

    Parameters
    ----------
    sink:
        The message sink whose queue should be exposed.
    serializer:
        Serializer for encoding messages in HTTP responses.
    logger:
        Optional logger instance.

    Returns
    -------
    :
        FastAPI application instance.

    Examples
    --------
    >>> from ess.livedata.http import QueueBasedMessageSink, JSONMessageSerializer
    >>> sink = QueueBasedMessageSink()
    >>> serializer = JSONMessageSerializer()
    >>> app = create_message_api(sink, serializer)
    >>> # Run with: uvicorn module:app --host 0.0.0.0 --port 8000
    """
    app = FastAPI(title="ESS Livedata Message API")
    _logger = logger or logging.getLogger(__name__)

    @app.get("/messages")
    def get_messages(response: Response):
        """
        Get all available messages from the queue.

        Returns all messages currently in the queue and clears it.
        Returns 204 No Content if no messages are available.
        """
        messages = sink.get_messages()

        if not messages:
            response.status_code = 204
            return None

        _logger.debug("Serving %d messages", len(messages))
        serialized = serializer.serialize(messages)
        return Response(content=serialized, media_type="application/json")

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "healthy", "queue_size": sink.size}

    return app


class MessageAPIWrapper(Generic[T]):
    """
    Wrapper that combines sink, serializer, and FastAPI app.

    Convenient for managing all components of an HTTP message endpoint together.

    Parameters
    ----------
    sink:
        The message sink to expose.
    serializer:
        Serializer for messages.
    logger:
        Optional logger instance.

    Examples
    --------
    >>> from ess.livedata.http import QueueBasedMessageSink, JSONMessageSerializer
    >>> sink = QueueBasedMessageSink()
    >>> serializer = JSONMessageSerializer()
    >>> api = MessageAPIWrapper(sink, serializer)
    >>> # Run with: uvicorn module:api.app --host 0.0.0.0 --port 8000
    """

    def __init__(
        self,
        sink: QueueBasedMessageSink[T],
        serializer: MessageSerializer[T],
        logger: logging.Logger | None = None,
    ):
        self.sink = sink
        self.serializer = serializer
        self.app = create_message_api(sink, serializer, logger)
