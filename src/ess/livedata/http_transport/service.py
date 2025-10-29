"""HTTP service integration for data services.

Provides a combined sink and FastAPI application server.
"""
# ruff: noqa: S104  # Binding to 0.0.0.0 is intentional for HTTP services
# ruff: noqa: S101  # assert is acceptable in this context

import logging
import threading
from contextlib import contextmanager
from typing import Generic, TypeVar

import uvicorn
from fastapi import FastAPI

from ..core import MessageSink
from ..core.message import Message
from .app import create_message_api
from .serialization import MessageSerializer
from .sink import QueueBasedMessageSink

T = TypeVar("T")

logger = logging.getLogger(__name__)


class HTTPServiceSink(MessageSink[T], Generic[T]):
    """Message sink that exposes messages via HTTP API.

    Combines a queue-based sink with a FastAPI server.
    """

    def __init__(
        self,
        serializer: MessageSerializer[T],
        host: str = "0.0.0.0",
        port: int = 8000,
        max_queue_size: int = 1000,
        max_age_seconds: float | None = None,
        logger_: logging.Logger | None = None,
    ):
        """
        Parameters
        ----------
        serializer:
            The serializer to use for HTTP responses.
        host:
            The host to bind the HTTP server to.
        port:
            The port to bind the HTTP server to.
        max_queue_size:
            Maximum number of messages to keep in the queue.
        max_age_seconds:
            Maximum age of messages in seconds before they are dropped.
        logger_:
            Logger to use for logging.
        """
        self._logger = logger_ or logger
        self._host = host
        self._port = port

        # Create the underlying queue-based sink
        self._queue_sink: QueueBasedMessageSink[T] = QueueBasedMessageSink(
            max_size=max_queue_size,
            max_age_seconds=max_age_seconds,
            logger=self._logger,
        )

        # Create FastAPI app
        self._app: FastAPI = create_message_api(
            sink=self._queue_sink, serializer=serializer, logger=self._logger
        )

        # Server thread management
        self._server_thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._started = False

    @property
    def app(self) -> FastAPI:
        """Returns the FastAPI application."""
        return self._app

    def publish_messages(self, messages: list[Message[T]]) -> None:
        """Publish messages to the queue."""
        self._queue_sink.publish_messages(messages)

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        if self._started:
            self._logger.warning("HTTP server already started")
            return

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)

        def run_server() -> None:
            """Run the server in a thread."""
            assert self._server is not None
            self._server.run()

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        self._started = True
        self._logger.info(
            "HTTP API server started on http://%s:%d", self._host, self._port
        )

    def stop(self) -> None:
        """Stop the HTTP server."""
        if not self._started:
            return

        if self._server is not None:
            self._server.should_exit = True

        if self._server_thread is not None:
            self._server_thread.join(timeout=5.0)

        self._started = False
        self._logger.info("HTTP API server stopped")

    def __enter__(self) -> "HTTPServiceSink[T]":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.stop()


@contextmanager
def http_service_sink(
    serializer: MessageSerializer[T],
    host: str = "0.0.0.0",
    port: int = 8000,
    max_queue_size: int = 1000,
    max_age_seconds: float | None = None,
    logger_: logging.Logger | None = None,
):
    """Context manager for HTTP service sink.

    Parameters
    ----------
    serializer:
        The serializer to use for HTTP responses.
    host:
        The host to bind the HTTP server to.
    port:
        The port to bind the HTTP server to.
    max_queue_size:
        Maximum number of messages to keep in the queue.
    max_age_seconds:
        Maximum age of messages in seconds before they are dropped.
    logger_:
        Logger to use for logging.

    Yields
    ------
    :
        The HTTP service sink with the server running.
    """
    sink = HTTPServiceSink(
        serializer=serializer,
        host=host,
        port=port,
        max_queue_size=max_queue_size,
        max_age_seconds=max_age_seconds,
        logger_=logger_,
    )
    try:
        sink.start()
        yield sink
    finally:
        sink.stop()
