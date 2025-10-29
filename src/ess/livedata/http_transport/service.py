"""HTTP service integration for data services.

Provides a combined sink and FastAPI application server.
"""
# ruff: noqa: S104  # Binding to 0.0.0.0 is intentional for HTTP services
# ruff: noqa: S101  # assert is acceptable in this context

import logging
import socket
import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI

from ..core import MessageSink
from ..core.message import Message, StreamKind
from .app import create_multi_endpoint_api
from .serialization import MessageSerializer
from .sink import QueueBasedMessageSink

logger = logging.getLogger(__name__)


def _wait_for_port(host: str, port: int, timeout: float = 5.0) -> bool:
    """
    Wait for a port to become available (server is listening).

    Parameters
    ----------
    host:
        Host to check
    port:
        Port to check
    timeout:
        Maximum time to wait in seconds

    Returns
    -------
    :
        True if port is available, False if timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            # Try to connect - if it succeeds, server is listening
            result = sock.connect_ex(('127.0.0.1' if host == '0.0.0.0' else host, port))
            sock.close()
            if result == 0:
                return True
        except OSError:
            pass
        time.sleep(0.1)
    return False


class HTTPMultiEndpointSink(MessageSink[Any]):
    """
    Message sink that exposes multiple HTTP endpoints for different message types.

    Routes messages to different queues based on StreamKind, mirroring Kafka's
    topic separation. Each StreamKind gets its own endpoint and serializer:
    - LIVEDATA_DATA → /data (DA00 format)
    - LIVEDATA_STATUS → /status (X5F2 format)
    - LIVEDATA_CONFIG → /config (JSON format)

    Parameters
    ----------
    data_serializer:
        Serializer for DATA messages.
    status_serializer:
        Serializer for STATUS messages.
    config_serializer:
        Serializer for CONFIG messages.
    host:
        The host to bind the HTTP server to.
    port:
        The port to bind the HTTP server to.
    max_queue_size:
        Maximum number of messages per queue.
    max_age_seconds:
        Maximum age of messages in seconds before they are dropped.
    logger_:
        Logger to use for logging.
    """

    def __init__(
        self,
        data_serializer: MessageSerializer[Any],
        status_serializer: MessageSerializer[Any],
        config_serializer: MessageSerializer[Any],
        host: str = "0.0.0.0",
        port: int = 8000,
        max_queue_size: int = 1000,
        max_age_seconds: float | None = None,
        logger_: logging.Logger | None = None,
    ):
        self._logger = logger_ or logger
        self._host = host
        self._port = port

        # Create separate sinks for each message type
        self._data_sink: QueueBasedMessageSink[Any] = QueueBasedMessageSink(
            max_size=max_queue_size,
            max_age_seconds=max_age_seconds,
            logger=self._logger,
        )
        self._status_sink: QueueBasedMessageSink[Any] = QueueBasedMessageSink(
            max_size=max_queue_size,
            max_age_seconds=max_age_seconds,
            logger=self._logger,
        )
        self._config_sink: QueueBasedMessageSink[Any] = QueueBasedMessageSink(
            max_size=max_queue_size,
            max_age_seconds=max_age_seconds,
            logger=self._logger,
        )

        # Create FastAPI app with multiple endpoints
        self._app: FastAPI = create_multi_endpoint_api(
            endpoints={
                "/data": (self._data_sink, data_serializer),
                "/status": (self._status_sink, status_serializer),
                "/config": (self._config_sink, config_serializer),
            },
            logger=self._logger,
        )

        # Server thread management
        self._server_thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._started = False

    @property
    def app(self) -> FastAPI:
        """Returns the FastAPI application."""
        return self._app

    def publish_messages(self, messages: list[Message[Any]]) -> None:
        """
        Publish messages to the appropriate queue based on StreamKind.

        Parameters
        ----------
        messages:
            List of messages to route to appropriate endpoints.
        """
        data_messages = []
        status_messages = []
        config_messages = []

        for msg in messages:
            if msg.stream.kind == StreamKind.LIVEDATA_STATUS:
                status_messages.append(msg)
            elif msg.stream.kind == StreamKind.LIVEDATA_CONFIG:
                config_messages.append(msg)
            else:
                # All other messages (DATA, MONITOR_COUNTS, DETECTOR_EVENTS, etc.)
                # go to /data endpoint
                data_messages.append(msg)

        # Publish to appropriate sinks
        if data_messages:
            self._data_sink.publish_messages(data_messages)
        if status_messages:
            self._status_sink.publish_messages(status_messages)
        if config_messages:
            self._config_sink.publish_messages(config_messages)

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

        # Wait for the server to be ready
        if not _wait_for_port(self._host, self._port, timeout=5.0):
            self._logger.error(
                "HTTP server failed to start on http://%s:%d within timeout",
                self._host,
                self._port,
            )
            raise RuntimeError(
                f"HTTP server failed to start on {self._host}:{self._port}"
            )

        self._started = True
        self._logger.info(
            "HTTP API server ready on http://%s:%d with endpoints: "
            "/data, /status, /config",
            self._host,
            self._port,
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

    def __enter__(self) -> "HTTPMultiEndpointSink":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.stop()
