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

from ..config.streams import stream_kind_to_topic
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
    topic separation. Each StreamKind is mapped to an endpoint derived from its
    topic name (with instrument prefix removed).

    Multiple StreamKinds can share the same endpoint if they map to the same
    topic (e.g., MONITOR_COUNTS and MONITOR_EVENTS both → /beam_monitor).

    Parameters
    ----------
    instrument:
        Instrument name used to derive endpoint paths from StreamKinds.
    stream_serializers:
        Mapping from StreamKind to serializer for that stream type.
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
        instrument: str,
        stream_serializers: dict[StreamKind, MessageSerializer[Any]],
        host: str = "0.0.0.0",
        port: int = 8000,
        max_queue_size: int = 1000,
        max_age_seconds: float | None = None,
        logger_: logging.Logger | None = None,
    ):
        self._logger = logger_ or logger
        self._host = host
        self._port = port
        self._instrument = instrument

        # Build endpoint mappings from StreamKinds
        # Multiple StreamKinds may map to the same endpoint
        endpoint_to_kinds: dict[str, list[StreamKind]] = {}
        endpoint_to_serializer: dict[str, MessageSerializer[Any]] = {}

        for kind, serializer in stream_serializers.items():
            topic = stream_kind_to_topic(instrument, kind)
            endpoint = f"/{topic.removeprefix(f'{instrument}_')}"

            if endpoint not in endpoint_to_kinds:
                endpoint_to_kinds[endpoint] = []
                endpoint_to_serializer[endpoint] = serializer
            else:
                # Verify all StreamKinds for same endpoint use same serializer
                if endpoint_to_serializer[endpoint] != serializer:
                    raise ValueError(
                        f"StreamKinds mapping to endpoint {endpoint} must use "
                        f"the same serializer"
                    )
            endpoint_to_kinds[endpoint].append(kind)

        # Create one sink per unique endpoint
        self._endpoint_to_sink: dict[str, QueueBasedMessageSink[Any]] = {
            endpoint: QueueBasedMessageSink(
                max_size=max_queue_size,
                max_age_seconds=max_age_seconds,
                logger=self._logger,
            )
            for endpoint in endpoint_to_kinds
        }

        # Build reverse lookup for fast routing: StreamKind → sink
        self._kind_to_sink: dict[StreamKind, QueueBasedMessageSink[Any]] = {}
        for endpoint, kinds in endpoint_to_kinds.items():
            sink = self._endpoint_to_sink[endpoint]
            for kind in kinds:
                self._kind_to_sink[kind] = sink

        # Create FastAPI app with all endpoints
        self._app: FastAPI = create_multi_endpoint_api(
            endpoints={
                endpoint: (sink, endpoint_to_serializer[endpoint])
                for endpoint, sink in self._endpoint_to_sink.items()
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
        # Group messages by sink
        sink_to_messages: dict[QueueBasedMessageSink[Any], list[Message[Any]]] = {}

        for msg in messages:
            sink = self._kind_to_sink.get(msg.stream.kind)
            if sink is None:
                self._logger.warning(
                    "No sink configured for StreamKind %s, dropping message",
                    msg.stream.kind,
                )
                continue

            if sink not in sink_to_messages:
                sink_to_messages[sink] = []
            sink_to_messages[sink].append(msg)

        # Publish to each sink
        for sink, sink_messages in sink_to_messages.items():
            sink.publish_messages(sink_messages)

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
        endpoints_str = ", ".join(sorted(self._endpoint_to_sink.keys()))
        self._logger.info(
            "HTTP API server ready on http://%s:%d with endpoints: %s",
            self._host,
            self._port,
            endpoints_str,
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
