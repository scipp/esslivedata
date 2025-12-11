# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""HTTP-based transport implementation for the dashboard.

This transport allows injecting data via HTTP POST requests instead of consuming
from Kafka. Useful for testing, development, and screenshot generation without
requiring a Kafka broker.

The HTTP endpoint accepts da00-serialized messages, using the same format as Kafka.
"""

import base64
import json
import logging
import threading
from collections.abc import Sequence
from types import TracebackType

import scipp as sc
from streaming_data_types import dataarray_da00

from ..core.message import Message, StreamId, StreamKind
from ..kafka.scipp_da00_compat import da00_to_scipp
from .transport import DashboardResources, NullMessageSink, Transport


class QueueableMessageSource:
    """
    Message source that returns messages from an internal queue.

    Messages can be injected via the `queue_message` method and will be
    returned by subsequent calls to `get_messages`.
    """

    def __init__(self) -> None:
        self._messages: list[Message[sc.DataArray]] = []
        self._lock = threading.Lock()

    def queue_message(self, message: Message[sc.DataArray]) -> None:
        """Add a message to the queue."""
        with self._lock:
            self._messages.append(message)

    def queue_messages(self, messages: Sequence[Message[sc.DataArray]]) -> None:
        """Add multiple messages to the queue."""
        with self._lock:
            self._messages.extend(messages)

    def get_messages(self) -> Sequence[Message[sc.DataArray]]:
        """Return and clear all queued messages."""
        with self._lock:
            messages = self._messages
            self._messages = []
            return messages


def deserialize_da00_to_message(
    payload: bytes, timestamp_ns: int | None = None
) -> Message[sc.DataArray]:
    """
    Deserialize a da00 payload to a Message with sc.DataArray value.

    Parameters
    ----------
    payload:
        Raw da00 bytes as received from Kafka or HTTP.
    timestamp_ns:
        Optional timestamp override. If None, uses timestamp from da00.

    Returns
    -------
    :
        Message with stream ID derived from da00 source_name and sc.DataArray value.
    """
    da00: dataarray_da00.da00_DataArray_t
    da00 = dataarray_da00.deserialise_da00(payload)  # type: ignore[reportAssignmentType]

    timestamp = timestamp_ns if timestamp_ns is not None else da00.timestamp_ns
    stream_id = StreamId(kind=StreamKind.LIVEDATA_DATA, name=da00.source_name)
    value = da00_to_scipp(da00.data)

    return Message(timestamp=timestamp, stream=stream_id, value=value)


class HttpTransport(Transport[DashboardResources]):
    """
    HTTP-based transport for the dashboard.

    Provides a message source that can receive data via HTTP POST requests.
    The HTTP endpoint is set up when the Panel server starts, using Panel's
    REST endpoint mechanism.

    Parameters
    ----------
    instrument:
        Instrument name (e.g., 'dummy', 'dream', 'bifrost')
    logger:
        Logger instance for logging
    """

    def __init__(
        self,
        *,
        instrument: str,
        logger: logging.Logger | None = None,
    ):
        self._instrument = instrument
        self._logger = logger or logging.getLogger(__name__)
        self._message_source = QueueableMessageSource()

    def __enter__(self) -> DashboardResources:
        """Set up HTTP transport and return dashboard resources."""
        self._logger.info("HttpTransport initialized for %s", self._instrument)

        return DashboardResources(
            message_source=self._message_source,
            command_sink=NullMessageSink(),
            roi_sink=NullMessageSink(),
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up HTTP transport resources."""
        self._logger.info("HttpTransport cleaned up")

    def start(self) -> None:
        """Start the transport (no-op for HTTP, endpoint is always available)."""
        pass

    def stop(self) -> None:
        """Stop the transport (no-op for HTTP)."""
        pass

    @property
    def message_source(self) -> QueueableMessageSource:
        """Get the message source for direct injection."""
        return self._message_source

    def inject_da00(self, payload: bytes, timestamp_ns: int | None = None) -> None:
        """
        Inject a da00-serialized message directly.

        Parameters
        ----------
        payload:
            Raw da00 bytes.
        timestamp_ns:
            Optional timestamp override.
        """
        message = deserialize_da00_to_message(payload, timestamp_ns)
        self._message_source.queue_message(message)
        self._logger.debug("Injected da00 message for stream %s", message.stream)

    def inject_from_json(self, json_data: dict) -> None:
        """
        Inject a message from JSON payload.

        Expected format:
        {
            "payload_base64": "<base64-encoded da00 bytes>",
            "timestamp_ns": <optional int>
        }

        Parameters
        ----------
        json_data:
            JSON dict with payload_base64 and optional timestamp_ns.
        """
        payload = base64.b64decode(json_data['payload_base64'])
        timestamp_ns = json_data.get('timestamp_ns')
        self.inject_da00(payload, timestamp_ns)

    def handle_post_request(self, body: bytes) -> dict:
        """
        Handle an HTTP POST request body.

        This method is called by the Panel REST endpoint handler.

        Parameters
        ----------
        body:
            Raw request body (JSON).

        Returns
        -------
        :
            Response dict with status.
        """
        try:
            json_data = json.loads(body)
            self.inject_from_json(json_data)
            return {'status': 'ok'}
        except Exception as e:
            self._logger.exception("Error handling POST request")
            return {'status': 'error', 'message': str(e)}
