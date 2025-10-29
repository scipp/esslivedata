# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""HTTP transport implementation for dashboard config messages."""

import json
import logging
from typing import Any

import requests

from ..config.models import ConfigKey
from ..core.message import CONFIG_STREAM_ID, Message
from ..handlers.config_handler import ConfigUpdate
from .message_transport import MessageTransport


class HTTPConfigTransport(MessageTransport[ConfigKey, dict[str, Any]]):
    """
    HTTP-based transport for config messages.

    Uses an HTTPServiceSink to expose config messages that backend services can poll,
    and polls backend HTTP endpoints for incoming config messages.
    This is the HTTP equivalent of KafkaTransport for the dashboard.

    Parameters
    ----------
    sink:
        HTTP sink for publishing config messages (dashboard exposes this)
    poll_url:
        Base URL for polling config messages from backend (GET endpoint)
    logger:
        Optional logger instance
    timeout:
        HTTP request timeout in seconds
    """

    def __init__(
        self,
        sink: Any,  # HTTPServiceSink
        poll_url: str,
        logger: logging.Logger | None = None,
        timeout: float = 1.0,
    ):
        self._sink = sink
        self._poll_url = poll_url
        self._poll_endpoint = '/config'  # Poll backend's /config endpoint
        self._logger = logger or logging.getLogger(__name__)
        self._timeout = timeout
        self._session = requests.Session()

    def send_messages(self, messages: list[tuple[ConfigKey, dict[str, Any]]]) -> None:
        """
        Send config messages by publishing to the HTTP sink.

        Backend services will poll this sink to receive config updates.

        Parameters
        ----------
        messages:
            List of (config_key, config_value) tuples to send
        """
        if not messages:
            return

        try:
            # Convert to Message format for HTTP sink
            http_messages = []
            for key, value in messages:
                config_dict = {
                    'key': str(key),
                    'value': value,
                }
                http_messages.append(
                    Message(stream=CONFIG_STREAM_ID, value=config_dict)
                )

            self._sink.publish_messages(http_messages)

            self._logger.debug(
                "Published %d config messages to HTTP sink", len(messages)
            )

        except Exception as e:
            self._logger.error("Error sending messages: %s", e)

    def receive_messages(self) -> list[tuple[ConfigKey, dict[str, Any]]]:
        """
        Poll for config messages via HTTP GET.

        In HTTP mode, config flow is unidirectional (dashboard → backend only).
        Backend services do not publish config back to the dashboard, so this
        method always returns an empty list.

        Returns
        -------
        :
            Empty list (no config loopback in HTTP mode)
        """
        # HTTP mode does not support config loopback from backend to dashboard
        # Backend services poll the dashboard for config, but don't publish back
        return []

    def _decode_update(self, item: dict[str, Any]) -> ConfigUpdate | None:
        """
        Decode a JSON message into a ConfigUpdate.

        Parameters
        ----------
        item:
            Dictionary with 'key' and 'value' fields

        Returns
        -------
        :
            Decoded ConfigUpdate or None if decoding fails
        """
        try:
            # Convert JSON dict to ConfigUpdate format
            key_bytes = item['key'].encode('utf-8')
            value_bytes = json.dumps(item['value']).encode('utf-8')

            from ..kafka.message_adapter import RawConfigItem

            return ConfigUpdate.from_raw(
                RawConfigItem(key=key_bytes, value=value_bytes)
            )
        except Exception as e:
            self._logger.exception("Failed to decode config message: %s", e)
            return None

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()
