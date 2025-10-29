# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""In-memory transport implementation for dashboard config messages."""

import json
import logging
from typing import Any

from ..config.models import ConfigKey
from ..handlers.config_handler import ConfigUpdate
from ..in_memory import InMemoryBroker
from ..kafka.message_adapter import RawConfigItem
from .throttling_message_handler import MessageTransport


class InMemoryConfigTransport(MessageTransport[ConfigKey, dict[str, Any]]):
    """
    In-memory transport for dashboard configuration messages.

    Uses the InMemoryBroker to publish and receive config updates without Kafka.
    Implements the same MessageTransport protocol as KafkaTransport.

    Parameters
    ----------
    broker:
        The in-memory broker instance
    topic:
        The topic name for config messages
    logger:
        Optional logger for transport events
    max_batch_size:
        Maximum number of messages to consume in one batch
    poll_timeout:
        Timeout in seconds when polling for messages. Default 0.1.
    """

    def __init__(
        self,
        broker: InMemoryBroker,
        topic: str,
        logger: logging.Logger | None = None,
        max_batch_size: int = 100,
        poll_timeout: float = 0.1,
    ):
        self._broker = broker
        self._topic = topic
        self._logger = logger or logging.getLogger(__name__)
        self._max_batch_size = max_batch_size
        self._poll_timeout = poll_timeout
        # Subscribe to the config topic - returns a Queue
        self._queue = self._broker.subscribe([topic])

    def send_messages(self, messages: list[tuple[ConfigKey, dict[str, Any]]]) -> None:
        """Send messages to the in-memory broker."""
        try:
            from ..in_memory.transport import FakeKafkaMessage
            import time

            fake_messages = []
            for key, value in messages:
                fake_msg = FakeKafkaMessage(
                    topic=self._topic,
                    key=str(key).encode("utf-8"),
                    value=json.dumps(value).encode("utf-8"),
                    timestamp=int(time.time() * 1_000_000_000),  # nanoseconds
                )
                fake_messages.append(fake_msg)

            if fake_messages:
                self._logger.debug(
                    "Publishing %d config messages to topic '%s'",
                    len(fake_messages),
                    self._topic,
                )
                self._broker.publish(self._topic, fake_messages)

        except Exception as e:
            self._logger.error("Error sending messages: %s", e)

    def receive_messages(self) -> list[tuple[ConfigKey, dict[str, Any]]]:
        """Receive messages from the in-memory broker."""
        from queue import Empty

        received = []

        try:
            # Get messages from queue - similar to InMemoryMessageSource
            # Block for first message with timeout
            try:
                first_msg = self._queue.get(timeout=self._poll_timeout)
                try:
                    decoded_update = self._decode_update(first_msg)
                    if decoded_update:
                        received.append(
                            (decoded_update.config_key, decoded_update.value)
                        )
                except Exception as e:
                    self._logger.error("Failed to process incoming message: %s", e)
            except Empty:
                # Timeout - no messages available
                return []

            # Drain remaining messages (non-blocking, up to max_batch_size)
            while len(received) < self._max_batch_size:
                try:
                    msg = self._queue.get_nowait()
                    try:
                        decoded_update = self._decode_update(msg)
                        if decoded_update:
                            received.append(
                                (decoded_update.config_key, decoded_update.value)
                            )
                    except Exception as e:
                        self._logger.error("Failed to process incoming message: %s", e)
                except Empty:
                    break

        except Exception as e:
            self._logger.error("Error receiving messages: %s", e)

        return received

    def _decode_update(self, msg) -> ConfigUpdate | None:
        """Decode a message into a ConfigUpdate."""
        try:
            return ConfigUpdate.from_raw(
                RawConfigItem(key=msg.key(), value=msg.value())
            )
        except Exception as e:
            self._logger.exception("Failed to decode config message: %s", e)
            return None
