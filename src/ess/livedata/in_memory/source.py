# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""In-memory message source implementation."""

from __future__ import annotations

import logging
from queue import Empty, Queue
from typing import Any, TypeVar

from ess.livedata.in_memory.broker import InMemoryBroker

logger = logging.getLogger(__name__)

T = TypeVar("T")


class InMemoryMessageSource:
    """
    In-memory implementation of MessageSource protocol.

    Receives messages from an InMemoryBroker subscription queue.

    Parameters
    ----------
    broker:
        Broker instance to subscribe to
    topics:
        List of topic names to subscribe to
    poll_timeout:
        Maximum time to wait for messages in seconds. Default 0.1.
    batch_size:
        Maximum messages to return per get_messages() call. Default 100.
    """

    def __init__(
        self,
        broker: InMemoryBroker,
        topics: list[str],
        poll_timeout: float = 0.1,
        batch_size: int = 100,
    ):
        self._broker = broker
        self._topics = topics
        self._poll_timeout = poll_timeout
        self._batch_size = batch_size
        self._queue: Queue = broker.subscribe(topics)
        logger.info("InMemoryMessageSource created for topics: %s", topics)

    def get_messages(self) -> list[Any]:
        """
        Get available messages from subscription queue.

        Returns FakeKafkaMessage objects that can be processed by Kafka adapters.

        Returns
        -------
        :
            List of FakeKafkaMessage objects (may be empty if timeout expires)
        """
        messages: list[Any] = []  # Actually FakeKafkaMessage, but avoid circular import

        try:
            # Block until first message or timeout
            first_message = self._queue.get(timeout=self._poll_timeout)
            messages.append(first_message)

            # Drain remaining messages (non-blocking, up to batch_size)
            while len(messages) < self._batch_size:
                try:
                    messages.append(self._queue.get_nowait())
                except Empty:
                    break

        except Empty:
            # Timeout - no messages available
            pass

        if messages:
            logger.debug("Got %d messages from queue", len(messages))

        return messages

    def close(self) -> None:
        """Unsubscribe from broker."""
        self._broker.unsubscribe(self._queue)
        logger.info("InMemoryMessageSource closed")
