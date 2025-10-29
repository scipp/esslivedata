# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Queue-based message sink for HTTP transport."""

import logging
import threading
import time
from collections import deque
from typing import Generic

from ..core.message import Message, MessageSink, T


class QueueBasedMessageSink(MessageSink[T], Generic[T]):
    """
    Message sink that stores messages in a bounded in-memory queue.

    Messages are stored until consumed via get_messages(). When the queue
    reaches max_size, oldest messages are dropped. Optionally, messages
    older than max_age_seconds are also dropped.

    This is designed to be exposed via HTTP endpoints for service-to-service
    communication as an alternative to Kafka.

    Parameters
    ----------
    max_size:
        Maximum number of messages to keep in the queue. When exceeded,
        oldest messages are dropped.
    max_age_seconds:
        If provided, messages older than this many seconds are dropped.
        If None, messages are kept until consumed or queue is full.
    logger:
        Optional logger instance.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_age_seconds: float | None = None,
        logger: logging.Logger | None = None,
    ):
        self._messages: deque[Message[T]] = deque(maxlen=max_size)
        self._max_age_seconds = max_age_seconds
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()

    def publish_messages(self, messages: list[Message[T]]) -> None:
        """
        Add messages to the queue.

        Parameters
        ----------
        messages:
            List of messages to add to the queue.
        """
        if not messages:
            return

        with self._lock:
            # Add messages to queue (deque automatically drops oldest if at maxlen)
            dropped_count = 0
            for msg in messages:
                if len(self._messages) == self._messages.maxlen:
                    dropped_count += 1
                self._messages.append(msg)

            if dropped_count > 0:
                self._logger.warning(
                    "Queue full, dropped %d oldest message(s)", dropped_count
                )

            self._logger.debug("Published %d messages to queue", len(messages))

    def get_messages(self) -> list[Message[T]]:
        """
        Get all available messages and clear the queue.

        If max_age_seconds is set, expired messages are filtered out.

        Returns
        -------
        :
            List of all messages currently in the queue.
        """
        with self._lock:
            if self._max_age_seconds is not None:
                cutoff_time = time.time() - self._max_age_seconds
                cutoff_ns = int(cutoff_time * 1_000_000_000)

                # Filter out expired messages
                valid_messages = [
                    msg for msg in self._messages if msg.timestamp >= cutoff_ns
                ]
                expired_count = len(self._messages) - len(valid_messages)

                if expired_count > 0:
                    self._logger.debug("Dropped %d expired message(s)", expired_count)

                messages = list(valid_messages)
            else:
                messages = list(self._messages)

            self._messages.clear()
            return messages

    def peek_messages(self, max_count: int | None = None) -> list[Message[T]]:
        """
        Peek at messages without removing them from the queue.

        Parameters
        ----------
        max_count:
            Maximum number of messages to return. If None, return all.

        Returns
        -------
        :
            List of messages (up to max_count).
        """
        with self._lock:
            if max_count is None:
                return list(self._messages)
            return list(self._messages)[:max_count]

    @property
    def size(self) -> int:
        """Returns the current number of messages in the queue."""
        with self._lock:
            return len(self._messages)
