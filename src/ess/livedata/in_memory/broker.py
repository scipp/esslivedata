# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Central message broker for in-memory transport."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from queue import Empty, Full, Queue
from typing import Any

logger = logging.getLogger(__name__)


class InMemoryBroker:
    """
    Thread-safe in-memory message broker.

    This broker implements a simple publish-subscribe pattern using Python queues.
    It is designed for testing and development, NOT for production use.

    Architecture:
    - Each topic has a list of subscriber queues
    - Publishing to a topic puts messages in all subscriber queues
    - Each subscriber gets its own queue for isolation
    - Queue overflow drops oldest messages (lossy behavior)

    Parameters
    ----------
    max_queue_size:
        Maximum messages per subscriber queue. When full, oldest messages
        are dropped. Default 1000.
    """

    def __init__(self, max_queue_size: int = 1000):
        self._max_queue_size = max_queue_size
        # Map topic name -> list of subscriber queues
        self._subscribers: dict[str, list[Queue]] = defaultdict(list)
        # Track which topics each queue is subscribed to (for cleanup)
        self._queue_topics: dict[Queue, set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        logger.info("InMemoryBroker initialized (max_queue_size=%d)", max_queue_size)

    def publish(self, topic: str, messages: list[Any]) -> None:
        """
        Publish messages to all subscribers of a topic.

        Messages are put into each subscriber's queue. If a queue is full,
        the oldest message is dropped (FIFO with overflow).

        Parameters
        ----------
        topic:
            Topic name to publish to
        messages:
            List of messages (Message objects or FakeKafkaMessage objects)
        """
        if not messages:
            return

        with self._lock:
            subscribers = self._subscribers.get(topic, [])
            if not subscribers:
                logger.debug(
                    "No subscribers for topic '%s', dropping %d messages",
                    topic,
                    len(messages),
                )
                return

            logger.debug(
                "Publishing %d messages to topic '%s' (%d subscribers)",
                len(messages),
                topic,
                len(subscribers),
            )

            for queue in subscribers:
                for message in messages:
                    try:
                        # Non-blocking put - if full, drop oldest and try again
                        queue.put_nowait(message)
                    except Full:
                        # Queue is full - drop oldest message and add new one
                        try:
                            queue.get_nowait()  # Drop oldest
                            queue.put_nowait(message)  # Add new
                            logger.warning(
                                "Queue overflow on topic '%s', dropped oldest message",
                                topic,
                            )
                        except (Empty, Full):
                            logger.error(
                                "Failed to handle queue overflow on topic '%s'", topic
                            )

    def subscribe(self, topics: list[str]) -> Queue:
        """
        Create a new subscription for specified topics.

        Returns a queue that will receive messages from all specified topics.
        The caller is responsible for calling unsubscribe() when done.

        Parameters
        ----------
        topics:
            List of topic names to subscribe to

        Returns
        -------
        :
            Queue that will receive messages from subscribed topics
        """
        queue: Queue = Queue(maxsize=self._max_queue_size)

        with self._lock:
            for topic in topics:
                self._subscribers[topic].append(queue)
                self._queue_topics[queue].add(topic)

            logger.info("Created subscription to %d topics: %s", len(topics), topics)

        return queue

    def unsubscribe(self, queue: Queue) -> None:
        """
        Remove a subscription queue.

        Parameters
        ----------
        queue:
            Queue to remove from all topic subscriptions
        """
        with self._lock:
            topics = self._queue_topics.pop(queue, set())
            for topic in topics:
                if queue in self._subscribers[topic]:
                    self._subscribers[topic].remove(queue)

            logger.info("Unsubscribed queue from %d topics", len(topics))

    def get_stats(self) -> dict[str, Any]:
        """
        Get broker statistics for debugging.

        Returns
        -------
        :
            Dictionary with topic counts, subscriber counts, and queue sizes
        """
        with self._lock:
            return {
                "topics": len(self._subscribers),
                "total_subscribers": sum(
                    len(subs) for subs in self._subscribers.values()
                ),
                "topic_details": {
                    topic: {
                        "subscribers": len(subs),
                        "queue_sizes": [sub.qsize() for sub in subs],
                    }
                    for topic, subs in self._subscribers.items()
                },
            }
