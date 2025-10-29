# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""In-memory transport implementation."""

from __future__ import annotations

import logging

from ess.livedata.in_memory.broker import InMemoryBroker

logger = logging.getLogger(__name__)


class FakeKafkaMessage:
    """
    Fake Kafka message for in-memory transport.

    Mimics the interface of confluent_kafka.Message that adapters expect.
    """

    def __init__(self, topic: str, key: bytes | None, value: bytes, timestamp: int):
        self._topic = topic
        self._key = key
        self._value = value
        self._timestamp = timestamp

    def topic(self) -> str:
        """Return the topic name."""
        return self._topic

    def key(self) -> bytes | None:
        """Return the message key."""
        return self._key

    def value(self) -> bytes:
        """Return the message value."""
        return self._value

    def timestamp(self) -> tuple[int, int]:
        """Return the timestamp as (type, timestamp_ms)."""
        # Type 0 = CreateTime
        return (0, self._timestamp // 1_000_000)  # Convert ns to ms


class InMemoryTransport:
    """
    In-memory implementation of MessageTransport.

    Wraps an InMemoryBroker and creates FakeKafkaMessage objects
    for compatibility with Kafka adapters.

    Parameters
    ----------
    broker:
        Broker instance to publish messages to
    logger:
        Optional logger for transport events
    """

    def __init__(
        self,
        broker: InMemoryBroker,
        logger: logging.Logger | None = None,
    ):
        self._broker = broker
        self._logger = logger or logging.getLogger(__name__)
        # Buffer messages by topic for batched publishing
        self._pending: dict[str, list[FakeKafkaMessage]] = {}

    def send(
        self, topic: str, value: bytes, key: bytes | None = None, timestamp: int | None = None
    ) -> None:
        """
        Buffer message for later publishing.

        Messages are batched and sent during flush() for efficiency.

        Parameters
        ----------
        topic:
            Topic name
        value:
            Serialized message value
        key:
            Optional message key
        timestamp:
            Optional message timestamp in nanoseconds. If not provided,
            current time will be used.
        """
        # Use provided timestamp or current time (nanoseconds since epoch)
        if timestamp is None:
            import time
            timestamp_ns = int(time.time() * 1_000_000_000)
        else:
            timestamp_ns = timestamp

        fake_msg = FakeKafkaMessage(
            topic=topic,
            key=key,
            value=value,
            timestamp=timestamp_ns,
        )

        self._pending.setdefault(topic, []).append(fake_msg)

    def flush(self) -> None:
        """
        Publish all pending messages to broker.

        Messages are grouped by topic and published in batches.
        """
        if not self._pending:
            return

        for topic, messages in self._pending.items():
            if messages:
                self._logger.debug(
                    "Publishing %d messages to topic '%s'",
                    len(messages),
                    topic,
                )
                self._broker.publish(topic, messages)

        # Clear pending messages
        self._pending.clear()
