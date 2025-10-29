# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Transport abstraction for message delivery."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import confluent_kafka as kafka

logger = logging.getLogger(__name__)


class MessageTransport(Protocol):
    """
    Protocol for message transport implementations.

    Abstracts the actual delivery mechanism (Kafka, in-memory, etc.)
    from the message serialization and routing logic.
    """

    def send(
        self, topic: str, value: bytes, key: bytes | None = None, timestamp: int | None = None
    ) -> None:
        """
        Send a serialized message to a topic.

        Parameters
        ----------
        topic:
            Topic name to send to
        value:
            Serialized message value
        key:
            Optional message key for partitioning
        timestamp:
            Optional message timestamp in nanoseconds
        """
        ...

    def flush(self) -> None:
        """
        Flush any pending messages.

        Blocks until all buffered messages are delivered or timeout.
        """
        ...


class KafkaTransport:
    """
    Kafka implementation of MessageTransport.

    Wraps a Kafka producer and handles Kafka-specific concerns
    like delivery callbacks, polling, and flushing.

    Parameters
    ----------
    producer:
        Confluent Kafka producer instance
    logger:
        Optional logger for transport events
    """

    def __init__(
        self,
        producer: kafka.Producer,
        logger: logging.Logger | None = None,
    ):
        self._producer = producer
        self._logger = logger or logging.getLogger(__name__)

    def send(
        self, topic: str, value: bytes, key: bytes | None = None, timestamp: int | None = None
    ) -> None:
        """
        Send message to Kafka topic.

        Parameters
        ----------
        topic:
            Topic name
        value:
            Serialized message value
        key:
            Optional message key
        timestamp:
            Optional timestamp (ignored for Kafka - Kafka producer sets its own)
        """

        def delivery_callback(err, msg):
            if err is not None:
                self._logger.error(
                    "Failed to deliver message to %s: %s", msg.topic(), err
                )

        try:
            if key is None:
                self._producer.produce(
                    topic=topic, value=value, callback=delivery_callback
                )
            else:
                self._producer.produce(
                    topic=topic,
                    key=key,
                    value=value,
                    callback=delivery_callback,
                )
            # Poll to trigger delivery callbacks
            self._producer.poll(0)
        except kafka.KafkaException as e:
            self._logger.error("Failed to publish message to %s: %s", topic, e)

    def flush(self) -> None:
        """Flush pending messages with timeout."""
        try:
            self._producer.flush(timeout=3)
        except kafka.KafkaException as e:
            self._logger.error("Error flushing producer: %s", e)

    @classmethod
    def from_config(
        cls,
        kafka_config: dict[str, Any],
        logger: logging.Logger | None = None,
    ) -> KafkaTransport:
        """
        Create KafkaTransport from Kafka configuration.

        Parameters
        ----------
        kafka_config:
            Configuration dictionary for confluent_kafka.Producer
        logger:
            Optional logger

        Returns
        -------
        :
            Configured KafkaTransport instance
        """
        producer = kafka.Producer(kafka_config)
        return cls(producer=producer, logger=logger)
