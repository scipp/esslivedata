# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import json
import logging
from typing import Any

from confluent_kafka import Consumer, KafkaError, Producer

from ..config.models import ConfigKey
from ..handlers.config_handler import ConfigUpdate
from ..kafka.message_adapter import RawConfigItem
from .throttling_message_handler import MessageTransport


class KafkaTransport(MessageTransport[ConfigKey, dict[str, Any]]):
    """Kafka-specific transport implementation."""

    def __init__(
        self,
        kafka_config: dict[str, Any],
        consumer: Consumer,
        publish_topic: str,
        logger: logging.Logger | None = None,
        max_batch_size: int = 100,
    ):
        if len(consumer.assignment()) != 1:
            raise ValueError(
                "KafkaTransport requires a single topic assignment for the consumer"
            )
        self._publish_topic = publish_topic
        self._logger = logger or logging.getLogger(__name__)
        self._producer = Producer(kafka_config)
        self._consumer = consumer
        self._max_batch_size = max_batch_size

    def send_messages(self, messages: list[tuple[ConfigKey, dict[str, Any]]]) -> None:
        """Send messages to Kafka."""
        try:
            for key, value in messages:
                self._producer.produce(
                    self._publish_topic,
                    key=str(key).encode("utf-8"),
                    value=json.dumps(value).encode("utf-8"),
                    callback=self._delivery_callback,
                )

            # Poll producer for delivery reports (non-blocking)
            if messages:
                self._producer.poll(0)

        except Exception as e:
            self._logger.error("Error sending messages: %s", e)

    def receive_messages(self) -> list[tuple[ConfigKey, dict[str, Any]]]:
        """Receive messages from Kafka."""
        received = []

        try:
            msgs = self._consumer.consume(
                num_messages=self._max_batch_size, timeout=0.1
            )

            for msg in msgs:
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        self._logger.error("Consumer error: %s", msg.error())
                        continue

                try:
                    decoded_update = self._decode_update(msg)
                    if decoded_update:
                        received.append(
                            (decoded_update.config_key, decoded_update.value)
                        )
                except Exception as e:
                    self._logger.error("Failed to process incoming message: %s", e)

        except Exception as e:
            self._logger.error("Error receiving messages: %s", e)

        return received

    def _decode_update(self, msg) -> ConfigUpdate | None:
        """Decode a Kafka message into a ConfigUpdate."""
        try:
            return ConfigUpdate.from_raw(
                RawConfigItem(key=msg.key(), value=msg.value())
            )
        except Exception as e:
            self._logger.exception("Failed to decode config message: %s", e)
            return None

    def _delivery_callback(self, err, msg) -> None:
        """Callback for message delivery confirmation."""
        if err:
            self._logger.error("Message delivery failed: %s", err)
        else:
            self._logger.debug(
                "Message delivered to %s [%s]", msg.topic(), msg.partition()
            )
