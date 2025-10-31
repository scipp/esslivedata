# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import json
import logging
from typing import Any

from confluent_kafka import KafkaError, Producer

from ..config.models import ConfigKey
from ..handlers.config_handler import ConfigUpdate
from ..kafka.message_adapter import RawConfigItem
from ..kafka.source import BackgroundMessageSource
from .message_bridge import MessageBridge


class SourceBasedMessageBridge(MessageBridge[ConfigKey, dict[str, Any]]):
    """
    Message bridge using BackgroundMessageSource for consuming messages.

    This implementation uses the backend's BackgroundMessageSource for message
    consumption, which handles background polling in its own thread. This is simpler
    than the previous implementation which had a separate background thread for both
    consuming and publishing.

    Publishing is done directly with non-blocking producer.poll(0), as config updates
    are infrequent and don't require batching or flushing.
    """

    def __init__(
        self,
        source: BackgroundMessageSource,
        producer: Producer,
        publish_topic: str,
        logger: logging.Logger | None = None,
    ):
        self._source = source
        self._producer = producer
        self._publish_topic = publish_topic
        self._logger = logger or logging.getLogger(__name__)
        self._running = False

    def start(self) -> None:
        """Start background message consumption."""
        if self._running:
            return
        self._source.start()
        self._running = True
        self._logger.info("SourceBasedMessageBridge started")

    def stop(self) -> None:
        """Stop background message consumption."""
        if not self._running:
            return
        self._source.stop()
        self._running = False
        self._logger.info("SourceBasedMessageBridge stopped")

    def publish(self, key: ConfigKey, value: dict[str, Any]) -> None:
        """Publish message to Kafka using non-blocking producer."""
        if not self._running:
            self._logger.warning("Cannot publish - bridge not running")
            return

        try:
            self._producer.produce(
                self._publish_topic,
                key=str(key).encode("utf-8"),
                value=json.dumps(value).encode("utf-8"),
            )
            # Non-blocking poll - config updates are infrequent so no need for flush
            self._producer.poll(0)
        except Exception as e:
            self._logger.error("Error publishing message: %s", e)

    def pop_all(self) -> dict[ConfigKey, dict[str, Any]]:
        """
        Pop all messages from BackgroundMessageSource and decode them.

        Returns a dict where the last value for each key wins, providing natural
        deduplication at the application layer.
        """
        messages = self._source.get_messages()
        decoded = {}

        for msg in messages:
            try:
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    self._logger.error("Consumer error: %s", msg.error())
                    continue

                update = ConfigUpdate.from_raw(
                    RawConfigItem(key=msg.key(), value=msg.value())
                )
                # Last write wins - naturally deduplicates by key
                decoded[update.config_key] = update.value
            except Exception as e:
                self._logger.error("Failed to decode message: %s", e)

        return decoded
