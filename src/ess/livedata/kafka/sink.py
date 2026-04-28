# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Any, Generic, Protocol, TypeVar

import confluent_kafka as kafka
import scipp as sc
import structlog

from ..config.workflow_spec import ResultKey
from ..core.message import Message, MessageSink
from .errors import is_fatal

logger = structlog.get_logger(__name__)


T = TypeVar("T")


class SerializationError(Exception):
    """Raised when serialization of a message fails."""


@dataclass(frozen=True, slots=True)
class SerializedMessage:
    """
    The result of serializing a :class:`Message` for publication to Kafka.

    Contains everything the sink needs to call ``producer.produce``: the destination
    topic, an optional Kafka message key, and the serialized value bytes.
    """

    topic: str
    key: bytes | None
    value: bytes


class MessageSerializer(Protocol, Generic[T]):
    """
    Protocol for converting a domain :class:`Message` into a :class:`SerializedMessage`.

    Mirrors the source-side :class:`~ess.livedata.kafka.message_adapter.MessageAdapter`
    protocol. Implementations are responsible for choosing the destination topic and
    producing the key/value bytes; :class:`KafkaSink` only handles producer I/O.
    """

    def serialize(self, message: Message[T]) -> SerializedMessage: ...


class KafkaSink(MessageSink[T]):
    """
    Publishes :class:`Message` instances to Kafka.

    Encoding is fully delegated to the injected :class:`MessageSerializer`, which
    produces a :class:`SerializedMessage` (topic + optional key + value bytes).
    This class only handles producer lifecycle, I/O, error logging, and metrics.
    """

    def __init__(
        self,
        *,
        kafka_config: dict[str, Any],
        serializer: MessageSerializer[T],
        producer_factory: Callable[[dict[str, Any]], kafka.Producer] = kafka.Producer,
    ):
        self._kafka_config = kafka_config
        self._producer: kafka.Producer | None = None
        self._producer_factory = producer_factory
        self._serializer = serializer
        self._fatal_error: kafka.KafkaError | None = None
        # Metrics tracking
        self._messages_published = 0
        self._publish_errors = 0
        self._last_metrics_time = time.monotonic()
        self._metrics_interval = 30.0

    def publish_messages(self, messages: list[Message[T]]) -> None:
        if self._producer is None:
            raise RuntimeError("KafkaSink must be used as a context manager")
        if self._fatal_error is not None:
            raise kafka.KafkaException(self._fatal_error)

        def delivery_callback(err, msg):
            if err is None:
                return
            self._publish_errors += 1
            if is_fatal(err):
                self._fatal_error = err
                logger.error("Fatal delivery error for topic %s: %s", msg.topic(), err)
            else:
                logger.error("Failed to deliver message to %s: %s", msg.topic(), err)

        logger.debug("Publishing %d messages", len(messages))
        for msg in messages:
            try:
                serialized = self._serializer.serialize(msg)
            except SerializationError as e:
                logger.error("Failed to serialize message: %s", e)
                continue
            try:
                self._producer.produce(
                    topic=serialized.topic,
                    key=serialized.key,
                    value=serialized.value,
                    callback=delivery_callback,
                )
                self._producer.poll(0)
            except kafka.KafkaException as e:
                err = e.args[0] if e.args else None
                if isinstance(err, kafka.KafkaError) and is_fatal(err):
                    raise
                logger.error("Failed to publish message to %s: %s", serialized.topic, e)

        try:
            self._producer.flush(timeout=3)
        except kafka.KafkaException as e:
            err = e.args[0] if e.args else None
            if isinstance(err, kafka.KafkaError) and is_fatal(err):
                raise
            logger.error("Error flushing producer: %s", e)

        if self._fatal_error is not None:
            raise kafka.KafkaException(self._fatal_error)

        self._messages_published += len(messages)
        self._maybe_log_metrics()

    def _maybe_log_metrics(self) -> None:
        """Log metrics if the interval has elapsed."""
        now = time.monotonic()
        if now - self._last_metrics_time >= self._metrics_interval:
            logger.info(
                "sink_metrics",
                messages_published=self._messages_published,
                errors=self._publish_errors,
                interval_seconds=self._metrics_interval,
            )
            self._messages_published = 0
            self._publish_errors = 0
            self._last_metrics_time = now

    def close(self) -> None:
        """Close the Kafka producer and release resources."""
        if self._producer is not None:
            try:
                self._producer.flush(timeout=5)
            except kafka.KafkaException as e:
                logger.error("Error flushing producer during close: %s", e)
            # The confluent_kafka Producer cleans up when deleted
            self._producer = None

    def __enter__(self) -> 'KafkaSink[T]':
        """Enter context manager - initialize the Kafka producer."""
        self._producer = self._producer_factory(self._kafka_config)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.close()


class UnrollingSinkAdapter(MessageSink[T | sc.DataGroup[T]]):
    def __init__(self, sink: MessageSink[T]):
        self._sink = sink

    def publish_messages(self, messages: list[Message[T | sc.DataGroup[T]]]) -> None:
        unrolled: list[Message[T]] = []
        for msg in messages:
            if isinstance(msg.value, sc.DataGroup):
                result_key = ResultKey.model_validate_json(msg.stream.name)
                for name, value in msg.value.items():
                    key = ResultKey(
                        workflow_id=result_key.workflow_id,
                        job_id=result_key.job_id,
                        output_name=name,
                    )
                    stream = replace(msg.stream, name=key.model_dump_json())
                    unrolled.append(replace(msg, stream=stream, value=value))
            else:
                unrolled.append(msg)
        self._sink.publish_messages(unrolled)
