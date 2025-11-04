# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import logging
from dataclasses import replace
from typing import Any, Generic, Protocol, TypeVar

import confluent_kafka as kafka
import scipp as sc
from streaming_data_types import dataarray_da00, logdata_f144

from ..config.streams import stream_kind_to_topic
from ..config.workflow_spec import ResultKey
from ..core.message import (
    Message,
    MessageSink,
    SchemaMessage,
    SchemaMessageSink,
    SchemaSerializer,
)
from .scipp_da00_compat import scipp_to_da00

T = TypeVar("T")
S = TypeVar("S")


class SerializationError(Exception):
    """Raised when serialization of a message fails."""


# ====== DEPRECATED: Use schema_codecs.py instead ======
# The following are legacy serializers kept for backward compatibility.
# New code should use the schema codec classes in kafka/schema_codecs.py.


class Serializer(Protocol, Generic[T]):
    """Deprecated: Use SchemaSerializer from schema_codecs.py instead."""

    def __call__(self, value: Message[T]) -> bytes: ...


def serialize_dataarray_to_da00(msg: Message[sc.DataArray]) -> bytes:
    """
    Deprecated: Use Da00Serializer and ScippDa00Converter from schema_codecs.py.

    This function combines domain conversion (scipp -> da00) with serialization
    (da00 -> bytes). The new architecture separates these concerns.
    """
    try:
        # We use the payload timestamp, which in turn was set from the result's
        # `start_time`. Depending on whether the result is a cumulative result or a
        # delta result, this is either the time of the first event in the result, or
        # the time of the first event since the last result was produced.
        da00 = dataarray_da00.serialise_da00(
            source_name=msg.stream.name,
            timestamp_ns=msg.timestamp,
            data=scipp_to_da00(msg.value),
        )
    except (ValueError, TypeError) as e:
        raise SerializationError(f"Failed to serialize message: {e}") from None
    return da00


def serialize_dataarray_to_f144(msg: Message[sc.DataArray]) -> bytes:
    """
    Deprecated: Use F144Serializer from schema_codecs.py.

    This function combines domain conversion with serialization.
    The new architecture separates these concerns.
    """
    try:
        da = msg.value
        f144 = logdata_f144.serialise_f144(
            source_name=msg.stream.name,
            value=da.values,
            timestamp_unix_ns=da.coords['time'].to(unit='ns', copy=False).value,
        )
    except (ValueError, TypeError) as e:
        raise SerializationError(f"Failed to serialize message: {e}") from None
    return f144


class KafkaSink(SchemaMessageSink[S]):
    """
    Pure Kafka transport layer for schema messages.

    Only knows about SchemaMessage and bytes. No domain knowledge.
    Can be reused with any schema format.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        instrument: str,
        kafka_config: dict[str, Any],
        schema_serializer: SchemaSerializer[S],
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._producer = kafka.Producer(kafka_config)
        self._serializer = schema_serializer
        self._instrument = instrument

    def publish_messages(self, messages: list[SchemaMessage[S]]) -> None:
        """Publish schema messages to Kafka."""

        def delivery_callback(err, msg):
            if err is not None:
                self._logger.error(
                    "Failed to deliver message to %s: %s", msg.topic(), err
                )

        self._logger.debug("Publishing %d messages", len(messages))

        for msg in messages:
            topic = stream_kind_to_topic(
                instrument=self._instrument, kind=msg.stream.kind
            )
            try:
                # Schema serializer handles SchemaMessage -> bytes
                value_bytes = self._serializer.serialize(msg)

                self._producer.produce(
                    topic=topic,
                    key=msg.key,  # Key already on SchemaMessage
                    value=value_bytes,
                    callback=delivery_callback,
                )
                self._producer.poll(0)

            except SerializationError as e:
                self._logger.error("Failed to serialize message: %s", e)
            except kafka.KafkaException as e:
                self._logger.error("Failed to publish message to %s: %s", topic, e)

        try:
            self._producer.flush(timeout=3)
        except kafka.KafkaException as e:
            self._logger.error("Error flushing producer: %s", e)


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
