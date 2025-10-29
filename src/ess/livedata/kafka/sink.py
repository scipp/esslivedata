# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import json
import logging
from dataclasses import replace
from typing import Any, Generic, Protocol, TypeVar

import scipp as sc
from streaming_data_types import dataarray_da00, logdata_f144

from ..config.streams import stream_kind_to_topic
from ..config.workflow_spec import ResultKey
from ..core.message import CONFIG_STREAM_ID, STATUS_STREAM_ID, Message, MessageSink
from .scipp_da00_compat import scipp_to_da00
from .transport import KafkaTransport, MessageTransport
from .x5f2_compat import job_status_to_x5f2

T = TypeVar("T")


class SerializationError(Exception):
    """Raised when serialization of a message fails."""


class Serializer(Protocol, Generic[T]):
    def __call__(self, value: Message[T]) -> bytes: ...


def serialize_dataarray_to_da00(msg: Message[sc.DataArray]) -> bytes:
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


class KafkaSink(MessageSink[T]):
    """
    Message sink using a transport abstraction.

    Handles message serialization and routing, delegating actual
    transport to a MessageTransport implementation.

    Parameters
    ----------
    transport:
        Transport implementation for message delivery
    instrument:
        Instrument name for topic generation
    serializer:
        Serializer for data messages
    logger:
        Optional logger
    """

    def __init__(
        self,
        *,
        transport: MessageTransport,
        instrument: str,
        serializer: Serializer[T] = serialize_dataarray_to_da00,
        logger: logging.Logger | None = None,
    ):
        self._transport = transport
        self._instrument = instrument
        self._serializer = serializer
        self._logger = logger or logging.getLogger(__name__)

    @classmethod
    def from_kafka_config(
        cls,
        *,
        kafka_config: dict[str, Any],
        instrument: str,
        serializer: Serializer[T] = serialize_dataarray_to_da00,
        logger: logging.Logger | None = None,
    ) -> "KafkaSink[T]":
        """
        Create KafkaSink with Kafka transport from configuration.

        Parameters
        ----------
        kafka_config:
            Kafka producer configuration
        instrument:
            Instrument name
        serializer:
            Serializer for data messages
        logger:
            Optional logger

        Returns
        -------
        :
            KafkaSink instance with KafkaTransport
        """
        transport = KafkaTransport.from_config(kafka_config=kafka_config, logger=logger)
        return cls(
            transport=transport,
            instrument=instrument,
            serializer=serializer,
            logger=logger,
        )

    def publish_messages(self, messages: Message[T]) -> None:
        """
        Publish messages using the transport.

        Serializes messages based on stream type and sends via transport.

        Parameters
        ----------
        messages:
            List of messages to publish
        """
        self._logger.debug("Publishing %d messages", len(messages))
        for msg in messages:
            try:
                topic = stream_kind_to_topic(
                    instrument=self._instrument, kind=msg.stream.kind
                )
                if msg.stream == CONFIG_STREAM_ID:
                    key_bytes = str(msg.value.config_key).encode('utf-8')
                    value = json.dumps(msg.value.value.model_dump()).encode('utf-8')
                elif msg.stream == STATUS_STREAM_ID:
                    key_bytes = None
                    value = job_status_to_x5f2(msg.value)
                else:
                    key_bytes = None
                    value = self._serializer(msg)
            except SerializationError as e:
                self._logger.error("Failed to serialize message: %s", e)
            else:
                self._transport.send(
                    topic=topic, value=value, key=key_bytes, timestamp=msg.timestamp
                )

        self._transport.flush()


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
