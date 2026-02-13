# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import importlib.metadata
import os
import socket
from dataclasses import replace
from types import TracebackType
from typing import Any, Generic, Protocol, TypeVar

import confluent_kafka as kafka
import scipp as sc
import structlog
from streaming_data_types import dataarray_da00, logdata_f144

from ..config.streams import stream_kind_to_topic
from ..config.workflow_spec import ResultKey
from ..core.job import ServiceStatus
from ..core.message import STATUS_STREAM_ID, Message, MessageSink, StreamKind
from .scipp_da00_compat import scipp_to_da00
from .x5f2_compat import job_status_to_x5f2, service_status_to_x5f2

logger = structlog.get_logger(__name__)


def _get_software_version() -> str:
    """Get the software version for x5f2 messages."""
    try:
        return importlib.metadata.version('esslivedata')
    except importlib.metadata.PackageNotFoundError:
        return '0.0.0'


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
    def __init__(
        self,
        *,
        instrument: str,
        kafka_config: dict[str, Any],
        serializer: Serializer[T] = serialize_dataarray_to_da00,
    ):
        self._kafka_config = kafka_config
        self._producer: kafka.Producer | None = None
        self._serializer = serializer
        self._instrument = instrument
        # Cache x5f2 metadata for status messages
        self._software_version = _get_software_version()
        self._host_name = socket.gethostname()
        self._process_id = os.getpid()

    def publish_messages(self, messages: Message[T]) -> None:
        if self._producer is None:
            raise RuntimeError("KafkaSink must be used as a context manager")

        def delivery_callback(err, msg):
            if err is not None:
                logger.error("Failed to deliver message to %s: %s", msg.topic(), err)

        logger.debug("Publishing %d messages", len(messages))
        for msg in messages:
            try:
                topic = stream_kind_to_topic(
                    instrument=self._instrument, kind=msg.stream.kind
                )
                if msg.stream.kind == StreamKind.LIVEDATA_COMMANDS:
                    key_bytes = str(msg.value.config_key).encode('utf-8')
                    value = msg.value.value.model_dump_json().encode('utf-8')
                elif msg.stream.kind == StreamKind.LIVEDATA_RESPONSES:
                    # Acknowledgements are events, not state - no key needed
                    key_bytes = None
                    value = msg.value.model_dump_json().encode('utf-8')
                elif msg.stream == STATUS_STREAM_ID:
                    key_bytes = None
                    if isinstance(msg.value, ServiceStatus):
                        value = service_status_to_x5f2(
                            msg.value,
                            software_version=self._software_version,
                            host_name=self._host_name,
                            process_id=self._process_id,
                        )
                    else:
                        value = job_status_to_x5f2(
                            msg.value,
                            software_version=self._software_version,
                            host_name=self._host_name,
                            process_id=self._process_id,
                        )
                else:
                    key_bytes = None
                    value = self._serializer(msg)
            except SerializationError as e:
                logger.error("Failed to serialize message: %s", e)
            else:
                try:
                    if key_bytes is None:
                        self._producer.produce(
                            topic=topic, value=value, callback=delivery_callback
                        )
                    else:
                        self._producer.produce(
                            topic=topic,
                            key=key_bytes,
                            value=value,
                            callback=delivery_callback,
                        )
                    self._producer.poll(0)
                except kafka.KafkaException as e:
                    logger.error("Failed to publish message to %s: %s", topic, e)

        try:
            self._producer.flush(timeout=3)
        except kafka.KafkaException as e:
            logger.error("Error flushing producer: %s", e)

    def close(self) -> None:
        """Close the Kafka producer and release resources."""
        if hasattr(self, '_producer'):
            try:
                self._producer.flush(timeout=5)
            except kafka.KafkaException as e:
                logger.error("Error flushing producer during close: %s", e)
            # The confluent_kafka Producer cleans up when deleted
            del self._producer

    def __enter__(self) -> 'KafkaSink[T]':
        """Enter context manager - initialize the Kafka producer."""
        self._producer = kafka.Producer(self._kafka_config)
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
