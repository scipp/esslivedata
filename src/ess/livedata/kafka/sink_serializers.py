# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Concrete :class:`MessageSerializer` implementations for publishing to Kafka.

Each serializer converts a single :class:`Message` value type into a
:class:`SerializedMessage` (topic + optional key + value bytes). Serializers are
composed via routers from :mod:`sink_routing` to form the serializer used by
:class:`KafkaSink`.

This mirrors the source-side architecture: one class per encoding, each
independently unit-testable, with a routing primitive on top.
"""

from __future__ import annotations

import importlib.metadata
import os
import socket
from typing import TypeVar

import scipp as sc
from streaming_data_types import dataarray_da00, logdata_f144

from ..config.acknowledgement import CommandAcknowledgement
from ..config.streams import stream_kind_to_topic
from ..core.job import JobStatus, ServiceStatus
from ..core.message import Message, StreamKind
from ..handlers.config_handler import ConfigUpdate
from .scipp_da00_compat import scipp_to_da00
from .sink import MessageSerializer, SerializationError, SerializedMessage
from .sink_routing import RouteByStatusTypeSerializer, RouteByStreamKindSerializer
from .x5f2_compat import job_status_to_x5f2, service_status_to_x5f2

T = TypeVar('T')


def _get_software_version() -> str:
    """Get the software version for x5f2 messages."""
    try:
        return importlib.metadata.version('esslivedata')
    except importlib.metadata.PackageNotFoundError:
        return '0.0.0'


class _TopicResolvingSerializer(MessageSerializer[T]):
    """
    Base class for serializers that resolve a Kafka topic from the instrument
    name and stream kind, and wrap encoding errors in :class:`SerializationError`.

    Subclasses implement :meth:`_encode` to produce the key/value bytes.
    """

    def __init__(self, *, instrument: str) -> None:
        self._instrument = instrument

    def _encode(self, message: Message[T]) -> tuple[bytes | None, bytes]:
        """Return ``(key, value)`` bytes for the message.

        Raises
        ------
        :
            Any exception from the underlying encoding library; the base class
            catches ``(AttributeError, KeyError, ValueError, TypeError)`` and
            wraps them in :class:`SerializationError`.
        """
        raise NotImplementedError

    def serialize(self, message: Message[T]) -> SerializedMessage:
        topic = stream_kind_to_topic(self._instrument, message.stream.kind)
        try:
            key, value = self._encode(message)
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            raise SerializationError(f"Failed to serialize message: {e}") from None
        return SerializedMessage(topic=topic, key=key, value=value)


class Da00Serializer(_TopicResolvingSerializer[sc.DataArray]):
    """Serializes scipp DataArrays to the ``da00`` flatbuffer schema."""

    def _encode(self, message: Message[sc.DataArray]) -> tuple[None, bytes]:
        # We use the payload timestamp, which in turn was set from the result's
        # `start_time`. Depending on whether the result is a cumulative result or
        # a delta result, this is either the time of the first event in the
        # result, or the time of the first event since the last result was
        # produced.
        value = dataarray_da00.serialise_da00(
            source_name=message.stream.name,
            timestamp_ns=message.timestamp.to_ns(),
            data=scipp_to_da00(message.value),
        )
        return None, value


class F144Serializer(_TopicResolvingSerializer[sc.DataArray]):
    """Serializes scipp log-data DataArrays to the ``f144`` flatbuffer schema."""

    def _encode(self, message: Message[sc.DataArray]) -> tuple[None, bytes]:
        da = message.value
        value = logdata_f144.serialise_f144(
            source_name=message.stream.name,
            value=da.values,
            timestamp_unix_ns=da.coords['time'].to(unit='ns', copy=False).value,
        )
        return None, value


class ServiceStatusToX5f2Serializer(_TopicResolvingSerializer[ServiceStatus]):
    """Serializes :class:`ServiceStatus` heartbeats to the ``x5f2`` schema."""

    def __init__(
        self,
        *,
        instrument: str,
        software_version: str,
        host_name: str,
        process_id: int,
    ) -> None:
        super().__init__(instrument=instrument)
        self._software_version = software_version
        self._host_name = host_name
        self._process_id = process_id

    def _encode(self, message: Message[ServiceStatus]) -> tuple[None, bytes]:
        value = service_status_to_x5f2(
            message.value,
            software_version=self._software_version,
            host_name=self._host_name,
            process_id=self._process_id,
        )
        return None, value


class JobStatusToX5f2Serializer(_TopicResolvingSerializer[JobStatus]):
    """Serializes :class:`JobStatus` heartbeats to the ``x5f2`` schema."""

    def __init__(
        self,
        *,
        instrument: str,
        software_version: str,
        host_name: str,
        process_id: int,
    ) -> None:
        super().__init__(instrument=instrument)
        self._software_version = software_version
        self._host_name = host_name
        self._process_id = process_id

    def _encode(self, message: Message[JobStatus]) -> tuple[None, bytes]:
        value = job_status_to_x5f2(
            message.value,
            software_version=self._software_version,
            host_name=self._host_name,
            process_id=self._process_id,
        )
        return None, value


class CommandSerializer(_TopicResolvingSerializer[ConfigUpdate]):
    """
    Serializes :class:`ConfigUpdate` messages for the commands topic.

    The Kafka message key is the encoded string representation of the
    :class:`ConfigKey`; the value carries the payload JSON. Consumers reconstruct
    the ``ConfigKey`` from the message key (see :class:`CommandsAdapter`).
    """

    def _encode(self, message: Message[ConfigUpdate]) -> tuple[bytes, bytes]:
        key = str(message.value.config_key).encode('utf-8')
        value = message.value.value.model_dump_json().encode('utf-8')
        return key, value


class ResponseSerializer(_TopicResolvingSerializer[CommandAcknowledgement]):
    """
    Serializes :class:`CommandAcknowledgement` events for the responses topic.

    Acknowledgements are events (not state), so no key is emitted.
    """

    def _encode(self, message: Message[CommandAcknowledgement]) -> tuple[None, bytes]:
        value = message.value.model_dump_json().encode('utf-8')
        return None, value


def make_default_sink_serializer(
    *,
    instrument: str,
    data_serializer: MessageSerializer | None = None,
) -> RouteByStreamKindSerializer:
    """
    Build the routing serializer used by production services.

    Dispatches by stream kind to the appropriate encoder: ``data_serializer`` for
    all data-carrying streams (results, ROIs, monitor counts, area detector),
    the command/response encoders for the livedata control topics, and the
    x5f2 encoders for service/job heartbeats. Log streams are not routed here;
    producers that emit ``StreamKind.LOG`` wire an :class:`F144Serializer`
    directly into their sink.

    Environment-dependent metadata for status messages (software version,
    hostname, process id) is resolved **once** here at factory time, so the
    serializer instances themselves remain pure and trivially testable.

    Parameters
    ----------
    instrument:
        Instrument name used to resolve Kafka topics via ``stream_kind_to_topic``.
    data_serializer:
        Serializer for data-carrying streams. Defaults to :class:`Da00Serializer`
        which is the standard encoding for all production services.
    """
    if data_serializer is None:
        data_serializer = Da00Serializer(instrument=instrument)
    software_version = _get_software_version()
    host_name = socket.gethostname()
    process_id = os.getpid()
    status_serializer = RouteByStatusTypeSerializer(
        service=ServiceStatusToX5f2Serializer(
            instrument=instrument,
            software_version=software_version,
            host_name=host_name,
            process_id=process_id,
        ),
        job=JobStatusToX5f2Serializer(
            instrument=instrument,
            software_version=software_version,
            host_name=host_name,
            process_id=process_id,
        ),
    )
    return RouteByStreamKindSerializer(
        {
            StreamKind.LIVEDATA_COMMANDS: CommandSerializer(instrument=instrument),
            StreamKind.LIVEDATA_RESPONSES: ResponseSerializer(instrument=instrument),
            StreamKind.LIVEDATA_STATUS: status_serializer,
            StreamKind.LIVEDATA_DATA: data_serializer,
            StreamKind.LIVEDATA_ROI: data_serializer,
            StreamKind.MONITOR_COUNTS: data_serializer,
            StreamKind.MONITOR_EVENTS: data_serializer,
            StreamKind.DETECTOR_EVENTS: data_serializer,
            StreamKind.AREA_DETECTOR: data_serializer,
        }
    )
