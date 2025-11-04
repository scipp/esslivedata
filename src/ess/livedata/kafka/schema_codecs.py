# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Kafka-specific schema codecs for wire format serialization and deserialization.

This module contains schema serializers and deserializers for converting between
wire formats (da00, ev44, f144, x5f2) and Kafka messages. Domain conversions are
handled separately in core.converters to keep domain logic Kafka-independent.
"""

from streaming_data_types import (
    dataarray_da00,
    deserialise_x5f2,
    eventdata_ev44,
    logdata_f144,
    serialise_x5f2,
)
from streaming_data_types.status_x5f2 import StatusMessage as StatusX5f2

from ..core.message import (
    STATUS_STREAM_ID,
    SchemaDeserializer,
    SchemaMessage,
    SchemaSerializer,
    StreamId,
    StreamKind,
)
from .message_adapter import RawConfigItem
from .stream_mapping import InputStreamKey, StreamLUT


class StreamResolver:
    """
    Helper for resolving Kafka (topic, source_name) to StreamId.

    This centralizes the stream resolution logic used by all deserializers.
    """

    def __init__(self, *, stream_lut: StreamLUT | None, stream_kind: StreamKind):
        self._stream_lut = stream_lut
        self._stream_kind = stream_kind

    def resolve(self, topic: str, source_name: str) -> StreamId:
        """
        Resolve Kafka topic and source_name to internal StreamId.

        Parameters
        ----------
        topic:
            Kafka topic name.
        source_name:
            Source name from the message schema.

        Returns
        -------
        :
            Internal stream identifier.
        """
        if self._stream_lut is None:
            # No lookup table - use source_name directly
            return StreamId(kind=self._stream_kind, name=source_name)
        input_key = InputStreamKey(topic=topic, source_name=source_name)
        return StreamId(kind=self._stream_kind, name=self._stream_lut[input_key])


class Da00Serializer(SchemaSerializer[list[dataarray_da00.Variable]]):
    """Schema serializer for da00 DataArray format."""

    def serialize(self, msg: SchemaMessage[list[dataarray_da00.Variable]]) -> bytes:
        return dataarray_da00.serialise_da00(
            source_name=msg.stream.name,
            timestamp_ns=msg.timestamp,
            data=msg.data,
        )


class Da00Deserializer(SchemaDeserializer[list[dataarray_da00.Variable]]):
    """Schema deserializer for da00 DataArray format."""

    def __init__(self, *, stream_lut: StreamLUT | None = None, stream_kind: StreamKind):
        self._resolver = StreamResolver(stream_lut=stream_lut, stream_kind=stream_kind)

    def deserialize(
        self,
        data: bytes,
        *,
        topic: str,
        _timestamp: int,
        key: bytes | None,
    ) -> SchemaMessage[list[dataarray_da00.Variable]]:
        da00_data: dataarray_da00.da00_DataArray_t
        da00_data = dataarray_da00.deserialise_da00(data)  # type: ignore[reportAssignmentType]

        stream = self._resolver.resolve(topic, da00_data.source_name)

        # Use timestamp from da00 payload, not transport timestamp
        return SchemaMessage(
            stream=stream,
            data=da00_data.data,
            timestamp=da00_data.timestamp_ns,
            key=key,
        )


class Ev44Serializer(SchemaSerializer[eventdata_ev44.EventData]):
    """Schema serializer for ev44 EventData format."""

    def serialize(self, msg: SchemaMessage[eventdata_ev44.EventData]) -> bytes:
        return eventdata_ev44.serialise_ev44(
            source_name=msg.stream.name,
            message_id=msg.data.message_id,
            reference_time=msg.data.reference_time,
            reference_time_index=msg.data.reference_time_index,
            time_of_flight=msg.data.time_of_flight,
            pixel_id=msg.data.pixel_id,
        )


class Ev44Deserializer(SchemaDeserializer[eventdata_ev44.EventData]):
    """Schema deserializer for ev44 EventData format."""

    def __init__(self, *, stream_lut: StreamLUT | None = None, stream_kind: StreamKind):
        self._resolver = StreamResolver(stream_lut=stream_lut, stream_kind=stream_kind)

    def deserialize(
        self,
        data: bytes,
        *,
        topic: str,
        timestamp: int,
        key: bytes | None,
    ) -> SchemaMessage[eventdata_ev44.EventData]:
        ev44 = eventdata_ev44.deserialise_ev44(data)
        stream = self._resolver.resolve(topic, ev44.source_name)

        # Use reference_time from schema if available,
        # otherwise fallback to transport timestamp
        if ev44.reference_time.size > 0:
            msg_timestamp = ev44.reference_time[-1]
        else:
            msg_timestamp = timestamp

        return SchemaMessage(stream=stream, data=ev44, timestamp=msg_timestamp, key=key)


class F144Serializer(SchemaSerializer[logdata_f144.ExtractedLogData]):
    """Schema serializer for f144 log data format."""

    def serialize(self, msg: SchemaMessage[logdata_f144.ExtractedLogData]) -> bytes:
        return logdata_f144.serialise_f144(
            source_name=msg.stream.name,
            value=msg.data.value,
            timestamp_unix_ns=msg.timestamp,
        )


class F144Deserializer(SchemaDeserializer[logdata_f144.ExtractedLogData]):
    """Schema deserializer for f144 log data format."""

    def __init__(self, *, stream_lut: StreamLUT | None = None):
        self._resolver = StreamResolver(
            stream_lut=stream_lut, stream_kind=StreamKind.LOG
        )

    def deserialize(
        self,
        data: bytes,
        *,
        topic: str,
        _timestamp: int,
        key: bytes | None,
    ) -> SchemaMessage[logdata_f144.ExtractedLogData]:
        log_data = logdata_f144.deserialise_f144(data)
        stream = self._resolver.resolve(topic, log_data.source_name)

        # Use timestamp from f144 payload, not transport timestamp
        return SchemaMessage(
            stream=stream, data=log_data, timestamp=log_data.timestamp_unix_ns, key=key
        )


class X5f2Serializer(SchemaSerializer[StatusX5f2]):
    """Schema serializer for x5f2 status format."""

    def serialize(self, msg: SchemaMessage[StatusX5f2]) -> bytes:
        return serialise_x5f2(**msg.data._asdict())


class X5f2Deserializer(SchemaDeserializer[StatusX5f2]):
    """Schema deserializer for x5f2 status format."""

    def deserialize(
        self,
        data: bytes,
        *,
        topic: str,
        timestamp: int,
        key: bytes | None,
    ) -> SchemaMessage[StatusX5f2]:
        _ = topic  # Unused for status messages
        status_x5f2 = deserialise_x5f2(data)

        return SchemaMessage(
            stream=STATUS_STREAM_ID, data=status_x5f2, timestamp=timestamp, key=key
        )


class ConfigSerializer(SchemaSerializer[RawConfigItem]):
    """
    Schema serializer for configuration commands/responses.

    Returns only the value bytes; key is already on SchemaMessage.
    """

    def serialize(self, msg: SchemaMessage[RawConfigItem]) -> bytes:
        return msg.data.value


class ConfigDeserializer(SchemaDeserializer[RawConfigItem]):
    """Schema deserializer for configuration commands/responses."""

    def __init__(self, *, stream_id: StreamId):
        self._stream_id = stream_id

    def deserialize(
        self,
        data: bytes,
        *,
        topic: str,
        timestamp: int,
        key: bytes | None,
    ) -> SchemaMessage[RawConfigItem]:
        _ = topic  # Unused for config messages
        if key is None:
            raise ValueError("Config messages must have a key")

        return SchemaMessage(
            stream=self._stream_id,
            data=RawConfigItem(key=key, value=data),
            timestamp=timestamp,
            key=key,
        )
