# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Message serialization protocols and implementations for HTTP transport."""

import json
import struct
from typing import Any, Generic, Protocol, TypeVar

import scipp as sc
from streaming_data_types import dataarray_da00

from ..core.message import (
    CONFIG_STREAM_ID,
    STATUS_STREAM_ID,
    Message,
    StreamId,
    StreamKind,
)
from ..kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00

T = TypeVar('T')


class MessageSerializer(Protocol, Generic[T]):
    """Protocol for serializing and deserializing messages."""

    def serialize(self, messages: list[Message[T]]) -> bytes:
        """Serialize a list of messages to bytes."""
        ...

    def deserialize(self, data: bytes) -> list[Message[T]]:
        """Deserialize bytes to a list of messages."""
        ...


class DA00MessageSerializer(MessageSerializer[sc.DataArray]):
    """
    Binary serializer for messages containing scipp DataArrays using da00 format.

    Uses the same da00 format as Kafka transport for compatibility.
    Messages are encoded as:
    [message_count][msg1_len][msg1_data][msg2_len][msg2_data]...
    Each message is prefixed with its length as a 4-byte unsigned integer.
    """

    def serialize(self, messages: list[Message[sc.DataArray]]) -> bytes:
        """
        Serialize messages using da00 format.

        Parameters
        ----------
        messages:
            List of messages to serialize.

        Returns
        -------
        :
            Binary-encoded messages with length prefixes.
        """
        parts = [struct.pack('<I', len(messages))]

        for msg in messages:
            da00 = dataarray_da00.serialise_da00(
                source_name=msg.stream.name,
                timestamp_ns=msg.timestamp,
                data=scipp_to_da00(msg.value),
            )
            stream_kind = msg.stream.kind.value.encode('utf-8')
            stream_kind_len = struct.pack('<I', len(stream_kind))

            parts.append(stream_kind_len)
            parts.append(stream_kind)
            parts.append(struct.pack('<I', len(da00)))
            parts.append(da00)

        return b''.join(parts)

    def deserialize(self, data: bytes) -> list[Message[sc.DataArray]]:
        """
        Deserialize da00-encoded messages.

        Parameters
        ----------
        data:
            Binary-encoded messages with length prefixes.

        Returns
        -------
        :
            List of deserialized messages.
        """
        messages = []
        offset = 0

        msg_count = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        for _ in range(msg_count):
            stream_kind_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            stream_kind = data[offset : offset + stream_kind_len].decode('utf-8')
            offset += stream_kind_len

            da00_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            da00_data = data[offset : offset + da00_len]
            offset += da00_len

            da00_msg = dataarray_da00.deserialise_da00(da00_data)
            stream = StreamId(kind=StreamKind(stream_kind), name=da00_msg.source_name)
            value = da00_to_scipp(da00_msg.data)

            messages.append(
                Message(timestamp=da00_msg.timestamp_ns, stream=stream, value=value)
            )

        return messages


class GenericJSONMessageSerializer(MessageSerializer[Any]):
    """
    Generic JSON serializer for messages with JSON-serializable values.

    Use this for simple types (dict, list, str, int, etc.) that don't need
    special handling like scipp DataArrays.
    """

    def serialize(self, messages: list[Message[Any]]) -> bytes:
        """
        Serialize messages to JSON bytes.

        Parameters
        ----------
        messages:
            List of messages to serialize.

        Returns
        -------
        :
            JSON-encoded bytes.
        """
        serialized = [
            {
                'timestamp': msg.timestamp,
                'stream': {'kind': msg.stream.kind.value, 'name': msg.stream.name},
                'value': msg.value,
            }
            for msg in messages
        ]
        return json.dumps(serialized).encode('utf-8')

    def deserialize(self, data: bytes) -> list[Message[Any]]:
        """
        Deserialize JSON bytes to messages.

        Parameters
        ----------
        data:
            JSON-encoded bytes.

        Returns
        -------
        :
            List of deserialized messages.
        """
        from ..core.message import StreamId, StreamKind

        decoded = json.loads(data.decode('utf-8'))
        messages = []
        for item in decoded:
            stream = StreamId(
                kind=StreamKind(item['stream']['kind']), name=item['stream']['name']
            )
            messages.append(
                Message(timestamp=item['timestamp'], stream=stream, value=item['value'])
            )
        return messages


class ConfigMessageSerializer(MessageSerializer[Any]):
    """
    JSON serializer specifically for CONFIG messages.

    Serializes config messages with their ConfigKey and value in JSON format.
    """

    def serialize(self, messages: list[Message[Any]]) -> bytes:
        """
        Serialize config messages to JSON.

        Parameters
        ----------
        messages:
            List of config messages to serialize.

        Returns
        -------
        :
            JSON-encoded bytes.
        """
        serialized = []
        for msg in messages:
            # Config messages have ConfigUpdate as value
            value_data = (
                msg.value.value.model_dump()
                if hasattr(msg.value.value, 'model_dump')
                else msg.value.value
            )
            config_dict = {
                'key': str(msg.value.config_key),
                'value': value_data,
            }
            serialized.append(
                {
                    'timestamp': msg.timestamp,
                    'stream': {'kind': msg.stream.kind.value, 'name': msg.stream.name},
                    'value': config_dict,
                }
            )
        return json.dumps(serialized).encode('utf-8')

    def deserialize(self, data: bytes) -> list[Message[Any]]:
        """
        Deserialize JSON bytes to config messages.

        Parameters
        ----------
        data:
            JSON-encoded bytes.

        Returns
        -------
        :
            List of deserialized config messages.
        """
        decoded = json.loads(data.decode('utf-8'))
        messages = []
        for item in decoded:
            stream = StreamId(
                kind=StreamKind(item['stream']['kind']), name=item['stream']['name']
            )
            # Keep value as dict (will be converted to ConfigUpdate by consumer)
            messages.append(
                Message(timestamp=item['timestamp'], stream=stream, value=item['value'])
            )
        return messages


class StatusMessageSerializer(MessageSerializer[Any]):
    """
    Binary serializer for STATUS messages using x5f2 format.

    Uses the same x5f2 format as Kafka transport for compatibility.
    """

    def serialize(self, messages: list[Message[Any]]) -> bytes:
        """
        Serialize status messages using x5f2 format.

        Parameters
        ----------
        messages:
            List of status messages to serialize.

        Returns
        -------
        :
            Binary-encoded messages with length prefixes.
        """
        from ..kafka.x5f2_compat import job_status_to_x5f2

        parts = [struct.pack('<I', len(messages))]

        for msg in messages:
            x5f2_data = job_status_to_x5f2(msg.value)
            parts.append(struct.pack('<I', len(x5f2_data)))
            parts.append(x5f2_data)

        return b''.join(parts)

    def deserialize(self, data: bytes) -> list[Message[Any]]:
        """
        Deserialize x5f2-encoded status messages.

        Parameters
        ----------
        data:
            Binary-encoded messages with length prefixes.

        Returns
        -------
        :
            List of deserialized status messages.
        """
        from ..kafka.x5f2_compat import x5f2_to_job_status

        messages = []
        offset = 0

        msg_count = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        for _ in range(msg_count):
            data_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            msg_data = data[offset : offset + data_len]
            offset += data_len

            job_status = x5f2_to_job_status(msg_data)
            messages.append(
                Message(
                    stream=STATUS_STREAM_ID,
                    value=job_status,
                )
            )

        return messages


class RoutingMessageSerializer(MessageSerializer[Any]):
    """
    Routes messages to appropriate serializers based on stream type.

    This matches the routing logic in KafkaSink but is transport-agnostic.
    - CONFIG messages → JSON with special config format
    - STATUS messages → x5f2 format
    - DATA messages → da00 format (default)
    """

    def serialize(self, messages: list[Message[Any]]) -> bytes:
        """
        Serialize messages using appropriate serializer for each type.

        Parameters
        ----------
        messages:
            List of messages to serialize (may have mixed types)

        Returns
        -------
        :
            Binary-encoded messages with type-specific serialization
        """
        from ..kafka.x5f2_compat import job_status_to_x5f2

        parts = [struct.pack('<I', len(messages))]

        for msg in messages:
            # Determine serialization format based on stream
            if msg.stream == CONFIG_STREAM_ID:
                # Config messages: JSON format with key/value
                config_dict = {
                    'key': str(msg.value.config_key),
                    'value': msg.value.value.model_dump(),
                }
                serialized = json.dumps(config_dict).encode('utf-8')
                format_id = b'config'
            elif msg.stream == STATUS_STREAM_ID:
                # Status messages: x5f2 format
                serialized = job_status_to_x5f2(msg.value)
                format_id = b'x5f2'
            else:
                # Data messages: da00 format (scipp DataArray)
                da00 = dataarray_da00.serialise_da00(
                    source_name=msg.stream.name,
                    timestamp_ns=msg.timestamp,
                    data=scipp_to_da00(msg.value),
                )
                serialized = da00
                format_id = b'da00'

            # Encode message with format ID, stream kind, and data
            stream_kind_bytes = msg.stream.kind.value.encode('utf-8')

            parts.append(struct.pack('<I', len(format_id)))
            parts.append(format_id)
            parts.append(struct.pack('<I', len(stream_kind_bytes)))
            parts.append(stream_kind_bytes)
            parts.append(struct.pack('<I', len(serialized)))
            parts.append(serialized)

        return b''.join(parts)

    def deserialize(self, data: bytes) -> list[Message[Any]]:
        """
        Deserialize messages using format-specific deserializers.

        Parameters
        ----------
        data:
            Binary data with mixed message types

        Returns
        -------
        :
            List of deserialized messages
        """
        from ..kafka.x5f2_compat import x5f2_to_job_status

        messages = []
        offset = 0

        msg_count = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        for _ in range(msg_count):
            # Read format ID
            format_id_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            format_id = data[offset : offset + format_id_len].decode('utf-8')
            offset += format_id_len

            # Read stream kind
            stream_kind_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            stream_kind = data[offset : offset + stream_kind_len].decode('utf-8')
            offset += stream_kind_len

            # Read serialized data
            data_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            msg_data = data[offset : offset + data_len]
            offset += data_len

            # Deserialize based on format
            if format_id == 'config':
                config_dict = json.loads(msg_data.decode('utf-8'))
                messages.append(
                    Message(
                        stream=CONFIG_STREAM_ID,
                        value=config_dict,  # Keep as dict for now
                    )
                )
            elif format_id == 'x5f2':
                job_status = x5f2_to_job_status(msg_data)
                messages.append(
                    Message(
                        stream=STATUS_STREAM_ID,
                        value=job_status,
                    )
                )
            elif format_id == 'da00':
                da00_msg = dataarray_da00.deserialise_da00(msg_data)
                stream = StreamId(
                    kind=StreamKind(stream_kind), name=da00_msg.source_name
                )
                value = da00_to_scipp(da00_msg.data)
                messages.append(
                    Message(timestamp=da00_msg.timestamp_ns, stream=stream, value=value)
                )

        return messages
