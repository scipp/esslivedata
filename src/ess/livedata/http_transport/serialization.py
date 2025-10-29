# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Message serialization protocols and implementations for HTTP transport."""

import json
import struct
from typing import Any, Generic, Protocol, TypeVar

import scipp as sc
from streaming_data_types import dataarray_da00

from ..core.message import Message, StreamId, StreamKind
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
