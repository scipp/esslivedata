# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Protocol, TypeVar

T = TypeVar('T')
Tin = TypeVar('Tin')
Tout = TypeVar('Tout')


class StreamKind(str, Enum):
    __slots__ = ()
    UNKNOWN = "unknown"
    MONITOR_COUNTS = "monitor_counts"
    MONITOR_EVENTS = "monitor_events"
    DETECTOR_EVENTS = "detector_events"
    LOG = "log"
    LIVEDATA_COMMANDS = "livedata_commands"
    LIVEDATA_RESPONSES = "livedata_responses"
    LIVEDATA_DATA = "livedata_data"
    LIVEDATA_ROI = "livedata_roi"
    LIVEDATA_STATUS = "livedata_status"


@dataclass(frozen=True, slots=True, kw_only=True)
class StreamId:
    kind: StreamKind = StreamKind.UNKNOWN
    name: str


COMMANDS_STREAM_ID = StreamId(kind=StreamKind.LIVEDATA_COMMANDS, name='')
RESPONSES_STREAM_ID = StreamId(kind=StreamKind.LIVEDATA_RESPONSES, name='')
STATUS_STREAM_ID = StreamId(kind=StreamKind.LIVEDATA_STATUS, name='')


@dataclass(frozen=True, slots=True, kw_only=True)
class Message(Generic[T]):
    """
    A message with a timestamp and a stream key.

    Parameters
    ----------
    timestamp:
        The timestamp of the message in nanoseconds since the epoch in UTC.
        If not provided, the current time is used.
    stream:
        The stream key of the message. Identifies which stream the message belongs to.
        This can be used to distinguish messages from different sources or types.
    value:
        The value of the message.
    """

    timestamp: int = field(
        default_factory=lambda: int(
            datetime.datetime.now(datetime.UTC).timestamp() * 1_000_000_000
        )
    )
    stream: StreamId
    value: T

    def __lt__(self, other: Message[T]) -> bool:
        return self.timestamp < other.timestamp


S = TypeVar('S')


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaMessage(Generic[S]):
    """
    Message with schema-specific data format.

    This sits at the boundary between transport layer (Kafka) and domain layer.
    The schema format S comes from streaming_data_types library (EventData,
    da00 Variables, etc.).

    Parameters
    ----------
    stream:
        The stream identifier for this message.
    data:
        Schema-specific data format from streaming_data_types.
    timestamp:
        The timestamp in nanoseconds since the epoch in UTC.
    key:
        Optional message key for Kafka partitioning/compaction.
    """

    stream: StreamId
    data: S
    timestamp: int
    key: bytes | None = None


class SchemaSerializer(Protocol[S]):
    """Serializes schema-specific format to bytes for wire transmission."""

    def serialize(self, msg: SchemaMessage[S]) -> bytes:
        """
        Serialize schema message to wire format.

        Parameters
        ----------
        msg:
            The schema message to serialize.

        Returns
        -------
        :
            Serialized bytes ready for transport.
        """
        ...


class SchemaDeserializer(Protocol[S]):
    """Deserializes bytes to schema-specific format with metadata extraction."""

    def deserialize(
        self,
        data: bytes,
        *,
        topic: str,
        timestamp: int,
        key: bytes | None,
    ) -> SchemaMessage[S]:
        """
        Deserialize bytes to schema message.

        Parameters
        ----------
        data:
            Raw bytes from transport layer.
        topic:
            Source topic (used for stream resolution).
        timestamp:
            Message timestamp from transport layer.
        key:
            Optional message key from transport layer.

        Returns
        -------
        :
            Schema message with extracted metadata.
        """
        ...


class DomainConverter(Protocol[S, T]):
    """Converts between schema format and domain model."""

    def to_domain(self, schema_msg: SchemaMessage[S]) -> Message[T]:
        """
        Convert schema message to domain message.

        Parameters
        ----------
        schema_msg:
            Schema message to convert.

        Returns
        -------
        :
            Domain message.
        """
        ...

    def to_schema(self, msg: Message[T]) -> SchemaMessage[S]:
        """
        Convert domain message to schema message.

        Parameters
        ----------
        msg:
            Domain message to convert.

        Returns
        -------
        :
            Schema message.
        """
        ...


class MessageSource(Protocol, Generic[Tin]):
    # Note that Tin is often (but not always) Message[T]
    def get_messages(self) -> Sequence[Tin]: ...


class MessageSink(Protocol, Generic[Tout]):
    def publish_messages(self, messages: list[Message[Tout]]) -> None:
        """
        Publish messages to the producer.

        Args:
            messages: A list of messages to publish.
        """


class SchemaMessageSink(Protocol, Generic[S]):
    """Sink for schema-level messages."""

    def publish_messages(self, messages: list[SchemaMessage[S]]) -> None:
        """
        Publish schema messages.

        Parameters
        ----------
        messages:
            A list of schema messages to publish.
        """
        ...


def compact_messages(messages: list[Message[T]]) -> list[Message[T]]:
    """
    Compact messages by removing outdates ones, keeping only the latest for each key.
    """
    latest = {}
    for msg in sorted(messages, reverse=True):  # Newest first
        if msg.stream not in latest:
            latest[msg.stream] = msg
    return sorted(latest.values())
