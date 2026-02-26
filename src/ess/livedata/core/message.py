# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, Protocol, TypeVar

T = TypeVar('T')
Tin = TypeVar('Tin')
Tout = TypeVar('Tout')


class StreamKind(StrEnum):
    __slots__ = ()
    UNKNOWN = "unknown"
    MONITOR_COUNTS = "monitor_counts"
    MONITOR_EVENTS = "monitor_events"
    DETECTOR_EVENTS = "detector_events"
    AREA_DETECTOR = "area_detector"
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


def compact_messages(messages: list[Message[T]]) -> list[Message[T]]:
    """
    Compact messages by removing outdates ones, keeping only the latest for each key.
    """
    latest = {}
    for msg in sorted(messages, reverse=True):  # Newest first
        if msg.stream not in latest:
            latest[msg.stream] = msg
    return sorted(latest.values())
