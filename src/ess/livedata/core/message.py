# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, Protocol, TypeVar

from .timestamp import Timestamp

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
    LIVEDATA_FOM = "livedata_fom"
    LIVEDATA_ROI = "livedata_roi"
    LIVEDATA_STATUS = "livedata_status"
    RUN_CONTROL = "run_control"


@dataclass(frozen=True, slots=True, kw_only=True)
class StreamId:
    kind: StreamKind = StreamKind.UNKNOWN
    name: str


COMMANDS_STREAM_ID = StreamId(kind=StreamKind.LIVEDATA_COMMANDS, name='')
RESPONSES_STREAM_ID = StreamId(kind=StreamKind.LIVEDATA_RESPONSES, name='')
STATUS_STREAM_ID = StreamId(kind=StreamKind.LIVEDATA_STATUS, name='')
RUN_CONTROL_STREAM_ID = StreamId(kind=StreamKind.RUN_CONTROL, name='')


@dataclass(frozen=True, slots=True)
class RunStart:
    """Run start event from the ESS control system."""

    run_name: str
    start_time: Timestamp
    stop_time: Timestamp | None = None

    def __str__(self) -> str:
        return f"RunStart(run_name={self.run_name!r})"


@dataclass(frozen=True, slots=True)
class RunStop:
    """Run stop event from the ESS control system."""

    run_name: str
    stop_time: Timestamp

    def __str__(self) -> str:
        return f"RunStop(run_name={self.run_name!r})"


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

    timestamp: Timestamp = field(default_factory=Timestamp.now)
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
