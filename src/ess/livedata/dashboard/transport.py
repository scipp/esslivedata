# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Transport abstraction for message sources and sinks."""

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Protocol, TypeVar

from ..core.message import Message, MessageSink, MessageSource

TResources = TypeVar('TResources', covariant=True)


class Transport(Protocol, Generic[TResources]):
    """
    Protocol for message transport implementations.

    This abstraction allows swapping between different transport mechanisms
    (e.g., Kafka, null/fake for testing) without changing dashboard code.

    The transport is a context manager that sets up resources on entry
    and cleans them up on exit. It also provides start/stop methods for
    managing background tasks (e.g., background message polling).
    """

    def __enter__(self) -> TResources:
        """
        Set up transport and return resources.

        Returns
        -------
        :
            Transport-specific resources (sources and sinks).
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Clean up transport resources.

        Parameters
        ----------
        exc_type:
            Exception type if an exception occurred.
        exc_val:
            Exception value if an exception occurred.
        exc_tb:
            Exception traceback if an exception occurred.
        """
        ...

    def start(self) -> None:
        """
        Start any background tasks.

        For Kafka transport, this starts background message polling.
        For null transport, this is a no-op.
        """
        ...

    def stop(self) -> None:
        """
        Stop any background tasks.

        For Kafka transport, this stops background message polling.
        For null transport, this is a no-op.
        """
        ...


@dataclass
class DashboardResources:
    """
    Resources provided by a transport for the dashboard.

    Attributes
    ----------
    message_source:
        Source for consuming messages from the transport.
    command_sink:
        Sink for publishing command messages.
    roi_sink:
        Sink for publishing ROI update messages.
    status_sink:
        Sink for publishing status/heartbeat messages.
    """

    message_source: MessageSource
    command_sink: MessageSink
    roi_sink: MessageSink
    status_sink: MessageSink


class NullMessageSource:
    """Message source that returns no messages (no-op implementation)."""

    def get_messages(self) -> Sequence[Message]:
        """Return empty list of messages."""
        return []


class NullMessageSink:
    """Message sink that discards all messages (no-op implementation)."""

    def publish_messages(self, messages: list[Message]) -> None:
        """Discard messages without doing anything."""
        pass


class NullTransport:
    """
    Null transport that provides no-op implementations.

    Useful for testing or when message transport is not needed.
    All resources do nothing - messages are discarded and none are produced.
    """

    def __enter__(self) -> DashboardResources:
        """Return no-op resources."""
        return DashboardResources(
            message_source=NullMessageSource(),
            command_sink=NullMessageSink(),
            roi_sink=NullMessageSink(),
            status_sink=NullMessageSink(),
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Nothing to clean up."""
        pass

    def start(self) -> None:
        """Nothing to start."""
        pass

    def stop(self) -> None:
        """Nothing to stop."""
        pass
