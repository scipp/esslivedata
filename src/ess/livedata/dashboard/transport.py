# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Transport abstraction for message sources and sinks."""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from ..core.message import MessageSink, MessageSource

TResources = TypeVar('TResources', covariant=True)


class Transport(Protocol, Generic[TResources]):
    """
    Protocol for message transport implementations.

    This abstraction allows swapping between different transport mechanisms
    (e.g., Kafka, null/fake for testing) without changing dashboard code.

    The transport is a context manager that sets up resources on entry
    and cleans them up on exit.
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

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
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
    """

    message_source: MessageSource
    command_sink: MessageSink
    roi_sink: MessageSink


class NullMessageSource:
    """Message source that returns no messages (no-op implementation)."""

    def get_messages(self):
        """Return empty list of messages."""
        return []


class NullMessageSink:
    """Message sink that discards all messages (no-op implementation)."""

    def publish_messages(self, messages):
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
        )

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """Nothing to clean up."""
        pass
