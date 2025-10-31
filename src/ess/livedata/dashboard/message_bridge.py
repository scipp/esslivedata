# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import Generic, Protocol, TypeVar

K = TypeVar('K')
V = TypeVar('V')
Serialized = TypeVar('Serialized')


class MessageBridge(Protocol, Generic[K, Serialized]):
    """Protocol for publishing and consuming configuration messages."""

    def publish(self, key: K, value: Serialized) -> None:
        """Publish a configuration update message."""

    def pop_all(self) -> dict[K, Serialized]:
        """Pop all available configuration update messages."""


class FakeMessageBridge(MessageBridge[K, V], Generic[K, V]):
    """Fake message bridge for testing purposes."""

    def __init__(self):
        self._published_messages: list[tuple[K, V]] = []
        self._incoming_messages: list[tuple[K, V]] = []

    def publish(self, key: K, value: V) -> None:
        """Store published messages for inspection."""
        self._published_messages.append((key, value))

    def pop_all(self) -> dict[K, V]:
        """Pop the next message from the incoming queue."""
        messages = self._incoming_messages.copy()
        self._incoming_messages.clear()
        return dict(messages)

    def add_incoming_message(self, update: tuple[K, V]) -> None:
        """Add a message to the incoming queue for testing."""
        self._incoming_messages.append(update)

    def get_published_messages(self) -> list[tuple[K, V]]:
        """Get all published messages for inspection."""
        return self._published_messages.copy()

    def clear(self) -> None:
        """Clear all stored messages."""
        self._published_messages.clear()
        self._incoming_messages.clear()
