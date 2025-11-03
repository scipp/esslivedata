# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import Generic, TypeVar

from .core import Message
from .core.message import STATUS_STREAM_ID

T = TypeVar('T')


class FakeMessageSource(Generic[T]):
    """
    A fake message source that returns messages from memory for testing purposes.
    """

    def __init__(self, messages: list[list[Message[T]]]) -> None:
        self._messages = messages
        self._index = 0

    def get_messages(self) -> list[Message[T]]:
        messages = (
            self._messages[self._index] if self._index < len(self._messages) else []
        )
        self._index += 1
        return messages


class FakeMessageSink(Generic[T]):
    """
    A fake message sink that stores messages in memory for testing purposes.

    Provides two views of published messages:
    - published_messages: Each publish_messages() call stored as a separate batch
    - messages: Flattened list of all non-status messages
    - status_messages: Flattened list of all status messages
    """

    def __init__(self) -> None:
        self.published_messages: list[list[Message[T]]] = []
        self.messages: list[Message[T]] = []
        self.status_messages: list[Message[T]] = []

    def publish_messages(self, messages: list[Message[T]]) -> None:
        # Store batch as-is
        self.published_messages.append(messages)
        # Also store flattened views for backward compatibility
        for msg in messages:
            if msg.stream == STATUS_STREAM_ID:
                self.status_messages.append(msg)
            else:
                self.messages.append(msg)
