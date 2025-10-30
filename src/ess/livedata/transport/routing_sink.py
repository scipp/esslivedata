# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TypeVar

from ..core.message import Message, MessageSink, StreamKind

T = TypeVar('T')


class RoutingSink(MessageSink[T]):
    """
    Routes messages to different sinks based on their stream kind.

    Messages are grouped by their stream kind and published to the corresponding
    sink. Messages with stream kinds that have no route are logged as warnings
    and skipped.

    Parameters
    ----------
    routes:
        Dictionary mapping stream kinds to their corresponding message sinks.
    logger:
        Optional logger for logging warnings and debug information.
    """

    def __init__(
        self,
        routes: dict[StreamKind, MessageSink[T]],
        *,
        logger: logging.Logger | None = None,
    ):
        self._routes = routes
        self._logger = logger or logging.getLogger(__name__)

    def publish_messages(self, messages: list[Message[T]]) -> None:
        """
        Group messages by stream kind and publish each group to its sink.

        Parameters
        ----------
        messages:
            List of messages to publish.
        """
        grouped = self._group_by_stream_kind(messages)

        for stream_kind, group_messages in grouped.items():
            if stream_kind not in self._routes:
                self._logger.warning(
                    "No route configured for stream kind '%s', skipping %d message(s)",
                    stream_kind.value,
                    len(group_messages),
                )
                continue

            sink = self._routes[stream_kind]
            self._logger.debug(
                "Publishing %d message(s) of kind '%s'",
                len(group_messages),
                stream_kind.value,
            )
            sink.publish_messages(group_messages)

    def _group_by_stream_kind(
        self, messages: Iterable[Message[T]]
    ) -> dict[StreamKind, list[Message[T]]]:
        """
        Group messages by their stream kind.

        Parameters
        ----------
        messages:
            Messages to group.

        Returns
        -------
        :
            Dictionary mapping stream kinds to lists of messages.
        """
        grouped: dict[StreamKind, list[Message[T]]] = {}
        for msg in messages:
            stream_kind = msg.stream.kind
            if stream_kind not in grouped:
                grouped[stream_kind] = []
            grouped[stream_kind].append(msg)
        return grouped
