# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""In-memory message sink implementation."""

from __future__ import annotations

import logging
from typing import TypeVar

from ess.livedata.core.message import Message
from ess.livedata.in_memory.broker import InMemoryBroker
from ess.livedata.in_memory.transport import InMemoryTransport
from ess.livedata.kafka.sink import KafkaSink, Serializer

logger = logging.getLogger(__name__)

T = TypeVar("T")


class InMemoryMessageSink(KafkaSink[T]):
    """
    In-memory implementation of MessageSink protocol.

    Uses the same serialization and routing logic as KafkaSink,
    but publishes messages to an InMemoryBroker instead of Kafka.

    Parameters
    ----------
    broker:
        Broker instance to publish to
    instrument:
        Instrument name for topic generation
    serializer:
        Serializer for data messages. If not provided, uses default da00 serializer.
    logger:
        Optional logger
    """

    def __init__(
        self,
        broker: InMemoryBroker,
        instrument: str,
        serializer: Serializer[T] | None = None,
        logger: logging.Logger | None = None,
    ):
        # Import default serializer if not provided
        if serializer is None:
            from ess.livedata.kafka.sink import serialize_dataarray_to_da00

            serializer = serialize_dataarray_to_da00

        transport = InMemoryTransport(broker=broker, logger=logger)
        super().__init__(
            transport=transport,
            instrument=instrument,
            serializer=serializer,
            logger=logger,
        )
        self._logger.info("InMemoryMessageSink created for instrument: %s", instrument)

    def close(self) -> None:
        """No cleanup needed for in-memory sink."""
        self._logger.info("InMemoryMessageSink closed")


class NullMessageSink:
    """
    A message sink that discards all messages.

    Used for services that don't publish output.
    """

    def publish_messages(self, messages: list[Message[T]]) -> None:
        """Discard messages."""
        pass

    def close(self) -> None:
        """No cleanup needed."""
        pass
