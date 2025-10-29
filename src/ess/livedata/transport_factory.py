# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Factory functions for creating transport-aware components."""

from __future__ import annotations

import logging
from typing import TypeVar

from .config import config_names
from .config.config_loader import load_config
from .core.message import MessageSink
from .kafka.sink import KafkaSink, Serializer, serialize_dataarray_to_da00

T = TypeVar("T")


def create_message_sink(
    instrument: str,
    serializer: Serializer[T] = serialize_dataarray_to_da00,
    logger: logging.Logger | None = None,
) -> MessageSink[T]:
    """
    Create appropriate message sink based on transport context.

    Checks the transport context for an in-memory broker. If found,
    returns InMemoryMessageSink. Otherwise, returns KafkaSink configured
    with upstream Kafka settings.

    Parameters
    ----------
    instrument:
        Instrument name for topic generation
    serializer:
        Serializer for data messages
    logger:
        Optional logger

    Returns
    -------
    :
        MessageSink implementation appropriate for current transport context
    """
    from . import transport_context

    broker = transport_context.get_broker()
    if broker is not None:
        from .in_memory import InMemoryMessageSink

        logger_msg = logger or logging.getLogger(__name__)
        logger_msg.debug("Creating InMemoryMessageSink for %s", instrument)
        return InMemoryMessageSink(broker, instrument, serializer, logger)
    else:
        kafka_config = load_config(namespace=config_names.kafka_upstream)
        logger_msg = logger or logging.getLogger(__name__)
        logger_msg.debug("Creating KafkaSink for %s", instrument)
        return KafkaSink.from_kafka_config(
            kafka_config=kafka_config,
            instrument=instrument,
            serializer=serializer,
            logger=logger,
        )
