# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Unified factory for creating transport sources and sinks from configuration.

This module provides high-level factory functions that create message sources
and sinks based on TransportConfig. It handles grouping streams by transport
type and URL, instantiating appropriate strategies, and combining multiple
sources/sinks as needed.
"""

from __future__ import annotations

from itertools import groupby
from typing import Any

from ..config.transport_config import StreamTransportConfig, TransportConfig
from ..core.message import MessageSink, MessageSource, StreamKind
from ..kafka.message_adapter import MessageAdapter
from .routing_sink import RoutingSink
from .strategy import HttpStrategy, KafkaStrategy, TransportStrategy


def create_strategies_from_config(
    transport_config: TransportConfig,
) -> dict[tuple[str, str | None], TransportStrategy]:
    """
    Create strategy instances for each unique (transport_type, url) pair.

    Groups the transport config by transport type and URL, then instantiates
    the appropriate strategy (KafkaStrategy or HttpStrategy) for each group.

    Parameters
    ----------
    transport_config:
        Transport configuration containing stream transport settings.

    Returns
    -------
    :
        Map of (transport_type, url) to strategy instance.
        For Kafka, url is None in the key.
        For HTTP, url is the base URL.
    """
    strategies: dict[tuple[str, str | None], TransportStrategy] = {}

    # Group streams by (transport, url)
    def group_key(stream_config: StreamTransportConfig) -> tuple[str, str | None]:
        return (stream_config.transport, stream_config.url)

    sorted_streams = sorted(transport_config.streams, key=group_key)

    for (transport_type, url), stream_group in groupby(sorted_streams, key=group_key):
        # Consume the iterator to get the actual configs
        list(stream_group)

        # Create strategy for this (transport, url) combination
        if transport_type == 'kafka':
            # For Kafka, use empty dict for now - caller may need to provide config
            strategies[(transport_type, url)] = KafkaStrategy(kafka_config={})
        elif transport_type == 'http':
            if url is None:
                raise ValueError(
                    f"HTTP transport requires a URL, but none provided for "
                    f"transport type '{transport_type}'"
                )
            strategies[(transport_type, url)] = HttpStrategy(base_url=url)
        else:
            raise ValueError(f"Unknown transport type: {transport_type}")

    return strategies


def create_source_from_config(
    instrument: str,
    adapter: MessageAdapter | None,
    transport_config: TransportConfig,
    kafka_config: dict[str, Any] | None = None,
) -> MessageSource:
    """
    Create a message source from transport configuration.

    Groups streams by (transport_type, url), instantiates appropriate strategies,
    and creates sources for each group. If multiple sources are created, they are
    combined into a single source.

    Only creates sources for stream kinds that the adapter needs to consume.
    If adapter is None, creates sources for all configured streams.

    Parameters
    ----------
    instrument:
        Instrument name for topic/endpoint derivation.
    adapter:
        Optional message adapter for transforming raw messages. If provided,
        only stream kinds corresponding to the adapter's topics will be used.
    transport_config:
        Transport configuration defining how each stream is transported.
    kafka_config:
        Optional Kafka configuration to use for Kafka strategies.
        If not provided, uses empty dict (strategies load defaults).

    Returns
    -------
    :
        Message source. If multiple transport groups exist, returns a combined
        source (currently using MultiHTTPSource, which works for heterogeneous
        source types).
    """
    from ..config.streams import stream_kind_to_topic

    # Filter streams to only those the adapter needs (input streams)
    if adapter is not None:
        # Get topics the adapter needs
        adapter_topics = set(adapter.topics)
        # Filter to stream kinds that match adapter's topics
        filtered_streams = [
            s
            for s in transport_config.streams
            if stream_kind_to_topic(instrument, s.kind) in adapter_topics
        ]
    else:
        filtered_streams = transport_config.streams

    if not filtered_streams:
        raise ValueError("No input streams found in transport config for this adapter")

    sources = []

    # Group streams by (transport, url)
    def group_key(stream_config: StreamTransportConfig) -> tuple[str, str | None]:
        return (stream_config.transport, stream_config.url)

    sorted_streams = sorted(filtered_streams, key=group_key)

    for (transport_type, url), stream_group in groupby(sorted_streams, key=group_key):
        # Extract stream kinds from this group
        stream_kinds = [s.kind for s in stream_group]

        # Create appropriate strategy
        if transport_type == 'kafka':
            strategy = KafkaStrategy(kafka_config=kafka_config or {})
        elif transport_type == 'http':
            if url is None:
                raise ValueError("HTTP transport requires a URL")
            strategy = HttpStrategy(base_url=url)
        else:
            raise ValueError(f"Unknown transport type: {transport_type}")

        # Create source for this group
        source = strategy.create_source(
            stream_kinds=stream_kinds,
            instrument=instrument,
            adapter=adapter,
        )
        sources.append(source)

    # Return single source or combine multiple
    if len(sources) == 0:
        raise ValueError("No sources created from transport config")
    if len(sources) == 1:
        return sources[0]

    # Use MultiHTTPSource to combine - it works for heterogeneous sources
    from ..http_transport.source import MultiHTTPSource

    return MultiHTTPSource(sources)


def create_sink_from_config(
    instrument: str,
    transport_config: TransportConfig,
    output_stream_kinds: list[StreamKind],
    kafka_config: dict[str, Any] | None = None,
) -> MessageSink:
    """
    Create a routing sink from transport configuration.

    Groups streams by (transport_type, url), instantiates appropriate strategies,
    and creates sinks for each group. Returns a RoutingSink that routes messages
    to the appropriate sink based on their stream kind.

    Only creates sinks for the specified output stream kinds.

    Parameters
    ----------
    instrument:
        Instrument name for topic/endpoint derivation.
    transport_config:
        Transport configuration defining how each stream is transported.
    output_stream_kinds:
        List of stream kinds this service will output (produce messages to).
    kafka_config:
        Optional Kafka configuration to use for Kafka strategies.
        If not provided, uses empty dict (strategies load defaults).

    Returns
    -------
    :
        RoutingSink that routes messages to appropriate sinks based on stream kind.
    """
    # Filter to only the output stream kinds this service produces
    output_kinds_set = set(output_stream_kinds)
    filtered_streams = [
        s for s in transport_config.streams if s.kind in output_kinds_set
    ]

    if not filtered_streams:
        raise ValueError(
            f"No output streams found in transport config. "
            f"Requested: {[k.value for k in output_stream_kinds]}, "
            f"Available: {[s.kind.value for s in transport_config.streams]}"
        )

    routing_map: dict[StreamKind, MessageSink] = {}

    # Group streams by (transport, url)
    def group_key(stream_config: StreamTransportConfig) -> tuple[str, str | None]:
        return (stream_config.transport, stream_config.url)

    sorted_streams = sorted(filtered_streams, key=group_key)

    for (transport_type, url), stream_group in groupby(sorted_streams, key=group_key):
        # Extract stream kinds from this group
        group_list = list(stream_group)
        stream_kinds = [s.kind for s in group_list]

        # Create appropriate strategy
        if transport_type == 'kafka':
            strategy = KafkaStrategy(kafka_config=kafka_config or {})
        elif transport_type == 'http':
            if url is None:
                raise ValueError("HTTP transport requires a URL")
            strategy = HttpStrategy(base_url=url)
        else:
            raise ValueError(f"Unknown transport type: {transport_type}")

        # Create sink for this group
        sink = strategy.create_sink(stream_kinds=stream_kinds, instrument=instrument)

        # Map each stream kind in this group to the sink
        for stream_kind in stream_kinds:
            routing_map[stream_kind] = sink

    if not routing_map:
        raise ValueError("No sinks created from transport config")

    return RoutingSink(routes=routing_map)
