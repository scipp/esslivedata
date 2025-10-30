# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Shared utilities for transport operations."""

from ..core.message import MessageSink
from ..http_transport.service import HTTPMultiEndpointSink
from .routing_sink import RoutingSink


def extract_http_sinks(sink: MessageSink) -> list[HTTPMultiEndpointSink]:
    """
    Extract HTTP sink instances that need lifecycle management.

    Parameters
    ----------
    sink:
        The message sink to extract HTTP sinks from.

    Returns
    -------
    :
        List of HTTP sinks found in the sink hierarchy.
    """
    http_sinks = []

    # Check if the sink is wrapped in an adapter (e.g., UnrollingSinkAdapter)
    unwrapped_sink = getattr(sink, '_sink', sink)

    if isinstance(unwrapped_sink, RoutingSink):
        # Check each route for HTTP sinks
        http_sinks.extend(
            route_sink
            for route_sink in unwrapped_sink.routes.values()
            if isinstance(route_sink, HTTPMultiEndpointSink)
        )
    elif isinstance(unwrapped_sink, HTTPMultiEndpointSink):
        http_sinks.append(unwrapped_sink)

    return http_sinks
