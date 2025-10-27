# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for RoutingAdapterBuilder to ensure all routes are registered correctly."""

import pytest

from ess.livedata.config.instruments import available_instruments
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.kafka.routes import RoutingAdapterBuilder


@pytest.mark.parametrize('instrument', available_instruments())
def test_routing_adapter_builder_with_livedata_roi_route(instrument: str) -> None:
    """Verify with_livedata_roi_route() adds ROI topic to adapter routes."""
    stream_mapping = get_stream_mapping(instrument=instrument, dev=True)

    adapter = (
        RoutingAdapterBuilder(stream_mapping=stream_mapping)
        .with_livedata_roi_route()
        .build()
    )

    expected_topic = stream_mapping.livedata_roi_topic
    assert expected_topic in adapter._routes, (
        f"ROI topic '{expected_topic}' not found in adapter routes. "
        f"Available routes: {list(adapter._routes.keys())}"
    )


@pytest.mark.parametrize('instrument', available_instruments())
def test_routing_adapter_builder_all_livedata_routes(instrument: str) -> None:
    """Verify all livedata routes can be added together."""
    stream_mapping = get_stream_mapping(instrument=instrument, dev=True)

    adapter = (
        RoutingAdapterBuilder(stream_mapping=stream_mapping)
        .with_livedata_config_route()
        .with_livedata_data_route()
        .with_livedata_roi_route()
        .with_livedata_status_route()
        .build()
    )

    # Check all livedata topics are routed
    assert stream_mapping.livedata_config_topic in adapter._routes
    assert stream_mapping.livedata_data_topic in adapter._routes
    assert stream_mapping.livedata_roi_topic in adapter._routes
    assert stream_mapping.livedata_status_topic in adapter._routes
