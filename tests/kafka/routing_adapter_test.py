# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for RoutingAdapterBuilder to ensure all routes are registered correctly."""

import pytest

from ess.livedata.config.instruments import available_instruments
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.stream_mapping import InputStreamKey, StreamMapping


@pytest.mark.parametrize('instrument', available_instruments())
def test_routing_adapter_builder_with_livedata_roi_route(instrument: str) -> None:
    """Verify with_livedata_roi_route() adds ROI topic to adapter routes."""
    stream_mapping = get_stream_mapping(instrument=instrument, dev=True)

    adapter = (
        RoutingAdapterBuilder(stream_mapping=stream_mapping)
        .with_livedata_roi_route()
        .build()
    )

    expected_topic = stream_mapping.topics.livedata_roi
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
        .with_livedata_commands_route()
        .with_livedata_data_route()
        .with_livedata_roi_route()
        .with_livedata_status_route()
        .build()
    )

    # Check all livedata topics are routed
    assert stream_mapping.topics.livedata_commands in adapter._routes
    assert stream_mapping.topics.livedata_data in adapter._routes
    assert stream_mapping.topics.livedata_roi in adapter._routes
    assert stream_mapping.topics.livedata_status in adapter._routes


class TestWithRoutesFromMapping:
    def test_adds_detector_route_when_detectors_present(
        self, infra_kwargs: dict
    ) -> None:
        mapping = StreamMapping(
            instrument="test",
            detectors={InputStreamKey(topic="det", source_name="s"): "d"},
            monitors={},
            **infra_kwargs,
        )
        adapter = (
            RoutingAdapterBuilder(stream_mapping=mapping)
            .with_routes_from_mapping()
            .build()
        )
        assert "det" in adapter._routes

    def test_adds_monitor_route_when_monitors_present(self, infra_kwargs: dict) -> None:
        mapping = StreamMapping(
            instrument="test",
            detectors={},
            monitors={InputStreamKey(topic="mon", source_name="s"): "m"},
            **infra_kwargs,
        )
        adapter = (
            RoutingAdapterBuilder(stream_mapping=mapping)
            .with_routes_from_mapping()
            .build()
        )
        assert "mon" in adapter._routes

    def test_adds_log_route_when_logs_present(self, infra_kwargs: dict) -> None:
        mapping = StreamMapping(
            instrument="test",
            detectors={},
            monitors={},
            logs={InputStreamKey(topic="log", source_name="s"): "l"},
            **infra_kwargs,
        )
        adapter = (
            RoutingAdapterBuilder(stream_mapping=mapping)
            .with_routes_from_mapping()
            .build()
        )
        assert "log" in adapter._routes

    def test_adds_area_detector_route_when_present(self, infra_kwargs: dict) -> None:
        mapping = StreamMapping(
            instrument="test",
            detectors={},
            monitors={},
            area_detectors={InputStreamKey(topic="area", source_name="s"): "a"},
            **infra_kwargs,
        )
        adapter = (
            RoutingAdapterBuilder(stream_mapping=mapping)
            .with_routes_from_mapping()
            .build()
        )
        assert "area" in adapter._routes

    def test_skips_empty_luts(self, infra_kwargs: dict) -> None:
        mapping = StreamMapping(
            instrument="test",
            detectors={},
            monitors={},
            **infra_kwargs,
        )
        adapter = (
            RoutingAdapterBuilder(stream_mapping=mapping)
            .with_routes_from_mapping()
            .build()
        )
        assert adapter._routes == {}

    def test_does_not_add_infrastructure_routes(self, infra_kwargs: dict) -> None:
        mapping = StreamMapping(
            instrument="test",
            detectors={InputStreamKey(topic="det", source_name="s"): "d"},
            monitors={},
            **infra_kwargs,
        )
        adapter = (
            RoutingAdapterBuilder(stream_mapping=mapping)
            .with_routes_from_mapping()
            .build()
        )
        assert "cmd" not in adapter._routes
        assert "roi" not in adapter._routes
