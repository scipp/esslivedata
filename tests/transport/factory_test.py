# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unit tests for transport factory functions."""

import pytest

from ess.livedata.config.transport_config import StreamTransportConfig, TransportConfig
from ess.livedata.core.message import StreamKind
from ess.livedata.http_transport import HTTPMessageSource, MultiHTTPSource
from ess.livedata.transport import (
    HttpStrategy,
    KafkaStrategy,
    RoutingSink,
    create_sink_from_config,
    create_source_from_config,
    create_strategies_from_config,
)


class TestCreateStrategiesFromConfig:
    """Tests for create_strategies_from_config()."""

    def test_creates_http_strategies_for_different_urls(self):
        """Should create separate strategies for different HTTP URLs."""
        config = TransportConfig(
            streams=[
                StreamTransportConfig(
                    kind=StreamKind.LIVEDATA_CONFIG,
                    transport='http',
                    url='http://localhost:5011',
                ),
                StreamTransportConfig(
                    kind=StreamKind.LIVEDATA_DATA,
                    transport='http',
                    url='http://localhost:8000',
                ),
            ]
        )

        strategies = create_strategies_from_config(config)

        assert len(strategies) == 2
        assert ('http', 'http://localhost:5011') in strategies
        assert ('http', 'http://localhost:8000') in strategies
        assert isinstance(strategies[('http', 'http://localhost:5011')], HttpStrategy)

    def test_groups_streams_with_same_url(self):
        """Should create single strategy for streams sharing same URL."""
        config = TransportConfig(
            streams=[
                StreamTransportConfig(
                    kind=StreamKind.LIVEDATA_DATA,
                    transport='http',
                    url='http://localhost:8000',
                ),
                StreamTransportConfig(
                    kind=StreamKind.LIVEDATA_STATUS,
                    transport='http',
                    url='http://localhost:8000',
                ),
            ]
        )

        strategies = create_strategies_from_config(config)

        # Should create only one strategy for the shared URL
        assert len(strategies) == 1
        assert ('http', 'http://localhost:8000') in strategies

    def test_creates_kafka_strategy(self):
        """Should create Kafka strategy with None URL."""
        config = TransportConfig(
            streams=[
                StreamTransportConfig(
                    kind=StreamKind.DETECTOR_EVENTS,
                    transport='kafka',
                    url=None,
                ),
            ]
        )

        strategies = create_strategies_from_config(config)

        assert len(strategies) == 1
        assert ('kafka', None) in strategies
        assert isinstance(strategies[('kafka', None)], KafkaStrategy)

    def test_handles_mixed_transports(self):
        """Should create strategies for both HTTP and Kafka."""
        config = TransportConfig(
            streams=[
                StreamTransportConfig(
                    kind=StreamKind.DETECTOR_EVENTS,
                    transport='kafka',
                    url=None,
                ),
                StreamTransportConfig(
                    kind=StreamKind.LIVEDATA_CONFIG,
                    transport='http',
                    url='http://localhost:5011',
                ),
            ]
        )

        strategies = create_strategies_from_config(config)

        assert len(strategies) == 2
        assert ('kafka', None) in strategies
        assert ('http', 'http://localhost:5011') in strategies


class TestCreateSourceFromConfig:
    """Tests for create_source_from_config()."""

    def test_creates_http_source_single_url(self):
        """Should create HTTPMessageSource or MultiHTTPSource for single URL."""
        config = TransportConfig(
            streams=[
                StreamTransportConfig(
                    kind=StreamKind.LIVEDATA_DATA,
                    transport='http',
                    url='http://localhost:8000',
                ),
            ]
        )

        source = create_source_from_config(
            instrument='dummy',
            adapter=None,
            transport_config=config,
        )

        # HttpStrategy creates single HTTPMessageSource for single stream kind
        assert isinstance(source, HTTPMessageSource)

    def test_creates_multi_source_for_multiple_urls(self):
        """Should create MultiHTTPSource when multiple URLs exist."""
        config = TransportConfig(
            streams=[
                StreamTransportConfig(
                    kind=StreamKind.LIVEDATA_CONFIG,
                    transport='http',
                    url='http://localhost:5011',
                ),
                StreamTransportConfig(
                    kind=StreamKind.LIVEDATA_DATA,
                    transport='http',
                    url='http://localhost:8000',
                ),
            ]
        )

        source = create_source_from_config(
            instrument='dummy',
            adapter=None,
            transport_config=config,
        )

        # Multiple URL groups should result in MultiHTTPSource
        assert isinstance(source, MultiHTTPSource)

    def test_raises_on_empty_config(self):
        """Should raise ValueError if no streams configured."""
        config = TransportConfig(streams=[])

        with pytest.raises(ValueError, match="No sources created"):
            create_source_from_config(
                instrument='dummy',
                adapter=None,
                transport_config=config,
            )


class TestCreateSinkFromConfig:
    """Tests for create_sink_from_config()."""

    def test_creates_routing_sink(self):
        """Should always create RoutingSink."""
        config = TransportConfig(
            streams=[
                StreamTransportConfig(
                    kind=StreamKind.LIVEDATA_DATA,
                    transport='http',
                    url='http://localhost:8000',
                ),
            ]
        )

        sink = create_sink_from_config(
            instrument='dummy',
            transport_config=config,
        )

        assert isinstance(sink, RoutingSink)

    def test_routing_sink_has_correct_routes(self):
        """Should create routes for all configured stream kinds."""
        config = TransportConfig(
            streams=[
                StreamTransportConfig(
                    kind=StreamKind.LIVEDATA_DATA,
                    transport='http',
                    url='http://localhost:8000',
                ),
                StreamTransportConfig(
                    kind=StreamKind.LIVEDATA_STATUS,
                    transport='http',
                    url='http://localhost:8000',
                ),
            ]
        )

        sink = create_sink_from_config(
            instrument='dummy',
            transport_config=config,
        )

        # RoutingSink should have routes for both stream kinds
        assert isinstance(sink, RoutingSink)
        # Check that both kinds are routed (by accessing private attribute for testing)
        assert StreamKind.LIVEDATA_DATA in sink._routes
        assert StreamKind.LIVEDATA_STATUS in sink._routes

    def test_raises_on_empty_config(self):
        """Should raise ValueError if no streams configured."""
        config = TransportConfig(streams=[])

        with pytest.raises(ValueError, match="No sinks created"):
            create_sink_from_config(
                instrument='dummy',
                transport_config=config,
            )
