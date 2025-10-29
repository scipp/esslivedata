# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Factory functions for creating transport layers based on configuration."""

import logging
from typing import Any, Literal

from confluent_kafka import Consumer

from ..config import config_names
from ..config.config_loader import load_config
from ..config.models import ConfigKey
from ..config.streams import get_stream_mapping, stream_kind_to_topic
from ..core.message import StreamKind
from ..kafka import consumer as kafka_consumer
from ..kafka.message_adapter import AdaptingMessageSource
from ..kafka.routes import RoutingAdapterBuilder
from ..kafka.sink import KafkaSink
from ..kafka.source import BackgroundMessageSource
from .kafka_transport import KafkaTransport
from .message_transport import MessageTransport

TransportType = Literal['kafka', 'http']


def create_config_transport(
    *,
    transport_type: TransportType,
    instrument: str,
    logger: logging.Logger,
    http_config_url: str | None = None,
    http_config_sink_port: int = 5011,
    consumer: Consumer | None = None,
) -> tuple[MessageTransport[ConfigKey, dict[str, Any]], Any | None]:
    """
    Create a config message transport based on the specified type.

    Parameters
    ----------
    transport_type:
        Type of transport ('kafka' or 'http')
    instrument:
        Instrument name
    logger:
        Logger instance
    http_config_url:
        Base URL to poll for config messages from backend (HTTP mode only)
    http_config_sink_port:
        Port for dashboard's config sink (HTTP mode only)
    consumer:
        Kafka consumer for config messages (required for Kafka mode)

    Returns
    -------
    :
        Tuple of (transport, http_sink_or_none) - HTTP sink for lifecycle management
    """
    if transport_type == 'kafka':
        if consumer is None:
            raise ValueError("consumer is required for Kafka transport")

        kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)
        transport = KafkaTransport(
            kafka_config=kafka_downstream_config,
            consumer=consumer,
            logger=logger,
        )
        return transport, None

    elif transport_type == 'http':
        if http_config_url is None:
            raise ValueError("http_config_url is required for HTTP transport")

        from ..http_transport import (
            GenericJSONMessageSerializer,
            HTTPMultiEndpointSink,
        )

        # Create HTTP multi-endpoint sink for dashboard (only /config is used)
        # Dashboard uses GenericJSONMessageSerializer for all endpoints for simplicity
        config_sink = HTTPMultiEndpointSink(
            data_serializer=GenericJSONMessageSerializer(),
            status_serializer=GenericJSONMessageSerializer(),
            config_serializer=GenericJSONMessageSerializer(),
            host='0.0.0.0',  # noqa: S104
            port=http_config_sink_port,
        )

        from .http_config_transport import HTTPConfigTransport

        transport = HTTPConfigTransport(
            sink=config_sink,
            poll_url=http_config_url,
            logger=logger,
        )
        return transport, config_sink

    else:
        raise ValueError(f"Unknown transport type: {transport_type}")


def create_data_source(
    *,
    transport_type: TransportType,
    instrument: str,
    dev: bool,
    http_data_url: str | None = None,
    exit_stack: Any,  # ExitStack from contextlib
):
    """
    Create a data message source based on the specified type.

    Parameters
    ----------
    transport_type:
        Type of transport ('kafka' or 'http')
    instrument:
        Instrument name
    dev:
        Development mode flag
    http_data_url:
        Base URL for HTTP data service (required for HTTP mode)
    exit_stack:
        ExitStack for resource management

    Returns
    -------
    :
        Message source for data and status messages
    """
    if transport_type == 'kafka':
        stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
        adapter = (
            RoutingAdapterBuilder(stream_mapping=stream_mapping)
            .with_livedata_data_route()
            .with_livedata_status_route()
            .build()
        )

        consumer_config = load_config(
            namespace=config_names.reduced_data_consumer, env=''
        )
        kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)
        data_topic = stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_DATA
        )
        status_topic = stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_STATUS
        )
        consumer = exit_stack.enter_context(
            kafka_consumer.make_consumer_from_config(
                topics=[data_topic, status_topic],
                config={**consumer_config, **kafka_downstream_config},
                group='dashboard',
            )
        )
        source = exit_stack.enter_context(BackgroundMessageSource(consumer=consumer))
        return AdaptingMessageSource(source=source, adapter=adapter)

    elif transport_type == 'http':
        if http_data_url is None:
            raise ValueError("http_data_url is required for HTTP transport")

        from ..http_transport import (
            DA00MessageSerializer,
            HTTPMessageSource,
            MultiHTTPSource,
            StatusMessageSerializer,
        )

        # Poll separate endpoints for data and status (mirrors Kafka topics)
        data_source = HTTPMessageSource(
            base_url=http_data_url,
            endpoint='/data',
            serializer=DA00MessageSerializer(),
        )
        status_source = HTTPMessageSource(
            base_url=http_data_url,
            endpoint='/status',
            serializer=StatusMessageSerializer(),
        )

        # Combine sources similar to Kafka's MultiConsumer
        combined_source = MultiHTTPSource([data_source, status_source])
        return exit_stack.enter_context(combined_source)

    else:
        raise ValueError(f"Unknown transport type: {transport_type}")


def create_roi_sink(
    *,
    transport_type: TransportType,
    instrument: str,
    logger: logging.Logger,
    http_sink_host: str = '0.0.0.0',  # noqa: S104
    http_sink_port: int = 5010,
):
    """
    Create a sink for publishing ROI updates.

    Parameters
    ----------
    transport_type:
        Type of transport ('kafka' or 'http')
    instrument:
        Instrument name
    logger:
        Logger instance
    http_sink_host:
        Host for HTTP sink (HTTP mode only)
    http_sink_port:
        Port for HTTP sink (HTTP mode only)

    Returns
    -------
    :
        Sink implementation for ROI messages
    """
    if transport_type == 'kafka':
        kafka_upstream_config = load_config(namespace=config_names.kafka_upstream)
        from ..kafka.sink import serialize_dataarray_to_da00

        return KafkaSink(
            kafka_config=kafka_upstream_config,
            instrument=instrument,
            serializer=serialize_dataarray_to_da00,
            logger=logger,
        )

    elif transport_type == 'http':
        from ..http_transport import (
            DA00MessageSerializer,
            GenericJSONMessageSerializer,
            HTTPMultiEndpointSink,
        )

        # ROI sink exposes /data endpoint (ROI data), /status, and /config unused
        return HTTPMultiEndpointSink(
            data_serializer=DA00MessageSerializer(),
            status_serializer=GenericJSONMessageSerializer(),
            config_serializer=GenericJSONMessageSerializer(),
            host=http_sink_host,
            port=http_sink_port,
        )

    else:
        raise ValueError(f"Unknown transport type: {transport_type}")
