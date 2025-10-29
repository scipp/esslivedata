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


def create_dashboard_sink(
    *,
    instrument: str,
    port: int,
    logger: logging.Logger,
) -> Any:
    """
    Create HTTP multi-endpoint sink for dashboard.

    Exposes both /config and /livedata_roi endpoints for backend services to poll.

    Parameters
    ----------
    instrument:
        Instrument name
    port:
        Port to expose HTTP endpoints on
    logger:
        Logger instance

    Returns
    -------
    :
        HTTPMultiEndpointSink exposing config and ROI endpoints
    """
    from ..http_transport import (
        DA00MessageSerializer,
        GenericJSONMessageSerializer,
        HTTPMultiEndpointSink,
    )

    return HTTPMultiEndpointSink(
        instrument=instrument,
        stream_serializers={
            StreamKind.LIVEDATA_CONFIG: GenericJSONMessageSerializer(),
            StreamKind.LIVEDATA_ROI: DA00MessageSerializer(),
        },
        host='0.0.0.0',  # noqa: S104
        port=port,
        logger_=logger,
    )


def create_config_transport(
    *,
    transport_type: TransportType,
    instrument: str,
    logger: logging.Logger,
    http_backend_url: str | None = None,
    http_sink: Any | None = None,
    consumer: Consumer | None = None,
) -> MessageTransport[ConfigKey, dict[str, Any]]:
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
    http_backend_url:
        Base URL to poll for config messages from backend (HTTP mode only)
    http_sink:
        HTTP multi-endpoint sink for publishing config (HTTP mode only)
    consumer:
        Kafka consumer for config messages (required for Kafka mode)

    Returns
    -------
    :
        Transport for config messages
    """
    if transport_type == 'kafka':
        if consumer is None:
            raise ValueError("consumer is required for Kafka transport")

        kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)
        return KafkaTransport(
            kafka_config=kafka_downstream_config,
            consumer=consumer,
            logger=logger,
        )

    elif transport_type == 'http':
        if http_backend_url is None:
            raise ValueError("http_backend_url is required for HTTP transport")
        if http_sink is None:
            raise ValueError("http_sink is required for HTTP transport")

        from .http_config_transport import HTTPConfigTransport

        return HTTPConfigTransport(
            sink=http_sink,
            poll_url=http_backend_url,
            logger=logger,
        )

    else:
        raise ValueError(f"Unknown transport type: {transport_type}")


def create_data_source(
    *,
    transport_type: TransportType,
    instrument: str,
    dev: bool,
    http_backend_url: str | None = None,
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
    http_backend_url:
        Base URL for HTTP backend service (required for HTTP mode)
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
        if http_backend_url is None:
            raise ValueError("http_backend_url is required for HTTP transport")

        from ..http_transport import (
            DA00MessageSerializer,
            HTTPMessageSource,
            MultiHTTPSource,
            StatusMessageSerializer,
        )

        # Poll separate endpoints for data and status (mirrors Kafka topics)
        # Endpoint names are derived from topic names with instrument prefix removed
        # e.g., dummy_livedata_data -> /livedata_data
        data_topic = stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_DATA
        )
        status_topic = stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_STATUS
        )
        data_endpoint = f"/{data_topic.removeprefix(f'{instrument}_')}"
        status_endpoint = f"/{status_topic.removeprefix(f'{instrument}_')}"

        data_source = HTTPMessageSource(
            base_url=http_backend_url,
            endpoint=data_endpoint,
            serializer=DA00MessageSerializer(),
        )
        status_source = HTTPMessageSource(
            base_url=http_backend_url,
            endpoint=status_endpoint,
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
    http_sink: Any | None = None,
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
    http_sink:
        HTTP multi-endpoint sink for publishing ROI (HTTP mode only)

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
        if http_sink is None:
            raise ValueError("http_sink is required for HTTP transport")
        # Reuse the same multi-endpoint sink created for config
        return http_sink

    else:
        raise ValueError(f"Unknown transport type: {transport_type}")
