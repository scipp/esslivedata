# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Kafka-based transport implementation for the dashboard."""

from contextlib import ExitStack
from types import TracebackType

import scipp as sc
import structlog

from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.config.streams import get_stream_mapping, stream_kind_to_topic
from ess.livedata.core.message import StreamKind
from ess.livedata.handlers.config_handler import ConfigUpdate
from ess.livedata.kafka import consumer as kafka_consumer
from ess.livedata.kafka.message_adapter import AdaptingMessageSource
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.sink import KafkaSink, serialize_dataarray_to_da00
from ess.livedata.kafka.source import BackgroundMessageSource

from .transport import DashboardResources, Transport

logger = structlog.get_logger(__name__)


class DashboardKafkaTransport(Transport[DashboardResources]):
    """
    Kafka-based transport for the dashboard.

    Sets up Kafka consumers and producers for dashboard message streams,
    including data consumption and command/ROI publishing.

    Parameters
    ----------
    instrument:
        Instrument name (e.g., 'dummy', 'dream', 'bifrost')
    dev:
        Use dev mode with simplified topic structure
    """

    def __init__(
        self,
        *,
        instrument: str,
        dev: bool,
    ):
        self._instrument = instrument
        self._dev = dev
        self._exit_stack = ExitStack()
        self._background_source = None

    def __enter__(self) -> DashboardResources:
        """Set up Kafka connections and return dashboard resources."""
        try:
            # Load configurations
            kafka_downstream_config = load_config(
                namespace=config_names.kafka_downstream
            )
            kafka_upstream_config = load_config(namespace=config_names.kafka_upstream)
            consumer_config = load_config(
                namespace=config_names.reduced_data_consumer, env=''
            )

            # Create message source
            message_source = self._create_message_source(
                consumer_config=consumer_config,
                kafka_downstream_config=kafka_downstream_config,
            )

            # Create command sink
            command_sink = self._exit_stack.enter_context(
                KafkaSink[ConfigUpdate](
                    kafka_config=kafka_downstream_config,
                    instrument=self._instrument,
                )
            )

            # Create ROI sink
            roi_sink = self._exit_stack.enter_context(
                KafkaSink[sc.DataArray](
                    kafka_config=kafka_upstream_config,
                    instrument=self._instrument,
                    serializer=serialize_dataarray_to_da00,
                )
            )

            logger.info(
                "dashboard_kafka_transport_initialized", instrument=self._instrument
            )

            return DashboardResources(
                message_source=message_source,
                command_sink=command_sink,
                roi_sink=roi_sink,
            )

        except Exception:
            # Clean up any resources created before the exception
            self._exit_stack.close()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up all Kafka resources."""
        self._exit_stack.close()
        logger.info("dashboard_kafka_transport_cleaned_up")

    def start(self) -> None:
        """Start background message polling."""
        if self._background_source is not None:
            self._background_source.start()
            logger.info("dashboard_message_polling_started")

    def stop(self) -> None:
        """Stop background message polling."""
        if self._background_source is not None:
            self._background_source.stop()
            logger.info("dashboard_message_polling_stopped")

    def _create_message_source(
        self,
        *,
        consumer_config: dict,
        kafka_downstream_config: dict,
    ) -> AdaptingMessageSource:
        """Create unified Kafka consumer for all dashboard message streams."""
        # Define topics to subscribe to
        topics = [
            stream_kind_to_topic(instrument=self._instrument, kind=kind)
            for kind in [
                StreamKind.LIVEDATA_DATA,
                StreamKind.LIVEDATA_STATUS,
                StreamKind.LIVEDATA_RESPONSES,
            ]
        ]

        # Create consumer
        consumer = self._exit_stack.enter_context(
            kafka_consumer.make_consumer_from_config(
                topics=topics,
                config={**consumer_config, **kafka_downstream_config},
                group='dashboard',
            )
        )

        # Create background source and store for lifecycle management
        self._background_source = self._exit_stack.enter_context(
            BackgroundMessageSource(consumer=consumer)
        )

        # Create adapter for message routing
        stream_mapping = get_stream_mapping(instrument=self._instrument, dev=self._dev)
        adapter = (
            RoutingAdapterBuilder(stream_mapping=stream_mapping)
            .with_livedata_data_route()
            .with_livedata_status_route()
            .with_livedata_responses_route()
            .build()
        )

        return AdaptingMessageSource(source=self._background_source, adapter=adapter)
