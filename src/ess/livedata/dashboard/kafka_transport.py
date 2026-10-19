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
from ess.livedata.core.job_manager import Command
from ess.livedata.core.message import StreamKind
from ess.livedata.kafka import consumer as kafka_consumer
from ess.livedata.kafka.message_adapter import AdaptingMessageSource
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.sink import KafkaSink
from ess.livedata.kafka.sink_serializers import CommandSerializer, Da00Serializer
from ess.livedata.kafka.source import KafkaMessageSource

from .transport import DashboardResources, Transport

logger = structlog.get_logger(__name__)

_MAX_MESSAGES_PER_PASS = 2_000
"""Cap on messages the ingestion loop takes from Kafka per pass.

Each pass consumes, then rebuilds every changed visible plot, so the loop keeps
up only while ``cap > input rate * pass duration``; otherwise messages pile up
in Kafka, unseen by any backend metric. A larger cap tolerates longer passes,
but a pass ingests its messages under the DataService lock, which session pulls
wait for, so the cap also bounds how long a catch-up pass blocks sessions.

``consume`` returns only what librdkafka has already fetched, which is bounded
in bytes by ``queued.max.messages.kbytes``, so large payloads hit that bound
before this one. ``consumer_lag`` in ``dashboard_ingest_metrics`` covers both.
"""


def _total_lag(source: KafkaMessageSource) -> int | None:
    lag = source.get_consumer_lag()
    return None if lag is None else lag['total_lag']


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

    def __enter__(self) -> DashboardResources:
        """Set up Kafka connections and return dashboard resources."""
        try:
            # Load configurations
            kafka_config = load_config(namespace=config_names.kafka)
            consumer_config = load_config(
                namespace=config_names.reduced_data_consumer, env=''
            )

            # Create message source
            message_source, kafka_source = self._create_message_source(
                consumer_config=consumer_config,
                kafka_config=kafka_config,
            )

            # Create command sink
            command_sink = self._exit_stack.enter_context(
                KafkaSink[Command](
                    kafka_config=kafka_config,
                    serializer=CommandSerializer(instrument=self._instrument),
                )
            )

            # Create ROI sink
            roi_sink = self._exit_stack.enter_context(
                KafkaSink[sc.DataArray](
                    kafka_config=kafka_config,
                    serializer=Da00Serializer(instrument=self._instrument),
                )
            )

            logger.info(
                "dashboard_kafka_transport_initialized", instrument=self._instrument
            )

            return DashboardResources(
                message_source=message_source,
                command_sink=command_sink,
                roi_sink=roi_sink,
                consumer_lag=lambda: _total_lag(kafka_source),
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
        """No background tasks to start."""

    def stop(self) -> None:
        """No background tasks to stop."""

    def _create_message_source(
        self,
        *,
        consumer_config: dict,
        kafka_config: dict,
    ) -> tuple[AdaptingMessageSource, KafkaMessageSource]:
        """Create unified Kafka consumer for all dashboard message streams.

        Returns the adapted source and the raw source it wraps, which reports
        consumer lag.
        """
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
                config={**consumer_config, **kafka_config},
                group='dashboard',
            )
        )

        source = KafkaMessageSource(
            consumer=consumer, num_messages=_MAX_MESSAGES_PER_PASS
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

        return AdaptingMessageSource(source=source, adapter=adapter), source
