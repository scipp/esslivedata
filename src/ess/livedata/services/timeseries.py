# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service that processes logdata into timeseries for plotting."""

import logging
from collections.abc import Mapping
from functools import partial
from typing import Any, NoReturn

from ess.livedata.config import instrument_registry
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.core.message_batcher import NaiveMessageBatcher
from ess.livedata.core.orchestrating_processor import OrchestratingProcessor
from ess.livedata.handlers.timeseries_handler import LogdataHandlerFactory
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.service_factory import DataServiceBuilder, DataServiceRunner


def make_timeseries_service_builder(
    *,
    instrument: str,
    dev: bool = True,
    log_level: int = logging.INFO,
    attribute_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> DataServiceBuilder:
    stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
    adapter = (
        RoutingAdapterBuilder(stream_mapping=stream_mapping)
        .with_logdata_route()
        .with_livedata_config_route()
        .build()
    )
    instrument_obj = instrument_registry[instrument]
    instrument_obj.load_factories()
    service_name = 'timeseries'
    preprocessor_factory = LogdataHandlerFactory(
        instrument=instrument_obj,
        attribute_registry=attribute_registry,
    )
    # The SimpleMessageBatcher used by default by OrchestratingProcessor) processes
    # messages in batches, not emitting messages unless the current batch is considered
    # "complete", by the first message after the batch interval arriving. This works for
    # monitor and detector processing (including for logs that are processed as part of
    # the overall stream of messages on the same service).
    # However, this service processes only logs, and the SimpleMessageBatcher would
    # indefinitely withhold the last log message. We use the NaiveMessageBatcher here,
    # which emits messages as soon as they arrive.
    processor = partial(OrchestratingProcessor, message_batcher=NaiveMessageBatcher())
    return DataServiceBuilder(
        instrument=instrument,
        name=service_name,
        log_level=log_level,
        adapter=adapter,
        preprocessor_factory=preprocessor_factory,
        processor_cls=processor,  # type: ignore[arg-type]
    )


def build_service(*, instrument: str, dev: bool = True, log_level: int = logging.INFO):
    """Build and return the timeseries service without starting it."""
    from ess.livedata import transport_context
    from ess.livedata.config import config_names
    from ess.livedata.config.config_loader import load_config
    from ess.livedata.kafka.sink import KafkaSink, UnrollingSinkAdapter

    builder = make_timeseries_service_builder(
        instrument=instrument, dev=dev, log_level=log_level
    )

    # Check if using in-memory broker
    if transport_context.get_broker() is not None:
        # Build with in-memory transport
        source_topics = builder._get_topic_names_from_adapter()
        sink_topic = f"{builder.instrument}_output"
        return builder.from_in_memory_broker(
            source_topics=source_topics,
            sink_topic=sink_topic,
        )
    else:
        # Build with Kafka transport
        consumer_config = load_config(namespace=config_names.raw_data_consumer, env='')
        kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)
        kafka_upstream_config = load_config(namespace=config_names.kafka_upstream)

        sink = KafkaSink.from_kafka_config(
            instrument=builder.instrument, kafka_config=kafka_downstream_config
        )
        sink = UnrollingSinkAdapter(sink)

        return builder.from_consumer_config(
            kafka_config={**consumer_config, **kafka_upstream_config},
            sink=sink,
            use_background_source=True,
        )


def main() -> NoReturn:
    runner = DataServiceRunner(
        pretty_name='Logdata to Timeseries',
        make_builder=make_timeseries_service_builder,
    )
    runner.run()


if __name__ == "__main__":
    main()
