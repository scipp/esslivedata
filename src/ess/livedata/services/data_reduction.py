# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service that runs a data reduction workflow."""

import logging
from typing import NoReturn

from ess.livedata.config import instrument_registry
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.handlers.data_reduction_handler import ReductionHandlerFactory
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.service_factory import DataServiceBuilder, DataServiceRunner


def make_reduction_service_builder(
    *, instrument: str, dev: bool = True, log_level: int = logging.INFO
) -> DataServiceBuilder:
    stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
    adapter = (
        RoutingAdapterBuilder(stream_mapping=stream_mapping)
        .with_beam_monitor_route()
        .with_detector_route()
        .with_logdata_route()
        .with_livedata_config_route()
        .build()
    )
    instrument_config = instrument_registry[instrument]
    instrument_config.load_factories()
    service_name = 'data_reduction'
    preprocessor_factory = ReductionHandlerFactory(
        instrument=instrument_config, namespace=service_name
    )
    return DataServiceBuilder(
        instrument=instrument,
        name=service_name,
        log_level=log_level,
        adapter=adapter,
        preprocessor_factory=preprocessor_factory,
    )


def build_service(*, instrument: str, dev: bool = True, log_level: int = logging.INFO):
    """Build and return the data reduction service without starting it."""
    from ess.livedata import transport_context
    from ess.livedata.config import config_names
    from ess.livedata.config.config_loader import load_config
    from ess.livedata.kafka.sink import KafkaSink, UnrollingSinkAdapter

    builder = make_reduction_service_builder(
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
        pretty_name='Data Reduction', make_builder=make_reduction_service_builder
    )
    runner.run()


if __name__ == "__main__":
    main()
