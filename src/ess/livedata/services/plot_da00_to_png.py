# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import logging
from typing import NoReturn

from ess.livedata import Service
from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.core import IdentityProcessor
from ess.livedata.kafka import consumer as kafka_consumer
from ess.livedata.kafka.message_adapter import (
    AdaptingMessageSource,
    ChainedAdapter,
    Da00ToScippAdapter,
    KafkaToDa00Adapter,
)
from ess.livedata.kafka.source import KafkaMessageSource
from ess.livedata.sinks import PlotToPngSink


def run_service(*, instrument: str, log_level: int = logging.INFO) -> NoReturn:
    consumer_config = load_config(namespace=config_names.reduced_data_consumer, env='')
    kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)
    config = load_config(namespace=config_names.visualization, env='')
    with kafka_consumer.make_consumer_from_config(
        topics=config['topics'],
        config={**consumer_config, **kafka_downstream_config},
        instrument=instrument,
        group='visualization',
    ) as consumer:
        processor = IdentityProcessor(
            source=AdaptingMessageSource(
                source=KafkaMessageSource(consumer=consumer),
                adapter=ChainedAdapter(
                    first=KafkaToDa00Adapter(), second=Da00ToScippAdapter()
                ),
            ),
            sink=PlotToPngSink(),
        )
        service = Service(
            processor=processor,
            name=f'{instrument}_plot_da00_to_png',
            log_level=log_level,
        )
        service.start()


def main() -> NoReturn:
    parser = Service.setup_arg_parser('Plot da00 data arrays to PNG')
    run_service(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
