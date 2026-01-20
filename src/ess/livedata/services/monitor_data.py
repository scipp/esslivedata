# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import logging
from typing import NoReturn

from ess.livedata.config import instrument_registry
from ess.livedata.config.environment import is_production
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.handlers import monitor_data_handler
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.logging_config import configure_logging
from ess.livedata.service_factory import DataServiceBuilder, DataServiceRunner


def make_monitor_service_builder(
    *, instrument: str, dev: bool = True, log_level: int = logging.INFO
) -> DataServiceBuilder:
    stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
    adapter = (
        RoutingAdapterBuilder(stream_mapping=stream_mapping)
        .with_beam_monitor_route()
        .with_livedata_commands_route()
        .build()
    )
    instrument_obj = instrument_registry[instrument]
    instrument_obj.load_factories()
    service_name = 'monitor_data'
    preprocessor_factory = monitor_data_handler.MonitorHandlerFactory(
        instrument=instrument_obj
    )
    return DataServiceBuilder(
        instrument=instrument,
        name=service_name,
        log_level=log_level,
        adapter=adapter,
        preprocessor_factory=preprocessor_factory,
    )


def main() -> NoReturn:
    configure_logging(production=is_production())
    runner = DataServiceRunner(
        pretty_name='Monitor Data', make_builder=make_monitor_service_builder
    )
    runner.run()


if __name__ == "__main__":
    main()
