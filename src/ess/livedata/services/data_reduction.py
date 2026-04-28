# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service that runs a data reduction workflow."""

import logging
from typing import NoReturn

from ess.livedata.config import instrument_registry
from ess.livedata.config.route_derivation import scope_stream_mapping
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.handlers.data_reduction_handler import ReductionHandlerFactory
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.stream_counter import StreamCounter
from ess.livedata.service_factory import DataServiceBuilder, DataServiceRunner


def make_reduction_service_builder(
    *,
    instrument: str,
    dev: bool = True,
    log_level: int = logging.INFO,
    group_by_pixel: bool = True,
) -> DataServiceBuilder:
    stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
    instrument_config = instrument_registry[instrument]
    instrument_config.load_factories()

    scoped = scope_stream_mapping(instrument_config, stream_mapping, 'data_reduction')

    stream_counter = StreamCounter()
    adapter = (
        RoutingAdapterBuilder(stream_mapping=scoped, stream_counter=stream_counter)
        .with_routes_from_mapping()
        .with_livedata_commands_route()
        .with_run_control_route()
        .build()
    )
    service_name = 'data_reduction'
    preprocessor_factory = ReductionHandlerFactory(
        instrument=instrument_config, group_by_pixel=group_by_pixel
    )
    return DataServiceBuilder(
        instrument=instrument,
        name=service_name,
        log_level=log_level,
        adapter=adapter,
        preprocessor_factory=preprocessor_factory,
        stream_counter=stream_counter,
        stream_mapping=scoped,
    )


def main() -> NoReturn:
    runner = DataServiceRunner(
        pretty_name='Data Reduction', make_builder=make_reduction_service_builder
    )
    runner.parser.add_argument(
        '--no-group-by-pixel',
        dest='group_by_pixel',
        action='store_false',
        default=True,
        help='Disable pixel grouping in the preprocessor',
    )
    runner.run()


if __name__ == "__main__":
    main()
