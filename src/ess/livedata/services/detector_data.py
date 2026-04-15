# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service that processes detector event data into 2-D data for plotting."""

import logging
from typing import NoReturn

from ess.livedata.config import instrument_registry
from ess.livedata.config.route_derivation import (
    gather_source_names,
    get_source_subset,
    resolve_stream_names,
)
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.handlers.detector_data_handler import DetectorHandlerFactory
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.stream_counter import StreamCounter
from ess.livedata.service_factory import DataServiceBuilder, DataServiceRunner


def make_detector_service_builder(
    *,
    instrument: str,
    dev: bool = True,
    log_level: int = logging.INFO,
    group_by_pixel: bool = True,
    num_shards: int = 1,
    shard: int = 0,
) -> DataServiceBuilder:
    stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
    instrument_obj = instrument_registry[instrument]
    instrument_obj.load_factories()

    source_subset = (
        get_source_subset(instrument_obj.detector_names, num_shards, shard)
        if num_shards > 1
        else None
    )
    needed = gather_source_names(
        instrument_obj, 'detector_data', source_subset=source_subset
    )
    needed = resolve_stream_names(needed, instrument_obj, stream_mapping)
    scoped = stream_mapping.filtered(needed)

    stream_counter = StreamCounter()
    adapter = (
        RoutingAdapterBuilder(stream_mapping=scoped, stream_counter=stream_counter)
        .with_routes_from_mapping()
        .with_livedata_commands_route()
        .with_livedata_roi_route()
        .with_run_control_route()
        .build()
    )
    service_name = 'detector_data'
    preprocessor_factory = DetectorHandlerFactory(
        instrument=instrument_obj, group_by_pixel=group_by_pixel
    )
    return DataServiceBuilder(
        instrument=instrument,
        name=service_name,
        log_level=log_level,
        adapter=adapter,
        preprocessor_factory=preprocessor_factory,
        stream_counter=stream_counter,
    )


def main() -> NoReturn:
    runner = DataServiceRunner(
        pretty_name='Detector Data', make_builder=make_detector_service_builder
    )
    runner.parser.add_argument(
        '--no-group-by-pixel',
        dest='group_by_pixel',
        action='store_false',
        default=True,
        help='Disable pixel grouping in the preprocessor',
    )
    runner.parser.add_argument(
        '--num-shards',
        type=int,
        default=1,
        help='Total number of shards (1 = no sharding)',
    )
    runner.parser.add_argument(
        '--shard',
        type=int,
        default=0,
        help='Zero-based shard index',
    )
    runner.run()


if __name__ == "__main__":
    main()
