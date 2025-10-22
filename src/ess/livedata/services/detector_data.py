# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service that processes detector event data into 2-D data for plotting."""

import logging
from typing import NoReturn

from ess.livedata.config import instrument_registry
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.handlers.detector_data_handler import DetectorHandlerFactory
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.service_factory import DataServiceBuilder, DataServiceRunner


def make_detector_service_builder(
    *, instrument: str, dev: bool = True, log_level: int = logging.INFO
) -> DataServiceBuilder:
    stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
    adapter = (
        RoutingAdapterBuilder(stream_mapping=stream_mapping)
        .with_detector_route()
        .with_livedata_config_route()
        .build()
    )
    instrument_obj = instrument_registry[instrument]
    instrument_obj.load_factories()
    service_name = 'detector_data'
    preprocessor_factory = DetectorHandlerFactory(instrument=instrument_obj)
    return DataServiceBuilder(
        instrument=instrument,
        name=service_name,
        log_level=log_level,
        adapter=adapter,
        preprocessor_factory=preprocessor_factory,
    )


def main() -> NoReturn:
    runner = DataServiceRunner(
        pretty_name='Detector Data', make_builder=make_detector_service_builder
    )
    runner.run()


if __name__ == "__main__":
    main()
