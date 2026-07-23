# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import logging
from typing import NoReturn

from ess.livedata.config import instrument_registry
from ess.livedata.config.route_derivation import scope_stream_mapping
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.kafka.device_synthesizer import DeviceSynthesizer
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.stream_counter import StreamCounter
from ess.livedata.preprocessors.data_reduction import ReductionPreprocessorFactory
from ess.livedata.service_factory import DataServiceBuilder, DataServiceRunner


def make_monitor_service_builder(
    *, instrument: str, dev: bool = True, log_level: int = logging.INFO
) -> DataServiceBuilder:
    stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
    instrument_obj = instrument_registry[instrument]
    instrument_obj.load_factories()

    scoped = scope_stream_mapping(instrument_obj, stream_mapping, 'monitor_data')

    stream_counter = StreamCounter(
        out_of_scope=stream_mapping.input_keys - scoped.input_keys
    )
    adapter = (
        RoutingAdapterBuilder(stream_mapping=scoped, stream_counter=stream_counter)
        .with_routes_from_mapping(
            pixellated_sources=instrument_obj.pixellated_monitor_sources
        )
        .with_livedata_commands_route()
        .with_run_control_route()
        .build()
    )
    service_name = 'monitor_data'
    preprocessor_factory = ReductionPreprocessorFactory(instrument=instrument_obj)
    devices = instrument_obj.devices
    return DataServiceBuilder(
        instrument=instrument,
        name=service_name,
        log_level=log_level,
        adapter=adapter,
        preprocessor_factory=preprocessor_factory,
        stream_counter=stream_counter,
        outer_source_wrapper=lambda src: DeviceSynthesizer(src, devices=devices),
    )


def main() -> NoReturn:
    runner = DataServiceRunner(
        pretty_name='Monitor Data', make_builder=make_monitor_service_builder
    )
    runner.run()


if __name__ == "__main__":
    main()
