# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service that computes a TOF wavelength lookup table from chopper setpoints.

v0 only supports chopperless operation: a ``ChopperSynthesizer`` emits a
single vacuous ``setpoints_reached`` tick at startup, which triggers the
lookup-table workflow once.
"""

import logging
from typing import NoReturn

from ess.livedata.config import instrument_registry
from ess.livedata.config.route_derivation import scope_stream_mapping
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.core.message_batcher import NaiveMessageBatcher
from ess.livedata.handlers.lookup_table_handler import LookupTableHandlerFactory
from ess.livedata.kafka.chopper_synthesizer import ChopperSynthesizer
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.stream_counter import StreamCounter
from ess.livedata.service_factory import DataServiceBuilder, DataServiceRunner


def make_tof_table_service_builder(
    *,
    instrument: str,
    dev: bool = True,
    log_level: int = logging.INFO,
) -> DataServiceBuilder:
    stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
    instrument_config = instrument_registry[instrument]
    instrument_config.load_factories()

    scoped = scope_stream_mapping(instrument_config, stream_mapping, 'tof_table')

    stream_counter = StreamCounter()
    adapter = (
        RoutingAdapterBuilder(stream_mapping=scoped, stream_counter=stream_counter)
        .with_routes_from_mapping()
        .with_livedata_commands_route()
        .with_run_control_route()
        .build()
    )
    preprocessor_factory = LookupTableHandlerFactory(instrument=instrument_config)
    # NaiveMessageBatcher: this service has no continuous data stream — the
    # primary signal is a one-shot synthesized chopper-cascade tick (and, in
    # v1, intermittent setpoint events). The default SimpleMessageBatcher
    # would buffer the lone tick indefinitely waiting for a "next batch"
    # message that never arrives. Same reason the timeseries service uses
    # NaiveMessageBatcher.
    return DataServiceBuilder(
        instrument=instrument,
        name='tof_table',
        log_level=log_level,
        adapter=adapter,
        preprocessor_factory=preprocessor_factory,
        stream_counter=stream_counter,
        message_batcher=NaiveMessageBatcher(),
        outer_source_wrapper=ChopperSynthesizer,
    )


def main() -> NoReturn:
    runner = DataServiceRunner(
        pretty_name='TOF Lookup Table',
        make_builder=make_tof_table_service_builder,
    )
    runner.run()


if __name__ == "__main__":
    main()
