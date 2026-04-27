# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service that processes logdata into timeseries for plotting."""

import logging
from collections.abc import Mapping
from typing import Any, NoReturn

from ess.livedata.config import instrument_registry
from ess.livedata.config.route_derivation import scope_stream_mapping
from ess.livedata.config.streams import get_stream_mapping
from ess.livedata.core.message_batcher import NaiveMessageBatcher
from ess.livedata.handlers.timeseries_handler import LogdataHandlerFactory
from ess.livedata.kafka.routes import RoutingAdapterBuilder
from ess.livedata.kafka.stream_counter import StreamCounter
from ess.livedata.service_factory import DataServiceBuilder, DataServiceRunner


def make_timeseries_service_builder(
    *,
    instrument: str,
    dev: bool = True,
    log_level: int = logging.INFO,
    attribute_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> DataServiceBuilder:
    stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
    instrument_obj = instrument_registry[instrument]
    instrument_obj.load_factories()

    scoped = scope_stream_mapping(instrument_obj, stream_mapping, 'timeseries')

    stream_counter = StreamCounter()
    adapter = (
        RoutingAdapterBuilder(stream_mapping=scoped, stream_counter=stream_counter)
        .with_routes_from_mapping()
        .with_livedata_commands_route()
        .with_run_control_route()
        .build()
    )
    service_name = 'timeseries'
    preprocessor_factory = LogdataHandlerFactory(
        instrument=instrument_obj,
        attribute_registry=attribute_registry,
    )
    # The default batcher processes messages in batches, not emitting messages unless
    # the current batch is considered "complete", by the first message after the batch
    # interval arriving. This works for monitor and detector processing (including for
    # logs that are processed as part of the overall stream of messages on the same
    # service). However, this service processes only logs, where that logic would
    # indefinitely withhold the last log message. We use the NaiveMessageBatcher here,
    # which emits messages as soon as they arrive.
    return DataServiceBuilder(
        instrument=instrument,
        name=service_name,
        log_level=log_level,
        adapter=adapter,
        preprocessor_factory=preprocessor_factory,
        stream_counter=stream_counter,
        message_batcher=NaiveMessageBatcher(),
    )


def main() -> NoReturn:
    runner = DataServiceRunner(
        pretty_name='Logdata to Timeseries',
        make_builder=make_timeseries_service_builder,
    )
    runner.run()


if __name__ == "__main__":
    main()
