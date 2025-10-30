# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import logging
from typing import NoReturn

import numpy as np
import scipp as sc

from ess.livedata import Message, MessageSource, Service, StreamId, StreamKind
from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.core import IdentityProcessor
from ess.livedata.http_transport.service import HTTPMultiEndpointSink


def _make_ramp(size: int) -> sc.DataArray:
    """Create a ramp pattern that can be used for continuous data generation.

    Returns a DataArray with values that ramp up and down between 0 and 90 degrees.
    """
    # Create values that go from 0 to 90 and back to 0
    values = np.concatenate(
        [
            np.linspace(0.0, 90.0, size // 2 + 1),
            np.linspace(90.0, 0.0, size // 2 + 1)[1:],
        ]
    )

    # Time offsets, one second between each point
    time_offsets = sc.linspace(
        'time', 0.0, float(len(values) - 1), num=len(values), unit='s'
    )

    return sc.DataArray(
        sc.array(dims=['time'], values=values, unit='deg'),
        coords={'time': time_offsets.to(unit='ns', dtype='int64')},
    )


class FakeLogdataSource(MessageSource[sc.DataArray]):
    """Fake message source that generates continuous monitor events in a loop."""

    def __init__(self, *, instrument: str):
        # Create the base ramp patterns
        self._ramp_patterns = {'detector_rotation': _make_ramp(size=100)}
        # Track the current time and cycle count for each log data
        self._current_time = {name: self._time_ns() for name in self._ramp_patterns}
        # Track the last index we produced for each log
        self._current_index = {name: 0 for name in self._ramp_patterns}
        # How often to produce new data points (in seconds)
        self._interval_ns = int(1e9)  # 1 second in nanoseconds
        self._last_produce_time = self._time_ns()

    def _time_ns(self) -> sc.Variable:
        """Return the current time in nanoseconds."""
        return sc.datetime('now', unit='ns') - sc.epoch(unit='ns')

    def get_messages(self) -> list[Message[sc.DataArray]]:
        messages = []
        current_time = self._time_ns()

        # Only produce new messages if enough time has passed
        elapsed_ns = current_time.value - self._last_produce_time.value
        if elapsed_ns < self._interval_ns:
            return messages

        self._last_produce_time = current_time

        for name, pattern in self._ramp_patterns.items():
            # Get the next point in the pattern
            idx = self._current_index[name]
            pattern_size = pattern.sizes['time']

            # Get position in the pattern cycle
            cycle_idx = idx % pattern_size

            # Calculate how many complete cycles we've done
            cycles = idx // pattern_size

            # Calculate the new time, ensuring it always increases
            # Base time + cycle duration + current offset within the pattern
            cycle_duration_ns = pattern.coords['time'][-1].value
            new_time = (
                self._current_time[name].value
                + cycle_duration_ns * cycles
                + pattern.coords['time'][cycle_idx].value
            )

            # Get the scalar value from the pattern
            value = pattern.values[cycle_idx]

            # Create the data point with the updated timestamp - using scalar values
            data_point = sc.DataArray(
                data=sc.scalar(value, unit=pattern.unit),
                coords={'time': sc.scalar(new_time, unit='ns')},
            )

            # Create and add the message
            messages.append(self._make_message(name=name, data=data_point))

            # Move to the next index
            self._current_index[name] = idx + 1

        return messages

    def _make_message(self, name: str, data: sc.DataArray) -> Message[sc.DataArray]:
        """Create a message with the given data and timestamp."""
        return Message(
            timestamp=self._time_ns().value,
            stream=StreamId(kind=StreamKind.LOG, name=name),
            value=data,
        )


def run_service(*, instrument: str, log_level: int = logging.INFO) -> NoReturn:
    """
    Run fake logdata service using YAML-based transport configuration.

    Parameters
    ----------
    instrument:
        Instrument name.
    log_level:
        Logging level.
    """
    from ..config.transport_config import load_transport_config
    from ..transport.factory import create_sink_from_config

    source = FakeLogdataSource(instrument=instrument)

    transport_config = load_transport_config(instrument)

    output_stream_kind = StreamKind.LOG

    configured_kinds = {s.kind for s in transport_config.streams}
    if output_stream_kind not in configured_kinds:
        raise ValueError(
            f"Stream kind {output_stream_kind.value} not found in transport config. "
            f"Available kinds: {[k.value for k in configured_kinds]}"
        )

    kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)

    sink = create_sink_from_config(
        instrument=instrument,
        transport_config=transport_config,
        output_stream_kinds=[output_stream_kind],
        kafka_config=kafka_downstream_config,
    )

    http_sink_instance = None
    if hasattr(sink, 'routes'):
        for route_sink in sink.routes.values():
            if isinstance(route_sink, HTTPMultiEndpointSink):
                http_sink_instance = route_sink
                break
    elif isinstance(sink, HTTPMultiEndpointSink):
        http_sink_instance = sink

    processor = IdentityProcessor(source=source, sink=sink)

    if http_sink_instance is not None:
        http_sink_instance.start()

    service = Service(
        processor=processor,
        name=f'{instrument}_fake_f144_producer',
        log_level=log_level,
    )
    try:
        service.start()
    finally:
        if http_sink_instance is not None:
            http_sink_instance.stop()


def main() -> NoReturn:
    parser = Service.setup_arg_parser(
        'Fake that publishes f144 logdata', dev_flag=False
    )
    run_service(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
