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
from ess.livedata.kafka.sink import KafkaSink, serialize_dataarray_to_f144
from ess.livedata.logging_config import configure_logging


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
        self._current_index = dict.fromkeys(self._ramp_patterns, 0)
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
    from contextlib import ExitStack

    kafka_config = load_config(namespace=config_names.kafka_upstream)
    serializer = serialize_dataarray_to_f144

    resources = ExitStack()
    with resources:
        sink = resources.enter_context(
            KafkaSink(
                instrument=instrument, kafka_config=kafka_config, serializer=serializer
            )
        )
        processor = IdentityProcessor(
            source=FakeLogdataSource(instrument=instrument),
            sink=sink,
        )
        service = Service(
            processor=processor,
            name=f'{instrument}_fake_f144_producer',
            log_level=log_level,
            resources=resources.pop_all(),
        )
        service.start()


def main() -> NoReturn:
    parser = Service.setup_arg_parser(
        'Fake that publishes f144 logdata', dev_flag=False
    )
    args = vars(parser.parse_args())

    # Configure logging with parsed arguments
    log_level = getattr(logging, args.pop('log_level'))
    log_json_file = args.pop('log_json_file')
    no_stdout_log = args.pop('no_stdout_log')
    configure_logging(
        level=log_level,
        json_file=log_json_file,
        disable_stdout=no_stdout_log,
    )

    run_service(log_level=log_level, **args)


if __name__ == "__main__":
    main()
