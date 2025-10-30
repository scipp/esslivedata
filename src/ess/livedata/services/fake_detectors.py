# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Fake that publishes random detector data"""

import logging
import time
from typing import NoReturn

import numpy as np
import scipp as sc
import scippnexus as snx

from ess.livedata import Message, MessageSource, Service, StreamId, StreamKind
from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.config.instruments import get_config
from ess.livedata.core import IdentityProcessor
from ess.livedata.http_transport.service import HTTPMultiEndpointSink


def events_from_nexus(file_path: str) -> dict[str, sc.DataGroup]:
    """Load events from a NeXus file and return as a DataGroup."""
    with snx.File(file_path, 'r', definitions={}) as f:
        entry = next(iter(f[snx.NXentry].values()))
        instrument = next(iter(entry[snx.NXinstrument].values()))
        detectors = instrument[snx.NXdetector]
        # Find detector groups - assuming detector names are unique
        detector_groups = instrument[snx.NXdetector_group]
        for group in detector_groups.values():
            detectors.update(group[snx.NXdetector])

        event_data = {name: det[snx.NXevent_data] for name, det in detectors.items()}
        return {
            name: next(iter(eg.values()))[...]
            for name, eg in event_data.items()
            if len(eg) == 1
        }


class FakeDetectorSource(MessageSource[sc.Dataset]):
    """Fake message source that generates random detector events."""

    def __init__(
        self,
        *,
        interval_ns: int = int(1e9 / 14),
        instrument: str,
        nexus_file: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._instrument = instrument
        self._config = get_config(instrument).detector_fakes
        self._rng = np.random.default_rng()
        self._tof = sc.linspace('tof', 0, 71_000_000, num=50, unit='ns')
        self._interval_ns = interval_ns
        self._bank_scales: dict[str, float] = {}

        # Load nexus data if file is provided
        self._nexus_data = None if nexus_file is None else events_from_nexus(nexus_file)
        if self._nexus_data is None:
            detector_names = list(self._config.keys())
            self._logger.info("Configured detectors: %s", detector_names)
        else:
            detector_names = list(self._nexus_data.keys())
            bank_sizes = {
                name: data['event_time_offset'].size
                for name, data in self._nexus_data.items()
            }
            self._logger.info("Loaded event data from %s", nexus_file)
            self._logger.info(
                "Loaded detectors:\n%s",
                '\n'.join(
                    f'    {detector}: {size} events'
                    for detector, size in bank_sizes.items()
                ),
            )
            largest_size = max(bank_sizes.values())
            self._bank_scales = {
                name: size / largest_size for name, size in bank_sizes.items()
            }

        self._last_message_time = {
            detector: time.time_ns() for detector in detector_names
        }
        self._offset = {name: 0 for name in detector_names}

    def _make_normal(self, mean: float, std: float, size: int) -> np.ndarray:
        return self._rng.normal(loc=mean, scale=std, size=size).astype(np.int64)

    def _make_ids(self, name: str, size: int) -> np.ndarray:
        low, high = self._config[name]
        return self._rng.integers(low=low, high=high + 1, size=size)

    def get_messages(self) -> list[Message[sc.Dataset]]:
        current_time = time.time_ns()
        messages = []

        for name in self._last_message_time:
            size = (
                1_000
                if self._instrument == 'bifrost'
                else int(10000 * self._bank_scales.get(name, 1.0))
            )
            elapsed = current_time - self._last_message_time[name]
            num_intervals = int(elapsed // self._interval_ns)

            for i in range(num_intervals):
                msg_time = self._last_message_time[name] + (i + 1) * self._interval_ns
                messages.append(
                    self._make_message(name=name, size=size, timestamp=msg_time)
                )
            self._last_message_time[name] += num_intervals * self._interval_ns

        return messages

    def _make_message(
        self, name: str, size: int, timestamp: int
    ) -> Message[sc.Variable]:
        if self._nexus_data is not None and name in self._nexus_data:
            # Use data from nexus file
            s = slice(self._offset[name], self._offset[name] + size)
            time_of_flight = self._nexus_data[name]['event_time_offset'].values[s]
            pixel_id = self._nexus_data[name]['event_id'].values[s]
            self._offset[name] += size
            if self._offset[name] >= self._nexus_data[name]['event_id'].size:
                self._offset[name] = 0
        else:
            # Generate random events
            time_of_flight = self._make_normal(
                mean=30_000_000, std=10_000_000, size=size
            )
            pixel_id = self._make_ids(name=name, size=size)

        ds = sc.Dataset(
            {
                'time_of_arrival': sc.array(
                    dims=['time_of_arrival'], values=time_of_flight, unit='ns'
                ),
                'pixel_id': sc.array(
                    dims=['time_of_arrival'], values=pixel_id, unit=None
                ),
            }
        )

        return Message(
            timestamp=timestamp,
            stream=StreamId(kind=StreamKind.DETECTOR_EVENTS, name=name),
            value=ds,
        )


def run_service(
    *, instrument: str, nexus_file: str | None = None, log_level: int = logging.INFO
) -> NoReturn:
    """
    Run fake detector service using YAML-based transport configuration.

    Parameters
    ----------
    instrument:
        Instrument name.
    nexus_file:
        Path to NeXus file containing event data to replay.
    log_level:
        Logging level.
    """
    from ..config.transport_config import load_transport_config
    from ..transport.factory import create_sink_from_config

    Service.configure_logging(log_level)
    logger = logging.getLogger(f'{instrument}_fake_detector_producer')

    source = FakeDetectorSource(
        instrument=instrument, nexus_file=nexus_file, logger=logger
    )

    transport_config = load_transport_config(instrument)

    output_stream_kind = StreamKind.DETECTOR_EVENTS

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
        name=f'{instrument}_fake_detector_producer',
        log_level=log_level,
    )
    try:
        service.start()
    finally:
        if http_sink_instance is not None:
            http_sink_instance.stop()


def main() -> NoReturn:
    parser = Service.setup_arg_parser(
        'Fake that publishes random detector data', dev_flag=False
    )
    parser.add_argument(
        '--nexus-file',
        type=str,
        help=(
            'Path to NeXus file containing event data to replay. '
            'The event data will be looped over indefinitely.'
        ),
    )
    run_service(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
