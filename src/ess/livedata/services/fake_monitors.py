# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: S104  # Binding to 0.0.0.0 is intentional for HTTP services
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import logging
import time
from dataclasses import replace
from typing import Literal, NoReturn

import numpy as np
import scipp as sc
from streaming_data_types import eventdata_ev44

from ess.livedata import Message, MessageSource, Service, StreamId, StreamKind
from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.core import IdentityProcessor
from ess.livedata.http_transport.serialization import DA00MessageSerializer
from ess.livedata.http_transport.service import HTTPMultiEndpointSink
from ess.livedata.kafka.message_adapter import AdaptingMessageSource, MessageAdapter
from ess.livedata.kafka.sink import (
    KafkaSink,
    SerializationError,
    serialize_dataarray_to_da00,
)


class FakeMonitorSource(MessageSource[sc.Variable]):
    """Fake message source that generates random monitor events."""

    def __init__(
        self,
        *,
        interval_ns: int = int(1e9 / 14),
        instrument: str,
        num_monitors: int = 2,
    ):
        self._rng = np.random.default_rng()
        self._interval_ns = interval_ns
        self._num_monitors = max(1, min(10, num_monitors))  # Clamp between 1 and 10
        self._last_message_time = {
            f"monitor{i + 1}": time.time_ns() for i in range(self._num_monitors)
        }

    def _make_normal(self, mean: float, std: float, size: int) -> np.ndarray:
        return self._rng.normal(loc=mean, scale=std, size=size).astype(np.int64)

    def get_messages(self) -> list[Message[sc.Variable]]:
        current_time = time.time_ns()
        messages = []

        # Generate monitor parameters with decreasing counts and increasing means
        for i in range(self._num_monitors):
            name = f"monitor{i + 1}"
            # Start with 10000 counts, decrease by ~30% each monitor
            size = max(100, int(10000 * (0.7**i)))
            # Mean goes from 20ms to 50ms across monitors
            mean_ms = 20 + (30 * i / max(1, self._num_monitors - 1))

            elapsed = current_time - self._last_message_time[name]
            num_intervals = int(elapsed // self._interval_ns)

            for j in range(num_intervals):
                msg_time = self._last_message_time[name] + (j + 1) * self._interval_ns
                messages.append(
                    self._make_message(
                        name=name, size=size, timestamp=msg_time, mean_ms=mean_ms
                    )
                )
            self._last_message_time[name] += num_intervals * self._interval_ns

        return messages

    def _make_message(
        self, name: str, size: int, timestamp: int, mean_ms: float
    ) -> Message[sc.Variable]:
        time_of_flight = self._make_normal(
            mean=int(1e6 * mean_ms), std=10_000_000, size=size
        )
        var = sc.array(dims=['time_of_arrival'], values=time_of_flight, unit='ns')
        return Message(
            timestamp=timestamp,
            stream=StreamId(kind=StreamKind.MONITOR_EVENTS, name=name),
            value=var,
        )


class EventsToHistogramAdapter(
    MessageAdapter[Message[sc.Variable], Message[sc.DataArray]]
):
    def __init__(self, toa: sc.Variable):
        self._toa = toa

    def adapt(self, message: Message[sc.Variable]) -> Message[sc.DataArray]:
        return replace(
            message,
            stream=replace(message.stream, kind=StreamKind.MONITOR_COUNTS),
            value=message.value.hist({self._toa.dim: self._toa}),
        )


def serialize_variable_to_monitor_ev44(msg: Message[sc.Variable]) -> bytes:
    if msg.value.unit != 'ns':
        raise SerializationError(f"Expected unit 'ns', got {msg.value.unit}")
    try:
        ev44 = eventdata_ev44.serialise_ev44(
            source_name=msg.stream.name,
            message_id=0,
            reference_time=msg.timestamp,
            reference_time_index=0,
            time_of_flight=msg.value.values,
            pixel_id=np.ones_like(msg.value.values),
        )
    except (ValueError, TypeError) as e:
        raise SerializationError(f"Failed to serialize message: {e}") from None
    return ev44


def run_service(
    *,
    instrument: str,
    mode: Literal['ev44', 'da00'],
    num_monitors: int = 2,
    log_level: int = logging.INFO,
    sink_type: str = 'kafka',
    http_host: str = '0.0.0.0',
    http_port: int = 8000,
) -> NoReturn:
    if mode == 'ev44':
        adapter = None
        serializer = serialize_variable_to_monitor_ev44
    else:
        adapter = EventsToHistogramAdapter(
            toa=sc.linspace('toa', 0, 71_000_000, num=1001, unit='ns')
        )
        serializer = serialize_dataarray_to_da00

    source = FakeMonitorSource(instrument=instrument, num_monitors=num_monitors)
    if adapter is not None:
        source = AdaptingMessageSource(source=source, adapter=adapter)

    # Create sink based on sink_type
    if sink_type == 'http':
        if mode == 'ev44':
            raise ValueError("HTTP sink only supports da00 mode (not ev44)")
        # Use multi-endpoint sink for monitor data (exposes /beam_monitor endpoint)
        sink = HTTPMultiEndpointSink(
            instrument=instrument,
            stream_serializers={
                StreamKind.MONITOR_COUNTS: DA00MessageSerializer(),
            },
            host=http_host,
            port=http_port,
        )
    else:
        kafka_config = load_config(namespace=config_names.kafka_upstream)
        sink = KafkaSink(
            instrument=instrument, kafka_config=kafka_config, serializer=serializer
        )

    processor = IdentityProcessor(source=source, sink=sink)

    # Start HTTP server if using HTTP sink
    if sink_type == 'http':
        sink.start()

    service = Service(
        processor=processor,
        name=f'{instrument}_fake_{mode}_producer',
        log_level=log_level,
    )
    try:
        service.start()
    finally:
        if sink_type == 'http':
            sink.stop()


def main() -> NoReturn:
    parser = Service.setup_arg_parser(
        'Fake that publishes random da00 or ev44 monitor data', dev_flag=False
    )
    parser.add_argument(
        '--mode',
        choices=['ev44', 'da00'],
        required=True,
        help='Select mode: ev44 or da00',
    )
    parser.add_argument(
        '--num-monitors',
        type=int,
        default=2,
        choices=range(1, 11),
        metavar='1-10',
        help='Number of monitors to simulate (1-10, default: 2)',
    )
    parser.add_argument(
        '--sink-type',
        choices=['kafka', 'http'],
        default='kafka',
        help='Select sink type: kafka or http',
    )
    parser.add_argument(
        '--http-host',
        default='0.0.0.0',
        help='HTTP server host (when using http sink)',
    )
    parser.add_argument(
        '--http-port',
        type=int,
        default=8000,
        help='HTTP server port (when using http sink)',
    )
    run_service(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
