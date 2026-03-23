# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Benchmarks for ev44/da00 adapter chains and the ToNXevent_data accumulator.

Run with: pytest tests/benchmarks/accumulator_bench.py --benchmark-only
"""

import numpy as np
import pytest
from streaming_data_types import dataarray_da00, eventdata_ev44

from ess.livedata.core.message import StreamKind
from ess.livedata.handlers.to_nxevent_data import (
    DetectorEvents,
    MonitorEvents,
    ToNXevent_data,
)
from ess.livedata.kafka.message_adapter import (
    ChainedAdapter,
    Da00ToScippAdapter,
    Ev44ToDetectorEventsAdapter,
    FakeKafkaMessage,
    KafkaToDa00Adapter,
    KafkaToEv44Adapter,
    KafkaToMonitorEventsAdapter,
)
from ess.livedata.kafka.stream_mapping import InputStreamKey, StreamLUT

TOPIC = "test_topic"
SOURCE = "test_source"


def _make_stream_lut() -> StreamLUT:
    return {InputStreamKey(topic=TOPIC, source_name=SOURCE): SOURCE}


def _make_ev44_bytes(n_events: int) -> bytes:
    rng = np.random.default_rng(42)
    return eventdata_ev44.serialise_ev44(
        source_name=SOURCE,
        message_id=1,
        reference_time=np.array([1_000_000_000], dtype=np.int64),
        reference_time_index=np.array([0], dtype=np.int32),
        time_of_flight=rng.integers(0, 50_000_000, size=n_events, dtype=np.int32),
        pixel_id=rng.integers(0, 100_000, size=n_events, dtype=np.int32),
    )


def _make_da00_bytes(array_length: int, dtype: type = np.float64) -> bytes:
    rng = np.random.default_rng(42)
    signal = rng.random(array_length).astype(dtype)
    tof = np.linspace(0, 50_000, array_length).astype(np.float64)
    ref_time = np.array([1_000_000_000], dtype=np.int64)
    variables = [
        dataarray_da00.Variable(
            name="signal",
            data=signal,
            axes=["tof"],
            shape=(array_length,),
            unit="counts",
            label="signal",
        ),
        dataarray_da00.Variable(
            name="tof",
            data=tof,
            axes=["tof"],
            shape=(array_length,),
            unit="us",
        ),
        dataarray_da00.Variable(
            name="reference_time",
            data=ref_time,
            axes=["time"],
            shape=(1,),
            unit="ns",
        ),
    ]
    return dataarray_da00.serialise_da00(
        source_name=SOURCE, timestamp_ns=1_000_000_000, data=variables
    )


@pytest.fixture
def stream_lut() -> StreamLUT:
    return _make_stream_lut()


# ---------------------------------------------------------------------------
# ev44 adapter chain benchmarks
# ---------------------------------------------------------------------------

EV44_EVENT_COUNTS = [1_000, 100_000, 714_000]


@pytest.mark.parametrize("n_events", EV44_EVENT_COUNTS)
def test_ev44_deserialise(benchmark, n_events: int) -> None:
    payload = _make_ev44_bytes(n_events)
    benchmark(eventdata_ev44.deserialise_ev44, payload)


@pytest.mark.parametrize("n_events", EV44_EVENT_COUNTS)
def test_ev44_to_detector_events(
    benchmark, n_events: int, stream_lut: StreamLUT
) -> None:
    payload = _make_ev44_bytes(n_events)
    msg = FakeKafkaMessage(value=payload, topic=TOPIC)
    adapter = ChainedAdapter(
        KafkaToEv44Adapter(
            stream_lut=stream_lut,
            stream_kind=StreamKind.DETECTOR_EVENTS,
        ),
        Ev44ToDetectorEventsAdapter(),
    )
    benchmark(adapter.adapt, msg)


@pytest.mark.parametrize("n_events", EV44_EVENT_COUNTS)
def test_ev44_to_monitor_events_direct(
    benchmark, n_events: int, stream_lut: StreamLUT
) -> None:
    payload = _make_ev44_bytes(n_events)
    msg = FakeKafkaMessage(value=payload, topic=TOPIC)
    adapter = KafkaToMonitorEventsAdapter(stream_lut=stream_lut)
    benchmark(adapter.adapt, msg)


# ---------------------------------------------------------------------------
# da00 adapter chain benchmarks
# ---------------------------------------------------------------------------

DA00_ARRAY_LENGTHS = [100, 1_000, 10_000]


@pytest.mark.parametrize("array_length", DA00_ARRAY_LENGTHS)
def test_da00_to_scipp(benchmark, array_length: int, stream_lut: StreamLUT) -> None:
    payload = _make_da00_bytes(array_length)
    msg = FakeKafkaMessage(value=payload, topic=TOPIC)
    adapter = ChainedAdapter(
        KafkaToDa00Adapter(
            stream_lut=stream_lut,
            stream_kind=StreamKind.MONITOR_EVENTS,
        ),
        Da00ToScippAdapter(),
    )
    benchmark(adapter.adapt, msg)


# ---------------------------------------------------------------------------
# ToNXevent_data accumulator benchmarks
# ---------------------------------------------------------------------------

ACCUMULATOR_PARAMS = [
    (14, 1_000),
    (14, 100_000),
    (14, 714_000),
]


@pytest.mark.parametrize(
    ("n_msgs", "n_events_per_msg"),
    ACCUMULATOR_PARAMS,
    ids=[f"{m}x{e}" for m, e in ACCUMULATOR_PARAMS],
)
def test_accumulator_get_detector(
    benchmark, n_msgs: int, n_events_per_msg: int
) -> None:
    rng = np.random.default_rng(42)
    chunks = [
        DetectorEvents(
            time_of_arrival=rng.integers(
                0, 50_000_000, size=n_events_per_msg, dtype=np.int32
            ),
            pixel_id=rng.integers(0, 100_000, size=n_events_per_msg, dtype=np.int32),
            unit="ns",
        )
        for _ in range(n_msgs)
    ]

    # Reuse the same accumulator instance across rounds (steady-state).
    acc = ToNXevent_data()

    def run():
        for i, chunk in enumerate(chunks):
            acc.add(i * 1000, chunk)
        result = acc.get()
        acc.release_buffers()
        return result

    benchmark(run)


@pytest.mark.parametrize(
    ("n_msgs", "n_events_per_msg"),
    ACCUMULATOR_PARAMS,
    ids=[f"{m}x{e}" for m, e in ACCUMULATOR_PARAMS],
)
def test_accumulator_get_monitor(benchmark, n_msgs: int, n_events_per_msg: int) -> None:
    rng = np.random.default_rng(42)
    chunks = [
        MonitorEvents(
            time_of_arrival=rng.integers(
                0, 50_000_000, size=n_events_per_msg, dtype=np.int32
            ),
            unit="ns",
        )
        for _ in range(n_msgs)
    ]

    # Reuse the same accumulator instance across rounds (steady-state).
    acc = ToNXevent_data()

    def run():
        for i, chunk in enumerate(chunks):
            acc.add(i * 1000, chunk)
        result = acc.get()
        acc.release_buffers()
        return result

    benchmark(run)
