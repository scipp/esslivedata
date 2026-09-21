# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Adapter-boundary robustness against hostile wire payloads.

Two invariants, driven by the corpus in ``tests/helpers/hostile_wire``:

1. **Containment**: a payload that cannot be adapted is dropped by
   ``AdaptingMessageSource`` without the exception escaping and without
   affecting subsequent messages.
2. **Verbatim timestamps**: a data-derived timestamp crosses the boundary
   unmodified, however insane it looks. It is the producer's claim about when
   the data was taken, and the lag reporter judges the producer by subtracting
   it from the broker's create time; clamping or dropping it here would erase
   the only evidence that a device's clock is wrong (#1133). Consumers that
   cannot use an insane time defend themselves instead: the batchers bound
   window placement, and a job re-latches a start time its own data
   contradicts.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ess.livedata.core.message import StreamKind
from ess.livedata.core.rate_aware_batcher import RateAwareMessageBatcher
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.kafka.message_adapter import (
    AdaptingMessageSource,
    FakeKafkaMessage,
    KafkaAdapter,
    KafkaMessage,
    KafkaToAd00Adapter,
    KafkaToDa00Adapter,
    KafkaToEv44Adapter,
    KafkaToF144Adapter,
    KafkaToMonitorEventsAdapter,
)
from ess.livedata.kafka.stream_mapping import InputStreamKey
from tests.helpers import hostile_wire

TOPIC = 'dummy_beam_monitor'
SOURCE = 'monitor1'
GOOD_TIME_NS = hostile_wire.REALISTIC_EPOCH_NS
SECOND_NS = 1_000_000_000


class ListSource:
    """Message source yielding a fixed list of raw Kafka messages once."""

    def __init__(self, messages: Sequence[KafkaMessage]) -> None:
        self._messages = list(messages)

    def get_messages(self) -> Sequence[KafkaMessage]:
        messages, self._messages = self._messages, []
        return messages


def _monitor_adapter() -> KafkaToMonitorEventsAdapter:
    lut = {InputStreamKey(topic=TOPIC, source_name=SOURCE): SOURCE}
    return KafkaToMonitorEventsAdapter(lut)


def _kafka_message(payload: bytes, timestamp_ms: int = 1234) -> FakeKafkaMessage:
    return FakeKafkaMessage(
        value=payload, topic=TOPIC, timestamp=timestamp_ms, timestamp_type=1
    )


@pytest.mark.parametrize('case', sorted(hostile_wire.malformed_corpus(SOURCE)))
def test_malformed_payload_is_contained_and_does_not_affect_next_message(
    case: str,
) -> None:
    payloads = hostile_wire.malformed_corpus(SOURCE)
    good = hostile_wire.ev44_events(SOURCE, reference_time_ns=GOOD_TIME_NS)
    source = AdaptingMessageSource(
        source=ListSource([_kafka_message(payloads[case]), _kafka_message(good)]),
        adapter=_monitor_adapter(),
    )
    adapted = source.get_messages()
    assert len(adapted) == 1
    assert adapted[0].timestamp == Timestamp.from_ns(GOOD_TIME_NS)


def test_ev44_mismatched_event_vectors_accepted_on_plain_monitor_path() -> None:
    """Pins current behavior: for non-pixellated monitors ``pixel_id`` is
    ignored, so a payload with disagreeing vector lengths adapts cleanly and
    all time-of-arrival entries survive. (The detector path, where the
    mismatch matters, is tracked in #1054.)
    """
    payload = hostile_wire.ev44_mismatched_event_vectors(
        SOURCE, reference_time_ns=GOOD_TIME_NS
    )
    adapted = _monitor_adapter().adapt(_kafka_message(payload))
    assert adapted.timestamp == Timestamp.from_ns(GOOD_TIME_NS)
    assert len(adapted.value.time_of_arrival) == 10


@pytest.mark.xfail(
    strict=True,
    reason='#1038 finding 2: absent event vectors raise deep in the adapter '
    'and the message is dropped instead of using the Kafka-timestamp fallback',
)
def test_ev44_without_event_vectors_falls_back_to_kafka_timestamp() -> None:
    payload = hostile_wire.ev44_without_event_vectors(SOURCE)
    message = _kafka_message(payload, timestamp_ms=5678)
    adapted = _monitor_adapter().adapt(message)
    assert adapted.timestamp == Timestamp.from_ms(5678)


def _far_future_cases() -> list[tuple[str, KafkaAdapter, bytes]]:
    """One (adapter, payload) pair per data-derived timestamp entry point."""
    far = hostile_wire.FAR_FUTURE_NS
    return [
        (
            'ev44',
            KafkaToEv44Adapter(stream_kind=StreamKind.DETECTOR_EVENTS),
            hostile_wire.ev44_events(SOURCE, reference_time_ns=far),
        ),
        (
            'ev44_monitor',
            _monitor_adapter(),
            hostile_wire.ev44_events(SOURCE, reference_time_ns=far),
        ),
        (
            'da00_reference_time',
            KafkaToDa00Adapter(stream_kind=StreamKind.MONITOR_COUNTS),
            hostile_wire.da00_array(
                SOURCE, timestamp_ns=GOOD_TIME_NS, reference_time_ns=far
            ),
        ),
        (
            'da00_timestamp_ns',
            KafkaToDa00Adapter(stream_kind=StreamKind.MONITOR_COUNTS),
            hostile_wire.da00_array(SOURCE, timestamp_ns=far),
        ),
        (
            'ad00',
            KafkaToAd00Adapter(stream_kind=StreamKind.AREA_DETECTOR),
            hostile_wire.ad00_frame(SOURCE, timestamp_ns=far),
        ),
    ]


@pytest.mark.parametrize(
    ('adapter', 'payload'),
    [pytest.param(a, p, id=name) for name, a, p in _far_future_cases()],
)
def test_far_future_data_timestamp_crosses_adapter_boundary_verbatim(
    adapter: KafkaAdapter, payload: bytes
) -> None:
    """A timestamp no producer can legitimately claim is still passed on as-is.

    The lag reporter measures a producer by the distance between this value
    and the broker's create time, so bounding it here would report a device
    with a badly wrong clock as perfectly punctual (#1133).
    """
    source = AdaptingMessageSource(
        source=ListSource([_kafka_message(payload)]), adapter=adapter
    )
    assert [m.timestamp.to_ns() for m in source.get_messages()] == [
        hostile_wire.FAR_FUTURE_NS
    ]


def test_da00_non_int64_reference_time_falls_back_to_timestamp_ns() -> None:
    """The existing dtype guard: int32 reference_time cannot hold nanosecond
    epochs, so the adapter must ignore it in favor of the top-level timestamp.
    """
    import numpy as np

    payload = hostile_wire.da00_array(
        SOURCE,
        timestamp_ns=GOOD_TIME_NS,
        reference_time_ns=12345,
        reference_time_dtype=np.int32,
    )
    adapter = KafkaToDa00Adapter(stream_kind=StreamKind.MONITOR_COUNTS)
    adapted = adapter.adapt(_kafka_message(payload))
    assert adapted.timestamp == Timestamp.from_ns(GOOD_TIME_NS)


def test_static_pv_heartbeats_do_not_drag_batch_window_into_the_past() -> None:
    """An f144 heartbeat repeats the EPICS time of the PV's last change, so
    while the detector is silent the only traffic carries payload times weeks
    old. With that as the envelope, the batcher's stall backstop re-placed
    the window there and results went back in time (#1313). The envelope is
    the Kafka time, so heartbeats read as the fresh traffic they are.
    """
    clock = {'now': 0.0}
    batcher = RateAwareMessageBatcher(batch_length_s=1.0, clock=lambda: clock['now'])
    detector = KafkaToEv44Adapter(stream_kind=StreamKind.DETECTOR_EVENTS)
    log = KafkaToF144Adapter()
    start_ns = GOOD_TIME_NS
    stale_ns = start_ns - 48 * 24 * 3600 * SECOND_NS
    pulse_ns = SECOND_NS // 14
    end_times: list[int] = []

    def run(seconds: int, detector_on: bool) -> None:
        """Poll at 10 Hz; detector pulses at 14 Hz, heartbeat every 10 s."""
        for _ in range(seconds * 10):
            clock['now'] += 0.1
            data_ns = start_ns + int(clock['now'] * SECOND_NS)
            kafka_ms = data_ns // 1_000_000
            messages = []
            if detector_on:
                pulse = hostile_wire.ev44_events(
                    SOURCE, reference_time_ns=data_ns - data_ns % pulse_ns
                )
                messages.append(detector.adapt(_kafka_message(pulse, kafka_ms)))
            if round(clock['now'] * 10) % 100 == 0:
                heartbeat = hostile_wire.f144_log('carriage', timestamp_ns=stale_ns)
                messages.append(log.adapt(_kafka_message(heartbeat, kafka_ms)))
            if (batch := batcher.batch(messages)) is not None:
                end_times.append(batch.end_time.to_ns())

    run(20, detector_on=True)
    run(60, detector_on=False)
    run(20, detector_on=True)

    assert end_times == sorted(end_times)
    assert end_times[-1] > start_ns + 90 * SECOND_NS
