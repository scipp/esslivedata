# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Adapter-boundary robustness against hostile wire payloads.

Two invariants, driven by the corpus in ``tests/helpers/hostile_wire``:

1. **Containment** (holds today, guarded here against regression): a payload
   that cannot be adapted is dropped by ``AdaptingMessageSource`` without the
   exception escaping and without affecting subsequent messages.
2. **Timestamp sanity** (does not hold today — strict xfail): data-derived
   timestamps should be bounded against the wall clock before crossing the
   adapter boundary, because the batcher uses them as its only clock and a
   single far-future value wedges the whole service (#1038 finding 1). These
   tests are the acceptance criterion for the boundary-validation layer
   proposed in #1047: when it lands, the xfails flip to XPASS (strict, so
   pytest will insist the markers are removed). Any resolution — dropping,
   clamping, or falling back to the Kafka broker timestamp — satisfies the
   assertion as written.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import pytest

from ess.livedata.core.message import StreamKind
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

# Data-derived timestamps further than this ahead of the wall clock have no
# legitimate producer; they must not cross the adapter boundary unmodified.
FUTURE_TOLERANCE_NS = 24 * 3600 * 1_000_000_000


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
            'f144',
            KafkaToF144Adapter(),
            hostile_wire.f144_log(SOURCE, timestamp_ns=far),
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
@pytest.mark.xfail(
    strict=True,
    reason='#1038 finding 1 / #1047: data-derived timestamps cross the '
    'adapter boundary unvalidated; a single far-future value wedges the '
    'batcher service-wide',
)
def test_far_future_data_timestamp_does_not_cross_adapter_boundary(
    adapter: KafkaAdapter, payload: bytes
) -> None:
    source = AdaptingMessageSource(
        source=ListSource([_kafka_message(payload)]), adapter=adapter
    )
    bound = time.time_ns() + FUTURE_TOLERANCE_NS
    for message in source.get_messages():
        assert message.timestamp.to_ns() <= bound


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
