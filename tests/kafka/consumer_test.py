# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for manual partition assignment in :mod:`ess.livedata.kafka.consumer`.

A running broker is unavoidable for a real ``confluent_kafka.Consumer``, so a
hand-rolled fake records ``assign`` calls and reports topic metadata.
"""

from __future__ import annotations

import pytest
from confluent_kafka import KafkaError, KafkaException

from ess.livedata.kafka import consumer as consumer_module
from ess.livedata.kafka.consumer import assign_all_partitions


class _Partition:
    pass


class _TopicMetadata:
    def __init__(self, partition_ids: list[int]) -> None:
        self.partitions = {p: _Partition() for p in partition_ids}


class _ClusterMetadata:
    def __init__(self, topics: dict[str, _TopicMetadata]) -> None:
        self.topics = topics


class _FakeConsumer:
    """Records ``assign`` calls; serves per-topic partition metadata."""

    def __init__(
        self,
        topic_partitions: dict[str, list[int]],
        high_watermarks: dict[tuple[str, int], int] | None = None,
        watermark_errors: list[int] | None = None,
    ) -> None:
        self._topic_partitions = topic_partitions
        self._high_watermarks = high_watermarks or {}
        self._watermark_errors = list(watermark_errors or [])
        self.watermark_calls = 0
        self.assignments: list[list] = []

    def list_topics(self, topic: str, timeout: float | None = None) -> _ClusterMetadata:
        ids = self._topic_partitions.get(topic, [])
        return _ClusterMetadata({topic: _TopicMetadata(ids)})

    def get_watermark_offsets(
        self, partition, timeout: float | None = None
    ) -> tuple[int, int]:
        self.watermark_calls += 1
        if self._watermark_errors:
            raise KafkaException(KafkaError(self._watermark_errors.pop(0)))
        high = self._high_watermarks.get((partition.topic, partition.partition), 0)
        return 0, high

    def assign(self, partitions: list) -> None:
        self.assignments.append(partitions)


def test_assigns_all_partitions_of_all_topics_in_one_call() -> None:
    consumer = _FakeConsumer({'a': [0, 1], 'b': [0]})

    assign_all_partitions(consumer, ['a', 'b'])

    # Single assign call: a per-topic loop would clobber earlier topics.
    assert len(consumer.assignments) == 1
    assigned = {(tp.topic, tp.partition) for tp in consumer.assignments[0]}
    assert assigned == {('a', 0), ('a', 1), ('b', 0)}


def test_assignment_pins_offsets_to_high_watermark() -> None:
    # Relying on auto.offset.reset would resolve "latest" only at first
    # fetch, silently skipping messages produced between assign() and that
    # fetch. The assignment must carry the high watermark explicitly.
    consumer = _FakeConsumer(
        {'a': [0, 1], 'b': [0]},
        high_watermarks={('a', 0): 42, ('a', 1): 7, ('b', 0): 0},
    )

    assign_all_partitions(consumer, ['a', 'b'])

    offsets = {(tp.topic, tp.partition): tp.offset for tp in consumer.assignments[0]}
    assert offsets == {('a', 0): 42, ('a', 1): 7, ('b', 0): 0}


def test_raises_when_topic_has_no_partitions() -> None:
    consumer = _FakeConsumer({'a': [0], 'b': []})

    with pytest.raises(ValueError, match="no partitions"):
        assign_all_partitions(consumer, ['a', 'b'])
    assert consumer.assignments == []


def test_waits_out_pending_partition_leadership(monkeypatch) -> None:
    # A topic created moments ago, or a broker that just restarted, answers
    # ListOffsets with NOT_LEADER_FOR_PARTITION until leadership settles.
    # Failing closed there would take down every service starting in that window.
    monkeypatch.setattr(consumer_module, '_LEADER_POLL_INTERVAL', 0.0)
    consumer = _FakeConsumer(
        {'a': [0]},
        high_watermarks={('a', 0): 42},
        watermark_errors=[
            KafkaError.LEADER_NOT_AVAILABLE,
            KafkaError.NOT_LEADER_FOR_PARTITION,
        ],
    )

    assign_all_partitions(consumer, ['a'])

    assert consumer.watermark_calls == 3
    assert [(tp.topic, tp.partition, tp.offset) for tp in consumer.assignments[0]] == [
        ('a', 0, 42)
    ]


def test_gives_up_when_partition_leadership_never_settles(monkeypatch) -> None:
    monkeypatch.setattr(consumer_module, '_LEADER_POLL_INTERVAL', 0.0)
    monkeypatch.setattr(consumer_module, '_LEADER_TIMEOUT', 0.0)
    consumer = _FakeConsumer(
        {'a': [0]}, watermark_errors=[KafkaError.NOT_LEADER_FOR_PARTITION]
    )

    with pytest.raises(ValueError, match="Failed to fetch watermark"):
        assign_all_partitions(consumer, ['a'])
    assert consumer.assignments == []


def test_does_not_wait_out_errors_unrelated_to_leadership() -> None:
    consumer = _FakeConsumer(
        {'a': [0]}, watermark_errors=[KafkaError.TOPIC_AUTHORIZATION_FAILED]
    )

    with pytest.raises(ValueError, match="Failed to fetch watermark"):
        assign_all_partitions(consumer, ['a'])
    assert consumer.watermark_calls == 1
