# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for manual partition assignment in :mod:`ess.livedata.kafka.consumer`.

A running broker is unavoidable for a real ``confluent_kafka.Consumer``, so a
hand-rolled fake records ``assign`` calls and reports topic metadata.
"""

from __future__ import annotations

import pytest

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
    ) -> None:
        self._topic_partitions = topic_partitions
        self._high_watermarks = high_watermarks or {}
        self.assignments: list[list] = []

    def list_topics(self, topic: str, timeout: float | None = None) -> _ClusterMetadata:
        ids = self._topic_partitions.get(topic, [])
        return _ClusterMetadata({topic: _TopicMetadata(ids)})

    def get_watermark_offsets(
        self, partition, timeout: float | None = None
    ) -> tuple[int, int]:
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
