# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Fake ``confluent_kafka.Consumer`` for :class:`KafkaDesiredStateReader`.

A real consumer needs a broker. :func:`make_reader` builds the real reader on a
:class:`FakeDesiredStateConsumer` that serves scripted ``consume`` batches of
:class:`FakeRecord`. Records are encoded with the real codec.
"""

from __future__ import annotations

import time
from typing import Any

from confluent_kafka import KafkaError

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.desired_state import DesiredWorkflow, encode_desired_record
from ess.livedata.kafka.desired_state import KafkaDesiredStateReader


class FakeRecord:
    """A consumed Kafka record: key and value, or an error event."""

    def __init__(
        self,
        key: bytes | None = None,
        value: bytes | None = None,
        *,
        offset: int = 0,
        error: KafkaError | None = None,
    ) -> None:
        self._key = key
        self._value = value
        self._offset = offset
        self._error = error

    def key(self) -> bytes | None:
        return self._key

    def value(self) -> bytes | None:
        return self._value

    def offset(self) -> int:
        return self._offset

    def error(self) -> KafkaError | None:
        return self._error


def record(workflow_id: WorkflowId, value: DesiredWorkflow | None) -> FakeRecord:
    """An encoded desired-state record; ``None`` is a tombstone."""
    return FakeRecord(*encode_desired_record(workflow_id, value))


def eof() -> FakeRecord:
    """The partition EOF event."""
    return FakeRecord(error=KafkaError(KafkaError._PARTITION_EOF))


class _Partition:
    pass


class _TopicMetadata:
    def __init__(self, partitions: int) -> None:
        self.partitions = {p: _Partition() for p in range(partitions)}


class _ClusterMetadata:
    def __init__(self, topics: dict[str, int]) -> None:
        self.topics = {name: _TopicMetadata(n) for name, n in topics.items()}


class FakeDesiredStateConsumer:
    """Serves scripted ``consume`` batches; an exhausted script means no data.

    Parameters
    ----------
    config:
        The configuration the reader created the consumer with.
    topics:
        Partition count per existing topic, reported by ``list_topics``.
    """

    def __init__(self, config: dict[str, Any], topics: dict[str, int]) -> None:
        self.config = config
        self.batches: list[list[FakeRecord]] = []
        self.assignments: list[list] = []
        self.closed = False
        self._topics = topics

    def list_topics(self, timeout: float | None = None) -> _ClusterMetadata:
        return _ClusterMetadata(self._topics)

    def assign(self, partitions: list) -> None:
        self.assignments.append(partitions)

    def consume(self, num_messages: int, timeout: float) -> list[FakeRecord]:
        if self.batches:
            return self.batches.pop(0)
        time.sleep(timeout)
        return []

    def close(self) -> None:
        self.closed = True


def make_reader(
    topic: str, topics: dict[str, int] | None = None
) -> tuple[KafkaDesiredStateReader, FakeDesiredStateConsumer]:
    """Build a reader of ``topic`` on a fake consumer.

    Parameters
    ----------
    topic:
        The topic the reader reads.
    topics:
        Partition count per existing topic. Defaults to ``topic`` with one
        partition.
    """
    consumers: list[FakeDesiredStateConsumer] = []

    def factory(config: dict[str, Any]) -> FakeDesiredStateConsumer:
        consumers.append(
            FakeDesiredStateConsumer(config, {topic: 1} if topics is None else topics)
        )
        return consumers[0]

    reader = KafkaDesiredStateReader(
        kafka_config={'bootstrap.servers': 'test'},
        topic=topic,
        consumer_factory=factory,
    )
    return reader, consumers[0]
