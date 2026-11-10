# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for the Kafka reader and writer of the desired-state topic.

A real consumer or producer needs a broker, so hand-rolled fakes stand in for
``confluent_kafka.Consumer`` and ``confluent_kafka.Producer``. Records are
encoded with the real codec.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from confluent_kafka import OFFSET_BEGINNING, KafkaError, KafkaException

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.desired_state import (
    DesiredJob,
    DesiredStateWriteError,
    DesiredWorkflow,
    encode_desired_record,
)
from ess.livedata.kafka.desired_state import (
    KafkaDesiredStateReader,
    KafkaDesiredStateWriter,
)
from tests.helpers.hostile_wire import malformed_desired_records

TOPIC = 'dummy_livedata_desired'
WF_A = WorkflowId(instrument='dummy', name='a', version=1)
WF_B = WorkflowId(instrument='dummy', name='b', version=1)


def desired(workflow_id: WorkflowId) -> DesiredWorkflow:
    return DesiredWorkflow(
        workflow_id=workflow_id,
        job_number=uuid.uuid4(),
        jobs={'det': DesiredJob(params={'x': 1}, aux_source_names={})},
    )


class _Msg:
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


def record(workflow_id: WorkflowId, value: DesiredWorkflow | None) -> _Msg:
    return _Msg(*encode_desired_record(workflow_id, value))


def eof() -> _Msg:
    return _Msg(error=KafkaError(KafkaError._PARTITION_EOF))


class _Partition:
    pass


class _TopicMetadata:
    def __init__(self, partitions: int) -> None:
        self.partitions = {p: _Partition() for p in range(partitions)}


class _ClusterMetadata:
    def __init__(self, topics: dict[str, int]) -> None:
        self.topics = {name: _TopicMetadata(n) for name, n in topics.items()}


class FakeConsumer:
    """Serves scripted ``consume`` batches; an exhausted script means no data."""

    def __init__(self, config: dict[str, Any], topics: dict[str, int]) -> None:
        self.config = config
        self.batches: list[list[_Msg]] = []
        self.assignments: list[list] = []
        self.closed = False
        self._topics = topics

    def list_topics(self, timeout: float | None = None) -> _ClusterMetadata:
        return _ClusterMetadata(self._topics)

    def assign(self, partitions: list) -> None:
        self.assignments.append(partitions)

    def consume(self, num_messages: int, timeout: float) -> list[_Msg]:
        if self.batches:
            return self.batches.pop(0)
        time.sleep(timeout)
        return []

    def close(self) -> None:
        self.closed = True


def make_reader(
    topics: dict[str, int] | None = None,
) -> tuple[KafkaDesiredStateReader, FakeConsumer]:
    consumers: list[FakeConsumer] = []

    def factory(config: dict[str, Any]) -> FakeConsumer:
        consumers.append(FakeConsumer(config, {TOPIC: 1} if topics is None else topics))
        return consumers[0]

    reader = KafkaDesiredStateReader(
        kafka_config={'bootstrap.servers': 'test'},
        topic=TOPIC,
        consumer_factory=factory,
    )
    return reader, consumers[0]


class TestKafkaDesiredStateReader:
    def test_assigns_partition_zero_from_beginning_without_commits(self) -> None:
        _, consumer = make_reader()
        [[partition]] = consumer.assignments
        assert (partition.topic, partition.partition) == (TOPIC, 0)
        assert partition.offset == OFFSET_BEGINNING
        assert consumer.config['enable.auto.commit'] is False
        assert consumer.config['enable.partition.eof'] is True
        assert consumer.config['bootstrap.servers'] == 'test'

    def test_group_id_is_unique_per_reader(self) -> None:
        _, first = make_reader()
        _, second = make_reader()
        assert first.config['group.id'] != second.config['group.id']

    def test_missing_topic_raises_and_closes_consumer(self) -> None:
        consumers: list[FakeConsumer] = []

        def factory(config: dict[str, Any]) -> FakeConsumer:
            consumers.append(FakeConsumer(config, {}))
            return consumers[0]

        with pytest.raises(ValueError, match='not found'):
            KafkaDesiredStateReader(
                kafka_config={}, topic=TOPIC, consumer_factory=factory
            )
        assert consumers[0].closed

    def test_topic_with_several_partitions_raises(self) -> None:
        with pytest.raises(ValueError, match='1 partition'):
            make_reader({TOPIC: 3})

    def test_bootstrap_of_empty_topic_returns_at_eof(self) -> None:
        reader, consumer = make_reader()
        consumer.batches = [[eof()]]
        assert reader.bootstrap(timeout_s=5.0) == []

    def test_bootstrap_returns_records_up_to_eof(self) -> None:
        reader, consumer = make_reader()
        a, b = desired(WF_A), desired(WF_B)
        consumer.batches = [[record(WF_A, a)], [], [record(WF_B, b), eof()]]
        assert reader.bootstrap(timeout_s=5.0) == [(WF_A, a), (WF_B, b)]

    def test_bootstrap_keeps_tombstones(self) -> None:
        reader, consumer = make_reader()
        consumer.batches = [[record(WF_A, desired(WF_A)), record(WF_A, None), eof()]]
        updates = reader.bootstrap(timeout_s=5.0)
        assert updates[-1] == (WF_A, None)

    def test_bootstrap_raises_if_eof_is_not_reached_in_time(self) -> None:
        reader, consumer = make_reader()
        consumer.batches = [[record(WF_A, desired(WF_A))]]
        with pytest.raises(TimeoutError, match=TOPIC):
            reader.bootstrap(timeout_s=0.05)

    def test_bootstrap_waits_through_transient_consumer_error(self) -> None:
        reader, consumer = make_reader()
        a = desired(WF_A)
        transient = _Msg(error=KafkaError(KafkaError._TRANSPORT))
        consumer.batches = [[transient], [record(WF_A, a), eof()]]
        assert reader.bootstrap(timeout_s=5.0) == [(WF_A, a)]

    def test_fatal_consumer_error_raises(self) -> None:
        reader, consumer = make_reader()
        fatal = _Msg(error=KafkaError(KafkaError.TOPIC_AUTHORIZATION_FAILED))
        consumer.batches = [[fatal]]
        with pytest.raises(KafkaException):
            reader.bootstrap(timeout_s=5.0)

    def test_undecodable_records_are_skipped_and_counted(self) -> None:
        reader, consumer = make_reader()
        malformed = malformed_desired_records('dummy')
        a = desired(WF_A)
        consumer.batches = [
            [_Msg(key, value) for key, value in malformed.values()]
            + [record(WF_A, a), eof()]
        ]
        assert reader.bootstrap(timeout_s=5.0) == [(WF_A, a)]
        assert reader.undecodable_records == len(malformed)

    def test_poll_returns_records_since_bootstrap_without_blocking(self) -> None:
        reader, consumer = make_reader()
        consumer.batches = [[eof()]]
        reader.bootstrap(timeout_s=5.0)
        b = desired(WF_B)
        consumer.batches = [[record(WF_B, b), record(WF_A, None), eof()]]
        assert reader.poll() == [(WF_B, b), (WF_A, None)]
        start = time.monotonic()
        assert reader.poll() == []
        assert time.monotonic() - start < 0.1

    def test_poll_reports_consumer_error_as_no_updates(self) -> None:
        reader, consumer = make_reader()
        consumer.batches = [[_Msg(error=KafkaError(KafkaError._TRANSPORT))]]
        assert reader.poll() == []

    def test_close_closes_consumer(self) -> None:
        reader, consumer = make_reader()
        with reader:
            pass
        assert consumer.closed


class FakeProducer:
    """Serves the delivery report on ``flush``, or never if ``deliver`` is False."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.produced: list[tuple[str, bytes, bytes | None]] = []
        self.flush_timeouts: list[float] = []
        self.delivery_error: KafkaError | None = None
        self.deliver = True
        self.raise_on_produce: BaseException | None = None
        self._pending: list[Callable[[KafkaError | None, Any], None]] = []

    def produce(
        self,
        *,
        topic: str,
        key: bytes,
        value: bytes | None,
        on_delivery: Callable[[KafkaError | None, Any], None],
    ) -> None:
        if self.raise_on_produce is not None:
            raise self.raise_on_produce
        self.produced.append((topic, key, value))
        self._pending.append(on_delivery)

    def flush(self, timeout: float) -> int:
        self.flush_timeouts.append(timeout)
        if self.deliver:
            for callback in self._pending:
                callback(self.delivery_error, None)
            self._pending = []
        return len(self._pending)


@pytest.fixture
def producer() -> FakeProducer:
    return FakeProducer({})


@pytest.fixture
def writer(producer: FakeProducer) -> KafkaDesiredStateWriter:
    def factory(config: dict[str, Any]) -> FakeProducer:
        producer.config = config
        return producer

    return KafkaDesiredStateWriter(
        kafka_config={'bootstrap.servers': 'test'},
        topic=TOPIC,
        timeout_s=2.0,
        producer_factory=factory,
    )


class TestKafkaDesiredStateWriter:
    def test_producer_is_idempotent_and_times_out_messages_before_flush(
        self, writer: KafkaDesiredStateWriter, producer: FakeProducer
    ) -> None:
        writer.write(WF_A, desired(WF_A))
        assert producer.config['enable.idempotence'] is True
        assert producer.config['bootstrap.servers'] == 'test'
        assert producer.config['message.timeout.ms'] == 2000
        [flush_timeout] = producer.flush_timeouts
        assert flush_timeout > 2.0

    def test_write_produces_keyed_record(
        self, writer: KafkaDesiredStateWriter, producer: FakeProducer
    ) -> None:
        a = desired(WF_A)
        writer.write(WF_A, a)
        assert producer.produced == [(TOPIC, *encode_desired_record(WF_A, a))]

    def test_write_none_produces_tombstone(
        self, writer: KafkaDesiredStateWriter, producer: FakeProducer
    ) -> None:
        writer.write(WF_A, None)
        assert producer.produced == [(TOPIC, b'dummy/a/1', None)]

    def test_delivery_error_raises(
        self, writer: KafkaDesiredStateWriter, producer: FakeProducer
    ) -> None:
        producer.delivery_error = KafkaError(KafkaError.NOT_ENOUGH_REPLICAS)
        with pytest.raises(DesiredStateWriteError, match='dummy/a/1'):
            writer.write(WF_A, desired(WF_A))

    def test_fatal_delivery_error_raises_kafka_exception(
        self, writer: KafkaDesiredStateWriter, producer: FakeProducer
    ) -> None:
        producer.delivery_error = KafkaError(KafkaError.TOPIC_AUTHORIZATION_FAILED)
        with pytest.raises(KafkaException):
            writer.write(WF_A, desired(WF_A))

    def test_missing_delivery_report_raises(
        self, writer: KafkaDesiredStateWriter, producer: FakeProducer
    ) -> None:
        producer.deliver = False
        with pytest.raises(DesiredStateWriteError, match='No delivery report'):
            writer.write(WF_A, desired(WF_A))

    @pytest.mark.parametrize(
        'error',
        [
            BufferError('queue full'),
            KafkaException(KafkaError(KafkaError.MSG_SIZE_TOO_LARGE)),
        ],
    )
    def test_produce_error_raises(
        self,
        writer: KafkaDesiredStateWriter,
        producer: FakeProducer,
        error: BaseException,
    ) -> None:
        producer.raise_on_produce = error
        with pytest.raises(DesiredStateWriteError):
            writer.write(WF_A, desired(WF_A))
