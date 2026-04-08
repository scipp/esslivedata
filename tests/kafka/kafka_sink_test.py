# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Producer-interaction tests for :class:`KafkaSink`.

These tests focus on the sink's behavior *as a producer client*: they use a
hand-rolled fake ``confluent_kafka.Producer`` to capture ``produce`` calls and
verify dispatch, error handling, and lifecycle. Encoding is not exercised here
— it is covered exhaustively in :mod:`sink_serializers_test`. A stub
:class:`MessageSerializer` returning canned bytes keeps these tests focused on
the producer-interaction contract.
"""

from __future__ import annotations

import pytest
from confluent_kafka import KafkaException

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.kafka.sink import (
    KafkaSink,
    MessageSerializer,
    SerializationError,
    SerializedMessage,
)


class _FakeProducer:
    """
    Minimal stand-in for ``confluent_kafka.Producer``.

    Records ``produce`` calls. ``poll`` and ``flush`` are no-ops unless
    configured to raise via ``raise_on_produce`` or ``raise_on_flush``.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.produced: list[dict] = []
        self.flushed = 0
        self.raise_on_produce: BaseException | None = None
        self.raise_on_flush: BaseException | None = None

    def produce(
        self,
        *,
        topic: str,
        value: bytes,
        key: bytes | None = None,
        callback=None,
    ) -> None:
        if self.raise_on_produce is not None:
            raise self.raise_on_produce
        self.produced.append(
            {'topic': topic, 'key': key, 'value': value, 'callback': callback}
        )

    def poll(self, timeout: float) -> None:
        pass

    def flush(self, timeout: float | None = None) -> None:
        if self.raise_on_flush is not None:
            raise self.raise_on_flush
        self.flushed += 1


class _StubSerializer(MessageSerializer):
    """Returns canned :class:`SerializedMessage` values per message."""

    def __init__(
        self,
        *,
        topic: str = 'stub_topic',
        key: bytes | None = None,
        value: bytes = b'stub_value',
        raise_error: Exception | None = None,
    ) -> None:
        self._topic = topic
        self._key = key
        self._value = value
        self._raise_error = raise_error
        self.calls: list[Message] = []

    def serialize(self, message: Message) -> SerializedMessage:
        self.calls.append(message)
        if self._raise_error is not None:
            raise self._raise_error
        return SerializedMessage(topic=self._topic, key=self._key, value=self._value)


def _make_message(value: str = 'payload') -> Message:
    return Message(
        timestamp=Timestamp.from_ns(0),
        stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name='s'),
        value=value,
    )


@pytest.fixture
def producer() -> _FakeProducer:
    return _FakeProducer(config={})


@pytest.fixture
def sink_factory(producer: _FakeProducer):
    def _make(serializer: MessageSerializer) -> KafkaSink:
        return KafkaSink(
            kafka_config={'bootstrap.servers': 'test'},
            serializer=serializer,
            producer_factory=lambda config: producer,
        )

    return _make


class TestPublish:
    def test_forwards_topic_key_and_value_to_producer(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        serializer = _StubSerializer(topic='t1', key=b'k1', value=b'v1')
        with sink_factory(serializer) as sink:
            sink.publish_messages([_make_message()])

        assert len(producer.produced) == 1
        call = producer.produced[0]
        assert call['topic'] == 't1'
        assert call['key'] == b'k1'
        assert call['value'] == b'v1'
        assert call['callback'] is not None

    def test_publishes_each_message_once(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        serializer = _StubSerializer()
        with sink_factory(serializer) as sink:
            sink.publish_messages([_make_message(), _make_message(), _make_message()])
        assert len(producer.produced) == 3

    def test_key_none_is_forwarded(self, sink_factory, producer: _FakeProducer) -> None:
        serializer = _StubSerializer(key=None)
        with sink_factory(serializer) as sink:
            sink.publish_messages([_make_message()])
        assert producer.produced[0]['key'] is None

    def test_flush_called_after_batch(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        serializer = _StubSerializer()
        with sink_factory(serializer) as sink:
            sink.publish_messages([_make_message()])
        # One flush during publish, one during close
        assert producer.flushed >= 1


class TestContextManager:
    def test_publish_outside_context_raises(self) -> None:
        sink = KafkaSink(
            kafka_config={},
            serializer=_StubSerializer(),
            producer_factory=lambda config: _FakeProducer(config),
        )
        with pytest.raises(RuntimeError, match='context manager'):
            sink.publish_messages([_make_message()])

    def test_enter_builds_producer_via_factory(self) -> None:
        captured: list[dict] = []

        def factory(config: dict) -> _FakeProducer:
            captured.append(config)
            return _FakeProducer(config)

        cfg = {'bootstrap.servers': 'host:9092'}
        with KafkaSink(
            kafka_config=cfg,
            serializer=_StubSerializer(),
            producer_factory=factory,
        ):
            pass
        assert captured == [cfg]


class TestErrorHandling:
    def test_serialization_error_is_logged_and_skipped(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        serializer = _StubSerializer(raise_error=SerializationError('boom'))
        with sink_factory(serializer) as sink:
            # Must not raise
            sink.publish_messages([_make_message(), _make_message()])
        # Neither message reaches the producer
        assert producer.produced == []
        # Serializer was still called for both
        assert len(serializer.calls) == 2

    def test_one_serialization_error_does_not_block_other_messages(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        class _PartialSerializer(_StubSerializer):
            def __init__(self) -> None:
                super().__init__(topic='t', value=b'ok')
                self._count = 0

            def serialize(self, message: Message) -> SerializedMessage:
                self._count += 1
                if self._count == 2:
                    raise SerializationError('bad second message')
                return SerializedMessage(topic='t', key=None, value=b'ok')

        serializer = _PartialSerializer()
        with sink_factory(serializer) as sink:
            sink.publish_messages(
                [_make_message('a'), _make_message('b'), _make_message('c')]
            )
        assert len(producer.produced) == 2

    def test_kafka_exception_from_produce_is_logged(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        producer.raise_on_produce = KafkaException('broker down')
        serializer = _StubSerializer()
        with sink_factory(serializer) as sink:
            # Must not raise
            sink.publish_messages([_make_message()])
        assert producer.produced == []  # raise happens before append

    def test_kafka_exception_from_flush_is_logged(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        serializer = _StubSerializer()
        with sink_factory(serializer) as sink:
            producer.raise_on_flush = KafkaException('flush failed')
            # Must not raise
            sink.publish_messages([_make_message()])
