# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Producer-interaction tests for :class:`KafkaSink`.

These tests use real serializers wherever possible, so that encoding output
and sink wiring are exercised together end-to-end. A hand-rolled fake
:class:`confluent_kafka.Producer` captures ``produce`` calls for assertion.
Stubs appear only where there is a specific reason — here, deterministic
failure injection for the :class:`SerializationError` handling tests.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import scipp as sc
from confluent_kafka import KafkaError, KafkaException
from pydantic import BaseModel
from streaming_data_types import dataarray_da00

from ess.livedata.config.models import ConfigKey
from ess.livedata.core.message import (
    COMMANDS_STREAM_ID,
    Message,
    StreamId,
    StreamKind,
)
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.handlers.config_handler import ConfigUpdate
from ess.livedata.kafka.sink import (
    KafkaSink,
    MessageSerializer,
    SerializationError,
    SerializedMessage,
)
from ess.livedata.kafka.sink_serializers import CommandSerializer, Da00Serializer

INSTRUMENT = 'dummy'


class _FakeMsg:
    def __init__(self, topic: str) -> None:
        self._topic = topic

    def topic(self) -> str:
        return self._topic


class _FakeProducer:
    """
    Minimal stand-in for ``confluent_kafka.Producer``.

    Faking the broker is unavoidable — a real producer would need a running
    Kafka cluster. Records ``produce`` calls for assertion. ``raise_on_produce``
    / ``raise_on_flush`` inject broker errors. ``delivery_error`` simulates
    the broker reporting a delivery failure asynchronously: pending callbacks
    fire with the given error on the next ``poll``/``flush``.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.produced: list[dict] = []
        self.flushed = 0
        self.raise_on_produce: BaseException | None = None
        self.raise_on_flush: BaseException | None = None
        self.delivery_error: KafkaError | None = None
        self._pending: list[tuple[Any, _FakeMsg]] = []

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
        if callback is not None:
            self._pending.append((callback, _FakeMsg(topic)))

    def _drain_callbacks(self) -> None:
        for cb, msg in self._pending:
            cb(self.delivery_error, msg)
        self._pending = []

    def poll(self, timeout: float) -> None:
        self._drain_callbacks()

    def flush(self, timeout: float | None = None) -> None:
        if self.raise_on_flush is not None:
            raise self.raise_on_flush
        self._drain_callbacks()
        self.flushed += 1


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


def _data_message(name: str = 'detector') -> Message[sc.DataArray]:
    return Message(
        timestamp=Timestamp.from_ns(1_234_567_890),
        stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name=name),
        value=sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[0, 1, 2], unit='mm')},
        ),
    )


class _Payload(BaseModel):
    foo: str


def _command_message() -> Message[ConfigUpdate]:
    return Message(
        timestamp=Timestamp.from_ns(0),
        stream=COMMANDS_STREAM_ID,
        value=ConfigUpdate(
            config_key=ConfigKey(
                source_name='detector_1',
                service_name='data_reduction',
                key='workflow',
            ),
            value=_Payload(foo='bar'),
        ),
    )


class TestPublish:
    def test_forwards_real_serializer_output_to_producer(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        serializer = Da00Serializer(instrument=INSTRUMENT)
        msg = _data_message()
        with sink_factory(serializer) as sink:
            sink.publish_messages([msg])

        assert len(producer.produced) == 1
        call = producer.produced[0]
        assert call['topic'] == f'{INSTRUMENT}_livedata_data'
        assert call['key'] is None
        # The bytes on the wire must decode back to the original data.
        decoded = dataarray_da00.deserialise_da00(call['value'])
        assert decoded.source_name == 'detector'
        assert decoded.timestamp_ns == 1_234_567_890
        assert call['callback'] is not None

    def test_command_serializer_emits_key(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        serializer = CommandSerializer(instrument=INSTRUMENT)
        msg = _command_message()
        with sink_factory(serializer) as sink:
            sink.publish_messages([msg])

        call = producer.produced[0]
        assert call['topic'] == f'{INSTRUMENT}_livedata_commands'
        assert call['key'] == str(msg.value.config_key).encode('utf-8')
        assert json.loads(call['value'].decode('utf-8')) == {'foo': 'bar'}

    def test_publishes_each_message_once(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        serializer = Da00Serializer(instrument=INSTRUMENT)
        with sink_factory(serializer) as sink:
            sink.publish_messages(
                [_data_message('a'), _data_message('b'), _data_message('c')]
            )
        assert len(producer.produced) == 3
        names = [
            dataarray_da00.deserialise_da00(call['value']).source_name
            for call in producer.produced
        ]
        assert names == ['a', 'b', 'c']

    def test_flush_called_after_batch(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        serializer = Da00Serializer(instrument=INSTRUMENT)
        with sink_factory(serializer) as sink:
            sink.publish_messages([_data_message()])
        # One flush during publish, one during close.
        assert producer.flushed >= 1


class TestContextManager:
    def test_publish_outside_context_raises(self) -> None:
        sink = KafkaSink(
            kafka_config={},
            serializer=Da00Serializer(instrument=INSTRUMENT),
            producer_factory=lambda config: _FakeProducer(config),
        )
        with pytest.raises(RuntimeError, match='context manager'):
            sink.publish_messages([_data_message()])

    def test_enter_builds_producer_via_factory(self) -> None:
        captured: list[dict] = []

        def factory(config: dict) -> _FakeProducer:
            captured.append(config)
            return _FakeProducer(config)

        cfg = {'bootstrap.servers': 'host:9092'}
        with KafkaSink(
            kafka_config=cfg,
            serializer=Da00Serializer(instrument=INSTRUMENT),
            producer_factory=factory,
        ):
            pass
        assert captured == [cfg]


class TestErrorHandling:
    def test_serialization_error_is_skipped(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        # Deterministic failure injection: justifies a local ad-hoc serializer.
        class _AlwaysFails:
            def serialize(self, message: Message) -> SerializedMessage:
                raise SerializationError('boom')

        with sink_factory(_AlwaysFails()) as sink:
            # Must not raise.
            sink.publish_messages([_data_message(), _data_message()])
        assert producer.produced == []

    def test_one_serialization_error_does_not_block_other_messages(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        class _FailsOnSecond:
            def __init__(self) -> None:
                self._count = 0
                self._inner = Da00Serializer(instrument=INSTRUMENT)

            def serialize(self, message: Message) -> SerializedMessage:
                self._count += 1
                if self._count == 2:
                    raise SerializationError('bad second message')
                return self._inner.serialize(message)

        with sink_factory(_FailsOnSecond()) as sink:
            sink.publish_messages(
                [_data_message('a'), _data_message('b'), _data_message('c')]
            )
        assert len(producer.produced) == 2
        names = [
            dataarray_da00.deserialise_da00(call['value']).source_name
            for call in producer.produced
        ]
        assert names == ['a', 'c']

    def test_kafka_exception_from_produce_does_not_propagate(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        producer.raise_on_produce = KafkaException('broker down')
        with sink_factory(Da00Serializer(instrument=INSTRUMENT)) as sink:
            # Must not raise.
            sink.publish_messages([_data_message()])
        assert producer.produced == []

    def test_kafka_exception_from_flush_does_not_propagate(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        with sink_factory(Da00Serializer(instrument=INSTRUMENT)) as sink:
            producer.raise_on_flush = KafkaException('flush failed')
            # Must not raise.
            sink.publish_messages([_data_message()])


class TestFailFast:
    """
    Authorization/auth-misconfiguration errors must crash the sink instead of
    being silently logged. confluent_kafka does not flag these as fatal, so
    the sink applies its own auth-code list (see ``_FATAL_ERROR_CODES``).
    """

    @pytest.mark.parametrize(
        "code",
        [
            KafkaError.TOPIC_AUTHORIZATION_FAILED,
            KafkaError.CLUSTER_AUTHORIZATION_FAILED,
            KafkaError.SASL_AUTHENTICATION_FAILED,
            KafkaError.TRANSACTIONAL_ID_AUTHORIZATION_FAILED,
        ],
    )
    def test_fatal_delivery_callback_raises_after_publish(
        self, sink_factory, producer: _FakeProducer, code: int
    ) -> None:
        producer.delivery_error = KafkaError(code)
        with sink_factory(Da00Serializer(instrument=INSTRUMENT)) as sink:
            with pytest.raises(KafkaException):
                sink.publish_messages([_data_message()])

    def test_subsequent_publish_raises_after_fatal(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        producer.delivery_error = KafkaError(KafkaError.TOPIC_AUTHORIZATION_FAILED)
        with sink_factory(Da00Serializer(instrument=INSTRUMENT)) as sink:
            with pytest.raises(KafkaException):
                sink.publish_messages([_data_message('a')])
            # Even after clearing the broker-side fault, the sink stays poisoned:
            # operator must restart the service.
            producer.delivery_error = None
            with pytest.raises(KafkaException):
                sink.publish_messages([_data_message('b')])

    def test_fatal_kafka_exception_from_produce_propagates(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        producer.raise_on_produce = KafkaException(
            KafkaError(KafkaError.SASL_AUTHENTICATION_FAILED)
        )
        with sink_factory(Da00Serializer(instrument=INSTRUMENT)) as sink:
            with pytest.raises(KafkaException):
                sink.publish_messages([_data_message()])

    def test_fatal_kafka_exception_from_flush_propagates(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        producer.raise_on_flush = KafkaException(
            KafkaError(KafkaError.CLUSTER_AUTHORIZATION_FAILED)
        )
        with sink_factory(Da00Serializer(instrument=INSTRUMENT)) as sink:
            with pytest.raises(KafkaException):
                sink.publish_messages([_data_message()])

    def test_nonfatal_delivery_error_does_not_raise(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        # NETWORK_EXCEPTION is a transient delivery failure, not auth/misconfig.
        producer.delivery_error = KafkaError(KafkaError.NETWORK_EXCEPTION)
        with sink_factory(Da00Serializer(instrument=INSTRUMENT)) as sink:
            # Must not raise.
            sink.publish_messages([_data_message()])
