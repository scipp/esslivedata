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

import pytest
import scipp as sc
from confluent_kafka import KafkaException
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


class _FakeProducer:
    """
    Minimal stand-in for ``confluent_kafka.Producer``.

    Faking the broker is unavoidable — a real producer would need a running
    Kafka cluster. Records ``produce`` calls for assertion. ``raise_on_produce``
    / ``raise_on_flush`` inject broker errors so the sink's error handling can
    be verified without a real broker.
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


def _command_message() -> Message[ConfigUpdate]:
    class _Payload:
        def model_dump_json(self) -> str:
            return json.dumps({'foo': 'bar'})

    return Message(
        timestamp=Timestamp.from_ns(0),
        stream=COMMANDS_STREAM_ID,
        value=ConfigUpdate(
            config_key=ConfigKey(
                source_name='detector_1',
                service_name='data_reduction',
                key='workflow',
            ),
            value=_Payload(),
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
    def test_serialization_error_is_logged_and_skipped(
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

    def test_kafka_exception_from_produce_is_logged(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        producer.raise_on_produce = KafkaException('broker down')
        with sink_factory(Da00Serializer(instrument=INSTRUMENT)) as sink:
            # Must not raise.
            sink.publish_messages([_data_message()])
        assert producer.produced == []

    def test_kafka_exception_from_flush_is_logged(
        self, sink_factory, producer: _FakeProducer
    ) -> None:
        with sink_factory(Da00Serializer(instrument=INSTRUMENT)) as sink:
            producer.raise_on_flush = KafkaException('flush failed')
            # Must not raise.
            sink.publish_messages([_data_message()])
