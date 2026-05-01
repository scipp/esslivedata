# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Round-trip tests for the sink-side serializers.

Each test pairs a :class:`MessageSerializer` with its source-side
:class:`MessageAdapter` counterpart and asserts that encoding followed by
decoding yields an equivalent message. These tests are the primary defense
against the class of type-drift bugs that slipped through in issue #847: the
``started_at`` field changed type on the payload without any test exercising
the full encode/decode cycle.
"""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest
import scipp as sc
from pydantic import BaseModel
from streaming_data_types import dataarray_da00

from ess.livedata.config.acknowledgement import (
    AcknowledgementResponse,
    CommandAcknowledgement,
)
from ess.livedata.config.models import ConfigKey
from ess.livedata.config.workflow_spec import JobId, WorkflowId
from ess.livedata.core.job import (
    JobState,
    JobStatus,
    ServiceState,
    ServiceStatus,
)
from ess.livedata.core.message import (
    COMMANDS_STREAM_ID,
    RESPONSES_STREAM_ID,
    STATUS_STREAM_ID,
    Message,
    StreamId,
    StreamKind,
)
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.handlers.config_handler import ConfigUpdate
from ess.livedata.kafka.message_adapter import (
    ChainedAdapter,
    CommandsAdapter,
    Da00ToScippAdapter,
    FakeKafkaMessage,
    KafkaToDa00Adapter,
    KafkaToF144Adapter,
    ResponsesAdapter,
    X5f2ToStatusAdapter,
)
from ess.livedata.kafka.sink import SerializationError, SerializedMessage
from ess.livedata.kafka.sink_routing import (
    RouteByStatusTypeSerializer,
    RouteByStreamKindSerializer,
)
from ess.livedata.kafka.sink_serializers import (
    CommandSerializer,
    Da00Serializer,
    F144Serializer,
    JobStatusToX5f2Serializer,
    ResponseSerializer,
    ServiceStatusToX5f2Serializer,
    make_default_sink_serializer,
)

INSTRUMENT = 'dummy'


class TestDa00Serializer:
    def test_round_trip_via_source_adapter(self) -> None:
        data = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='m'),
            coords={'x': sc.array(dims=['x'], values=[0, 1, 2], unit='mm')},
        )
        msg = Message(
            timestamp=Timestamp.from_ns(1_234_567_890),
            stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name='detector'),
            value=data,
        )
        serializer = Da00Serializer(instrument=INSTRUMENT)

        result = serializer.serialize(msg)

        assert isinstance(result, SerializedMessage)
        assert result.key is None
        assert result.topic == f'{INSTRUMENT}_livedata_data'

        adapter = ChainedAdapter(
            first=KafkaToDa00Adapter(stream_kind=StreamKind.LIVEDATA_DATA),
            second=Da00ToScippAdapter(),
        )
        decoded = adapter.adapt(
            FakeKafkaMessage(value=result.value, topic=result.topic)
        )
        assert sc.identical(decoded.value, data)
        assert decoded.timestamp == Timestamp.from_ns(1_234_567_890)
        assert decoded.stream.name == 'detector'

    def test_topic_follows_stream_kind(self) -> None:
        data = sc.DataArray(
            data=sc.array(dims=['x', 'y'], values=[[1, 2], [3, 4]], unit='counts')
        )
        msg = Message(
            timestamp=Timestamp.from_ns(0),
            stream=StreamId(kind=StreamKind.MONITOR_COUNTS, name='m1'),
            value=data,
        )
        result = Da00Serializer(instrument=INSTRUMENT).serialize(msg)
        assert result.topic == f'{INSTRUMENT}_beam_monitor'

    def test_round_trip_multidimensional_data_with_monitor_counts(self) -> None:
        data = sc.DataArray(
            data=sc.array(dims=['x', 'y'], values=[[1, 2], [3, 4]], unit='counts'),
            coords={
                'x': sc.array(dims=['x'], values=[0.1, 0.2], unit='m'),
                'y': sc.array(dims=['y'], values=[10, 20], unit='s'),
            },
        )
        msg = Message(
            timestamp=Timestamp.from_ns(9_876_543_210),
            stream=StreamId(kind=StreamKind.MONITOR_COUNTS, name='monitor_1'),
            value=data,
        )
        result = Da00Serializer(instrument=INSTRUMENT).serialize(msg)

        adapter = ChainedAdapter(
            first=KafkaToDa00Adapter(stream_kind=StreamKind.MONITOR_COUNTS),
            second=Da00ToScippAdapter(),
        )
        decoded = adapter.adapt(
            FakeKafkaMessage(value=result.value, topic=result.topic)
        )
        assert sc.identical(decoded.value, data)
        assert decoded.timestamp == Timestamp.from_ns(9_876_543_210)
        assert decoded.stream.name == 'monitor_1'


class TestF144Serializer:
    def test_round_trip_via_source_adapter(self) -> None:
        time_ns = 1_234_567_890_123_456_789
        data = sc.DataArray(
            data=sc.array(dims=[], values=42.5, unit='m/s'),
            coords={'time': sc.scalar(time_ns, unit='ns')},
        )
        msg = Message(
            timestamp=Timestamp.from_ns(0),
            stream=StreamId(kind=StreamKind.LOG, name='log1'),
            value=data,
        )
        serializer = F144Serializer(instrument=INSTRUMENT)

        result = serializer.serialize(msg)

        assert result.key is None
        assert result.topic == f'{INSTRUMENT}_motion'

        decoded = KafkaToF144Adapter().adapt(
            FakeKafkaMessage(value=result.value, topic=result.topic)
        )
        assert decoded.value.value == 42.5
        assert decoded.timestamp == Timestamp.from_ns(time_ns)
        assert decoded.stream.name == 'log1'

    def test_round_trip_with_array_data(self) -> None:
        time_ns = 9_876_543_210
        data = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'time': sc.scalar(time_ns, unit='ns')},
        )
        msg = Message(
            timestamp=Timestamp.from_ns(0),
            stream=StreamId(kind=StreamKind.LOG, name='array_log'),
            value=data,
        )
        result = F144Serializer(instrument=INSTRUMENT).serialize(msg)

        decoded = KafkaToF144Adapter().adapt(
            FakeKafkaMessage(value=result.value, topic=result.topic)
        )
        np.testing.assert_array_equal(decoded.value.value, [1.0, 2.0, 3.0])
        assert decoded.timestamp == Timestamp.from_ns(time_ns)
        assert decoded.stream.name == 'array_log'

    def test_time_coord_in_microseconds_is_converted_to_ns(self) -> None:
        time_us = 1_234_567_890_123_456
        data = sc.DataArray(
            data=sc.array(dims=[], values=7.5, unit='V'),
            coords={'time': sc.array(dims=[], values=time_us, unit='us')},
        )
        msg = Message(
            timestamp=Timestamp.from_ns(0),
            stream=StreamId(kind=StreamKind.LOG, name='time_unit_log'),
            value=data,
        )
        result = F144Serializer(instrument=INSTRUMENT).serialize(msg)

        decoded = KafkaToF144Adapter().adapt(
            FakeKafkaMessage(value=result.value, topic=result.topic)
        )
        assert decoded.value.value == 7.5
        assert decoded.timestamp == Timestamp.from_ns(time_us * 1000)

    def test_missing_time_coord_raises_serialization_error(self) -> None:
        msg = Message(
            timestamp=Timestamp.from_ns(0),
            stream=StreamId(kind=StreamKind.LOG, name='log1'),
            value=sc.DataArray(data=sc.array(dims=[], values=1.0, unit='K')),
        )
        with pytest.raises(SerializationError):
            F144Serializer(instrument=INSTRUMENT).serialize(msg)


def _make_service_status(**overrides) -> ServiceStatus:
    defaults: dict = {
        'instrument': 'dream',
        'service_name': 'data_reduction',
        'worker_id': str(uuid.uuid4()),
        'state': ServiceState.running,
        'started_at': Timestamp.from_ns(1_700_000_000_000_000_000),
        'active_job_count': 3,
        'error': None,
    }
    defaults.update(overrides)
    return ServiceStatus(**defaults)


def _make_job_status(**overrides) -> JobStatus:
    defaults: dict = {
        'job_id': JobId(source_name='detector_1', job_number=uuid.uuid4()),
        'workflow_id': WorkflowId(
            instrument='test_inst',
            name='test_workflow',
            version=1,
        ),
        'state': JobState.active,
        'start_time': Timestamp.from_ns(1_700_000_000_000_000_000),
        'end_time': None,
    }
    defaults.update(overrides)
    return JobStatus(**defaults)


def _status_message(value) -> Message:
    return Message(
        timestamp=Timestamp.from_ns(1_700_000_000_000_000_000),
        stream=STATUS_STREAM_ID,
        value=value,
    )


class TestServiceStatusToX5f2Serializer:
    def test_round_trip_preserves_timestamp_type(self) -> None:
        """
        Regression test for issue #847.

        The original bug was ``ServiceStatusPayload.started_at: int`` vs
        ``ServiceStatus.started_at: Timestamp``. A round-trip asserting that
        the decoded ``started_at`` is a :class:`Timestamp` catches this
        immediately.
        """
        status = _make_service_status()
        serializer = ServiceStatusToX5f2Serializer(
            instrument=INSTRUMENT,
            software_version='1.2.3',
            host_name='test-host',
            process_id=4321,
        )

        result = serializer.serialize(_status_message(status))

        assert result.key is None
        assert result.topic == f'{INSTRUMENT}_livedata_heartbeat'

        decoded = X5f2ToStatusAdapter().adapt(
            FakeKafkaMessage(value=result.value, topic=result.topic)
        )
        decoded_status = decoded.value
        assert isinstance(decoded_status, ServiceStatus)
        assert isinstance(decoded_status.started_at, Timestamp)
        assert decoded_status.started_at == status.started_at
        assert decoded_status.state == status.state
        assert decoded_status.instrument == status.instrument
        assert decoded_status.worker_id == status.worker_id
        assert decoded_status.active_job_count == status.active_job_count


class TestJobStatusToX5f2Serializer:
    def test_round_trip_preserves_timestamp_type(self) -> None:
        status = _make_job_status(end_time=Timestamp.from_ns(1_700_000_001_000_000_000))
        serializer = JobStatusToX5f2Serializer(
            instrument=INSTRUMENT,
            software_version='1.2.3',
            host_name='test-host',
            process_id=4321,
        )

        result = serializer.serialize(_status_message(status))

        assert result.key is None
        assert result.topic == f'{INSTRUMENT}_livedata_heartbeat'

        decoded = X5f2ToStatusAdapter().adapt(
            FakeKafkaMessage(value=result.value, topic=result.topic)
        )
        decoded_status = decoded.value
        assert isinstance(decoded_status, JobStatus)
        assert isinstance(decoded_status.start_time, Timestamp)
        assert isinstance(decoded_status.end_time, Timestamp)
        assert decoded_status.start_time == status.start_time
        assert decoded_status.end_time == status.end_time
        assert decoded_status.state == status.state
        assert decoded_status.job_id == status.job_id


class _CommandPayload(BaseModel):
    hello: str
    n: int


class TestCommandSerializer:
    def test_round_trip_via_source_adapter(self) -> None:
        config_key = ConfigKey(
            source_name='detector_1', service_name='data_reduction', key='workflow'
        )
        payload = _CommandPayload(hello='world', n=7)
        update = ConfigUpdate(config_key=config_key, value=payload)
        msg = Message(
            timestamp=Timestamp.from_ns(0),
            stream=COMMANDS_STREAM_ID,
            value=update,
        )

        result = CommandSerializer(instrument=INSTRUMENT).serialize(msg)

        assert result.topic == f'{INSTRUMENT}_livedata_commands'
        assert result.key is not None
        assert result.key == str(config_key).encode('utf-8')
        assert json.loads(result.value.decode('utf-8')) == {'hello': 'world', 'n': 7}

        decoded = CommandsAdapter().adapt(
            FakeKafkaMessage(key=result.key, value=result.value, topic=result.topic)
        )
        assert decoded.stream == COMMANDS_STREAM_ID
        assert decoded.value.key == result.key
        assert decoded.value.value == result.value
        assert ConfigKey.from_string(decoded.value.key.decode('utf-8')) == config_key
        assert json.loads(decoded.value.value.decode('utf-8')) == {
            'hello': 'world',
            'n': 7,
        }


class TestResponseSerializer:
    def test_round_trip_via_source_adapter(self) -> None:
        ack = CommandAcknowledgement(
            message_id='msg-123',
            device='detector_1',
            response=AcknowledgementResponse.ACK,
            message=None,
        )
        msg = Message(
            timestamp=Timestamp.from_ns(0),
            stream=RESPONSES_STREAM_ID,
            value=ack,
        )

        result = ResponseSerializer(instrument=INSTRUMENT).serialize(msg)

        assert result.key is None
        assert result.topic == f'{INSTRUMENT}_livedata_responses'

        decoded = ResponsesAdapter().adapt(
            FakeKafkaMessage(value=result.value, topic=result.topic)
        )
        assert decoded.stream == RESPONSES_STREAM_ID
        assert decoded.value == ack


class _FixedSerializer:
    """Test stub that records calls and returns a canned SerializedMessage."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls: list[Message] = []

    def serialize(self, message: Message) -> SerializedMessage:
        self.calls.append(message)
        return SerializedMessage(
            topic=f'topic_{self.tag}', key=None, value=self.tag.encode('utf-8')
        )


class TestRouteByStreamKindSerializer:
    def test_dispatches_by_kind(self) -> None:
        a = _FixedSerializer('a')
        b = _FixedSerializer('b')
        router = RouteByStreamKindSerializer(
            {StreamKind.LIVEDATA_DATA: a, StreamKind.LIVEDATA_ROI: b}
        )
        msg_a = Message(
            timestamp=Timestamp.from_ns(0),
            stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name='x'),
            value=None,
        )
        msg_b = Message(
            timestamp=Timestamp.from_ns(0),
            stream=StreamId(kind=StreamKind.LIVEDATA_ROI, name='y'),
            value=None,
        )
        assert router.serialize(msg_a).value == b'a'
        assert router.serialize(msg_b).value == b'b'
        assert len(a.calls) == 1
        assert len(b.calls) == 1

    def test_unknown_kind_raises_serialization_error(self) -> None:
        router = RouteByStreamKindSerializer(
            {StreamKind.LIVEDATA_DATA: _FixedSerializer('a')}
        )
        msg = Message(
            timestamp=Timestamp.from_ns(0),
            stream=StreamId(kind=StreamKind.LIVEDATA_ROI, name='y'),
            value=None,
        )
        with pytest.raises(SerializationError, match='No serializer configured'):
            router.serialize(msg)


class TestRouteByStatusTypeSerializer:
    def test_dispatches_by_payload_type(self) -> None:
        svc = _FixedSerializer('svc')
        job = _FixedSerializer('job')
        router = RouteByStatusTypeSerializer(service=svc, job=job)

        svc_msg = _status_message(_make_service_status())
        job_msg = _status_message(_make_job_status())

        assert router.serialize(svc_msg).value == b'svc'
        assert router.serialize(job_msg).value == b'job'
        assert len(svc.calls) == 1
        assert len(job.calls) == 1


class TestMakeDefaultSinkSerializer:
    def test_routes_data_and_status_end_to_end(self) -> None:
        data_serializer = Da00Serializer(instrument=INSTRUMENT)
        serializer = make_default_sink_serializer(
            instrument=INSTRUMENT, data_serializer=data_serializer
        )

        data_msg = Message(
            timestamp=Timestamp.from_ns(0),
            stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name='d'),
            value=sc.DataArray(data=sc.array(dims=['x'], values=[1.0], unit='m')),
        )
        svc_msg = _status_message(_make_service_status())
        job_msg = _status_message(_make_job_status())

        data_result = serializer.serialize(data_msg)
        svc_result = serializer.serialize(svc_msg)
        job_result = serializer.serialize(job_msg)

        assert data_result.topic == f'{INSTRUMENT}_livedata_data'
        assert svc_result.topic == f'{INSTRUMENT}_livedata_heartbeat'
        assert job_result.topic == f'{INSTRUMENT}_livedata_heartbeat'

        # Decoded service status must round-trip with correct Timestamp types.
        decoded_svc = X5f2ToStatusAdapter().adapt(
            FakeKafkaMessage(value=svc_result.value, topic=svc_result.topic)
        )
        assert isinstance(decoded_svc.value, ServiceStatus)
        assert isinstance(decoded_svc.value.started_at, Timestamp)

        decoded_job = X5f2ToStatusAdapter().adapt(
            FakeKafkaMessage(value=job_result.value, topic=job_result.topic)
        )
        assert isinstance(decoded_job.value, JobStatus)

    def test_defaults_to_da00_serializer(self) -> None:
        serializer = make_default_sink_serializer(instrument=INSTRUMENT)
        msg = Message(
            timestamp=Timestamp.from_ns(0),
            stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name='d'),
            value=sc.DataArray(data=sc.array(dims=['x'], values=[1.0], unit='m')),
        )
        result = serializer.serialize(msg)
        assert result.topic == f'{INSTRUMENT}_livedata_data'
        # Verify the bytes are valid da00 by decoding.
        decoded = dataarray_da00.deserialise_da00(result.value)
        assert decoded.source_name == 'd'
