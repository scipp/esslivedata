# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for HttpTransport."""

import base64
import time
import uuid

import numpy as np
import scipp as sc
from streaming_data_types import dataarray_da00

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.core.message import StreamKind
from ess.livedata.dashboard.http_transport import (
    HttpTransport,
    QueueableMessageSource,
    deserialize_da00_to_message,
)
from ess.livedata.kafka.scipp_da00_compat import scipp_to_da00


def make_test_dataarray() -> sc.DataArray:
    """Create a simple test DataArray."""
    return sc.DataArray(
        sc.array(dims=['x', 'y'], values=np.arange(6).reshape(2, 3), unit='counts'),
        coords={
            'x': sc.array(dims=['x'], values=[0.0, 1.0], unit='m'),
            'y': sc.array(dims=['y'], values=[0.0, 1.0, 2.0], unit='m'),
        },
    )


def serialize_dataarray_to_da00(source_name: str, data: sc.DataArray) -> bytes:
    """Serialize a DataArray to da00 format using the same method as KafkaSink."""
    return dataarray_da00.serialise_da00(
        source_name=source_name,
        timestamp_ns=int(time.time_ns()),
        data=scipp_to_da00(data),
    )


class TestQueueableMessageSource:
    def test_empty_source_returns_empty_list(self) -> None:
        source = QueueableMessageSource()
        assert source.get_messages() == []

    def test_queue_message_returns_on_get(self) -> None:
        source = QueueableMessageSource()

        # Create a test message using da00 deserialization
        result_key = ResultKey(
            workflow_id=WorkflowId(
                instrument='test', namespace='ns', name='workflow', version=1
            ),
            job_id=JobId(job_number=uuid.uuid4(), source_name='detector1'),
            output_name='result',
        )
        payload = serialize_dataarray_to_da00(
            source_name=result_key.model_dump_json(),
            data=make_test_dataarray(),
        )
        message = deserialize_da00_to_message(payload)
        source.queue_message(message)

        messages = source.get_messages()
        assert len(messages) == 1
        assert messages[0].stream.kind == StreamKind.LIVEDATA_DATA

    def test_get_messages_clears_queue(self) -> None:
        source = QueueableMessageSource()

        result_key = ResultKey(
            workflow_id=WorkflowId(
                instrument='test', namespace='ns', name='workflow', version=1
            ),
            job_id=JobId(job_number=uuid.uuid4(), source_name='detector1'),
            output_name='result',
        )
        payload = serialize_dataarray_to_da00(
            source_name=result_key.model_dump_json(),
            data=make_test_dataarray(),
        )
        message = deserialize_da00_to_message(payload)
        source.queue_message(message)

        # First call returns messages
        messages = source.get_messages()
        assert len(messages) == 1

        # Second call returns empty
        messages = source.get_messages()
        assert len(messages) == 0


class TestDeserializeDa00ToMessage:
    def test_deserialize_creates_valid_message(self) -> None:
        result_key = ResultKey(
            workflow_id=WorkflowId(
                instrument='dummy', namespace='detector', name='view', version=1
            ),
            job_id=JobId(job_number=uuid.uuid4(), source_name='panel_0'),
            output_name='current',
        )
        data = make_test_dataarray()
        payload = serialize_dataarray_to_da00(
            source_name=result_key.model_dump_json(),
            data=data,
        )

        message = deserialize_da00_to_message(payload)

        assert message.stream.kind == StreamKind.LIVEDATA_DATA
        assert message.stream.name == result_key.model_dump_json()
        assert isinstance(message.value, sc.DataArray)
        assert message.value.dims == data.dims

    def test_timestamp_override(self) -> None:
        result_key = ResultKey(
            workflow_id=WorkflowId(
                instrument='test', namespace='ns', name='workflow', version=1
            ),
            job_id=JobId(job_number=uuid.uuid4(), source_name='detector1'),
            output_name='result',
        )
        payload = serialize_dataarray_to_da00(
            source_name=result_key.model_dump_json(),
            data=make_test_dataarray(),
        )

        custom_timestamp = 123456789
        message = deserialize_da00_to_message(payload, timestamp_ns=custom_timestamp)

        assert message.timestamp == custom_timestamp


class TestHttpTransport:
    def test_context_manager_returns_resources(self) -> None:
        transport = HttpTransport(instrument='dummy')

        with transport as resources:
            assert resources.message_source is not None
            assert resources.command_sink is not None
            assert resources.roi_sink is not None

    def test_inject_da00(self) -> None:
        transport = HttpTransport(instrument='dummy')

        with transport as resources:
            result_key = ResultKey(
                workflow_id=WorkflowId(
                    instrument='dummy', namespace='detector', name='view', version=1
                ),
                job_id=JobId(job_number=uuid.uuid4(), source_name='panel_0'),
                output_name='current',
            )
            payload = serialize_dataarray_to_da00(
                source_name=result_key.model_dump_json(),
                data=make_test_dataarray(),
            )

            transport.inject_da00(payload)

            messages = resources.message_source.get_messages()
            assert len(messages) == 1
            assert messages[0].stream.kind == StreamKind.LIVEDATA_DATA

    def test_inject_from_json(self) -> None:
        transport = HttpTransport(instrument='dummy')

        with transport as resources:
            result_key = ResultKey(
                workflow_id=WorkflowId(
                    instrument='dummy', namespace='detector', name='view', version=1
                ),
                job_id=JobId(job_number=uuid.uuid4(), source_name='panel_0'),
                output_name='current',
            )
            payload = serialize_dataarray_to_da00(
                source_name=result_key.model_dump_json(),
                data=make_test_dataarray(),
            )

            json_data = {
                'payload_base64': base64.b64encode(payload).decode('utf-8'),
                'timestamp_ns': 999999,
            }
            transport.inject_from_json(json_data)

            messages = resources.message_source.get_messages()
            assert len(messages) == 1
            assert messages[0].timestamp == 999999

    def test_handle_post_request_success(self) -> None:
        transport = HttpTransport(instrument='dummy')

        with transport:
            result_key = ResultKey(
                workflow_id=WorkflowId(
                    instrument='dummy', namespace='detector', name='view', version=1
                ),
                job_id=JobId(job_number=uuid.uuid4(), source_name='panel_0'),
                output_name='current',
            )
            payload = serialize_dataarray_to_da00(
                source_name=result_key.model_dump_json(),
                data=make_test_dataarray(),
            )

            body = (
                b'{"payload_base64": "'
                + base64.b64encode(payload)
                + b'", "timestamp_ns": 12345}'
            )
            result = transport.handle_post_request(body)

            assert result == {'status': 'ok'}

    def test_handle_post_request_invalid_json(self) -> None:
        transport = HttpTransport(instrument='dummy')

        with transport:
            result = transport.handle_post_request(b'not valid json')

            assert result['status'] == 'error'
            assert 'message' in result

    def test_handle_post_request_invalid_payload(self) -> None:
        transport = HttpTransport(instrument='dummy')

        with transport:
            body = b'{"payload_base64": "aW52YWxpZA=="}'  # "invalid" in base64
            result = transport.handle_post_request(body)

            assert result['status'] == 'error'
