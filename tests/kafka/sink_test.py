# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.kafka.message_adapter import (
    ChainedAdapter,
    Da00ToScippAdapter,
    FakeKafkaMessage,
    KafkaToDa00Adapter,
    KafkaToF144Adapter,
)
from ess.livedata.kafka.sink import (
    serialize_dataarray_to_da00,
    serialize_dataarray_to_f144,
)


class TestDa00Serializer:
    def test_serialize_dataarray_to_da00_roundtrip_preserves_data(self) -> None:
        # Create test data
        original_data = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='m'),
            coords={'x': sc.array(dims=['x'], values=[0, 1, 2], unit='mm')},
        )
        stream_id = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test_detector')
        payload_timestamp = 1234567890
        original_msg = Message(
            timestamp=payload_timestamp, stream=stream_id, value=original_data
        )

        # Serialize to da00
        serialized_bytes = serialize_dataarray_to_da00(original_msg)

        # Create adapter chain to deserialize back to scipp
        da00_adapter = KafkaToDa00Adapter(stream_kind=StreamKind.LIVEDATA_DATA)
        scipp_adapter = Da00ToScippAdapter()
        roundtrip_adapter = ChainedAdapter(da00_adapter, scipp_adapter)

        # Create fake Kafka message with different timestamp than payload
        kafka_timestamp = 9999999999  # Different from payload timestamp
        kafka_msg = FakeKafkaMessage(
            value=serialized_bytes,
            topic='test_topic',
            timestamp=kafka_timestamp,  # This should NOT be used
        )

        # Deserialize back to scipp
        [result_msg] = roundtrip_adapter.adapt(kafka_msg)

        # Verify the roundtrip preserved the data and payload timestamp
        assert sc.identical(result_msg.value, original_data)
        assert result_msg.timestamp == payload_timestamp  # Should use da00 timestamp_ns
        assert result_msg.stream.name == 'test_detector'  # From da00 source_name

    def test_serialize_dataarray_to_da00_with_multidimensional_data(self) -> None:
        # Test with more complex data structure
        original_data = sc.DataArray(
            data=sc.array(dims=['x', 'y'], values=[[1, 2], [3, 4]], unit='counts'),
            coords={
                'x': sc.array(dims=['x'], values=[0.1, 0.2], unit='m'),
                'y': sc.array(dims=['y'], values=[10, 20], unit='s'),
            },
        )
        stream_id = StreamId(kind=StreamKind.MONITOR_COUNTS, name='monitor_1')
        payload_timestamp = 9876543210
        original_msg = Message(
            timestamp=payload_timestamp, stream=stream_id, value=original_data
        )

        # Serialize and deserialize
        serialized_bytes = serialize_dataarray_to_da00(original_msg)

        da00_adapter = KafkaToDa00Adapter(stream_kind=StreamKind.MONITOR_COUNTS)
        scipp_adapter = Da00ToScippAdapter()
        roundtrip_adapter = ChainedAdapter(da00_adapter, scipp_adapter)

        # Use different Kafka timestamp
        kafka_timestamp = 5555555555
        kafka_msg = FakeKafkaMessage(
            value=serialized_bytes,
            topic='monitor_topic',
            timestamp=kafka_timestamp,  # Should be ignored
        )

        [result_msg] = roundtrip_adapter.adapt(kafka_msg)

        # Verify roundtrip uses payload timestamp, not Kafka timestamp
        assert sc.identical(result_msg.value, original_data)
        assert result_msg.timestamp == payload_timestamp  # From da00 payload
        assert result_msg.stream.name == 'monitor_1'


class TestF144Serializer:
    def test_roundtrip_preserves_value_and_payload_timestamp(
        self,
    ) -> None:
        # Create test data with time coordinate
        time_ns = 1234567890123456789
        original_data = sc.DataArray(
            data=sc.array(dims=[], values=42.5, unit='m/s'),
            coords={'time': sc.scalar(time_ns, unit='ns')},
        )
        stream_id = StreamId(kind=StreamKind.LOG, name='test_log')
        payload_timestamp = 9876543210  # Different from time coord
        original_msg = Message(
            timestamp=payload_timestamp,  # This should NOT be used
            stream=stream_id,
            value=original_data,
        )

        # Serialize to f144
        serialized_bytes = serialize_dataarray_to_f144(original_msg)

        # Create adapter to deserialize back
        f144_adapter = KafkaToF144Adapter()

        # Create fake Kafka message with different timestamp
        kafka_timestamp = 5555555555  # Should be ignored
        kafka_msg = FakeKafkaMessage(
            value=serialized_bytes, topic='test_topic', timestamp=kafka_timestamp
        )

        # Deserialize back to f144 log data
        [result_msg] = f144_adapter.adapt(kafka_msg)

        # Verify the roundtrip preserved the value and used time coordinate as timestamp
        assert result_msg.value.value == 42.5  # Value preserved
        assert (
            result_msg.timestamp == time_ns
        )  # Uses time coord, not payload or Kafka timestamp
        assert result_msg.stream.name == 'test_log'  # From f144 source_name

    def test_serialize_dataarray_to_f144_with_array_data(self) -> None:
        # Test with array data
        time_ns = 9876543210
        original_data = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'time': sc.scalar(time_ns, unit='ns')},
        )
        stream_id = StreamId(kind=StreamKind.LOG, name='array_log')
        original_msg = Message(
            timestamp=1111111111,  # Should be ignored
            stream=stream_id,
            value=original_data,
        )

        # Serialize and deserialize
        serialized_bytes = serialize_dataarray_to_f144(original_msg)
        f144_adapter = KafkaToF144Adapter()
        kafka_msg = FakeKafkaMessage(
            value=serialized_bytes,
            topic='log_topic',
            timestamp=2222222222,  # Should be ignored
        )

        [result_msg] = f144_adapter.adapt(kafka_msg)

        # Verify array values are preserved and timestamp comes from time coordinate
        np.testing.assert_array_equal(result_msg.value.value, [1.0, 2.0, 3.0])
        assert result_msg.timestamp == time_ns
        assert result_msg.stream.name == 'array_log'

    def test_serialize_dataarray_to_f144_different_time_units(self) -> None:
        # Test with different time units (should be converted to ns)
        time_us = 1234567890123456  # microseconds
        original_data = sc.DataArray(
            data=sc.array(dims=[], values=7.5, unit='V'),
            coords={'time': sc.array(dims=[], values=time_us, unit='us')},
        )
        stream_id = StreamId(kind=StreamKind.LOG, name='time_unit_log')
        original_msg = Message(timestamp=0, stream=stream_id, value=original_data)

        # Serialize and deserialize
        serialized_bytes = serialize_dataarray_to_f144(original_msg)
        f144_adapter = KafkaToF144Adapter()
        kafka_msg = FakeKafkaMessage(
            value=serialized_bytes, topic='log_topic', timestamp=0
        )

        [result_msg] = f144_adapter.adapt(kafka_msg)

        # Time should be converted to nanoseconds
        expected_time_ns = time_us * 1000  # us to ns conversion
        assert result_msg.value.value == 7.5
        assert result_msg.timestamp == expected_time_ns

    def test_serialize_dataarray_to_f144_missing_time_coordinate_raises_error(
        self,
    ) -> None:
        # Test error handling when time coordinate is missing
        original_data = sc.DataArray(
            data=sc.array(dims=[], values=42.0, unit='K'),
            # No time coordinate
        )
        stream_id = StreamId(kind=StreamKind.LOG, name='no_time_log')
        original_msg = Message(
            timestamp=1234567890, stream=stream_id, value=original_data
        )

        # Should raise an error when trying to serialize without time coordinate
        with pytest.raises(KeyError):  # KeyError or similar
            serialize_dataarray_to_f144(original_msg)
