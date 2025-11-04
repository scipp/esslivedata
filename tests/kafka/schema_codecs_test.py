# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for schema codecs and the new message architecture."""

import scipp as sc
from streaming_data_types import dataarray_da00

from ess.livedata.core.message import Message, SchemaMessage, StreamId, StreamKind
from ess.livedata.kafka.schema_codecs import (
    CompoundDomainConverter,
    CompoundSchemaSerializer,
    Da00Deserializer,
    Da00Serializer,
    ScippDa00Converter,
    make_standard_converter,
    make_standard_serializer,
)


def test_da00_serializer_roundtrip():
    """Test that Da00Serializer can serialize SchemaMessage to bytes."""
    # Create a schema message with da00 data
    da = sc.DataArray(
        sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='m'),
        coords={'x': sc.arange('x', 3, unit='m')},
    )

    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test_stream')
    converter = ScippDa00Converter()
    domain_msg = Message(stream=stream, value=da, timestamp=1000)

    # Convert to schema message
    schema_msg = converter.to_schema(domain_msg)

    # Serialize
    serializer = Da00Serializer()
    serialized_bytes = serializer.serialize(schema_msg)

    # Verify it's bytes
    assert isinstance(serialized_bytes, bytes)
    assert len(serialized_bytes) > 0

    # Deserialize
    deserializer = Da00Deserializer(stream_kind=StreamKind.LIVEDATA_DATA)
    deserialized_schema_msg = deserializer.deserialize(
        serialized_bytes, topic='test_topic', _timestamp=1000, key=None
    )

    # Verify stream
    assert deserialized_schema_msg.stream.kind == StreamKind.LIVEDATA_DATA
    assert deserialized_schema_msg.stream.name == 'test_stream'
    assert deserialized_schema_msg.timestamp == 1000

    # Convert back to domain
    recovered_msg = converter.to_domain(deserialized_schema_msg)

    # Verify the data
    assert sc.identical(recovered_msg.value, da)


def test_scipp_da00_converter_bidirectional():
    """Test ScippDa00Converter can convert both ways."""
    # Create domain message
    da = sc.DataArray(
        sc.array(dims=['y'], values=[10.0, 20.0], unit='counts'),
        coords={'y': sc.array(dims=['y'], values=[1.0, 2.0], unit='s')},
        name='test_data',
    )

    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='detector_1')
    domain_msg = Message(stream=stream, value=da, timestamp=5000)

    # Convert to schema
    converter = ScippDa00Converter()
    schema_msg = converter.to_schema(domain_msg)

    # Verify schema message structure
    assert isinstance(schema_msg, SchemaMessage)
    assert schema_msg.stream == stream
    assert schema_msg.timestamp == 5000
    assert schema_msg.key is None
    assert isinstance(schema_msg.data, list)
    assert all(isinstance(v, dataarray_da00.Variable) for v in schema_msg.data)

    # Convert back to domain
    recovered_msg = converter.to_domain(schema_msg)

    # Verify recovered message
    assert recovered_msg.stream == stream
    assert recovered_msg.timestamp == 5000
    assert sc.identical(recovered_msg.value, da)


def test_compound_schema_serializer():
    """Test CompoundSchemaSerializer dispatches to correct serializer."""
    da = sc.DataArray(
        sc.array(dims=['x'], values=[1.0, 2.0], unit='m'),
        coords={'x': sc.arange('x', 2, unit='m')},
    )

    # Create compound serializer
    compound = CompoundSchemaSerializer()
    da00_serializer = Da00Serializer()
    compound.register(StreamKind.LIVEDATA_DATA, da00_serializer)

    # Convert domain message to schema message
    converter = ScippDa00Converter()
    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test')
    domain_msg = Message(stream=stream, value=da, timestamp=1000)
    schema_msg = converter.to_schema(domain_msg)

    # Serialize using compound serializer
    serialized = compound.serialize(schema_msg)

    assert isinstance(serialized, bytes)
    assert len(serialized) > 0


def test_compound_schema_serializer_missing_kind():
    """Test CompoundSchemaSerializer raises error for unregistered StreamKind."""
    import pytest

    compound = CompoundSchemaSerializer()
    # Only register LIVEDATA_DATA
    compound.register(StreamKind.LIVEDATA_DATA, Da00Serializer())

    # Try to serialize a LOG stream
    stream = StreamId(kind=StreamKind.LOG, name='test')
    schema_msg = SchemaMessage(stream=stream, data=[], timestamp=1000, key=None)

    with pytest.raises(
        ValueError, match=r"No serializer registered for StreamKind.*LOG"
    ):
        compound.serialize(schema_msg)


def test_compound_domain_converter():
    """Test CompoundDomainConverter dispatches to correct converter."""
    da = sc.DataArray(
        sc.array(dims=['x'], values=[1.0, 2.0], unit='m'),
        coords={'x': sc.arange('x', 2, unit='m')},
    )

    # Create compound converter
    compound = CompoundDomainConverter()
    converter = ScippDa00Converter()
    compound.register(StreamKind.LIVEDATA_DATA, converter)

    # Convert domain to schema
    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test')
    domain_msg = Message(stream=stream, value=da, timestamp=1000)
    schema_msg = compound.to_schema(domain_msg)

    assert isinstance(schema_msg, SchemaMessage)
    assert schema_msg.stream == stream
    assert schema_msg.timestamp == 1000

    # Convert back to domain
    recovered = compound.to_domain(schema_msg)
    assert sc.identical(recovered.value, da)


def test_make_standard_serializer():
    """Test make_standard_serializer creates serializer for common stream kinds."""
    compound = make_standard_serializer()

    # Test da00 stream kinds
    da = sc.DataArray(sc.array(dims=['x'], values=[1.0], unit='m'))
    da00_converter = ScippDa00Converter()

    for kind in [
        StreamKind.LIVEDATA_DATA,
        StreamKind.LIVEDATA_ROI,
    ]:
        stream = StreamId(kind=kind, name='test')
        domain_msg = Message(stream=stream, value=da, timestamp=1000)
        schema_msg = da00_converter.to_schema(domain_msg)

        # Should not raise error
        serialized = compound.serialize(schema_msg)
        assert isinstance(serialized, bytes)


def test_make_standard_converter():
    """Test make_standard_converter creates converter for common stream kinds."""
    compound = make_standard_converter()

    # Verify common stream kinds are registered
    da = sc.DataArray(sc.array(dims=['x'], values=[1.0], unit='m'))

    for kind in [
        StreamKind.LIVEDATA_DATA,
        StreamKind.LIVEDATA_ROI,
    ]:
        stream = StreamId(kind=kind, name='test')
        domain_msg = Message(stream=stream, value=da, timestamp=1000)

        # Should not raise error
        schema_msg = compound.to_schema(domain_msg)
        assert isinstance(schema_msg, SchemaMessage)

        # And should be able to convert back
        recovered = compound.to_domain(schema_msg)
        assert sc.identical(recovered.value, da)
