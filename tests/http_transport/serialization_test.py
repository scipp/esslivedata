# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.http_transport.serialization import (
    DA00MessageSerializer,
    GenericJSONMessageSerializer,
)


def test_da00_serializer_roundtrip():
    """Test that DA00MessageSerializer can serialize and deserialize messages."""
    serializer = DA00MessageSerializer()

    da = sc.DataArray(
        data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
        coords={'x': sc.array(dims=['x'], values=[0, 1, 2])},
    )

    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test_stream')
    messages = [
        Message(timestamp=1000, stream=stream, value=da),
        Message(timestamp=2000, stream=stream, value=da * 2),
    ]

    serialized = serializer.serialize(messages)
    deserialized = serializer.deserialize(serialized)

    assert len(deserialized) == 2
    assert deserialized[0].timestamp == 1000
    assert deserialized[0].stream == stream
    assert sc.identical(deserialized[0].value, da)
    assert deserialized[1].timestamp == 2000
    assert sc.identical(deserialized[1].value, da * 2)


def test_generic_json_serializer_roundtrip():
    """Test that GenericJSONMessageSerializer handles simple types."""
    serializer = GenericJSONMessageSerializer()

    stream = StreamId(kind=StreamKind.LIVEDATA_CONFIG, name='config_stream')
    messages = [
        Message(timestamp=1000, stream=stream, value={'key': 'value1'}),
        Message(timestamp=2000, stream=stream, value={'key': 'value2', 'count': 42}),
    ]

    serialized = serializer.serialize(messages)
    deserialized = serializer.deserialize(serialized)

    assert len(deserialized) == 2
    assert deserialized[0].timestamp == 1000
    assert deserialized[0].stream == stream
    assert deserialized[0].value == {'key': 'value1'}
    assert deserialized[1].timestamp == 2000
    assert deserialized[1].value == {'key': 'value2', 'count': 42}


def test_da00_serializer_empty_list():
    """Test serialization of empty message list."""
    serializer = DA00MessageSerializer()

    serialized = serializer.serialize([])
    deserialized = serializer.deserialize(serialized)

    assert deserialized == []
