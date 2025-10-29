# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from unittest.mock import Mock

import pytest
import requests
import scipp as sc

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.http_transport.serialization import DA00MessageSerializer
from ess.livedata.http_transport.source import HTTPMessageSource


@pytest.fixture
def serializer():
    return DA00MessageSerializer()


@pytest.fixture
def mock_session(monkeypatch):
    """Mock the requests.Session to avoid real HTTP calls."""
    mock = Mock()
    monkeypatch.setattr('requests.Session', lambda: mock)
    return mock


def test_source_polls_and_deserializes(serializer, mock_session):
    """Test that HTTPMessageSource polls endpoint and deserializes messages."""
    stream = StreamId(kind=StreamKind.LIVEDATA_DATA, name='test')
    messages = [
        Message(timestamp=1000, stream=stream, value=sc.DataArray(sc.scalar(1.0))),
        Message(timestamp=2000, stream=stream, value=sc.DataArray(sc.scalar(2.0))),
    ]

    serialized = serializer.serialize(messages)

    response = Mock()
    response.status_code = 200
    response.content = serialized
    mock_session.get.return_value = response

    source = HTTPMessageSource(base_url='http://localhost:8000', serializer=serializer)

    retrieved = source.get_messages()

    assert len(retrieved) == 2
    assert retrieved[0].timestamp == 1000
    assert retrieved[1].timestamp == 2000
    mock_session.get.assert_called_once()


def test_source_returns_empty_on_no_content(serializer, mock_session):
    """Test that source returns empty list on 204 No Content."""
    response = Mock()
    response.status_code = 204
    response.content = b''
    mock_session.get.return_value = response

    source = HTTPMessageSource(base_url='http://localhost:8000', serializer=serializer)

    retrieved = source.get_messages()

    assert retrieved == []


def test_source_handles_timeout_gracefully(serializer, mock_session):
    """Test that timeouts are handled without raising."""
    mock_session.get.side_effect = requests.Timeout()

    source = HTTPMessageSource(base_url='http://localhost:8000', serializer=serializer)

    retrieved = source.get_messages()

    assert retrieved == []


def test_source_handles_connection_error(serializer, mock_session):
    """Test that connection errors are handled without raising."""
    mock_session.get.side_effect = requests.ConnectionError()

    source = HTTPMessageSource(base_url='http://localhost:8000', serializer=serializer)

    retrieved = source.get_messages()

    assert retrieved == []


def test_source_context_manager(serializer, mock_session):
    """Test that source can be used as context manager."""
    response = Mock()
    response.status_code = 204
    response.content = b''
    mock_session.get.return_value = response

    with HTTPMessageSource(
        base_url='http://localhost:8000', serializer=serializer
    ) as source:
        source.get_messages()

    mock_session.close.assert_called_once()
