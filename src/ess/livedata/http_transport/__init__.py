# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""HTTP-based transport layer for message passing."""

from .app import MessageAPIWrapper, create_message_api, create_multi_endpoint_api
from .serialization import (
    ConfigMessageSerializer,
    DA00MessageSerializer,
    GenericJSONMessageSerializer,
    MessageSerializer,
    RoutingMessageSerializer,
    StatusMessageSerializer,
)
from .service import HTTPMultiEndpointSink
from .sink import QueueBasedMessageSink
from .source import HTTPMessageSource, MultiHTTPSource

__all__ = [
    'ConfigMessageSerializer',
    'DA00MessageSerializer',
    'GenericJSONMessageSerializer',
    'HTTPMessageSource',
    'HTTPMultiEndpointSink',
    'MessageAPIWrapper',
    'MessageSerializer',
    'MultiHTTPSource',
    'QueueBasedMessageSink',
    'RoutingMessageSerializer',
    'StatusMessageSerializer',
    'create_message_api',
    'create_multi_endpoint_api',
]
