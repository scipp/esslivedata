# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""HTTP-based transport layer for message passing."""

from .app import MessageAPIWrapper, create_message_api
from .serialization import (
    DA00MessageSerializer,
    GenericJSONMessageSerializer,
    MessageSerializer,
)
from .service import HTTPServiceSink, http_service_sink
from .sink import QueueBasedMessageSink
from .source import HTTPMessageSource

__all__ = [
    'DA00MessageSerializer',
    'GenericJSONMessageSerializer',
    'HTTPMessageSource',
    'HTTPServiceSink',
    'MessageAPIWrapper',
    'MessageSerializer',
    'QueueBasedMessageSink',
    'create_message_api',
    'http_service_sink',
]
