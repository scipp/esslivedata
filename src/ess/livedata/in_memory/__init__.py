# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
In-memory message transport for testing and development.

NOT FOR PRODUCTION USE - use Kafka transport for production deployments.
"""

from .broker import InMemoryBroker
from .sink import InMemoryMessageSink, NullMessageSink
from .source import InMemoryMessageSource

__all__ = [
    "InMemoryBroker",
    "InMemoryMessageSource",
    "InMemoryMessageSink",
    "NullMessageSink",
]
