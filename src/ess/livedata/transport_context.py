# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Global transport context for dependency injection.

This module provides a way for the launcher to inject an in-memory broker
into services without modifying their command-line interfaces.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ess.livedata.in_memory import InMemoryBroker

# Thread-local storage for broker instance
_context = threading.local()


def set_broker(broker: InMemoryBroker | None) -> None:
    """
    Set the in-memory broker for the current thread.

    This is called by the launcher when --transport in-memory is used.
    Services will automatically use this broker if it's set.

    Parameters
    ----------
    broker:
        InMemoryBroker instance, or None to use Kafka transport
    """
    _context.broker = broker


def get_broker() -> InMemoryBroker | None:
    """
    Get the in-memory broker for the current thread.

    Returns None if no broker is set (use Kafka transport).

    Returns
    -------
    :
        InMemoryBroker instance or None
    """
    return getattr(_context, 'broker', None)
