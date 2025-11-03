# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Service for publishing commands and configuration updates.

Unified service for sending job commands, workflow configurations, and other
command messages to backend services via MessageSink abstraction.
"""

import logging
import time
from typing import Any

from ess.livedata.config.models import ConfigKey
from ess.livedata.core.message import COMMANDS_STREAM_ID, Message, MessageSink
from ess.livedata.handlers.config_handler import ConfigUpdate


class CommandService:
    """
    Service for publishing commands and configuration updates to backend services.

    Provides a unified interface for sending any type of command or configuration
    to backend services via the COMMANDS_STREAM_ID.
    """

    def __init__(
        self, sink: MessageSink[ConfigUpdate], logger: logging.Logger | None = None
    ):
        self._sink = sink
        self._logger = logger or logging.getLogger(__name__)

    def send(self, key: ConfigKey, value: Any) -> None:
        """
        Send a command or configuration update to backend services.

        Parameters
        ----------
        key:
            The configuration key identifying the command/config type and target.
        value:
            The command or configuration value to send.
        """
        update = ConfigUpdate(config_key=key, value=value)
        msg = Message(stream=COMMANDS_STREAM_ID, timestamp=time.time_ns(), value=update)
        self._sink.publish_messages([msg])
        self._logger.debug("Sent command for key %s", key)
