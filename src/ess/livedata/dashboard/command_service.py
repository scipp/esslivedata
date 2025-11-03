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
        self.send_batch([(key, value)])

    def send_batch(self, commands: list[tuple[ConfigKey, Any]]) -> None:
        """
        Send multiple commands or configuration updates in a single batch.

        Batching multiple commands into a single call is more efficient than
        sending them individually, as it requires only one Kafka flush operation.

        Parameters
        ----------
        commands:
            List of (key, value) tuples where each key is a ConfigKey identifying
            the command/config type and target, and value is the command or
            configuration value to send.
        """
        messages = [
            Message(
                stream=COMMANDS_STREAM_ID,
                timestamp=time.time_ns(),
                value=ConfigUpdate(config_key=key, value=val),
            )
            for key, val in commands
        ]
        self._sink.publish_messages(messages)
        self._logger.debug("Sent %d command(s)", len(commands))
