# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Service for publishing commands and configuration updates.
"""

import time

import structlog

from ess.livedata.core.job_manager import Command
from ess.livedata.core.message import COMMANDS_STREAM_ID, Message, MessageSink
from ess.livedata.core.timestamp import Timestamp

logger = structlog.get_logger(__name__)


class CommandService:
    """Publishes :class:`Command` messages to backend services."""

    def __init__(self, sink: MessageSink[Command]):
        self._sink = sink

    def send(self, command: Command) -> None:
        """Send a single command."""
        self.send_batch([command])

    def send_batch(self, commands: list[Command]) -> None:
        """Send multiple commands in a single Kafka flush."""
        if not commands:
            return
        messages = [
            Message(
                stream=COMMANDS_STREAM_ID,
                timestamp=Timestamp.from_ns(time.time_ns()),
                value=command,
            )
            for command in commands
        ]
        self._sink.publish_messages(messages)
        logger.debug("Sent %d command(s)", len(commands))
