# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Kafka-based service for sending job commands.

Uses backend MessageSink abstraction directly for job command publishing.
"""

import logging
import time

from ess.livedata.config.models import ConfigKey
from ess.livedata.core.job_manager import JobCommand
from ess.livedata.core.message import COMMANDS_STREAM_ID, Message, MessageSink
from ess.livedata.handlers.config_handler import ConfigUpdate


class KafkaJobCommandService:
    """
    Kafka-based service for publishing job commands.

    Provides a focused interface for JobController to send commands without
    depending on the full ConfigService.
    """

    def __init__(
        self, sink: MessageSink[ConfigUpdate], logger: logging.Logger | None = None
    ):
        self._sink = sink
        self._logger = logger or logging.getLogger(__name__)

    def send_command(self, key: ConfigKey, command: JobCommand) -> None:
        """Send a job command to backend services."""
        update = ConfigUpdate(config_key=key, value=command)
        msg = Message(stream=COMMANDS_STREAM_ID, timestamp=time.time_ns(), value=update)
        self._sink.publish_messages([msg])
        self._logger.debug("Sent job command %s for key %s", command.action, key)
