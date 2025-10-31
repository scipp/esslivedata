# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Kafka-based implementation of WorkflowConfigService.

Uses backend MessageSource/MessageSink abstractions directly for workflow
configuration management.
"""

import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable

import pydantic
from confluent_kafka import KafkaError

import ess.livedata.config.keys as keys
from ess.livedata.config.workflow_spec import WorkflowConfig, WorkflowStatus
from ess.livedata.core.message import (
    COMMANDS_STREAM_ID,
    Message,
    MessageSink,
    MessageSource,
)
from ess.livedata.dashboard.workflow_config_service import WorkflowConfigService
from ess.livedata.handlers.config_handler import ConfigUpdate


class KafkaWorkflowConfigService(WorkflowConfigService):
    """
    Kafka-based implementation of WorkflowConfigService.

    Publishes workflow configurations and subscribes to workflow status updates
    using backend MessageSource/MessageSink abstractions.
    """

    def __init__(
        self,
        source: MessageSource,
        sink: MessageSink[ConfigUpdate],
        logger: logging.Logger | None = None,
    ):
        self._source = source
        self._sink = sink
        self._logger = logger or logging.getLogger(__name__)
        self._subscribers: dict[str, list[Callable[[WorkflowStatus], None]]] = (
            defaultdict(list)
        )
        self._last_status: dict[str, WorkflowStatus] = {}

    def send_workflow_config(self, source_name: str, config: WorkflowConfig) -> None:
        """Send workflow configuration to backend services."""
        config_key = keys.WORKFLOW_CONFIG.create_key(source_name=source_name)
        update = ConfigUpdate(config_key=config_key, value=config)
        msg = Message(stream=COMMANDS_STREAM_ID, timestamp=time.time_ns(), value=update)
        self._sink.publish_messages([msg])
        self._logger.debug("Sent workflow config for source %s", source_name)

    def subscribe_to_workflow_status(
        self, source_name: str, callback: Callable[[WorkflowStatus], None]
    ) -> None:
        """Subscribe to workflow status updates for a source."""
        self._subscribers[source_name].append(callback)

        # If we already have status for this source, call immediately
        if source_name in self._last_status:
            callback(self._last_status[source_name])

    def process_incoming_messages(self) -> None:
        """Process incoming workflow status messages from backend."""
        messages = self._source.get_messages()

        for kafka_msg in messages:
            try:
                error = kafka_msg.error()
                if error is not None:
                    if error.code() == KafkaError._PARTITION_EOF:
                        continue
                    self._logger.error("Consumer error: %s", error)
                    continue

                # Decode message
                key_str = kafka_msg.key().decode('utf-8')
                value_dict = json.loads(kafka_msg.value().decode('utf-8'))

                # Parse ConfigKey
                from ess.livedata.config.models import ConfigKey

                config_key = ConfigKey.from_string(key_str)

                # Check if this is a workflow_status message
                if config_key.key == "workflow_status":
                    source_name = config_key.source_name
                    if source_name is None:
                        continue

                    # Validate to WorkflowStatus
                    try:
                        status = WorkflowStatus.model_validate(value_dict)
                    except pydantic.ValidationError as e:
                        self._logger.error(
                            "Invalid workflow status for %s: %s", source_name, e
                        )
                        continue

                    # Check if value changed
                    old_status = self._last_status.get(source_name)
                    if old_status == status:
                        continue

                    # Update cache
                    self._last_status[source_name] = status

                    # Notify subscribers
                    for callback in self._subscribers.get(source_name, []):
                        try:
                            callback(status)
                        except Exception:
                            self._logger.exception(
                                "Error in workflow status callback for %s", source_name
                            )

            except Exception:
                self._logger.exception("Error processing incoming message")
