# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Service for managing workflow status updates from backend services.

Processes workflow status response messages and notifies subscribers of changes.
"""

import json
import logging
from collections import defaultdict
from collections.abc import Callable

import pydantic

from ess.livedata.config.workflow_spec import WorkflowStatus
from ess.livedata.kafka.message_adapter import RawConfigItem


class WorkflowConfigService:
    """
    Service for managing workflow status updates.

    Processes workflow status response messages from backend services and
    maintains subscription callbacks for status changes.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self._logger = logger or logging.getLogger(__name__)
        self._subscribers: dict[str, list[Callable[[WorkflowStatus], None]]] = (
            defaultdict(list)
        )
        self._last_status: dict[str, WorkflowStatus] = {}

    def subscribe_to_workflow_status(
        self, source_name: str, callback: Callable[[WorkflowStatus], None]
    ) -> None:
        """Subscribe to workflow status updates for a source."""
        self._subscribers[source_name].append(callback)

        # If we already have status for this source, call immediately
        if source_name in self._last_status:
            callback(self._last_status[source_name])

    def process_response(self, raw_item: RawConfigItem) -> None:
        """
        Process a single config response message item.

        Parameters
        ----------
        raw_item:
            The raw config item value from the response message.
        """
        try:
            # Decode message
            key_str = raw_item.key.decode('utf-8')
            value_dict = json.loads(raw_item.value.decode('utf-8'))

            # Parse ConfigKey
            from ess.livedata.config.models import ConfigKey

            config_key = ConfigKey.from_string(key_str)

            # Check if this is a workflow_status message
            if config_key.key == "workflow_status":
                source_name = config_key.source_name
                if source_name is None:
                    return

                # Validate to WorkflowStatus
                try:
                    status = WorkflowStatus.model_validate(value_dict)
                except pydantic.ValidationError as e:
                    self._logger.error(
                        "Invalid workflow status for %s: %s", source_name, e
                    )
                    return

                # Check if value changed
                old_status = self._last_status.get(source_name)
                if old_status == status:
                    return

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
            self._logger.exception("Error processing config response")
