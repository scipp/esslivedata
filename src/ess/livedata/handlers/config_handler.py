# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ..config.models import ConfigKey
from ..core.job_manager import JobCommand
from ..core.job_manager_adapter import JobManagerAdapter
from ..core.message import RESPONSES_STREAM_ID, Message
from ..kafka.message_adapter import RawConfigItem


@dataclass
class ConfigUpdate:
    config_key: ConfigKey
    value: Any

    @property
    def source_name(self) -> str | None:
        return self.config_key.source_name

    @property
    def service_name(self) -> str | None:
        return self.config_key.service_name

    @property
    def key(self) -> str:
        return self.config_key.key

    @staticmethod
    def from_raw(item: RawConfigItem) -> ConfigUpdate:
        config_key = ConfigKey.from_string(item.key.decode('utf-8'))
        value = json.loads(item.value.decode('utf-8'))
        return ConfigUpdate(config_key=config_key, value=value)


class ConfigProcessor:
    """
    Simple config processor that handles workflow_config and start_time messages
    by delegating directly to JobManagerAdapter.
    """

    def __init__(
        self,
        *,
        job_manager_adapter: JobManagerAdapter,
        logger: logging.Logger | None = None,
    ) -> None:
        self._job_manager_adapter = job_manager_adapter
        self._actions = {
            'workflow_config': self._job_manager_adapter.set_workflow_with_config,
            JobCommand.key: self._job_manager_adapter.job_command,
        }
        self._logger = logger or logging.getLogger(__name__)

    def process_messages(
        self, messages: list[Message[RawConfigItem]]
    ) -> list[Message[Any]]:
        """
        Process config messages and handle workflow_config and start_time updates.

        Parameters
        ----------
        messages:
            List of messages containing configuration updates

        Returns
        -------
        :
            List of response messages
        """
        # Group latest updates by key and source
        latest_updates: defaultdict[str, dict[str | None, ConfigUpdate]] = defaultdict(
            dict
        )

        for message in messages:
            try:
                update = ConfigUpdate.from_raw(message.value)
                source_name = update.source_name
                config_key = update.key
                value = update.value

                self._logger.info(
                    'Processing config message for source_name = %s: %s = %s',
                    source_name,
                    config_key,
                    value,
                )

                if source_name is None:
                    # source_name=None overrides all previous source-specific updates
                    latest_updates[config_key].clear()

                latest_updates[config_key][source_name] = update

            except Exception:
                self._logger.exception('Error processing config message')

        # Process the latest updates
        response_messages: list[Message[Any]] = []

        for config_key, source_updates in latest_updates.items():
            for source_name, update in source_updates.items():
                self._logger.debug(
                    'Processing config key %s for source %s', config_key, source_name
                )
                try:
                    if (action := self._actions.get(config_key)) is None:
                        # Echo back unrecognized config keys for dashboard persistence
                        self._logger.debug(
                            'Unknown config key: %s - echoing back', config_key
                        )
                        response_messages.append(
                            Message(stream=RESPONSES_STREAM_ID, value=update)
                        )
                        continue
                    results = action(source_name, update.value)
                    # Convert results to messages
                    updates = [ConfigUpdate(*result) for result in results or []]
                    response_messages.extend(
                        Message(stream=RESPONSES_STREAM_ID, value=up) for up in updates
                    )

                except Exception:
                    self._logger.exception(
                        'Error processing config key %s for source %s',
                        config_key,
                        source_name,
                    )

        return response_messages
