# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import structlog

from ..config.acknowledgement import CommandAcknowledgement
from ..config.models import ConfigKey
from ..core.job_manager import JobCommand
from ..core.job_manager_adapter import JobManagerAdapter
from ..core.message import RESPONSES_STREAM_ID, Message
from ..kafka.message_adapter import RawConfigItem

logger = structlog.get_logger(__name__)

# Re-export for backwards compatibility with tests
__all__ = ['ConfigProcessor', 'ConfigUpdate']


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
    Simple config processor that handles workflow_config and job_command messages
    by delegating directly to JobManagerAdapter.
    """

    def __init__(
        self,
        *,
        job_manager_adapter: JobManagerAdapter,
    ) -> None:
        self._job_manager_adapter = job_manager_adapter
        self._actions = {
            'workflow_config': self._job_manager_adapter.set_workflow_with_config,
            JobCommand.key: self._job_manager_adapter.job_command,
        }

    def process_messages(
        self, messages: list[Message[RawConfigItem]]
    ) -> list[Message[CommandAcknowledgement]]:
        """
        Process config messages and handle workflow_config and job_command updates.

        Parameters
        ----------
        messages:
            List of messages containing configuration updates

        Returns
        -------
        :
            List of response messages containing CommandAcknowledgements.
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

                logger.info(
                    'processing_config_message',
                    source_name=source_name,
                    config_key=config_key,
                    value=value,
                )

                if source_name is None:
                    # source_name=None overrides all previous source-specific updates
                    latest_updates[config_key].clear()

                latest_updates[config_key][source_name] = update

            except Exception:
                logger.exception('error_processing_config_message')

        # Process the latest updates
        response_messages: list[Message[CommandAcknowledgement]] = []

        for config_key, source_updates in latest_updates.items():
            for source_name, update in source_updates.items():
                logger.debug(
                    'processing_config_key', config_key=config_key, source=source_name
                )
                try:
                    if (action := self._actions.get(config_key)) is None:
                        logger.debug('unknown_config_key', config_key=config_key)
                        continue
                    result = action(source_name, update.value)
                    if result is not None:
                        response_messages.append(
                            Message(stream=RESPONSES_STREAM_ID, value=result)
                        )

                except Exception:
                    logger.exception(
                        'error_processing_config_key',
                        config_key=config_key,
                        source=source_name,
                    )

        return response_messages
