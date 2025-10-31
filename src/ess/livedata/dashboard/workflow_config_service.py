# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Provides a protocol and an adapter for managing workflow configurations.

This is not a real service but an adapter that wraps :py:class:`ConfigService` to
make it compatible with the :py:class:`WorkflowConfigService` protocol. This simplifies
the implementation and testing of :py:class:`WorkflowController`.
"""

from collections.abc import Callable
from typing import Protocol

import ess.livedata.config.keys as keys
from ess.livedata.config.workflow_spec import (
    WorkflowConfig,
    WorkflowStatus,
)

from .config_service import ConfigService


class WorkflowConfigService(Protocol):
    """
    Protocol for workflow controller runtime communication dependencies.

    This protocol handles runtime coordination with backend services via Kafka.
    For persistent UI state storage, see ConfigStore protocol instead.
    """

    def send_workflow_config(self, source_name: str, config: WorkflowConfig) -> None:
        """Send workflow configuration to a source."""
        ...

    def subscribe_to_workflow_status(
        self, source_name: str, callback: Callable[[WorkflowStatus], None]
    ) -> None:
        """Subscribe to workflow status updates for a source."""
        ...


class ConfigServiceAdapter(WorkflowConfigService):
    """
    Adapter to make ConfigService compatible with WorkflowConfigService protocol.
    """

    def __init__(self, config_service: ConfigService):
        self._config_service = config_service

    def send_workflow_config(self, source_name: str, config: WorkflowConfig) -> None:
        """Send workflow configuration to a source."""
        config_key = keys.WORKFLOW_CONFIG.create_key(source_name=source_name)
        self._config_service.update_config(config_key, config)

    def subscribe_to_workflow_status(
        self, source_name: str, callback: Callable[[WorkflowStatus], None]
    ) -> None:
        """Subscribe to workflow status updates for a source."""
        workflow_status_key = keys.WORKFLOW_STATUS.create_key(source_name=source_name)
        self._config_service.subscribe(workflow_status_key, callback)
