# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Protocol for managing workflow configurations.

This protocol defines the interface for workflow controller runtime communication
with backend services via Kafka. For persistent UI state storage, see ConfigStore
protocol instead.
"""

from collections.abc import Callable
from typing import Protocol

from ess.livedata.config.workflow_spec import (
    WorkflowConfig,
    WorkflowStatus,
)


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
