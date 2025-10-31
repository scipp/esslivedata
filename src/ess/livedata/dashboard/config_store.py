# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Abstraction for persisting dashboard UI state.

This module provides the ConfigStore protocol for storing dashboard preferences
and UI state across sessions. This is separate from ConfigService, which handles
runtime communication with backend services via Kafka.

ConfigStore is for persistent storage (e.g., files, local storage) while
ConfigService is for transient runtime coordination.
"""

from typing import Protocol

from ess.livedata.config.workflow_spec import PersistedUIConfig, WorkflowId


class ConfigStore(Protocol):
    """Protocol for persisting dashboard UI configuration state."""

    def save_workflow_config(
        self, workflow_id: WorkflowId, config: PersistedUIConfig
    ) -> None:
        """
        Save workflow configuration for persistence across sessions.

        Parameters
        ----------
        workflow_id:
            Unique identifier for the workflow.
        config:
            Configuration to persist, including source names and parameters.
        """
        ...

    def load_workflow_config(self, workflow_id: WorkflowId) -> PersistedUIConfig | None:
        """
        Load persisted workflow configuration.

        Parameters
        ----------
        workflow_id:
            Unique identifier for the workflow.

        Returns
        -------
        :
            The persisted configuration, or None if not found.
        """
        ...

    def save_plotter_config(
        self, plotter_id: WorkflowId, config: PersistedUIConfig
    ) -> None:
        """
        Save plotter configuration for persistence across sessions.

        Parameters
        ----------
        plotter_id:
            Unique identifier for the plotter (derived from workflow + output + plot).
        config:
            Configuration to persist, including source names and parameters.
        """
        ...

    def load_plotter_config(self, plotter_id: WorkflowId) -> PersistedUIConfig | None:
        """
        Load persisted plotter configuration.

        Parameters
        ----------
        plotter_id:
            Unique identifier for the plotter.

        Returns
        -------
        :
            The persisted configuration, or None if not found.
        """
        ...

    def cleanup_missing_workflows(self, current_workflow_ids: set[WorkflowId]) -> None:
        """
        Remove configurations for workflows that no longer exist.

        Parameters
        ----------
        current_workflow_ids:
            Set of currently available workflow IDs.
        """
        ...

    def cleanup_old_plotter_configs(
        self, max_configs: int, cleanup_fraction: float = 0.1
    ) -> None:
        """
        Remove oldest plotter configurations when limit is exceeded.

        Parameters
        ----------
        max_configs:
            Maximum number of plotter configs to keep.
        cleanup_fraction:
            Fraction of configs to remove when limit exceeded (default 0.1 = 10%).
        """
        ...


class InMemoryConfigStore(ConfigStore):
    """Simple in-memory implementation of ConfigStore for testing and development."""

    def __init__(self):
        self._workflow_configs: dict[WorkflowId, PersistedUIConfig] = {}
        self._plotter_configs: dict[WorkflowId, PersistedUIConfig] = {}

    def save_workflow_config(
        self, workflow_id: WorkflowId, config: PersistedUIConfig
    ) -> None:
        """Save workflow configuration in memory."""
        self._workflow_configs[workflow_id] = config

    def load_workflow_config(self, workflow_id: WorkflowId) -> PersistedUIConfig | None:
        """Load workflow configuration from memory."""
        return self._workflow_configs.get(workflow_id)

    def save_plotter_config(
        self, plotter_id: WorkflowId, config: PersistedUIConfig
    ) -> None:
        """Save plotter configuration in memory."""
        self._plotter_configs[plotter_id] = config

    def load_plotter_config(self, plotter_id: WorkflowId) -> PersistedUIConfig | None:
        """Load plotter configuration from memory."""
        return self._plotter_configs.get(plotter_id)

    def cleanup_missing_workflows(self, current_workflow_ids: set[WorkflowId]) -> None:
        """Remove configurations for workflows that no longer exist."""
        missing_ids = set(self._workflow_configs.keys()) - current_workflow_ids
        for workflow_id in missing_ids:
            del self._workflow_configs[workflow_id]

    def cleanup_old_plotter_configs(
        self, max_configs: int, cleanup_fraction: float = 0.1
    ) -> None:
        """Remove oldest plotter configurations when limit exceeded."""
        if len(self._plotter_configs) <= max_configs:
            return

        num_to_remove = int(len(self._plotter_configs) * cleanup_fraction)
        if num_to_remove == 0:
            num_to_remove = 1

        # Remove oldest configs (dict maintains insertion order)
        oldest_keys = list(self._plotter_configs.keys())[:num_to_remove]
        for key in oldest_keys:
            del self._plotter_configs[key]
