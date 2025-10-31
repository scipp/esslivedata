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

    def save_config(self, config_id: WorkflowId, config: PersistedUIConfig) -> None:
        """
        Save configuration for persistence across sessions.

        Parameters
        ----------
        config_id:
            Unique identifier (workflow ID or plotter ID).
        config:
            Configuration to persist, including source names and parameters.
        """
        ...

    def load_config(self, config_id: WorkflowId) -> PersistedUIConfig | None:
        """
        Load persisted configuration.

        Parameters
        ----------
        config_id:
            Unique identifier (workflow ID or plotter ID).

        Returns
        -------
        :
            The persisted configuration, or None if not found.
        """
        ...

    def remove_not_in_set(self, valid_ids: set[WorkflowId]) -> None:
        """
        Remove configurations whose IDs are not in the valid set.

        Parameters
        ----------
        valid_ids:
            Set of IDs to keep. All other configs will be removed.
        """
        ...

    def remove_oldest(self, max_configs: int, cleanup_fraction: float = 0.1) -> None:
        """
        Remove oldest configurations when limit is exceeded.

        Parameters
        ----------
        max_configs:
            Maximum number of configs to keep.
        cleanup_fraction:
            Fraction of configs to remove when limit exceeded (default 0.1 = 10%).
        """
        ...


class InMemoryConfigStore(ConfigStore):
    """Simple in-memory implementation of ConfigStore for testing and development."""

    def __init__(self):
        self._configs: dict[WorkflowId, PersistedUIConfig] = {}

    def save_config(self, config_id: WorkflowId, config: PersistedUIConfig) -> None:
        """Save configuration in memory."""
        self._configs[config_id] = config

    def load_config(self, config_id: WorkflowId) -> PersistedUIConfig | None:
        """Load configuration from memory."""
        return self._configs.get(config_id)

    def remove_not_in_set(self, valid_ids: set[WorkflowId]) -> None:
        """Remove configurations whose IDs are not in the valid set."""
        missing_ids = set(self._configs.keys()) - valid_ids
        for config_id in missing_ids:
            del self._configs[config_id]

    def remove_oldest(self, max_configs: int, cleanup_fraction: float = 0.1) -> None:
        """Remove oldest configurations when limit exceeded."""
        if len(self._configs) <= max_configs:
            return

        num_to_remove = int(len(self._configs) * cleanup_fraction)
        if num_to_remove == 0:
            num_to_remove = 1

        # Remove oldest configs (dict maintains insertion order)
        oldest_keys = list(self._configs.keys())[:num_to_remove]
        for key in oldest_keys:
            del self._configs[key]
