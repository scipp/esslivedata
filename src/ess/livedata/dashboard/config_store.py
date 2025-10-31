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

from typing import Any, Protocol

from ess.livedata.config.workflow_spec import WorkflowId


class ConfigStore(Protocol):
    """Protocol for persisting dashboard UI configuration state."""

    def save_config(self, config_id: WorkflowId, config: dict[str, Any]) -> None:
        """
        Save configuration for persistence across sessions.

        Parameters
        ----------
        config_id:
            Unique identifier (WorkflowId for workflows, or synthetic WorkflowId
            for plotters).
        config:
            Configuration to persist as a JSON-serializable dict.
        """
        ...

    def load_config(self, config_id: WorkflowId) -> dict[str, Any] | None:
        """
        Load persisted configuration.

        Parameters
        ----------
        config_id:
            Unique identifier (WorkflowId for workflows, or synthetic WorkflowId
            for plotters).

        Returns
        -------
        :
            The persisted configuration dict, or None if not found.
        """
        ...


class InMemoryConfigStore(ConfigStore):
    """
    In-memory implementation of ConfigStore with optional LRU eviction.

    Parameters
    ----------
    max_configs:
        Maximum number of configurations to keep. When exceeded, oldest
        configurations are automatically evicted. If None, no limit is enforced.
    cleanup_fraction:
        Fraction of configs to remove when limit exceeded (default 0.2 = 20%).
    """

    def __init__(
        self, max_configs: int | None = None, cleanup_fraction: float = 0.2
    ) -> None:
        self._configs: dict[WorkflowId, dict[str, Any]] = {}
        self._max_configs = max_configs
        self._cleanup_fraction = cleanup_fraction

    def save_config(self, config_id: WorkflowId, config: dict[str, Any]) -> None:
        """Save configuration in memory with automatic LRU eviction."""
        self._configs[config_id] = config

        # Automatic LRU eviction if limit exceeded
        if self._max_configs and len(self._configs) > self._max_configs:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Remove oldest configs based on cleanup fraction."""
        num_to_remove = max(1, int(len(self._configs) * self._cleanup_fraction))
        # Dict maintains insertion order in Python 3.7+
        oldest_keys = list(self._configs.keys())[:num_to_remove]
        for key in oldest_keys:
            del self._configs[key]

    def load_config(self, config_id: WorkflowId) -> dict[str, Any] | None:
        """Load configuration from memory."""
        return self._configs.get(config_id)
