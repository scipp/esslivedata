# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Abstraction for persisting dashboard UI state.

This module provides the ConfigStore type alias for storing dashboard preferences
and UI state across sessions. This is separate from ConfigService, which handles
runtime communication with backend services via Kafka.

ConfigStore is for persistent storage (e.g., files, local storage) while
ConfigService is for transient runtime coordination.
"""

from collections.abc import MutableMapping
from typing import Any

from ess.livedata.config.workflow_spec import WorkflowId

# Type alias for config stores - any mutable mapping from WorkflowId to config dict
ConfigStore = MutableMapping[WorkflowId, dict[str, Any]]


class InMemoryConfigStore(MutableMapping[WorkflowId, dict[str, Any]]):
    """
    In-memory implementation of ConfigStore with optional LRU eviction.

    This store uses LRU (Least Recently Used) eviction because it needs to handle
    configurations for multiple use cases with different cleanup strategies:

    - Workflow configs: Could theoretically be cleaned up by removing non-existent
      workflows, but this requires tracking workflow registry state.
    - Plotter configs: More complex - for each workflow there can be multiple outputs,
      and for each output multiple applicable plotters, each needing its own config.
      Tracking existence is impractical.

    LRU eviction provides a simple, uniform policy that works for both cases and can
    be configured per store instance based on expected usage patterns.

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

    def __getitem__(self, key: WorkflowId) -> dict[str, Any]:
        """Get configuration by key."""
        return self._configs[key]

    def __setitem__(self, key: WorkflowId, value: dict[str, Any]) -> None:
        """Save configuration with automatic LRU eviction."""
        self._configs[key] = value

        # Automatic LRU eviction if limit exceeded
        if self._max_configs and len(self._configs) > self._max_configs:
            self._evict_oldest()

    def __delitem__(self, key: WorkflowId) -> None:
        """Delete configuration by key."""
        del self._configs[key]

    def __iter__(self):
        """Iterate over configuration keys."""
        return iter(self._configs)

    def __len__(self) -> int:
        """Return number of stored configurations."""
        return len(self._configs)

    def _evict_oldest(self) -> None:
        """Remove oldest configs based on cleanup fraction."""
        num_to_remove = max(1, int(len(self._configs) * self._cleanup_fraction))
        # Dict maintains insertion order in Python 3.7+
        oldest_keys = list(self._configs.keys())[:num_to_remove]
        for key in oldest_keys:
            del self._configs[key]
