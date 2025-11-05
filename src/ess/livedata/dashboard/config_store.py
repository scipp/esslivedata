# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Abstraction for persisting dashboard UI state.

This module provides config stores for persisting dashboard preferences and UI state
across sessions. This is separate from ConfigService, which handles runtime
communication with backend services via Kafka.

Key Components
--------------
- **ConfigStore**: Type alias for any mutable mapping from WorkflowId to config dict
- **InMemoryConfigStore**: Transient store with LRU eviction (for testing)
- **FileBackedConfigStore**: Persistent YAML-based store with atomic writes
- **ConfigStoreManager**: Thread-safe factory ensuring singleton stores per name

Design Principles
-----------------
1. **Two-level thread safety**: Thread safety is implemented at BOTH levels:
   - **Manager level**: Lock ensures get_store('name') returns the same singleton
     instance to all threads, preventing separate cache instances from desyncing.
   - **Store level**: Lock protects operations on the shared store instance, ensuring
     concurrent reads/writes/evictions don't corrupt the in-memory dict or file.

2. **No file locking**: FileBackedConfigStore uses atomic file operations (write to
   temp, then rename) for crash safety, but not file locks. Store locks protect
   in-memory operations; atomic writes handle the file safely.

3. **Singleton per store name**: ConfigStoreManager ensures get_store('name') always
   returns the same instance, so all dashboard components stay synchronized.
"""

import logging
import os
import threading
from collections import UserDict
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Literal

import platformdirs
import yaml

from ess.livedata.config.workflow_spec import WorkflowId

logger = logging.getLogger(__name__)

# Type alias for config stores - any mutable mapping from WorkflowId to config dict
ConfigStore = MutableMapping[WorkflowId, dict[str, Any]]


class InMemoryConfigStore(UserDict[WorkflowId, dict[str, Any]]):
    """
    Thread-safe in-memory implementation of ConfigStore with optional LRU eviction.

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
        super().__init__()
        self._max_configs = max_configs
        self._cleanup_fraction = cleanup_fraction
        # Lock to protect all operations on this store instance
        self._lock = threading.Lock()

    def __setitem__(self, key: WorkflowId, value: dict[str, Any]) -> None:
        """Save configuration with automatic LRU eviction."""
        with self._lock:
            super().__setitem__(key, value)

            # Automatic LRU eviction if limit exceeded
            if self._max_configs and len(self.data) > self._max_configs:
                self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Remove oldest configs based on cleanup fraction."""
        num_to_remove = max(1, int(len(self.data) * self._cleanup_fraction))
        # Dict maintains insertion order in Python 3.7+
        oldest_keys = list(self.data.keys())[:num_to_remove]
        for key in oldest_keys:
            del self.data[key]


class FileBackedConfigStore(UserDict[WorkflowId, dict[str, Any]]):
    """
    Thread-safe file-backed implementation of ConfigStore with optional LRU eviction.

    Persists configurations to a YAML file on disk, with the same LRU eviction
    policy as InMemoryConfigStore. Configurations are loaded from the file on
    initialization and saved after each modification.

    This implementation uses atomic file operations (write to temp file, then
    rename) for crash safety, and a lock to protect concurrent operations on the
    in-memory cache. It does NOT use file locking - the lock protects in-memory
    operations (dict access, eviction logic), while atomic writes handle file
    persistence safely. This design works in conjunction with ConfigStoreManager,
    which ensures all threads share the same store instance rather than creating
    separate caches that could desync.

    Parameters
    ----------
    file_path:
        Path to the YAML file for storing configurations. Parent directory
        will be created if it doesn't exist.
    max_configs:
        Maximum number of configurations to keep. When exceeded, oldest
        configurations are automatically evicted. If None, no limit is enforced.
    cleanup_fraction:
        Fraction of configs to remove when limit exceeded (default 0.2 = 20%).
    """

    def __init__(
        self,
        file_path: Path | str,
        max_configs: int | None = None,
        cleanup_fraction: float = 0.2,
    ) -> None:
        super().__init__()
        self._file_path = Path(file_path)
        self._max_configs = max_configs
        self._cleanup_fraction = cleanup_fraction
        # Lock to protect all operations on this store instance
        self._lock = threading.Lock()

        # Ensure parent directory exists
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing configs from file
        self._load_from_file()

    def __setitem__(self, key: WorkflowId, value: dict[str, Any]) -> None:
        """Save configuration with automatic LRU eviction and file persistence."""
        with self._lock:
            super().__setitem__(key, value)

            # Automatic LRU eviction if limit exceeded
            if self._max_configs and len(self.data) > self._max_configs:
                self._evict_oldest()

            # Persist to file after modification
            self._save_to_file()

    def __delitem__(self, key: WorkflowId) -> None:
        """Delete configuration and persist to file."""
        with self._lock:
            super().__delitem__(key)
            self._save_to_file()

    def _evict_oldest(self) -> None:
        """Remove oldest configs based on cleanup fraction."""
        num_to_remove = max(1, int(len(self.data) * self._cleanup_fraction))
        # Dict maintains insertion order in Python 3.7+
        oldest_keys = list(self.data.keys())[:num_to_remove]
        for key in oldest_keys:
            # Use dict.pop to avoid triggering __delitem__ which would save repeatedly
            self.data.pop(key, None)

    def _load_from_file(self) -> None:
        """Load configurations from YAML file."""
        if not self._file_path.exists():
            logger.debug(
                "Config file %s does not exist, starting empty", self._file_path
            )
            return

        try:
            with open(self._file_path) as f:
                raw_data = yaml.safe_load(f)
                if raw_data is None:
                    logger.debug("Config file %s is empty", self._file_path)
                    return

                # Deserialize WorkflowId keys from strings
                for key_str, value in raw_data.items():
                    try:
                        workflow_id = WorkflowId.from_string(key_str)
                        self.data[workflow_id] = value
                    except (ValueError, KeyError) as e:
                        logger.warning(
                            "Skipping invalid config entry '%s': %s", key_str, e
                        )

        except yaml.YAMLError as e:
            logger.error("Failed to parse config file %s: %s", self._file_path, e)
            logger.warning("Starting with empty config store")
        except Exception as e:
            logger.error(
                "Unexpected error loading config file %s: %s", self._file_path, e
            )
            logger.warning("Starting with empty config store")

    def _save_to_file(self) -> None:
        """Save configurations to YAML file using atomic write operation."""
        try:
            # Serialize WorkflowId keys to strings for YAML
            serialized_data = {str(key): value for key, value in self.data.items()}

            # Write atomically: write to temp file, then rename
            temp_path = self._file_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                yaml.safe_dump(
                    serialized_data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,  # Preserve insertion order for LRU
                )

            # Atomic rename
            temp_path.replace(self._file_path)

        except Exception as e:
            logger.error("Failed to save config file %s: %s", self._file_path, e)
            # Don't raise - continue operation even if persistence fails


def get_config_dir(instrument: str, config_dir: Path | str | None = None) -> Path:
    """
    Resolve the configuration directory path for an instrument.

    Uses LIVEDATA_CONFIG_DIR environment variable if set, otherwise uses
    platformdirs to determine the appropriate user config directory.

    Parameters
    ----------
    instrument:
        The instrument name, used for per-instrument config directory.
    config_dir:
        Override config directory path. If None, resolves from LIVEDATA_CONFIG_DIR
        environment variable or platform-specific config directory.

    Returns
    -------
    :
        Path to the instrument-specific config directory.
    """
    if config_dir is not None:
        base_dir = Path(config_dir)
    elif (env_dir := os.environ.get('LIVEDATA_CONFIG_DIR')) is not None:
        base_dir = Path(env_dir)
    else:
        # Use platformdirs for cross-platform config directory resolution
        base_dir = Path(platformdirs.user_config_dir('esslivedata', appauthor=False))

    return base_dir / instrument


class ConfigStoreManager:
    """
    Thread-safe manager for config stores ensuring singleton behavior per store name.

    The manager maintains references to created stores, ensuring that multiple
    calls to get_store() with the same name return the same instance. This is
    critical for maintaining consistency - if different parts of the dashboard
    created separate store instances, each would have its own in-memory cache
    that could fall out of sync with the others.

    Thread safety is implemented at the manager level (not at individual stores)
    because the shared resource is the in-memory store cache, not the file on disk.
    A lock protects the _stores dict from race conditions during concurrent access,
    ensuring true singleton behavior even when multiple threads call get_store()
    simultaneously. This approach is simpler and more efficient than file-level
    locking, which would be the wrong abstraction layer.

    Parameters
    ----------
    instrument:
        The instrument name, used for per-instrument config directory.
    store_type:
        Type of stores to create: "file" for persistent file-backed storage,
        or "memory" for transient in-memory storage. Default is "file".
    max_configs:
        Maximum number of configurations per store. Default is 100.
    cleanup_fraction:
        Fraction of configs to remove when limit exceeded. Default is 0.2 (20%).
    config_dir:
        Override config directory path. If None, resolves from LIVEDATA_CONFIG_DIR
        environment variable or platform-specific config directory.

    Examples
    --------
    >>> # Create manager for an instrument
    >>> manager = ConfigStoreManager(instrument='dummy')
    >>> workflow_store = manager.get_store('workflow_configs')
    >>> plotter_store = manager.get_store('plotter_configs')
    >>>
    >>> # Multiple calls return the same instance
    >>> same_store = manager.get_store('workflow_configs')
    >>> assert same_store is workflow_store
    """

    def __init__(
        self,
        instrument: str,
        store_type: Literal["file", "memory"] = "file",
        max_configs: int = 100,
        cleanup_fraction: float = 0.2,
        config_dir: Path | str | None = None,
    ) -> None:
        self._instrument = instrument
        self._store_type = store_type
        self._max_configs = max_configs
        self._cleanup_fraction = cleanup_fraction
        self._config_dir = get_config_dir(instrument, config_dir)
        # Cache of created stores to ensure singleton behavior
        self._stores: dict[str, ConfigStore] = {}
        # Lock to protect the store cache from concurrent access
        self._lock = threading.Lock()

    def get_store(self, name: str) -> ConfigStore:
        """
        Get or create a config store by name.

        Returns the same instance for repeated calls with the same name,
        ensuring all parts of the dashboard share the same in-memory cache.
        Thread-safe: uses a lock to prevent race conditions when creating stores.

        Parameters
        ----------
        name:
            Name of the config store. For file-backed stores, this becomes
            the filename (with .yaml extension). For memory stores, this is
            used for identification only.

        Returns
        -------
        :
            A config store instance (either FileBackedConfigStore or
            InMemoryConfigStore depending on store_type).

        Examples
        --------
        >>> manager = ConfigStoreManager(instrument='dummy')
        >>> workflow_store = manager.get_store('workflow_configs')
        >>> # Creates file at ~/.config/esslivedata/dummy/workflow_configs.yaml
        >>>
        >>> # Second call returns the same instance
        >>> same_store = manager.get_store('workflow_configs')
        >>> assert same_store is workflow_store
        """
        with self._lock:
            if name not in self._stores:
                self._stores[name] = self._create_store(name)
            return self._stores[name]

    def _create_store(self, name: str) -> ConfigStore:
        """Internal method to create a new config store instance."""
        if self._store_type == "file":
            file_path = self._config_dir / f"{name}.yaml"
            return FileBackedConfigStore(
                file_path=file_path,
                max_configs=self._max_configs,
                cleanup_fraction=self._cleanup_fraction,
            )
        else:  # memory
            return InMemoryConfigStore(
                max_configs=self._max_configs,
                cleanup_fraction=self._cleanup_fraction,
            )
