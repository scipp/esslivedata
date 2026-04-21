# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Data subscription and assembly for streaming plot updates.

This module provides the core data flow components:
- DataSubscriber: Connects to DataService, assembles data, invokes callback
- Output is always role-grouped: dict[role, dict[ResultKey, data]]
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ess.livedata.config.workflow_spec import ResultKey
from ess.livedata.dashboard.data_service import DataServiceSubscriber
from ess.livedata.dashboard.extractors import UpdateExtractor


class DataSubscriber(DataServiceSubscriber[ResultKey]):
    """Subscriber that groups data by role and invokes a callback.

    Output shape is always ``dict[role, dict[ResultKey, data]]``. Consumers
    that care about a single role (e.g. standard plotters using ``primary``)
    extract it explicitly.

    The ready_condition is built internally: requires at least one key from each role.
    """

    def __init__(
        self,
        keys_by_role: dict[str, list[ResultKey]],
        extractors: Mapping[ResultKey, UpdateExtractor],
        on_data: Callable[[dict[str, dict[ResultKey, Any]]], None],
    ) -> None:
        """
        Initialize the subscriber.

        Parameters
        ----------
        keys_by_role
            Dict mapping role names to lists of ResultKeys. For standard plots,
            this is {"primary": [keys...]}. For correlation plots, includes
            additional roles like "x_axis", "y_axis".
        extractors
            Mapping from keys to their UpdateExtractor instances.
        on_data
            Callback invoked on every data update with the grouped data.
            Called when at least one key from each role has data.
        """
        self._keys_by_role = keys_by_role
        self._all_keys = {key for keys in keys_by_role.values() for key in keys}

        # Build ready_condition: need at least one key from each role
        self._key_sets_by_role = [set(keys) for keys in keys_by_role.values()]

        self._extractors = extractors
        self._on_data = on_data

        # Initialize parent class to cache keys
        super().__init__()

    @property
    def keys(self) -> set[ResultKey]:
        """Return all keys this subscriber depends on."""
        return self._all_keys

    @property
    def extractors(self) -> Mapping[ResultKey, UpdateExtractor]:
        """Return extractors for obtaining data views."""
        return self._extractors

    def _is_ready(self, available_keys: set[ResultKey]) -> bool:
        """Check if we have at least one key from each role."""
        return all(bool(available_keys & ks) for ks in self._key_sets_by_role)

    def _assemble(self, data: dict[ResultKey, Any]) -> dict[str, dict[ResultKey, Any]]:
        """Group data by role, sorted deterministically within each role."""
        result: dict[str, dict[ResultKey, Any]] = {}
        for role, role_keys in self._keys_by_role.items():
            sorted_keys = sorted(
                role_keys,
                key=lambda k: (str(k.workflow_id), str(k.job_id), k.output_name),
            )
            role_data = {k: data[k] for k in sorted_keys if k in data}
            if role_data:
                result[role] = role_data
        return result

    def trigger(self, store: dict[ResultKey, Any]) -> None:
        """Trigger the subscriber with the current data store."""
        data = {key: store[key] for key in self._all_keys if key in store}

        # Only invoke callback when we have data and are ready
        if data and self._is_ready(set(data.keys())):
            assembled = self._assemble(data)
            self._on_data(assembled)
