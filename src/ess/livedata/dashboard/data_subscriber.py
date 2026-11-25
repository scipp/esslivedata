# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Data subscription and assembly for streaming plot updates.

This module provides the core data flow components:
- DataSubscriber: Watches DataService keys, assembles pulled data by role
- Output is always role-grouped: dict[role, dict[DataKey, data]]
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ess.livedata.config.workflow_spec import DataKey
from ess.livedata.dashboard.data_service import DataServiceSubscriber
from ess.livedata.dashboard.extractors import UpdateExtractor


class DataSubscriber(DataServiceSubscriber[DataKey]):
    """Subscriber that reports key updates and assembles pulled data by role.

    Update notifications carry no data; ``on_update`` just signals that the
    consumer should schedule a pull. The pull applies this subscriber's
    extractors via ``DataService.snapshot`` and is grouped with
    :py:meth:`assemble`, whose output shape is always
    ``dict[role, dict[DataKey, data]]``. Consumers that care about a single
    role (e.g. standard plotters using ``primary``) extract it explicitly.

    Data is ready once each role has data for at least one of its keys.
    """

    def __init__(
        self,
        keys_by_role: dict[str, list[DataKey]],
        extractors: Mapping[DataKey, UpdateExtractor],
        on_update: Callable[[], None],
        select_by_role: Mapping[str, Callable[[Any], Any | None]] | None = None,
    ) -> None:
        """
        Initialize the subscriber.

        Parameters
        ----------
        keys_by_role
            Dict mapping role names to lists of DataKeys. For standard plots,
            this is {"primary": [keys...]}. For correlation plots, includes
            additional roles like "x_axis", "y_axis".
        extractors
            Mapping from keys to their UpdateExtractor instances.
        on_update
            Callback invoked when any of this subscriber's keys changed.
            Must be cheap (it runs on the ingestion thread per update batch);
            the consumer pulls data later via ``DataService.snapshot`` +
            :py:meth:`assemble`.
        select_by_role
            Per role, a function selecting the part of each key's data that
            the consumer wants, e.g. one ROI of a per-ROI output. A key for
            which it returns None has no such part and is left out. Roles
            without an entry get their data as extracted.
        """
        self._keys_by_role = {
            role: sorted(
                keys, key=lambda k: (str(k.workflow_id), k.source_name, k.output_name)
            )
            for role, keys in keys_by_role.items()
        }
        self._all_keys = {key for keys in keys_by_role.values() for key in keys}
        self._select_by_role = select_by_role or {}
        self._extractors = extractors
        self._on_update = on_update

        # Initialize parent class to cache keys
        super().__init__()

    @property
    def keys(self) -> set[DataKey]:
        """Return all keys this subscriber depends on."""
        return self._all_keys

    @property
    def extractors(self) -> Mapping[DataKey, UpdateExtractor]:
        """Return extractors for obtaining data views."""
        return self._extractors

    def assemble(
        self, store: dict[DataKey, Any]
    ) -> dict[str, dict[DataKey, Any]] | None:
        """
        Group this subscriber's keys from ``store`` by role.

        Returns None unless each role has data for at least one of its keys,
        after the selection of ``select_by_role``. Within each role, keys are
        sorted deterministically.
        """
        result: dict[str, dict[DataKey, Any]] = {}
        for role, role_keys in self._keys_by_role.items():
            select = self._select_by_role.get(role)
            role_data = {}
            for key in role_keys:
                if key not in store:
                    continue
                value = store[key] if select is None else select(store[key])
                if value is not None:
                    role_data[key] = value
            if not role_data:
                return None
            result[role] = role_data
        return result or None

    def on_updated(self, updated_keys: set[DataKey]) -> None:
        """Signal the consumer that a pull is due (see ``on_update`` in init)."""
        self._on_update()
