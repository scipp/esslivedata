# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Data subscription and assembly for streaming plot updates.

This module provides the core data flow components:
- DataSubscriber: Connects to DataService, assembles data, and feeds pipes
- Assembly is role-aware: single-role outputs flat dict, multi-role outputs grouped dict
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Generic, Protocol, TypeVar

import panel as pn

from ess.livedata.config.workflow_spec import ResultKey
from ess.livedata.dashboard.data_service import DataServiceSubscriber
from ess.livedata.dashboard.extractors import UpdateExtractor


class PipeBase(Protocol):
    """
    Protocol for downstream pipes that can receive data from upstream pipes.
    """

    def send(self, data: Any) -> None:
        """
        Send data to the downstream pipe.

        Parameters
        ----------
        data:
            The data to be sent.
        """


class Pipe(PipeBase):
    """Protocol for holoviews pipes, which need to be initialized with data."""

    def __init__(self, data: Any) -> None:
        """
        Initialize the pipe with its data.

        Parameters
        ----------
        data:
            The initial data for the pipe.
        """


P = TypeVar('P', bound=PipeBase)


class DataSubscriber(DataServiceSubscriber[ResultKey], Generic[P]):
    """Subscriber that assembles data by role and feeds it to a pipe.

    Handles both single-role (standard plots) and multi-role (correlation plots):
    - Single role: outputs flat dict[ResultKey, data] for backward compatibility
    - Multiple roles: outputs dict[str, dict[ResultKey, data]] grouped by role

    The ready_condition is built internally: requires at least one key from each role.
    """

    def __init__(
        self,
        keys_by_role: dict[str, list[ResultKey]],
        pipe_factory: Callable[[Any], P],
        extractors: Mapping[ResultKey, UpdateExtractor],
        on_first_data: Callable[[P], None] | None = None,
    ) -> None:
        """
        Initialize the subscriber.

        Parameters
        ----------
        keys_by_role
            Dict mapping role names to lists of ResultKeys. For standard plots,
            this is {"primary": [keys...]}. For correlation plots, includes
            additional roles like "x_axis", "y_axis".
        pipe_factory
            Factory function to create the pipe on first trigger.
        extractors:
            Mapping from keys to their UpdateExtractor instances.
        on_first_data:
            Optional callback invoked when first data arrives with the created pipe.
        """
        self._keys_by_role = keys_by_role
        self._all_keys = {key for keys in keys_by_role.values() for key in keys}
        self._single_role = len(keys_by_role) == 1

        # Build ready_condition: need at least one key from each role
        self._key_sets_by_role = [set(keys) for keys in keys_by_role.values()]

        self._pipe_factory = pipe_factory
        self._pipe: P | None = None
        self._extractors = extractors
        self._on_first_data = on_first_data
        self._first_data_callback_invoked = False

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

    @property
    def pipe(self) -> P:
        """Return the pipe (must be created by first trigger)."""
        if self._pipe is None:
            raise RuntimeError("Pipe not yet initialized - subscriber not triggered")
        return self._pipe

    def _is_ready(self, available_keys: set[ResultKey]) -> bool:
        """Check if we have at least one key from each role."""
        return all(bool(available_keys & ks) for ks in self._key_sets_by_role)

    def _assemble(self, data: dict[ResultKey, Any]) -> Any:
        """Assemble data based on role structure.

        Single role: returns flat dict[ResultKey, data] for standard plotters.
        Multiple roles: returns dict[str, dict[ResultKey, data]] for
        correlation plotters.
        """
        if self._single_role:
            # Flat output for standard plotters (sorted for deterministic ordering)
            sorted_keys = sorted(
                (k for k in self._all_keys if k in data),
                key=lambda k: (str(k.workflow_id), str(k.job_id), k.output_name),
            )
            return {k: data[k] for k in sorted_keys}
        else:
            # Grouped output for correlation plotters
            result: dict[str, dict[ResultKey, Any]] = {}
            for role, role_keys in self._keys_by_role.items():
                sorted_keys = sorted(
                    role_keys, key=lambda k: (str(k.workflow_id), str(k.job_id))
                )
                role_data = {k: data[k] for k in sorted_keys if k in data}
                if role_data:
                    result[role] = role_data
            return result

    def trigger(self, store: dict[ResultKey, Any]) -> None:
        """Trigger the subscriber with the current data store."""
        data = {key: store[key] for key in self._all_keys if key in store}
        assembled_data = self._assemble(data)

        if self._pipe is None:
            # First trigger - always create pipe (even with empty data)
            self._pipe = self._pipe_factory(assembled_data)
        else:
            # Subsequent triggers - send to existing pipe
            self._pipe.send(assembled_data)

        # Invoke first-data callback when we have actual data for the first time
        # AND the ready_condition is satisfied (at least one key from each role).
        # IMPORTANT: We defer this callback to the next event loop iteration using
        # pn.state.execute(). This breaks the synchronous callback chain that occurs
        # when subscribing to a workflow that already has data, preventing UI blocking
        # during plot creation. The chain would otherwise be:
        #   subscribe_to_workflow() → on_all_jobs_ready() → setup_pipeline()
        #   → register_subscriber() → trigger() → on_first_data() → create_plot()
        # All running synchronously before returning control to the event loop.
        if (
            data
            and not self._first_data_callback_invoked
            and self._on_first_data
            and self._is_ready(set(data.keys()))
        ):
            self._first_data_callback_invoked = True
            pipe = self._pipe  # Capture for lambda
            pn.state.execute(lambda: self._on_first_data(pipe))
