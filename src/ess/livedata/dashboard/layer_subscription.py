# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
LayerSubscription - Manages subscription to one or more workflows for a plot layer.

This abstraction hides single vs multi-job complexity from PlotOrchestrator,
enabling uniform handling of both simple plots and correlation histograms.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from ess.livedata.config.workflow_spec import JobId, JobNumber, ResultKey

if TYPE_CHECKING:
    from ess.livedata.dashboard.job_orchestrator import SubscriptionId
    from ess.livedata.dashboard.plot_orchestrator import (
        DataSourceConfig,
        JobOrchestratorProtocol,
    )


@dataclass
class SubscriptionReady:
    """Data provided when all workflows are ready."""

    keys: list[ResultKey]
    ready_condition: Callable[[set[ResultKey]], bool]


class LayerSubscription:
    """
    Manages subscription to one or more workflows for a plot layer.

    Handles:
    - Subscribing to all required workflows via JobOrchestrator
    - Tracking which workflows have started (job_numbers available)
    - Notifying when ALL workflows are ready (with ResultKeys and ready_condition)
    - Propagating stop notifications (fires on ANY workflow stop)

    Subscribes once per DataSourceConfig. If multiple DataSourceConfigs share the
    same workflow_id, this results in redundant subscriptions, which is harmless
    and simpler than de-duplicating.

    Use start() to begin subscriptions after construction. This avoids callbacks
    firing during __init__ before the caller has stored the subscription.
    """

    def __init__(
        self,
        data_sources: list[DataSourceConfig],
        job_orchestrator: JobOrchestratorProtocol,
        on_ready: Callable[[SubscriptionReady], None],
        on_stopped: Callable[[JobNumber], None] | None = None,
    ):
        """
        Initialize the layer subscription.

        Parameters
        ----------
        data_sources
            List of data source configurations. Each describes one workflow's
            contribution to the layer's data requirements.
        job_orchestrator
            Orchestrator for subscribing to workflow lifecycle.
        on_ready
            Called when ALL workflows have started. Receives SubscriptionReady
            containing ResultKeys and a ready_condition for gating plot creation.
        on_stopped
            Called when ANY subscribed workflow stops. For correlation plots,
            losing any data source means the correlation cannot be computed.
        """
        self._data_sources = data_sources
        self._job_orchestrator = job_orchestrator
        self._on_ready = on_ready
        self._on_stopped = on_stopped

        # Track job_numbers per data_source index (not workflow_id to handle dups)
        self._job_numbers: dict[int, JobNumber] = {}
        self._subscription_ids: list[SubscriptionId] = []

    def start(self) -> None:
        """
        Subscribe to all workflows.

        Call after construction. May fire on_ready synchronously if all
        workflows are already running.
        """
        from .job_orchestrator import WorkflowCallbacks

        for idx, ds in enumerate(self._data_sources):
            sub_id, _ = self._job_orchestrator.subscribe_to_workflow(
                workflow_id=ds.workflow_id,
                callbacks=WorkflowCallbacks(
                    on_started=partial(self._on_workflow_started, idx),
                    on_stopped=partial(self._handle_workflow_stopped, idx),
                ),
            )
            self._subscription_ids.append(sub_id)

    def _on_workflow_started(self, ds_index: int, job_number: JobNumber) -> None:
        """Handle workflow start for a specific data_source."""
        self._job_numbers[ds_index] = job_number

        if len(self._job_numbers) == len(self._data_sources):
            ready = SubscriptionReady(
                keys=self._build_result_keys(),
                ready_condition=self._build_ready_condition(),
            )
            self._on_ready(ready)

    def _handle_workflow_stopped(self, ds_index: int, job_number: JobNumber) -> None:
        """Handle workflow stop for a specific data_source.

        Clears the job_number for the stopped source so the length check in
        _on_workflow_started will wait for restart before firing on_ready.
        """
        if ds_index in self._job_numbers:
            del self._job_numbers[ds_index]

        if self._on_stopped is not None:
            self._on_stopped(job_number)

    def _keys_by_data_source(self) -> list[set[ResultKey]]:
        """Return ResultKeys grouped by data_source."""
        return [
            {
                ResultKey(
                    workflow_id=ds.workflow_id,
                    job_id=JobId(source_name=sn, job_number=self._job_numbers[idx]),
                    output_name=ds.output_name,
                )
                for sn in ds.source_names
            }
            for idx, ds in enumerate(self._data_sources)
        ]

    def _build_result_keys(self) -> list[ResultKey]:
        """Build ResultKeys for all data_sources using collected job_numbers."""
        return [key for group in self._keys_by_data_source() for key in group]

    def _build_ready_condition(self) -> Callable[[set[ResultKey]], bool]:
        """
        Build a ready_condition that requires at least one key from each source.

        This ensures correlation plots wait for all required data sources, while
        allowing progressive display within each data source (e.g., if a data source
        has multiple source_names, we don't wait for all of them).

        Returns
        -------
        :
            A callable that takes a set of available keys and returns True if
            at least one key from each DataSourceConfig is present.
        """
        key_groups = self._keys_by_data_source()

        def ready(available: set[ResultKey]) -> bool:
            return all(bool(available & group) for group in key_groups)

        return ready

    def unsubscribe(self) -> None:
        """Unsubscribe from all workflow notifications."""
        for sub_id in self._subscription_ids:
            self._job_orchestrator.unsubscribe(sub_id)
        self._subscription_ids.clear()
