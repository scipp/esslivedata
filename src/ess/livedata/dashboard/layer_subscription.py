# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
LayerSubscription - Manages subscription to one or more workflows for a plot layer.

This abstraction hides single vs multi-job complexity from PlotOrchestrator,
enabling uniform handling of both simple plots and correlation histograms.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
    """Data provided when all workflows are ready.

    The keys_by_role dict maps role names (e.g., "primary", "x_axis") to lists
    of ResultKeys for that role. This structure is preserved through the data
    pipeline, allowing plotters to receive pre-structured data.
    """

    keys_by_role: dict[str, list[ResultKey]] = field(default_factory=dict)


class LayerSubscription:
    """
    Manages subscription to one or more workflows for a plot layer.

    Handles:
    - Subscribing to all required workflows via JobOrchestrator
    - Tracking which workflows have started (job_numbers available)
    - Notifying when ALL workflows are ready (with keys_by_role)
    - Propagating stop notifications (fires on ANY workflow stop)

    Subscribes once per role in data_sources. If multiple roles share the same
    workflow_id, this results in redundant subscriptions, which is harmless
    and simpler than de-duplicating.

    Use start() to begin subscriptions after construction. This avoids callbacks
    firing during __init__ before the caller has stored the subscription.
    """

    def __init__(
        self,
        data_sources: dict[str, DataSourceConfig],
        job_orchestrator: JobOrchestratorProtocol,
        on_ready: Callable[[SubscriptionReady], None],
        on_stopped: Callable[[JobNumber], None] | None = None,
    ):
        """
        Initialize the layer subscription.

        Parameters
        ----------
        data_sources
            Dict mapping role names to data source configurations. Each describes
            one workflow's contribution to the layer's data requirements.
        job_orchestrator
            Orchestrator for subscribing to workflow lifecycle.
        on_ready
            Called when ALL workflows have started. Receives SubscriptionReady
            containing keys_by_role for structured data access.
        on_stopped
            Called when ANY subscribed workflow stops. For correlation plots,
            losing any data source means the correlation cannot be computed.
        """
        self._data_sources = data_sources
        self._roles = list(data_sources.keys())
        self._job_orchestrator = job_orchestrator
        self._on_ready = on_ready
        self._on_stopped = on_stopped

        # Track job_numbers per role name
        self._job_numbers: dict[str, JobNumber] = {}
        self._subscription_ids: list[SubscriptionId] = []

    def start(self) -> None:
        """
        Subscribe to all workflows.

        Call after construction. May fire on_ready synchronously if all
        workflows are already running.
        """
        from .job_orchestrator import WorkflowCallbacks

        for role, ds in self._data_sources.items():
            sub_id, _ = self._job_orchestrator.subscribe_to_workflow(
                workflow_id=ds.workflow_id,
                callbacks=WorkflowCallbacks(
                    on_started=partial(self._on_workflow_started, role),
                    on_stopped=partial(self._handle_workflow_stopped, role),
                ),
            )
            self._subscription_ids.append(sub_id)

    def _on_workflow_started(self, role: str, job_number: JobNumber) -> None:
        """Handle workflow start for a specific role."""
        self._job_numbers[role] = job_number

        if len(self._job_numbers) == len(self._data_sources):
            ready = SubscriptionReady(keys_by_role=self._build_keys_by_role())
            self._on_ready(ready)

    def _handle_workflow_stopped(self, role: str, job_number: JobNumber) -> None:
        """Handle workflow stop for a specific role.

        Clears the job_number for the stopped source so the length check in
        _on_workflow_started will wait for restart before firing on_ready.
        """
        if role in self._job_numbers:
            del self._job_numbers[role]

        if self._on_stopped is not None:
            self._on_stopped(job_number)

    def _build_keys_by_role(self) -> dict[str, list[ResultKey]]:
        """Build ResultKeys grouped by role."""
        return {
            role: [
                ResultKey(
                    workflow_id=ds.workflow_id,
                    job_id=JobId(source_name=sn, job_number=self._job_numbers[role]),
                    output_name=ds.output_name,
                )
                for sn in ds.source_names
            ]
            for role, ds in self._data_sources.items()
        }

    def unsubscribe(self) -> None:
        """Unsubscribe from all workflow notifications."""
        for sub_id in self._subscription_ids:
            self._job_orchestrator.unsubscribe(sub_id)
        self._subscription_ids.clear()
