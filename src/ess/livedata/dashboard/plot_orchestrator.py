# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
PlotOrchestrator - Manages plot grid configurations and plot lifecycle.

Coordinates plot creation and management across multiple plot grids:
- Configuration staging and persistence
- Plot grid lifecycle (create, remove)
- Plot cell management (add, remove)
- Event-driven plot creation via JobOrchestrator subscriptions
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NewType, Protocol
from uuid import UUID, uuid4

from ess.livedata.config.workflow_spec import JobNumber, WorkflowId

from .config_store import ConfigStore
from .plotting_controller import PlottingController

SubscriptionId = NewType('SubscriptionId', UUID)
GridId = NewType('GridId', UUID)
CellId = NewType('CellId', UUID)


class JobOrchestratorProtocol(Protocol):
    """Protocol for JobOrchestrator interface needed by PlotOrchestrator."""

    def subscribe_to_workflow(
        self, workflow_id: WorkflowId, callback: Callable[[JobNumber], None]
    ) -> SubscriptionId:
        """
        Subscribe to workflow availability notifications.

        The callback will be called with the job_number whenever the workflow
        is committed (started or restarted).

        Parameters
        ----------
        workflow_id
            The workflow to subscribe to.
        callback
            Called with job_number when workflow becomes available.

        Returns
        -------
        :
            Subscription ID that can be used to unsubscribe.
        """
        ...

    def unsubscribe(self, subscription_id: SubscriptionId) -> None:
        """
        Unsubscribe from workflow availability notifications.

        Parameters
        ----------
        subscription_id
            The subscription ID returned from subscribe_to_workflow.
        """
        ...


@dataclass
class PlotConfig:
    """Configuration for a single plot in a grid cell."""

    workflow_id: WorkflowId
    output_name: str | None
    source_names: list[str]
    plot_name: str
    params: dict


@dataclass
class PlotCell:
    """Configuration for a plot cell (position and what to plot)."""

    row: int
    col: int
    row_span: int
    col_span: int
    config: PlotConfig


@dataclass
class PlotGridConfig:
    """A plot grid tab configuration."""

    title: str = ""
    nrows: int = 3
    ncols: int = 3
    cells: dict[CellId, PlotCell] = field(default_factory=dict)


class PlotOrchestrator:
    """Manages plot grid configurations and plot lifecycle."""

    def __init__(
        self,
        *,
        plotting_controller: PlottingController,
        job_orchestrator: JobOrchestratorProtocol,
        config_store: ConfigStore | None = None,
    ) -> None:
        """
        Initialize the plot orchestrator.

        Parameters
        ----------
        plotting_controller
            Controller for creating plots.
        job_orchestrator
            Orchestrator for subscribing to workflow availability.
        config_store
            Optional store for persisting plot grid configurations across sessions.
        """
        self._plotting_controller = plotting_controller
        self._job_orchestrator = job_orchestrator
        self._config_store = config_store
        self._logger = logging.getLogger(__name__)

        self._grids: dict[GridId, PlotGridConfig] = {}
        self._cell_to_grid: dict[CellId, GridId] = {}
        self._cell_to_subscription: dict[CellId, SubscriptionId] = {}

    def add_grid(self, title: str, nrows: int, ncols: int) -> GridId:
        """
        Add a new plot grid.

        Parameters
        ----------
        title
            Display title for the grid.
        nrows
            Number of rows in the grid.
        ncols
            Number of columns in the grid.

        Returns
        -------
        :
            ID of the created grid.
        """
        grid_id = GridId(uuid4())
        grid = PlotGridConfig(title=title, nrows=nrows, ncols=ncols)
        self._grids[grid_id] = grid
        self._persist_to_store()
        self._logger.info(
            'Added plot grid %s (%s) with size %dx%d', grid_id, title, nrows, ncols
        )
        return grid_id

    def remove_grid(self, grid_id: GridId) -> None:
        """
        Remove a plot grid and unsubscribe all plots.

        Parameters
        ----------
        grid_id
            ID of the grid to remove.
        """
        if grid_id in self._grids:
            grid = self._grids[grid_id]
            title = grid.title

            # Unsubscribe all cells and clean up mappings
            for cell_id in grid.cells.keys():
                self._job_orchestrator.unsubscribe(self._cell_to_subscription[cell_id])
                del self._cell_to_subscription[cell_id]
                del self._cell_to_grid[cell_id]

            del self._grids[grid_id]
            self._persist_to_store()
            self._logger.info('Removed plot grid %s (%s)', grid_id, title)

    def add_plot(self, grid_id: GridId, cell: PlotCell) -> CellId:
        """
        Add a plot configuration to a grid and subscribe to workflow availability.

        Parameters
        ----------
        grid_id
            ID of the grid to add the plot to.
        cell
            Plot cell configuration.

        Returns
        -------
        :
            ID of the added plot cell.
        """
        cell_id = CellId(uuid4())
        grid = self._grids[grid_id]
        grid.cells[cell_id] = cell

        # Map cell to its grid for fast lookup
        self._cell_to_grid[cell_id] = grid_id

        # Subscribe to workflow availability and store subscription ID
        subscription_id = self._job_orchestrator.subscribe_to_workflow(
            workflow_id=cell.config.workflow_id,
            callback=lambda job_number: self._on_job_available(cell_id, job_number),
        )
        self._cell_to_subscription[cell_id] = subscription_id

        self._persist_to_store()
        self._logger.info(
            'Added plot %s to grid %s at (%d,%d) for workflow %s',
            cell_id,
            grid_id,
            cell.row,
            cell.col,
            cell.config.workflow_id,
        )
        return cell_id

    def remove_plot(self, cell_id: CellId) -> None:
        """
        Remove a plot by its unique ID.

        Parameters
        ----------
        cell_id
            ID of the cell to remove.
        """
        grid_id = self._cell_to_grid[cell_id]
        grid = self._grids[grid_id]
        cell = grid.cells[cell_id]

        # Unsubscribe from workflow notifications
        self._job_orchestrator.unsubscribe(self._cell_to_subscription[cell_id])
        del self._cell_to_subscription[cell_id]

        # Remove from grid and mapping
        del grid.cells[cell_id]
        del self._cell_to_grid[cell_id]

        self._persist_to_store()
        self._logger.info(
            'Removed plot %s from grid %s at (%d,%d)',
            cell_id,
            grid_id,
            cell.row,
            cell.col,
        )

    def get_plot_config(self, cell_id: CellId) -> PlotConfig:
        """
        Get configuration for a plot.

        Parameters
        ----------
        cell_id
            ID of the plot cell.

        Returns
        -------
        :
            The plot configuration.
        """
        grid_id = self._cell_to_grid[cell_id]
        return self._grids[grid_id].cells[cell_id].config

    def update_plot_config(self, cell_id: CellId, new_config: PlotConfig) -> None:
        """
        Update plot configuration and resubscribe to workflow.

        This resubscribes to the workflow.
        When the workflow is next committed, the plot will be recreated
        with the new configuration.

        Parameters
        ----------
        cell_id
            ID of the plot cell to update.
        new_config
            New plot configuration.
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        # Unsubscribe from old workflow notifications
        self._job_orchestrator.unsubscribe(self._cell_to_subscription[cell_id])

        # Update configuration
        cell.config = new_config

        # Re-subscribe to workflow (in case workflow_id changed)
        subscription_id = self._job_orchestrator.subscribe_to_workflow(
            workflow_id=new_config.workflow_id,
            callback=lambda job_number: self._on_job_available(cell_id, job_number),
        )
        self._cell_to_subscription[cell_id] = subscription_id

        # Persist updated configuration
        self._persist_to_store()

        self._logger.info('Updated plot config for cell %s', cell_id)

    def _on_job_available(self, cell_id: CellId, job_number: JobNumber) -> None:
        """
        Handle workflow availability notification from JobOrchestrator.

        Called when a workflow is committed (started or restarted). Passes
        the cell configuration to PlottingController to create the plot
        and notify subscribers.

        Parameters
        ----------
        cell_id
            ID of the plot cell to create plot for.
        job_number
            Job number for the workflow.
        """
        # Defensive check: cell may have been removed before callback fires
        if cell_id not in self._cell_to_grid:
            self._logger.debug(
                'Ignoring workflow availability for removed cell %s', cell_id
            )
            return

        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        # Pass config to controller - it will create plot and notify subscribers
        self._plotting_controller.create_and_notify_cell_plot(
            cell_id=cell_id,
            job_number=job_number,
            source_names=cell.config.source_names,
            output_name=cell.config.output_name,
            plot_name=cell.config.plot_name,
            params=cell.config.params,
        )

    def _persist_to_store(self) -> None:
        """Persist plot grid configurations to config store."""
        if self._config_store is None:
            return

        # TODO: Implement persistence when needed
        self._logger.debug('Plot grid configs would be persisted to store')

    def get_grid(self, grid_id: GridId) -> PlotGridConfig | None:
        """
        Get a plot grid configuration.

        Parameters
        ----------
        grid_id
            ID of the grid to retrieve.

        Returns
        -------
        :
            Plot grid configuration if found, None otherwise.
        """
        return self._grids.get(grid_id)

    def get_all_grids(self) -> dict[GridId, PlotGridConfig]:
        """
        Get all plot grid configurations.

        Returns
        -------
        :
            Dictionary mapping grid IDs to configurations.
        """
        return self._grids.copy()

    def shutdown(self) -> None:
        """
        Shutdown the orchestrator and unsubscribe from all workflows.

        This removes all grids and unsubscribes from all workflow notifications.
        Call this method when the orchestrator is no longer needed to prevent
        memory leaks.
        """
        # Remove all grids (which unsubscribes all plots)
        grid_ids = list(self._grids.keys())
        for grid_id in grid_ids:
            self.remove_grid(grid_id)

        self._logger.info('PlotOrchestrator shutdown complete')
