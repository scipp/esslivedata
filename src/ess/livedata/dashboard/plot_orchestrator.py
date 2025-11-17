# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
PlotOrchestrator - Manages plot grid configurations and plot lifecycle.

Coordinates plot creation and management across multiple plot grids:
- Configuration staging and persistence
- Plot grid lifecycle (create, remove)
- Plot cell management (add, remove, refresh)
- Automatic plot creation when matching jobs become available
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import holoviews as hv
import pydantic

from ess.livedata.config.workflow_spec import JobNumber, WorkflowId, WorkflowSpec

from .config_store import ConfigStore
from .job_service import JobService
from .plotting_controller import PlottingController


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
    """A configured plot cell (may or may not have actual plot yet)."""

    row: int
    col: int
    row_span: int
    col_span: int
    config: PlotConfig
    id: UUID = field(default_factory=uuid4)
    job_number: JobNumber | None = None
    plot: hv.DynamicMap | hv.Layout | None = None
    error: str | None = None


@dataclass
class PlotGridConfig:
    """A plot grid tab configuration."""

    id: UUID = field(default_factory=uuid4)
    title: str = ""
    nrows: int = 3
    ncols: int = 3
    cells: list[PlotCell] = field(default_factory=list)


class PlotOrchestrator:
    """Manages plot grid configurations and plot lifecycle."""

    def __init__(
        self,
        *,
        job_service: JobService,
        plotting_controller: PlottingController,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        config_store: ConfigStore | None = None,
    ) -> None:
        """
        Initialize the plot orchestrator.

        Parameters
        ----------
        job_service
            Service for accessing job data and information.
        plotting_controller
            Controller for creating plots.
        workflow_registry
            Registry of available workflows and their specifications.
        config_store
            Optional store for persisting plot grid configurations across sessions.
        """
        self._job_service = job_service
        self._plotting_controller = plotting_controller
        self._workflow_registry = workflow_registry
        self._config_store = config_store
        self._logger = logging.getLogger(__name__)

        self._grids: dict[UUID, PlotGridConfig] = {}
        self._cell_index: dict[UUID, tuple[UUID, int]] = {}

        self._load_from_store()

    def add_grid(self, title: str, nrows: int, ncols: int) -> UUID:
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
            UUID of the created grid.
        """
        grid = PlotGridConfig(title=title, nrows=nrows, ncols=ncols)
        self._grids[grid.id] = grid
        self._persist_to_store()
        self._logger.info(
            'Added plot grid %s (%s) with size %dx%d', grid.id, title, nrows, ncols
        )
        return grid.id

    def remove_grid(self, grid_id: UUID) -> None:
        """
        Remove a plot grid.

        Parameters
        ----------
        grid_id
            UUID of the grid to remove.
        """
        if grid_id in self._grids:
            title = self._grids[grid_id].title
            del self._grids[grid_id]
            self._persist_to_store()
            self._logger.info('Removed plot grid %s (%s)', grid_id, title)

    def add_plot(self, grid_id: UUID, cell: PlotCell) -> UUID:
        """
        Add a plot configuration to a grid and try to create plot if job exists.

        Parameters
        ----------
        grid_id
            UUID of the grid to add the plot to.
        cell
            Plot cell configuration.

        Returns
        -------
        :
            UUID of the added plot cell.
        """
        grid = self._grids[grid_id]
        grid.cells.append(cell)

        # Index the cell for fast lookup
        self._cell_index[cell.id] = (grid_id, len(grid.cells) - 1)

        self._try_create_plot(cell)

        self._persist_to_store()
        self._logger.info(
            'Added plot %s to grid %s at (%d,%d) for workflow %s',
            cell.id,
            grid_id,
            cell.row,
            cell.col,
            cell.config.workflow_id,
        )
        return cell.id

    def remove_plot(self, cell_id: UUID) -> None:
        """
        Remove a plot by its unique ID.

        Parameters
        ----------
        cell_id
            UUID of the cell to remove.
        """
        grid_id, cell_index = self._cell_index[cell_id]
        grid = self._grids[grid_id]
        cell = grid.cells.pop(cell_index)

        # Remove from index
        del self._cell_index[cell_id]

        # Rebuild index for cells after the removed one
        for i in range(cell_index, len(grid.cells)):
            self._cell_index[grid.cells[i].id] = (grid_id, i)

        self._persist_to_store()
        self._logger.info(
            'Removed plot %s from grid %s at (%d,%d)',
            cell_id,
            grid_id,
            cell.row,
            cell.col,
        )

    def get_plot_config(self, cell_id: UUID) -> PlotConfig:
        """
        Get configuration for a plot.

        Parameters
        ----------
        cell_id
            UUID of the plot cell.

        Returns
        -------
        :
            The plot configuration.
        """
        grid_id, cell_index = self._cell_index[cell_id]
        return self._grids[grid_id].cells[cell_index].config

    def update_plot_config(self, cell_id: UUID, new_config: PlotConfig) -> None:
        """
        Update plot configuration and regenerate plot.

        This clears the existing plot and attempts to create a new one
        with the updated configuration. If the matching job is available,
        the plot is created immediately. Otherwise, it remains in a
        pending state until refresh_plots() is called.

        Parameters
        ----------
        cell_id
            UUID of the plot cell to update.
        new_config
            New plot configuration.
        """
        grid_id, cell_index = self._cell_index[cell_id]
        cell = self._grids[grid_id].cells[cell_index]

        # Update configuration
        cell.config = new_config

        # Clear runtime state (plot needs to be regenerated)
        cell.plot = None
        cell.error = None
        cell.job_number = None

        # Try to create plot with new config
        self._try_create_plot(cell)

        # Persist updated configuration
        self._persist_to_store()

        self._logger.info('Updated plot config for cell %s', cell_id)

    def refresh_plots(self) -> None:
        """
        Refresh all plots.

        Try to create plots for any cells that are waiting for data.
        """
        for grid in self._grids.values():
            for cell in grid.cells:
                if cell.plot is None and cell.error is None:
                    self._try_create_plot(cell)

    def _try_create_plot(self, cell: PlotCell) -> None:
        """
        Try to create plot if matching job exists.

        Parameters
        ----------
        cell
            Plot cell to create plot for.
        """
        for job_number, workflow_id in self._job_service.job_info.items():
            if workflow_id == cell.config.workflow_id:
                cell.job_number = job_number
                try:
                    spec = self._plotting_controller.get_spec(cell.config.plot_name)
                    if spec.params is None:
                        params = pydantic.BaseModel()
                    else:
                        params = spec.params(**cell.config.params)

                    cell.plot = self._plotting_controller.create_plot(
                        job_number=job_number,
                        source_names=cell.config.source_names,
                        output_name=cell.config.output_name,
                        plot_name=cell.config.plot_name,
                        params=params,
                    )
                    cell.error = None
                    self._logger.info(
                        'Created plot for workflow %s at job %s',
                        cell.config.workflow_id,
                        job_number,
                    )
                except Exception as e:
                    cell.error = str(e)
                    self._logger.exception(
                        'Failed to create plot for workflow %s', cell.config.workflow_id
                    )
                break

    def _load_from_store(self) -> None:
        """Load plot grid configurations from config store."""
        if self._config_store is None:
            return

        for workflow_id in self._workflow_registry.keys():
            plot_grid_key = self._create_plot_grid_key(workflow_id)
            if self._config_store.get(plot_grid_key):
                self._logger.debug(
                    'Loaded plot grid config for %s from store', workflow_id
                )

    def _persist_to_store(self) -> None:
        """Persist plot grid configurations to config store."""
        if self._config_store is None:
            return

        self._logger.debug('Persisted plot grid configs to store')

    def _create_plot_grid_key(self, workflow_id: WorkflowId) -> WorkflowId:
        """
        Create a plot-grid-specific WorkflowId.

        Parameters
        ----------
        workflow_id
            The workflow ID to base the key on.

        Returns
        -------
        :
            A unique workflow ID for the plot grid configuration.
        """
        return WorkflowId(
            instrument=workflow_id.instrument,
            namespace="plot_grid",
            name=workflow_id.name,
            version=workflow_id.version,
        )

    def get_grid(self, grid_id: UUID) -> PlotGridConfig | None:
        """
        Get a plot grid configuration.

        Parameters
        ----------
        grid_id
            UUID of the grid to retrieve.

        Returns
        -------
        :
            Plot grid configuration if found, None otherwise.
        """
        return self._grids.get(grid_id)

    def get_all_grids(self) -> dict[UUID, PlotGridConfig]:
        """
        Get all plot grid configurations.

        Returns
        -------
        :
            Dictionary mapping grid UUIDs to configurations.
        """
        return self._grids.copy()
