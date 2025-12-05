# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
PlotOrchestrator - Manages plot grid configurations and plot lifecycle.

Coordinates plot creation and management across multiple plot grids:
- Configuration staging and persistence
- Plot grid lifecycle (create, remove)
- Plot cell management (add, remove)
- Event-driven plot creation via JobOrchestrator subscriptions
- Grid template loading and parsing
"""

from __future__ import annotations

import copy
import logging
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NewType, Protocol
from uuid import UUID, uuid4

import pydantic

from ess.livedata.config.grid_template import GridSpec
from ess.livedata.config.workflow_spec import JobNumber, WorkflowId

from .config_store import ConfigStore
from .data_service import DataService
from .plotting_controller import PlottingController

if TYPE_CHECKING:
    import holoviews as hv

SubscriptionId = NewType('SubscriptionId', UUID)
GridId = NewType('GridId', UUID)
CellId = NewType('CellId', UUID)


class JobOrchestratorProtocol(Protocol):
    """Protocol for JobOrchestrator interface needed by PlotOrchestrator."""

    def subscribe_to_workflow(
        self, workflow_id: WorkflowId, callback: Callable[[JobNumber], None]
    ) -> tuple[SubscriptionId, bool]:
        """
        Subscribe to workflow data availability notifications.

        The callback will be called with the job_number when workflow data
        becomes available (i.e., first result data arrives from the workflow).

        If workflow data already exists when you subscribe, the callback
        will be called immediately with the current job_number.

        Parameters
        ----------
        workflow_id
            The workflow to subscribe to.
        callback
            Called with job_number when workflow data becomes available.

        Returns
        -------
        :
            Tuple of (subscription_id, callback_invoked_immediately).
            subscription_id can be used to unsubscribe.
            callback_invoked_immediately is True if the workflow was already
            running and the callback was invoked synchronously during this call.
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


@dataclass(frozen=True)
class CellGeometry:
    """
    Grid cell geometry (position and size).

    Defines the location and span of a cell in a plot grid.
    """

    row: int
    col: int
    row_span: int
    col_span: int


@dataclass
class CellState:
    """
    State of a rendered plot cell.

    Either plot or error is set (mutually exclusive).
    Both None indicates cell is waiting for workflow data.
    """

    plot: hv.DynamicMap | hv.Layout | None = None
    error: str | None = None


@dataclass
class PlotConfig:
    """Configuration for a single plot."""

    workflow_id: WorkflowId
    source_names: list[str]
    plot_name: str
    params: pydantic.BaseModel
    output_name: str = 'result'


@dataclass
class PlotCell:
    """
    Configuration for a plot cell (position, size, and what to plot).

    The plots are placed in the given row and col of a :py:class:`PlotGrid`, spanning
    the given number of rows and columns.

    A cell can have multiple layers (overlay layers on top of the primary layer).
    The first layer (config) is the primary layer that determines the cell's
    display title and primary data source. Additional layers are overlaid on top.

    Parameters
    ----------
    geometry:
        Position and size of the cell in the grid.
    config:
        Configuration for the primary layer.
    additional_layers:
        Optional list of additional layer configurations to overlay.
    """

    geometry: CellGeometry
    config: PlotConfig
    additional_layers: list[PlotConfig] = field(default_factory=list)


@dataclass
class PlotGridConfig:
    """A plot grid tab configuration."""

    title: str = ""
    nrows: int = 3
    ncols: int = 3
    cells: dict[CellId, PlotCell] = field(default_factory=dict)


GridCreatedCallback = Callable[[GridId, PlotGridConfig], None]
GridRemovedCallback = Callable[[GridId], None]
CellRemovedCallback = Callable[[GridId, CellGeometry], None]


class CellUpdatedCallback(Protocol):
    """Callback for cell updates with all keyword-only parameters."""

    def __call__(
        self,
        *,
        grid_id: GridId,
        cell_id: CellId,
        cell: PlotCell,
        plot: Any = None,
        error: str | None = None,
    ) -> None:
        """
        Handle cell update.

        Parameters
        ----------
        grid_id
            ID of the grid containing the cell.
        cell_id
            ID of the cell being updated.
        cell
            Plot cell configuration.
        plot
            The plot widget, or None if not yet available.
        error
            Error message if plot creation failed, or None.
        """


@dataclass
class LifecycleSubscription:
    """Subscription to plot grid lifecycle events."""

    on_grid_created: GridCreatedCallback | None = None
    on_grid_removed: GridRemovedCallback | None = None
    on_cell_updated: CellUpdatedCallback | None = None
    on_cell_removed: CellRemovedCallback | None = None


class PlotOrchestrator:
    """Manages plot grid configurations and plot lifecycle."""

    def __init__(
        self,
        *,
        plotting_controller: PlottingController,
        job_orchestrator: JobOrchestratorProtocol,
        data_service: DataService,
        config_store: ConfigStore | None = None,
        raw_templates: Sequence[dict[str, Any]] = (),
    ) -> None:
        """
        Initialize the plot orchestrator.

        Parameters
        ----------
        plotting_controller
            Controller for creating plots.
        job_orchestrator
            Orchestrator for subscribing to workflow availability.
        data_service
            DataService for monitoring data arrival.
        config_store
            Optional store for persisting plot grid configurations across sessions.
        raw_templates
            Raw grid template dicts loaded from YAML files. These are parsed
            during initialization and made available via get_available_templates().
        """
        self._plotting_controller = plotting_controller
        self._job_orchestrator = job_orchestrator
        self._data_service = data_service
        self._config_store = config_store
        self._logger = logging.getLogger(__name__)

        self._grids: dict[GridId, PlotGridConfig] = {}
        self._cell_to_grid: dict[CellId, GridId] = {}
        self._cell_to_subscription: dict[CellId, SubscriptionId] = {}
        self._cell_state: dict[CellId, CellState] = {}
        self._lifecycle_subscribers: dict[SubscriptionId, LifecycleSubscription] = {}

        # Parse templates (requires plotter registry, so must be done here)
        self._templates = self._parse_grid_specs(list(raw_templates))

        # Load persisted configurations
        self._load_from_store()

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
        self._notify_grid_created(grid_id)
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
            self._notify_grid_removed(grid_id)

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
        self._cell_to_grid[cell_id] = grid_id

        # Subscribe to workflow
        self._subscribe_and_setup(grid_id, cell_id, cell.config.workflow_id)

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

        # Remove stored state
        self._cell_state.pop(cell_id, None)

        # Remove from grid and mapping
        del grid.cells[cell_id]
        del self._cell_to_grid[cell_id]

        self._persist_to_store()
        self._logger.info(
            'Removed plot %s from grid %s at (%d,%d)',
            cell_id,
            grid_id,
            cell.geometry.row,
            cell.geometry.col,
        )
        self._notify_cell_removed(grid_id, cell_id, cell)

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

    def get_cell_state(
        self, cell_id: CellId
    ) -> tuple[hv.DynamicMap | hv.Layout | None, str | None]:
        """
        Get the current plot and error state for a cell.

        This is used by UI components to retrieve the current state when
        initializing from existing cells (e.g., after page reload).

        Parameters
        ----------
        cell_id
            ID of the plot cell.

        Returns
        -------
        :
            Tuple of (plot, error) where plot is the HoloViews object if
            available (None otherwise), and error is the error message if
            plot creation failed (None otherwise).
        """
        state = self._cell_state.get(cell_id, CellState())
        return state.plot, state.error

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

        # Clear stored state since config changed (new plot will be created)
        self._cell_state.pop(cell_id, None)

        # Re-subscribe to workflow
        self._subscribe_and_setup(grid_id, cell_id, new_config.workflow_id)

    def add_layer(self, cell_id: CellId, layer_config: PlotConfig) -> None:
        """
        Add an overlay layer to a plot cell.

        The new layer is added on top of existing layers. The cell's plot
        will be recreated to include the new layer.

        Parameters
        ----------
        cell_id
            ID of the plot cell to add the layer to.
        layer_config
            Configuration for the new layer.
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        # Add the layer
        cell.additional_layers.append(layer_config)

        # Unsubscribe from current workflow notifications
        self._job_orchestrator.unsubscribe(self._cell_to_subscription[cell_id])

        # Clear stored state (plot needs to be recreated with new layer)
        self._cell_state.pop(cell_id, None)

        # Re-subscribe to workflow (will recreate plot with all layers)
        self._subscribe_and_setup(grid_id, cell_id, cell.config.workflow_id)

        self._persist_to_store()
        self._logger.info(
            'Added layer to cell %s (now %d layers)',
            cell_id,
            len(cell.additional_layers) + 1,
        )

    def remove_layer(self, cell_id: CellId, layer_index: int) -> None:
        """
        Remove an overlay layer from a plot cell.

        Parameters
        ----------
        cell_id
            ID of the plot cell to remove the layer from.
        layer_index
            Index of the layer to remove (0 = first additional layer).
            Cannot remove the primary layer (config).

        Raises
        ------
        IndexError
            If layer_index is out of range.
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        if layer_index < 0 or layer_index >= len(cell.additional_layers):
            raise IndexError(
                f"Layer index {layer_index} out of range "
                f"(0-{len(cell.additional_layers) - 1})"
            )

        # Remove the layer
        cell.additional_layers.pop(layer_index)

        # Unsubscribe from current workflow notifications
        self._job_orchestrator.unsubscribe(self._cell_to_subscription[cell_id])

        # Clear stored state (plot needs to be recreated)
        self._cell_state.pop(cell_id, None)

        # Re-subscribe to workflow
        self._subscribe_and_setup(grid_id, cell_id, cell.config.workflow_id)

        self._persist_to_store()
        self._logger.info(
            'Removed layer %d from cell %s (now %d layers)',
            layer_index,
            cell_id,
            len(cell.additional_layers) + 1,
        )

    def get_layer_count(self, cell_id: CellId) -> int:
        """
        Get the number of layers in a plot cell.

        Parameters
        ----------
        cell_id
            ID of the plot cell.

        Returns
        -------
        :
            Total number of layers (1 + number of additional layers).
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]
        return 1 + len(cell.additional_layers)

    def get_all_layers(self, cell_id: CellId) -> list[PlotConfig]:
        """
        Get all layer configurations for a plot cell.

        Parameters
        ----------
        cell_id
            ID of the plot cell.

        Returns
        -------
        :
            List of all layer configurations, with the primary layer first.
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]
        return [cell.config, *cell.additional_layers]

    def _subscribe_and_setup(
        self, grid_id: GridId, cell_id: CellId, workflow_id: WorkflowId
    ) -> None:
        """
        Subscribe to workflow availability and set up initial notification.

        This method handles two scenarios depending on workflow state:

        **Scenario A: Workflow not yet running**

        1. Subscribe to workflow (callback not invoked)
        2. Notify UI that cell is "waiting for workflow"
        3. Later, when workflow is committed, callback fires -> _on_job_available

        **Scenario B: Workflow already running**

        1. Subscribe to workflow (callback invoked immediately with job_number)
        2. _on_job_available sets up data pipeline
        3. If data exists: plot created immediately
           If no data yet: notify UI "waiting for data"

        In both scenarios, the UI receives exactly one notification from this method
        or from _on_job_available, never both.

        Parameters
        ----------
        grid_id
            ID of the grid containing the cell.
        cell_id
            ID of the cell to set up.
        workflow_id
            ID of the workflow to subscribe to.
        """

        def on_workflow_available(job_number: JobNumber) -> None:
            self._on_job_available(cell_id, job_number)

        # Subscribe to workflow availability.
        # Returns whether callback was invoked immediately (workflow already running).
        subscription_id, was_invoked = self._job_orchestrator.subscribe_to_workflow(
            workflow_id=workflow_id,
            callback=on_workflow_available,
        )
        self._cell_to_subscription[cell_id] = subscription_id

        # Scenario A: Workflow doesn't exist yet.
        # Notify UI that cell is waiting for workflow to be committed.
        # When workflow starts, _on_job_available will handle the rest.
        if not was_invoked:
            cell = self._grids[grid_id].cells[cell_id]
            self._notify_cell_updated(grid_id, cell_id, cell)

        # Persist updated state
        self._persist_to_store()

    def _on_job_available(self, cell_id: CellId, job_number: JobNumber) -> None:
        """
        Handle workflow availability notification from JobOrchestrator.

        Called when a workflow job becomes available (either immediately during
        subscription if workflow was already running, or later when committed).

        **Flow:**

        1. Set up data pipeline with on_first_data callback
        2. If data already exists in DataService:
           - on_first_data fires immediately -> plot created -> state stored
           - No notification here (on_first_data already notified)
        3. If no data yet:
           - Notify UI that cell is "waiting for data"
           - Later when data arrives, on_first_data fires -> plot created

        **State tracking:**

        - ``self._cell_state[cell_id]`` is set when plot is created or on error
        - The check ``if cell_id not in self._cell_state`` prevents double
          notification when on_first_data already ran synchronously

        Parameters
        ----------
        cell_id
            ID of the plot cell to create plot for.
        job_number
            Job number for the workflow.
        """
        # Defensive check: cell may have been removed before callback fires
        if cell_id not in self._cell_to_grid:
            self._logger.warning(
                'Ignoring workflow availability for removed cell_id=%s', cell_id
            )
            return

        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        # Check if this is a multi-layer cell
        has_multiple_layers = len(cell.additional_layers) > 0

        if has_multiple_layers:
            # Multi-layer path: use composed pipeline
            self._setup_multi_layer_pipeline(cell_id, grid_id, cell, job_number)
        else:
            # Single-layer path: use existing pipeline
            self._setup_single_layer_pipeline(cell_id, grid_id, cell, job_number)

    def _setup_single_layer_pipeline(
        self,
        cell_id: CellId,
        grid_id: GridId,
        cell: PlotCell,
        job_number: JobNumber,
    ) -> None:
        """Set up data pipeline for a single-layer cell."""

        def on_data_arrived(pipe) -> None:
            """Create plot when first data arrives for the pipeline."""
            self._logger.debug(
                'Data arrived for cell_id=%s, job_number=%s, creating plot',
                cell_id,
                job_number,
            )
            # Create the plot with the now-populated pipe
            try:
                plot = self._plotting_controller.create_plot_from_pipeline(
                    plot_name=cell.config.plot_name,
                    params=cell.config.params,
                    pipe=pipe,
                )
                # Store the plot so late subscribers can access it
                self._cell_state[cell_id] = CellState(plot=plot)
                self._notify_cell_updated(grid_id, cell_id, cell, plot=plot)
            except Exception:
                error_msg = traceback.format_exc()
                self._logger.exception('Failed to create plot for cell_id=%s', cell_id)
                # Store the error so late subscribers can see it
                self._cell_state[cell_id] = CellState(error=error_msg)
                self._notify_cell_updated(grid_id, cell_id, cell, error=error_msg)

        # Set up data pipeline with callback
        try:
            self._plotting_controller.setup_data_pipeline(
                job_number=job_number,
                workflow_id=cell.config.workflow_id,
                source_names=cell.config.source_names,
                output_name=cell.config.output_name,
                plot_name=cell.config.plot_name,
                params=cell.config.params,
                on_first_data=on_data_arrived,
            )
        except Exception:
            error_msg = traceback.format_exc()
            self._logger.exception(
                'Failed to set up data pipeline for cell_id=%s', cell_id
            )
            self._cell_state[cell_id] = CellState(error=error_msg)
            self._notify_cell_updated(grid_id, cell_id, cell, error=error_msg)
            return

        # Case 2: Workflow exists but data hasn't arrived yet
        # Notify UI that cell is waiting for first data to arrive.
        # When data arrives, on_data_arrived callback above will notify with the plot.
        # (If data was already present, on_data_arrived ran and stored state)
        if cell_id not in self._cell_state:
            self._notify_cell_updated(grid_id, cell_id, cell)

    def _setup_multi_layer_pipeline(
        self,
        cell_id: CellId,
        grid_id: GridId,
        cell: PlotCell,
        job_number: JobNumber,
    ) -> None:
        """Set up data pipelines for a multi-layer cell.

        Creates pipelines for all layers and composes them when all are ready.
        """

        def on_all_layers_ready(composed_plot) -> None:
            """Handle composed plot when all layers have data."""
            self._logger.debug(
                'All layers ready for cell_id=%s, job_number=%s',
                cell_id,
                job_number,
            )
            try:
                # Store the composed plot
                self._cell_state[cell_id] = CellState(plot=composed_plot)
                self._notify_cell_updated(grid_id, cell_id, cell, plot=composed_plot)
            except Exception:
                error_msg = traceback.format_exc()
                self._logger.exception(
                    'Failed to notify composed plot for cell_id=%s', cell_id
                )
                self._cell_state[cell_id] = CellState(error=error_msg)
                self._notify_cell_updated(grid_id, cell_id, cell, error=error_msg)

        # Build layer configs list: primary + additional layers
        layer_configs = [
            (
                cell.config.workflow_id,
                cell.config.source_names,
                cell.config.output_name,
                cell.config.plot_name,
                cell.config.params,
            ),
            *(
                (
                    layer.workflow_id,
                    layer.source_names,
                    layer.output_name,
                    layer.plot_name,
                    layer.params,
                )
                for layer in cell.additional_layers
            ),
        ]

        try:
            self._plotting_controller.setup_multi_layer_pipeline(
                job_number=job_number,
                layer_configs=layer_configs,
                on_all_layers_ready=on_all_layers_ready,
            )
        except Exception:
            error_msg = traceback.format_exc()
            self._logger.exception(
                'Failed to set up multi-layer pipeline for cell_id=%s', cell_id
            )
            self._cell_state[cell_id] = CellState(error=error_msg)
            self._notify_cell_updated(grid_id, cell_id, cell, error=error_msg)
            return

        # Notify UI that cell is waiting for data (if not already set)
        if cell_id not in self._cell_state:
            self._notify_cell_updated(grid_id, cell_id, cell)

    def _validate_params(
        self, plot_name: str, params: dict[str, Any]
    ) -> pydantic.BaseModel | None:
        """
        Validate params dict against the plotter's spec.

        Returns validated model, or None if plotter is unknown (cell should be skipped).
        Falls back to default-constructed model if validation fails.

        Parameters
        ----------
        plot_name
            Name of the plotter.
        params
            Parameters dict to validate.

        Returns
        -------
        :
            Validated params model, or None if plotter is unknown.
        """
        try:
            spec = self._plotting_controller.get_spec(plot_name)
        except KeyError:
            self._logger.warning('Skipping cell with unknown plotter %s', plot_name)
            return None

        if spec.params is None:
            return pydantic.BaseModel()

        try:
            return spec.params(**params)
        except Exception:
            self._logger.warning(
                'Params validation failed for plotter %s, using defaults', plot_name
            )
            return spec.params()

    def _parse_config_data(self, config_data: dict[str, Any]) -> PlotConfig | None:
        """
        Parse a raw config dict into a PlotConfig.

        Parameters
        ----------
        config_data
            Configuration dict with workflow_id, source_names, plot_name.
            Optional: output_name, params.

        Returns
        -------
        :
            Parsed PlotConfig, or None if the plotter is unknown.
        """
        plot_name = config_data['plot_name']

        # Validate params, returning None if plotter is unknown
        params = self._validate_params(plot_name, config_data.get('params', {}))
        if params is None:
            return None

        return PlotConfig(
            workflow_id=WorkflowId.from_string(config_data['workflow_id']),
            output_name=config_data.get('output_name'),
            source_names=config_data['source_names'],
            plot_name=plot_name,
            params=params,
        )

    def parse_raw_cell(self, cell_data: dict[str, Any]) -> PlotCell | None:
        """
        Parse a raw cell dict into a typed PlotCell.

        Use this to convert cells from templates or persisted configurations
        into typed objects that can be passed to :py:meth:`add_grid`.

        Parameters
        ----------
        cell_data
            Cell configuration dict with 'geometry' and 'config' keys.
            The config must contain: workflow_id, source_names, plot_name.
            Optional: output_name, params, additional_layers.

        Returns
        -------
        :
            Parsed PlotCell, or None if the plotter is unknown (cell skipped).
        """
        # Parse primary layer config
        config = self._parse_config_data(cell_data['config'])
        if config is None:
            return None

        geometry = CellGeometry(
            row=cell_data['geometry']['row'],
            col=cell_data['geometry']['col'],
            row_span=cell_data['geometry']['row_span'],
            col_span=cell_data['geometry']['col_span'],
        )

        # Parse additional layers (if present)
        additional_layers: list[PlotConfig] = []
        for layer_data in cell_data.get('additional_layers', []):
            layer_config = self._parse_config_data(layer_data)
            if layer_config is not None:
                additional_layers.append(layer_config)

        return PlotCell(
            geometry=geometry, config=config, additional_layers=additional_layers
        )

    def get_available_templates(self) -> list[GridSpec]:
        """
        Get available grid templates.

        Templates are parsed from raw YAML during initialization and cached.
        Use these to populate template selectors in the UI.

        Returns
        -------
        :
            List of validated GridSpec objects. Empty if no templates were
            provided or all templates failed to parse.
        """
        return list(self._templates)

    def _parse_grid_specs(self, raw_specs: list[dict[str, Any]]) -> list[GridSpec]:
        """
        Parse raw grid dicts into validated GridSpec objects.

        Validates cells using the plotter registry. Cells with unknown plotters
        are skipped (logged as warnings).

        Parameters
        ----------
        raw_specs
            List of raw grid dicts from template files or persisted configurations.

        Returns
        -------
        :
            List of validated GridSpec objects.
        """
        specs: list[GridSpec] = []

        for raw in raw_specs:
            spec = self._parse_single_spec(raw)
            if spec is not None:
                specs.append(spec)

        self._logger.info('Parsed %d grid spec(s)', len(specs))
        return specs

    def _parse_single_spec(self, raw: dict[str, Any]) -> GridSpec | None:
        """
        Parse a single raw dict into a GridSpec.

        Parameters
        ----------
        raw
            Raw grid dict.

        Returns
        -------
        :
            GridSpec if parsing succeeded, None otherwise.
        """
        try:
            # Use title as display name, falling back to 'Untitled'
            name = raw.get('title', 'Untitled')

            # Parse cells using orchestrator's validation
            raw_cells = raw.get('cells', [])
            cells = []
            for cell_data in raw_cells:
                parsed = self.parse_raw_cell(cell_data)
                if parsed is not None:
                    cells.append(parsed)

            return GridSpec(
                name=name,
                title=raw.get('title', name),
                description=raw.get('description', ''),
                nrows=raw.get('nrows', 3),
                ncols=raw.get('ncols', 3),
                cells=tuple(cells),
            )

        except Exception:
            self._logger.exception(
                'Failed to parse grid spec: %s', raw.get('title', 'unknown')
            )
            return None

    def _serialize_grids(self) -> list[dict[str, Any]]:
        """
        Serialize all grids to list for persistence.

        Returns
        -------
        :
            List of grid configurations. UUIDs are not persisted as they are
            runtime identity handles with no cross-session significance.
        """

        def serialize_config(config: PlotConfig) -> dict[str, Any]:
            return {
                'workflow_id': str(config.workflow_id),
                'output_name': config.output_name,
                'source_names': config.source_names,
                'plot_name': config.plot_name,
                'params': config.params.model_dump(mode='json'),
            }

        return [
            {
                'title': grid.title,
                'nrows': grid.nrows,
                'ncols': grid.ncols,
                'cells': [
                    {
                        'geometry': {
                            'row': cell.geometry.row,
                            'col': cell.geometry.col,
                            'row_span': cell.geometry.row_span,
                            'col_span': cell.geometry.col_span,
                        },
                        'config': serialize_config(cell.config),
                        'additional_layers': [
                            serialize_config(layer) for layer in cell.additional_layers
                        ],
                    }
                    for cell in grid.cells.values()
                ],
            }
            for grid in self._grids.values()
        ]

    def _load_from_store(self) -> None:
        """Load plot grid configurations from config store."""
        if self._config_store is None:
            return

        try:
            data = self._config_store.get('plot_grids')
            if data is None:
                self._logger.debug('No persisted plot grids found in store')
                return

            # Parse stored configs as GridSpecs (same parser as templates)
            raw_grids = data.get('grids', [])
            specs = self._parse_grid_specs(raw_grids)

            # Apply each spec through the normal API
            for spec in specs:
                grid_id = self.add_grid(spec.title, spec.nrows, spec.ncols)
                for cell in spec.cells:
                    self.add_plot(grid_id, cell)

            self._logger.info('Loaded %d plot grids from store', len(specs))
        except Exception:
            self._logger.exception('Failed to load plot grids from store')

    def _persist_to_store(self) -> None:
        """Persist plot grid configurations to config store."""
        if self._config_store is None:
            return

        try:
            serialized = self._serialize_grids()
            self._config_store['plot_grids'] = {'grids': serialized}
            self._logger.debug('Persisted %d plot grids to store', len(self._grids))
        except Exception as e:
            self._logger.error('Failed to persist plot grids: %s', e)

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
            Deep copy of plot grid configuration if found, None otherwise.
            Safe to modify without affecting internal state.
        """
        grid = self._grids.get(grid_id)
        return copy.deepcopy(grid) if grid is not None else None

    def get_all_grids(self) -> dict[GridId, PlotGridConfig]:
        """
        Get all plot grid configurations.

        Returns
        -------
        :
            Deep copy of dictionary mapping grid IDs to configurations.
            Safe to modify without affecting internal state.
        """
        return copy.deepcopy(self._grids)

    def subscribe_to_lifecycle(
        self,
        *,
        on_grid_created: GridCreatedCallback | None = None,
        on_grid_removed: GridRemovedCallback | None = None,
        on_cell_updated: CellUpdatedCallback | None = None,
        on_cell_removed: CellRemovedCallback | None = None,
    ) -> SubscriptionId:
        """
        Subscribe to plot grid lifecycle events.

        Subscribers will be notified when grids or cells are created, updated,
        or removed. At least one callback must be provided.

        Callbacks are fired in the order grids are created. Late subscribers
        (subscribing after grids already exist) should call `get_all_grids()`
        to get existing grids in their creation order before relying on callbacks
        for new grids.

        Parameters
        ----------
        on_grid_created
            Called when a new grid is created with (grid_id, grid_config).
        on_grid_removed
            Called when a grid is removed.
        on_cell_updated
            Called when a cell is added or updated.
        on_cell_removed
            Called when a cell is removed.

        Returns
        -------
        :
            Subscription ID that can be used to unsubscribe.
        """
        subscription_id = SubscriptionId(uuid4())
        self._lifecycle_subscribers[subscription_id] = LifecycleSubscription(
            on_grid_created=on_grid_created,
            on_grid_removed=on_grid_removed,
            on_cell_updated=on_cell_updated,
            on_cell_removed=on_cell_removed,
        )
        return subscription_id

    def unsubscribe_from_lifecycle(self, subscription_id: SubscriptionId) -> None:
        """
        Unsubscribe from plot grid lifecycle events.

        Parameters
        ----------
        subscription_id
            The subscription ID returned from subscribe_to_lifecycle.
        """
        if subscription_id in self._lifecycle_subscribers:
            del self._lifecycle_subscribers[subscription_id]

    def _notify_grid_created(self, grid_id: GridId) -> None:
        """Notify subscribers that a grid was created."""
        grid = self._grids[grid_id]
        for subscription in self._lifecycle_subscribers.values():
            if subscription.on_grid_created:
                try:
                    subscription.on_grid_created(grid_id, grid)
                except Exception:
                    self._logger.exception(
                        'Error in grid created callback for grid %s', grid_id
                    )

    def _notify_grid_removed(self, grid_id: GridId) -> None:
        """Notify subscribers that a grid was removed."""
        for subscription in self._lifecycle_subscribers.values():
            if subscription.on_grid_removed:
                try:
                    subscription.on_grid_removed(grid_id)
                except Exception:
                    self._logger.exception(
                        'Error in grid removed callback for grid %s', grid_id
                    )

    def _notify_cell_updated(
        self,
        grid_id: GridId,
        cell_id: CellId,
        cell: PlotCell,
        plot: hv.DynamicMap | hv.Layout | None = None,
        error: str | None = None,
    ) -> None:
        """Notify subscribers that a cell was added or updated."""
        for subscription in self._lifecycle_subscribers.values():
            if subscription.on_cell_updated:
                try:
                    subscription.on_cell_updated(
                        grid_id=grid_id,
                        cell_id=cell_id,
                        cell=cell,
                        plot=plot,
                        error=error,
                    )
                except Exception:
                    self._logger.exception(
                        'Error in cell updated callback for grid %s cell %s at (%d,%d)',
                        grid_id,
                        cell_id,
                        cell.geometry.row,
                        cell.geometry.col,
                    )

    def _notify_cell_removed(
        self, grid_id: GridId, cell_id: CellId, cell: PlotCell
    ) -> None:
        """Notify subscribers that a cell was removed."""
        for subscription in self._lifecycle_subscribers.values():
            if subscription.on_cell_removed:
                try:
                    subscription.on_cell_removed(grid_id, cell.geometry)
                except Exception:
                    self._logger.exception(
                        'Error in cell removed callback for grid %s cell %s at (%d,%d)',
                        grid_id,
                        cell_id,
                        cell.geometry.row,
                        cell.geometry.col,
                    )

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
