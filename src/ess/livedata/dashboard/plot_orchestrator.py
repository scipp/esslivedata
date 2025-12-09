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
- Layer composition for multi-layer plots

Layer Composition
-----------------

The orchestrator supports two types of layers:

**Pipeline layers**: Data from reduction workflows via StreamManager subscription.
These use existing Plotter infrastructure and create DynamicMaps that update
as new data arrives.

**Static layers**: Fixed data provided directly (e.g., peak marker positions,
threshold lines). These create DynamicMaps with Pipes that can be updated
programmatically but don't subscribe to workflow data.

Layers are composed via HoloViews' ``*`` operator, which creates overlays.
Each layer creates its own DynamicMap, enabling independent updates and
correct handling of interactive streams.
"""

from __future__ import annotations

import copy
import logging
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NewType, Protocol
from uuid import UUID, uuid4

import holoviews as hv
import pydantic

from ess.livedata.config.grid_template import GridSpec
from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    ResultKey,
    WorkflowId,
)

from .config_store import ConfigStore
from .data_service import DataService
from .plot_params import create_extractors_from_params
from .plotting import plotter_registry
from .plotting_controller import PlottingController
from .stream_manager import StreamManager

if TYPE_CHECKING:
    pass

SubscriptionId = NewType('SubscriptionId', UUID)
GridId = NewType('GridId', UUID)
CellId = NewType('CellId', UUID)

logger = logging.getLogger(__name__)


# =============================================================================
# Static Layer Infrastructure
# =============================================================================


@dataclass(frozen=True)
class StaticSource:
    """Fixed data provided directly for static layers.

    Use for data that doesn't change during the plot's lifetime,
    such as reference values, peak positions, or theoretical curves.

    Parameters
    ----------
    data:
        The static data. Type depends on the element factory:
        - For vlines: list[float] of x positions
        - For hlines: list[float] of y positions
    """

    data: Any


class ElementFactory(ABC):
    """Base class for factories that create HoloViews elements from data."""

    @abstractmethod
    def create_element(self, data: Any) -> hv.Element:
        """Create a HoloViews element from the provided data."""


class VLinesFactory(ElementFactory):
    """Factory for vertical line elements."""

    def __init__(self, **opts: Any) -> None:
        """Initialize with HoloViews options (e.g., color, line_dash)."""
        self._opts = opts

    def create_element(self, data: list[float] | None) -> hv.Overlay:
        """Create vertical lines at the given x positions."""
        if not data:
            return hv.Overlay([])
        return hv.Overlay([hv.VLine(x).opts(**self._opts) for x in data])


class HLinesFactory(ElementFactory):
    """Factory for horizontal line elements."""

    def __init__(self, **opts: Any) -> None:
        """Initialize with HoloViews options (e.g., color, line_dash)."""
        self._opts = opts

    def create_element(self, data: list[float] | None) -> hv.Overlay:
        """Create horizontal lines at the given y positions."""
        if not data:
            return hv.Overlay([])
        return hv.Overlay([hv.HLine(y).opts(**self._opts) for y in data])


_STATIC_ELEMENT_FACTORIES: dict[str, type[ElementFactory]] = {
    'vlines': VLinesFactory,
    'hlines': HLinesFactory,
}


@dataclass(frozen=True)
class StaticLayerConfig:
    """Configuration for a static layer (non-pipeline data).

    Parameters
    ----------
    name:
        Unique identifier for this layer within the cell.
    element:
        Element type: "vlines", "hlines", etc.
    source:
        Static data source.
    params:
        Element-specific styling parameters (passed to factory).
    """

    name: str
    element: str
    source: StaticSource
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StaticLayerState:
    """Runtime state for a static layer.

    Parameters
    ----------
    config:
        The layer configuration.
    pipe:
        HoloViews Pipe for pushing data updates.
    dmap:
        The DynamicMap that renders this layer.
    factory:
        The element factory for creating visuals.
    """

    config: StaticLayerConfig
    pipe: hv.streams.Pipe
    dmap: hv.DynamicMap
    factory: ElementFactory


# =============================================================================
# Core Types
# =============================================================================


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

    Tracks both the composed plot and individual layer DynamicMaps
    for multi-layer cells. For single-layer cells, layer_dmaps contains
    just the one DynamicMap.

    Either plot or error is set (mutually exclusive).
    Both None indicates cell is waiting for workflow data.

    Parameters
    ----------
    plot:
        The composed plot (overlay of all layers), or None if not ready.
    error:
        Error message if plot creation failed, or None.
    layer_dmaps:
        Dict mapping layer index to DynamicMap for pipeline layers.
    static_layer_states:
        Dict mapping layer name to StaticLayerState for static layers.
    """

    plot: hv.DynamicMap | hv.Layout | None = None
    error: str | None = None
    layer_dmaps: dict[int, hv.DynamicMap] = field(default_factory=dict)
    static_layer_states: dict[str, StaticLayerState] = field(default_factory=dict)


@dataclass
class PlotConfig:
    """Configuration for a single pipeline layer (workflow data)."""

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

    A cell can have multiple layers that are composed via overlay:
    - **Pipeline layers** (layers): Data from reduction workflows
    - **Static layers** (static_layers): Fixed data like peak markers

    All layers are peers - the first pipeline layer is used for display
    purposes (title, sizing) but has no special status otherwise.

    Parameters
    ----------
    geometry:
        Position and size of the cell in the grid.
    layers:
        List of pipeline layer configurations. Must have at least one.
    static_layers:
        List of static layer configurations (optional).
    """

    geometry: CellGeometry
    layers: list[PlotConfig] = field(default_factory=list)
    static_layers: list[StaticLayerConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("PlotCell must have at least one layer")


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
        stream_manager: StreamManager | None = None,
        config_store: ConfigStore | None = None,
        raw_templates: Sequence[dict[str, Any]] = (),
    ) -> None:
        """
        Initialize the plot orchestrator.

        Parameters
        ----------
        plotting_controller
            Controller for creating single-layer plots.
        job_orchestrator
            Orchestrator for subscribing to workflow availability.
        data_service
            DataService for monitoring data arrival.
        stream_manager
            Manager for creating data streams for multi-layer composition.
            If None, multi-layer cells will fall back to PlottingController.
        config_store
            Optional store for persisting plot grid configurations across sessions.
        raw_templates
            Raw grid template dicts loaded from YAML files. These are parsed
            during initialization and made available via get_available_templates().
        """
        self._plotting_controller = plotting_controller
        self._job_orchestrator = job_orchestrator
        self._data_service = data_service
        self._stream_manager = stream_manager
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

        # Subscribe to workflow(s) - use first layer's workflow for job notifications
        # All layers will set up their own data subscriptions in _on_job_available
        self._subscribe_and_setup(grid_id, cell_id, cell.layers[0].workflow_id)

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
        Get configuration for the first layer of a plot.

        For multi-layer cells, use get_all_layers() to get all configurations.

        Parameters
        ----------
        cell_id
            ID of the plot cell.

        Returns
        -------
        :
            The first layer's plot configuration.
        """
        grid_id = self._cell_to_grid[cell_id]
        return self._grids[grid_id].cells[cell_id].layers[0]

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

    def update_plot_config(
        self, cell_id: CellId, new_config: PlotConfig, layer_index: int = 0
    ) -> None:
        """
        Update configuration for a specific layer and resubscribe to workflow.

        This resubscribes to the workflow.
        When the workflow is next committed, the plot will be recreated
        with the new configuration.

        Parameters
        ----------
        cell_id
            ID of the plot cell to update.
        new_config
            New plot configuration for the layer.
        layer_index
            Index of the layer to update (default: 0, the first layer).
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        if layer_index < 0 or layer_index >= len(cell.layers):
            raise IndexError(
                f"Layer index {layer_index} out of range (0-{len(cell.layers) - 1})"
            )

        # Unsubscribe from old workflow notifications
        self._job_orchestrator.unsubscribe(self._cell_to_subscription[cell_id])

        # Update configuration for the specified layer
        cell.layers[layer_index] = new_config

        # Clear stored state since config changed (new plot will be created)
        self._cell_state.pop(cell_id, None)

        # Re-subscribe to workflow (use first layer's workflow)
        self._subscribe_and_setup(grid_id, cell_id, cell.layers[0].workflow_id)

    def add_layer(self, cell_id: CellId, layer_config: PlotConfig) -> None:
        """
        Add a layer to a plot cell.

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
        cell.layers.append(layer_config)

        # Unsubscribe from current workflow notifications
        self._job_orchestrator.unsubscribe(self._cell_to_subscription[cell_id])

        # Clear stored state (plot needs to be recreated with new layer)
        self._cell_state.pop(cell_id, None)

        # Re-subscribe to workflow (will recreate plot with all layers)
        self._subscribe_and_setup(grid_id, cell_id, cell.layers[0].workflow_id)

        self._persist_to_store()
        self._logger.info(
            'Added layer to cell %s (now %d layers)',
            cell_id,
            len(cell.layers),
        )

    def remove_layer(self, cell_id: CellId, layer_index: int) -> None:
        """
        Remove a layer from a plot cell.

        If this removes the last layer, the entire cell is removed.

        Parameters
        ----------
        cell_id
            ID of the plot cell to remove the layer from.
        layer_index
            Index of the layer to remove.

        Raises
        ------
        IndexError
            If layer_index is out of range.
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        if layer_index < 0 or layer_index >= len(cell.layers):
            raise IndexError(
                f"Layer index {layer_index} out of range (0-{len(cell.layers) - 1})"
            )

        # If this is the last layer, remove the entire cell
        if len(cell.layers) == 1:
            self._logger.info(
                'Removing last layer from cell %s, removing cell', cell_id
            )
            self.remove_plot(cell_id)
            return

        # Remove the layer
        cell.layers.pop(layer_index)

        # Unsubscribe from current workflow notifications
        self._job_orchestrator.unsubscribe(self._cell_to_subscription[cell_id])

        # Clear stored state (plot needs to be recreated)
        self._cell_state.pop(cell_id, None)

        # Re-subscribe to workflow (use first layer's workflow)
        self._subscribe_and_setup(grid_id, cell_id, cell.layers[0].workflow_id)

        self._persist_to_store()
        self._logger.info(
            'Removed layer %d from cell %s (now %d layers)',
            layer_index,
            cell_id,
            len(cell.layers),
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
            Total number of layers.
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]
        return len(cell.layers)

    def get_all_layers(self, cell_id: CellId) -> list[PlotConfig]:
        """
        Get all pipeline layer configurations for a plot cell.

        Parameters
        ----------
        cell_id
            ID of the plot cell.

        Returns
        -------
        :
            List of all pipeline layer configurations.
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]
        return list(cell.layers)

    def add_static_layer(
        self, cell_id: CellId, layer_config: StaticLayerConfig
    ) -> None:
        """
        Add a static layer to a plot cell.

        Static layers display fixed data (e.g., peak markers, threshold lines)
        that doesn't come from workflow pipelines.

        Parameters
        ----------
        cell_id
            ID of the plot cell to add the layer to.
        layer_config
            Configuration for the static layer.
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        # Check for duplicate name
        for existing in cell.static_layers:
            if existing.name == layer_config.name:
                raise ValueError(
                    f"Static layer '{layer_config.name}' already exists in cell"
                )

        # Add the static layer
        cell.static_layers.append(layer_config)

        # Unsubscribe from current workflow notifications
        self._job_orchestrator.unsubscribe(self._cell_to_subscription[cell_id])

        # Clear stored state (plot needs to be recreated with new layer)
        self._cell_state.pop(cell_id, None)

        # Re-subscribe to workflow (will recreate plot with all layers)
        self._subscribe_and_setup(grid_id, cell_id, cell.layers[0].workflow_id)

        self._persist_to_store()
        self._logger.info(
            'Added static layer %s to cell %s',
            layer_config.name,
            cell_id,
        )

    def remove_static_layer(self, cell_id: CellId, layer_name: str) -> None:
        """
        Remove a static layer from a plot cell by name.

        Parameters
        ----------
        cell_id
            ID of the plot cell to remove the layer from.
        layer_name
            Name of the static layer to remove.

        Raises
        ------
        ValueError
            If no static layer with the given name exists.
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        # Find and remove the layer
        for i, layer in enumerate(cell.static_layers):
            if layer.name == layer_name:
                cell.static_layers.pop(i)
                break
        else:
            raise ValueError(f"Static layer '{layer_name}' not found in cell")

        # Unsubscribe from current workflow notifications
        self._job_orchestrator.unsubscribe(self._cell_to_subscription[cell_id])

        # Clear stored state (plot needs to be recreated)
        self._cell_state.pop(cell_id, None)

        # Re-subscribe to workflow
        self._subscribe_and_setup(grid_id, cell_id, cell.layers[0].workflow_id)

        self._persist_to_store()
        self._logger.info(
            'Removed static layer %s from cell %s',
            layer_name,
            cell_id,
        )

    def update_static_layer_data(
        self, cell_id: CellId, layer_name: str, new_data: Any
    ) -> None:
        """
        Update the data for a static layer without recreating the plot.

        This allows live updates to static layers (e.g., moving peak markers)
        without disrupting the pipeline layers.

        Parameters
        ----------
        cell_id
            ID of the plot cell containing the layer.
        layer_name
            Name of the static layer to update.
        new_data
            New data for the layer (type depends on element type).

        Raises
        ------
        ValueError
            If no static layer with the given name exists.
        KeyError
            If the cell has no rendered state yet.
        """
        state = self._cell_state.get(cell_id)
        if state is None:
            raise KeyError(f"Cell {cell_id} has no rendered state yet")

        static_state = state.static_layer_states.get(layer_name)
        if static_state is None:
            raise ValueError(
                f"Static layer '{layer_name}' not found in cell state. "
                f"Available: {list(state.static_layer_states.keys())}"
            )

        # Update the pipe - this triggers the DynamicMap to update
        static_state.pipe.send(new_data)
        self._logger.debug(
            'Updated static layer %s data for cell %s',
            layer_name,
            cell_id,
        )

    def get_static_layers(self, cell_id: CellId) -> list[StaticLayerConfig]:
        """
        Get all static layer configurations for a plot cell.

        Parameters
        ----------
        cell_id
            ID of the plot cell.

        Returns
        -------
        :
            List of all static layer configurations.
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]
        return list(cell.static_layers)

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

        # Determine if this needs multi-layer composition:
        # - More than one pipeline layer, OR
        # - Has static layers
        needs_composition = len(cell.layers) > 1 or len(cell.static_layers) > 0

        if needs_composition and self._stream_manager is not None:
            # Multi-layer path: compose layers directly
            self._setup_composed_layers(cell_id, grid_id, cell, job_number)
        elif len(cell.layers) == 1 and not cell.static_layers:
            # Single-layer path: use PlottingController
            self._setup_single_layer_pipeline(cell_id, grid_id, cell, job_number)
        else:
            # Fallback for multi-layer without stream_manager
            self._setup_multi_layer_fallback(cell_id, grid_id, cell, job_number)

    def _setup_single_layer_pipeline(
        self,
        cell_id: CellId,
        grid_id: GridId,
        cell: PlotCell,
        job_number: JobNumber,
    ) -> None:
        """Set up data pipeline for a single-layer cell."""
        layer = cell.layers[0]

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
                    plot_name=layer.plot_name,
                    params=layer.params,
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
                workflow_id=layer.workflow_id,
                source_names=layer.source_names,
                output_name=layer.output_name,
                plot_name=layer.plot_name,
                params=layer.params,
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

    def _setup_composed_layers(
        self,
        cell_id: CellId,
        grid_id: GridId,
        cell: PlotCell,
        job_number: JobNumber,
    ) -> None:
        """Set up composed layers directly using StreamManager.

        Creates pipelines for all pipeline layers, creates static layers,
        and composes them when all pipeline layers have data.
        """
        if self._stream_manager is None:
            raise RuntimeError('Stream manager not initialized')

        # Initialize cell state to track layer DynamicMaps
        state = CellState()
        self._cell_state[cell_id] = state

        # Create static layers immediately (they don't wait for data)
        for static_config in cell.static_layers:
            try:
                static_state = self._create_static_layer(static_config)
                state.static_layer_states[static_config.name] = static_state
            except Exception:
                error_msg = traceback.format_exc()
                self._logger.exception(
                    'Failed to create static layer %s for cell_id=%s',
                    static_config.name,
                    cell_id,
                )
                state.error = error_msg
                self._notify_cell_updated(grid_id, cell_id, cell, error=error_msg)
                return

        # Track pipeline layer readiness
        pipeline_layer_count = len(cell.layers)

        def on_layer_ready(layer_idx: int, pipe: Any) -> None:
            """Handle individual pipeline layer becoming ready."""
            layer = cell.layers[layer_idx]

            try:
                # Create plotter and DynamicMap for this layer
                plotter = plotter_registry.create_plotter(
                    layer.plot_name, params=layer.params
                )
                plotter.initialize_from_data(pipe.data)

                dmap = hv.DynamicMap(
                    plotter, streams=[pipe], kdims=plotter.kdims, cache_size=1
                )

                # Store in cell state
                state.layer_dmaps[layer_idx] = dmap

                self._logger.debug(
                    'Pipeline layer %d ready for cell_id=%s', layer_idx, cell_id
                )

                # Check if all pipeline layers are ready
                if len(state.layer_dmaps) == pipeline_layer_count:
                    self._compose_and_notify(cell_id, grid_id, cell, state)

            except Exception:
                error_msg = traceback.format_exc()
                self._logger.exception(
                    'Failed to create layer %d for cell_id=%s', layer_idx, cell_id
                )
                state.error = error_msg
                self._notify_cell_updated(grid_id, cell_id, cell, error=error_msg)

        # Set up pipeline for each pipeline layer
        for idx, layer in enumerate(cell.layers):
            try:
                self._setup_layer_pipeline(
                    job_number=job_number,
                    layer=layer,
                    layer_idx=idx,
                    on_ready=on_layer_ready,
                )
            except Exception:
                error_msg = traceback.format_exc()
                self._logger.exception(
                    'Failed to set up pipeline for layer %d cell_id=%s', idx, cell_id
                )
                state.error = error_msg
                self._notify_cell_updated(grid_id, cell_id, cell, error=error_msg)
                return

        # If no pipeline layers (only static), compose immediately
        if pipeline_layer_count == 0:
            self._compose_and_notify(cell_id, grid_id, cell, state)
        elif cell_id not in self._cell_state or state.plot is None:
            # Notify UI that cell is waiting for data
            self._notify_cell_updated(grid_id, cell_id, cell)

    def _setup_layer_pipeline(
        self,
        job_number: JobNumber,
        layer: PlotConfig,
        layer_idx: int,
        on_ready: Callable[[int, Any], None],
    ) -> None:
        """Set up data pipeline for a single pipeline layer."""
        if self._stream_manager is None:
            raise RuntimeError('Stream manager not initialized')

        # Build result keys for all sources
        keys = [
            ResultKey(
                workflow_id=layer.workflow_id,
                job_id=JobId(job_number=job_number, source_name=source_name),
                output_name=layer.output_name,
            )
            for source_name in layer.source_names
        ]

        # Get plotter spec for extractor creation
        spec = plotter_registry.get_spec(layer.plot_name)
        window = getattr(layer.params, 'window', None)
        extractors = create_extractors_from_params(keys, window, spec)

        def on_first_data(pipe: Any) -> None:
            on_ready(layer_idx, pipe)

        self._stream_manager.make_merging_stream(
            extractors, on_first_data=on_first_data
        )

    def _create_static_layer(self, config: StaticLayerConfig) -> StaticLayerState:
        """Create a static layer with its DynamicMap."""
        if config.element not in _STATIC_ELEMENT_FACTORIES:
            raise ValueError(
                f"Unknown static element type: {config.element}. "
                f"Supported: {list(_STATIC_ELEMENT_FACTORIES.keys())}"
            )

        factory_cls = _STATIC_ELEMENT_FACTORIES[config.element]
        factory = factory_cls(**config.params)

        # Create pipe with initial data
        pipe = hv.streams.Pipe(data=config.source.data)

        # Closure capture for Panel session safety
        _factory = factory

        def make_element(data: Any, _f: ElementFactory = _factory) -> hv.Element:
            return _f.create_element(data)

        dmap = hv.DynamicMap(make_element, streams=[pipe])

        return StaticLayerState(config=config, pipe=pipe, dmap=dmap, factory=factory)

    def _compose_and_notify(
        self,
        cell_id: CellId,
        grid_id: GridId,
        cell: PlotCell,
        state: CellState,
    ) -> None:
        """Compose all layers and notify with the result."""
        try:
            composed = self._compose_layers(state)
            state.plot = composed
            self._notify_cell_updated(grid_id, cell_id, cell, plot=composed)
            self._logger.info(
                'Composed %d pipeline + %d static layers for cell_id=%s',
                len(state.layer_dmaps),
                len(state.static_layer_states),
                cell_id,
            )
        except Exception:
            error_msg = traceback.format_exc()
            self._logger.exception('Failed to compose layers for cell_id=%s', cell_id)
            state.error = error_msg
            self._notify_cell_updated(grid_id, cell_id, cell, error=error_msg)

    def _compose_layers(self, state: CellState) -> hv.DynamicMap | hv.Layout:
        """Compose all layer DynamicMaps into a single overlay.

        Pipeline layers are composed in order, then static layers are added on top.
        """
        # Collect all DynamicMaps
        # KEY INSIGHT: DynamicMap evaluates to False when empty, use `is not None`
        dmaps: list[hv.DynamicMap] = []

        # Add pipeline layers in order
        for idx in sorted(state.layer_dmaps.keys()):
            dmap = state.layer_dmaps[idx]
            if dmap is not None:
                dmaps.append(dmap)

        # Add static layers
        dmaps.extend(
            static_state.dmap
            for static_state in state.static_layer_states.values()
            if static_state.dmap is not None
        )

        if not dmaps:
            # Return empty DynamicMap if no layers
            return hv.Layout([hv.DynamicMap(lambda: hv.Curve([]))]).opts(
                shared_axes=False
            )

        if len(dmaps) == 1:
            return hv.Layout([dmaps[0]]).opts(shared_axes=False)

        # Compose via * operator - first layer is "bottom", subsequent on top
        composed = dmaps[0]
        for dmap in dmaps[1:]:
            composed = composed * dmap

        return hv.Layout([composed]).opts(shared_axes=False)

    def _setup_multi_layer_fallback(
        self,
        cell_id: CellId,
        grid_id: GridId,
        cell: PlotCell,
        job_number: JobNumber,
    ) -> None:
        """Fallback: Set up multi-layer via PlottingController (no static layers).

        Used when stream_manager is not available.
        """

        def on_all_layers_ready(composed_plot: Any) -> None:
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

        # Build layer configs list from pipeline layers only (no static support)
        layer_configs = [
            (
                layer.workflow_id,
                layer.source_names,
                layer.output_name,
                layer.plot_name,
                layer.params,
            )
            for layer in cell.layers
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
            output_name=config_data.get('output_name', 'result'),
            source_names=config_data['source_names'],
            plot_name=plot_name,
            params=params,
        )

    def parse_raw_cell(self, cell_data: dict[str, Any]) -> PlotCell | None:
        """
        Parse a raw cell dict into a typed PlotCell.

        Use this to convert cells from templates or persisted configurations
        into typed objects that can be passed to :py:meth:`add_grid`.

        Supports two formats:
        - New format: 'layers' list containing all layer configs
        - Legacy format: 'config' (primary) + optional 'additional_layers'

        Parameters
        ----------
        cell_data
            Cell configuration dict with 'geometry' and either 'layers' or 'config'.
            Each layer config must contain: workflow_id, source_names, plot_name.
            Optional per layer: output_name, params.

        Returns
        -------
        :
            Parsed PlotCell, or None if all plotters are unknown (cell skipped).
        """
        geometry = CellGeometry(
            row=cell_data['geometry']['row'],
            col=cell_data['geometry']['col'],
            row_span=cell_data['geometry']['row_span'],
            col_span=cell_data['geometry']['col_span'],
        )

        # Parse layers - support both new and legacy formats
        layers: list[PlotConfig] = []

        if 'layers' in cell_data:
            # New format: all layers in a list
            for layer_data in cell_data['layers']:
                layer_config = self._parse_config_data(layer_data)
                if layer_config is not None:
                    layers.append(layer_config)
        else:
            # Legacy format: 'config' + optional 'additional_layers'
            primary_config = self._parse_config_data(cell_data['config'])
            if primary_config is not None:
                layers.append(primary_config)

            for layer_data in cell_data.get('additional_layers', []):
                layer_config = self._parse_config_data(layer_data)
                if layer_config is not None:
                    layers.append(layer_config)

        # Skip cell if no valid layers
        if not layers:
            return None

        return PlotCell(geometry=geometry, layers=layers)

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
                        'layers': [serialize_config(layer) for layer in cell.layers],
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
