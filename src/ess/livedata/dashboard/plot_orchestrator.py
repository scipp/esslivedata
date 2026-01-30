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
from .data_roles import PRIMARY
from .data_service import DataService
from .job_orchestrator import WorkflowCallbacks
from .layer_subscription import LayerSubscription, SubscriptionReady
from .plot_data_service import LayerId, PlotDataService
from .plotting_controller import PlottingController

if TYPE_CHECKING:
    from ess.livedata.config import Instrument

SubscriptionId = NewType('SubscriptionId', UUID)
GridId = NewType('GridId', UUID)
CellId = NewType('CellId', UUID)


class JobOrchestratorProtocol(Protocol):
    """Protocol for JobOrchestrator interface needed by PlotOrchestrator."""

    def subscribe_to_workflow(
        self, workflow_id: WorkflowId, callbacks: WorkflowCallbacks
    ) -> tuple[SubscriptionId, bool]:
        """
        Subscribe to workflow job lifecycle notifications.

        The on_started callback will be called with the job_number when:
        1. A workflow is committed (immediately after commit)
        2. Immediately if subscribing and workflow already has an active job

        The on_stopped callback (if provided) will be called when the workflow
        is stopped, with the job_number of the stopped job.

        Parameters
        ----------
        workflow_id
            The workflow to subscribe to.
        callbacks
            Callbacks for job lifecycle events (on_started, on_stopped).

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
        Unsubscribe from workflow lifecycle notifications.

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
class DataSourceConfig:
    """Configuration for a single data source in a plot layer.

    This defines how to connect a layer to a workflow's output.
    """

    workflow_id: WorkflowId
    source_names: list[str]
    output_name: str = 'result'


@dataclass
class PlotConfig:
    """Configuration for a single plot layer.

    The data_sources dict maps role names to DataSourceConfig:

    - **"primary"**: The main data source (required). For standard plots, this is
      the only entry. For correlation histograms, this is the data to be histogrammed.
    - **"x_axis"**: X-axis correlation data (optional). Used by correlation histograms.
    - **"y_axis"**: Y-axis correlation data (optional). For 2D correlation histograms.

    Static overlays (e.g., geometric shapes) have a primary source with empty
    source_names, a synthetic workflow ID, and store a user-defined name in output_name.

    Convenience properties (workflow_id, source_names, output_name) provide direct
    access to the primary data source.
    """

    data_sources: dict[str, DataSourceConfig]
    plot_name: str
    params: pydantic.BaseModel

    @property
    def workflow_id(self) -> WorkflowId:
        """Workflow ID from the primary data source."""
        if PRIMARY not in self.data_sources:
            raise ValueError("Cannot access workflow_id: no primary data source")
        return self.data_sources[PRIMARY].workflow_id

    @property
    def source_names(self) -> list[str]:
        """Source names from the primary data source."""
        if PRIMARY not in self.data_sources:
            raise ValueError("Cannot access source_names: no primary data source")
        return self.data_sources[PRIMARY].source_names

    @property
    def output_name(self) -> str:
        """Output name from the primary data source."""
        if PRIMARY not in self.data_sources:
            raise ValueError("Cannot access output_name: no primary data source")
        return self.data_sources[PRIMARY].output_name

    def is_static(self) -> bool:
        """Return True if this is a static overlay (no workflow subscription needed).

        Static overlays have only a primary data source with empty source_names.
        They use a synthetic workflow ID and store the user-defined overlay name
        in output_name.
        """
        if PRIMARY not in self.data_sources or len(self.data_sources) != 1:
            return False
        return len(self.data_sources[PRIMARY].source_names) == 0


@dataclass
class Layer:
    """A layer within a plot cell, combining identity with configuration."""

    layer_id: LayerId
    config: PlotConfig


@dataclass
class PlotCell:
    """
    Configuration for a plot cell (position, size, and layers to plot).

    The plots are placed in the given row and col of a :py:class:`PlotGrid`, spanning
    the given number of rows and columns. A cell can contain multiple layers that
    are composed via hv.Overlay.
    """

    geometry: CellGeometry
    layers: list[Layer]


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
    """Callback for cell configuration changes.

    Called when a cell's configuration changes (layer added/removed/updated).
    Subscribers should query PlotDataService for layer state (error, stopped, data).
    """

    def __call__(
        self,
        *,
        grid_id: GridId,
        cell_id: CellId,
        cell: PlotCell,
    ) -> None:
        """
        Handle cell configuration update.

        Parameters
        ----------
        grid_id
            ID of the grid containing the cell.
        cell_id
            ID of the cell being updated.
        cell
            Plot cell configuration with all layers.
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
        instrument: str,
        plot_data_service: PlotDataService,
        config_store: ConfigStore | None = None,
        raw_templates: Sequence[dict[str, Any]] = (),
        instrument_config: Instrument | None = None,
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
        instrument
            Name of the instrument (e.g., 'dummy', 'dream').
        plot_data_service
            Service for storing plot state for multi-session support.
        config_store
            Optional store for persisting plot grid configurations across sessions.
        raw_templates
            Raw grid template dicts loaded from YAML files. These are parsed
            during initialization and made available via get_available_templates().
        instrument_config
            Optional instrument configuration for source metadata lookup.
        """
        self._plotting_controller = plotting_controller
        self._job_orchestrator = job_orchestrator
        self._data_service = data_service
        self._instrument = instrument
        self._instrument_config = instrument_config
        self._config_store = config_store
        self._plot_data_service = plot_data_service
        self._logger = logging.getLogger(__name__)

        self._grids: dict[GridId, PlotGridConfig] = {}
        self._cell_to_grid: dict[CellId, GridId] = {}
        self._layer_subscriptions: dict[LayerId, LayerSubscription] = {}
        self._layer_to_cell: dict[LayerId, CellId] = {}
        self._lifecycle_subscribers: dict[SubscriptionId, LifecycleSubscription] = {}
        self._data_subscriptions: dict[LayerId, Any] = {}  # DataServiceSubscriber

        # Parse templates (requires plotter registry, so must be done here)
        self._templates = self._parse_grid_specs(list(raw_templates))

        # Load persisted configurations
        self._load_from_store()

    @property
    def instrument(self) -> str:
        """The instrument name for this orchestrator."""
        return self._instrument

    @property
    def instrument_config(self) -> Instrument | None:
        """The instrument configuration (if available)."""
        return self._instrument_config

    def get_source_title(self, source_name: str) -> str:
        """Get display title for a source name.

        Falls back to the source name if no instrument config is available
        or no title is defined for the source.
        """
        if self._instrument_config is not None:
            return self._instrument_config.get_source_title(source_name)
        return source_name

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

            # Unsubscribe all layers and clean up mappings
            for cell_id, cell in list(grid.cells.items()):
                self._remove_cell_and_cleanup(grid_id, cell_id, cell)

            del self._grids[grid_id]
            self._persist_to_store()
            self._logger.info('Removed plot grid %s (%s)', grid_id, title)
            self._notify_grid_removed(grid_id)

    def add_cell(self, grid_id: GridId, geometry: CellGeometry) -> CellId:
        """
        Add an empty cell to a grid.

        Use :py:meth:`add_layer` to add layers to the cell after creation.

        Parameters
        ----------
        grid_id
            ID of the grid to add the cell to.
        geometry
            Cell geometry (position and size in the grid).

        Returns
        -------
        :
            ID of the added cell.
        """
        cell_id = CellId(uuid4())
        cell = PlotCell(geometry=geometry, layers=[])
        grid = self._grids[grid_id]
        grid.cells[cell_id] = cell
        self._cell_to_grid[cell_id] = grid_id
        return cell_id

    def remove_cell(self, cell_id: CellId) -> None:
        """
        Remove a plot cell (with all its layers) by its unique ID.

        Parameters
        ----------
        cell_id
            ID of the cell to remove.
        """
        grid_id = self._cell_to_grid[cell_id]
        grid = self._grids[grid_id]
        cell = grid.cells[cell_id]

        self._remove_cell_and_cleanup(grid_id, cell_id, cell)

        self._persist_to_store()
        self._notify_cell_removed(grid_id, cell_id, cell)

    def get_layer_config(self, layer_id: LayerId) -> PlotConfig:
        """
        Get configuration for a layer.

        Parameters
        ----------
        layer_id
            ID of the layer.

        Returns
        -------
        :
            The layer's plot configuration.
        """
        cell_id = self._layer_to_cell[layer_id]
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]
        for layer in cell.layers:
            if layer.layer_id == layer_id:
                return layer.config
        raise KeyError(f'Layer {layer_id} not found in cell {cell_id}')

    def get_cell(self, cell_id: CellId) -> PlotCell:
        """
        Get the configuration for a cell.

        Parameters
        ----------
        cell_id
            ID of the plot cell.

        Returns
        -------
        :
            The plot cell configuration with all layers.
        """
        grid_id = self._cell_to_grid[cell_id]
        return self._grids[grid_id].cells[cell_id]

    def add_layer(self, cell_id: CellId, config: PlotConfig) -> LayerId:
        """
        Add a new layer to an existing cell.

        Parameters
        ----------
        cell_id
            ID of the cell to add the layer to.
        config
            Configuration for the new layer.

        Returns
        -------
        :
            ID of the added layer.
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        layer_id = LayerId(uuid4())
        layer = Layer(layer_id=layer_id, config=config)
        cell.layers.append(layer)

        self._layer_to_cell[layer_id] = cell_id
        self._subscribe_layer(grid_id, cell_id, layer)

        return layer_id

    def remove_layer(self, layer_id: LayerId) -> None:
        """
        Remove a layer from its cell.

        If this is the last layer in the cell, the entire cell is removed.

        Parameters
        ----------
        layer_id
            ID of the layer to remove.
        """
        cell_id = self._layer_to_cell[layer_id]
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        # Find and remove the layer
        cell.layers = [layer for layer in cell.layers if layer.layer_id != layer_id]

        # Unsubscribe and clean up layer state
        self._unsubscribe_and_cleanup_layer(layer_id)

        # If no layers left, remove the entire cell
        if not cell.layers:
            self.remove_cell(cell_id)
            return

        self._persist_to_store()

        # Notify with updated cell config
        self._notify_cell_updated(grid_id, cell_id, cell)

    def update_layer_config(self, layer_id: LayerId, new_config: PlotConfig) -> None:
        """
        Update a layer's configuration by replacing with a new layer.

        Creates a new layer with a fresh layer_id, which naturally invalidates
        any cached session state (DynamicMaps, Presenters) since they're keyed
        by layer_id. The old layer_id entries in session caches become orphaned
        and are cleaned up when sessions detect they're no longer in PlotDataService.

        Parameters
        ----------
        layer_id
            ID of the layer to update.
        new_config
            New plot configuration.
        """
        cell_id = self._layer_to_cell[layer_id]
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        # Generate new layer_id - this naturally invalidates session caches
        new_layer_id = LayerId(uuid4())

        # Find and replace the layer with new identity
        layer_index: int | None = None
        for i, layer in enumerate(cell.layers):
            if layer.layer_id == layer_id:
                layer_index = i
                break

        if layer_index is None:
            raise KeyError(f'Layer {layer_id} not found in cell {cell_id}')

        # Fully clean up old layer (including cell mapping)
        self._unsubscribe_and_cleanup_layer(layer_id, remove_from_cell_mapping=True)

        # Create new layer with new identity
        new_layer = Layer(layer_id=new_layer_id, config=new_config)
        cell.layers[layer_index] = new_layer

        # Set up mapping for new layer
        self._layer_to_cell[new_layer_id] = cell_id

        # Subscribe new layer to workflow
        self._subscribe_layer(grid_id, cell_id, new_layer)

    def _remove_cell_and_cleanup(
        self, grid_id: GridId, cell_id: CellId, cell: PlotCell
    ) -> None:
        """
        Remove a cell and unsubscribe all its layers.

        Parameters
        ----------
        grid_id
            ID of the grid containing the cell.
        cell_id
            ID of the cell to remove.
        cell
            The cell object to remove.
        """
        # Unsubscribe all layers in the cell
        for layer in cell.layers:
            self._unsubscribe_and_cleanup_layer(layer.layer_id)

        # Remove cell from grid and mapping
        grid = self._grids[grid_id]
        del grid.cells[cell_id]
        del self._cell_to_grid[cell_id]

    def _unsubscribe_and_cleanup_layer(
        self, layer_id: LayerId, *, remove_from_cell_mapping: bool = True
    ) -> None:
        """
        Unsubscribe a layer from workflow notifications and clean up state.

        Parameters
        ----------
        layer_id
            ID of the layer to unsubscribe and clean up.
        remove_from_cell_mapping
            If True, remove the layer from _layer_to_cell mapping.
            Set to False when the layer is being updated (still in the cell),
            True when removing the layer completely.
        """
        if layer_id in self._layer_subscriptions:
            self._layer_subscriptions[layer_id].unsubscribe()
            del self._layer_subscriptions[layer_id]
        # Clean up data subscription
        if layer_id in self._data_subscriptions:
            self._data_service.unregister_subscriber(self._data_subscriptions[layer_id])
            del self._data_subscriptions[layer_id]
        # Clean up from PlotDataService
        self._plot_data_service.remove(layer_id)
        if remove_from_cell_mapping:
            self._layer_to_cell.pop(layer_id, None)

    def _create_and_register_plotter(
        self, layer_id: LayerId, config: PlotConfig
    ) -> Any | None:
        """
        Create plotter and register with PlotDataService.

        Parameters
        ----------
        layer_id
            ID of the layer to create plotter for.
        config
            Plot configuration containing plot_name and params.

        Returns
        -------
        :
            The created plotter, or None if creation failed.
            On failure, error is logged and PlotDataService is notified.
        """
        try:
            plotter = self._plotting_controller.create_plotter(
                config.plot_name, params=config.params
            )
            self._plot_data_service.job_started(layer_id, plotter)
            return plotter
        except Exception:
            error_msg = traceback.format_exc()
            self._logger.exception('Failed to create plotter for layer_id=%s', layer_id)
            self._plot_data_service.error_occurred(layer_id, error_msg)
            return None

    def _run_compute(self, layer_id: LayerId, plotter: Any, data: dict) -> None:
        """
        Compute plot state and transition layer to READY.

        Calls plotter.compute() with the provided data and notifies PlotDataService.
        Handles errors by logging and transitioning to error state.

        Parameters
        ----------
        layer_id
            ID of the layer being computed.
        plotter
            The plotter instance to compute with.
        data
            Data dict to pass to plotter.compute(). Empty dict for static plotters.
        """
        if layer_id not in self._layer_to_cell:
            return

        try:
            plotter.compute(data)
            self._plot_data_service.data_arrived(layer_id)
        except Exception:
            error_msg = traceback.format_exc()
            self._logger.exception('Failed to compute state for layer_id=%s', layer_id)
            self._plot_data_service.error_occurred(layer_id, error_msg)

    def _subscribe_layer(self, grid_id: GridId, cell_id: CellId, layer: Layer) -> None:
        """
        Subscribe a layer to workflow lifecycle and set up initial notification.

        Uses LayerSubscription to handle both single and multi-source layers uniformly.
        LayerSubscription manages:
        - Subscribing to all required workflows
        - Tracking which workflows have started
        - Notifying when ALL workflows are ready (for multi-source layers)
        - Propagating stop notifications

        Branches based on layer type (determined by data sources):

        - **Static overlay** (single data source with empty source_names): Create plot
          immediately without workflow subscription.
        - **Dynamic layers** (single or multiple data sources): Use LayerSubscription.

        Parameters
        ----------
        grid_id
            ID of the grid containing the cell.
        cell_id
            ID of the cell containing the layer.
        layer
            The layer to subscribe.
        """
        layer_id = layer.layer_id
        config = layer.config

        if config.is_static():
            # Static overlay: create plot immediately without subscription.
            plotter = self._create_and_register_plotter(layer_id, config)
            if plotter is not None:
                self._run_compute(layer_id, plotter, {})
            cell = self._grids[grid_id].cells[cell_id]
            self._notify_cell_updated(grid_id, cell_id, cell)
            self._persist_to_store()
            return

        # Unified path for all non-static layers (single or multi-source)
        def on_all_jobs_ready(ready: SubscriptionReady) -> None:
            self._on_all_jobs_ready(layer_id, ready)

        def on_any_job_stopped(job_number: JobNumber) -> None:
            self._on_layer_job_stopped(layer_id, job_number)

        subscription = LayerSubscription(
            data_sources=config.data_sources,
            job_orchestrator=self._job_orchestrator,
            on_ready=on_all_jobs_ready,
            on_stopped=on_any_job_stopped,
        )
        self._layer_subscriptions[layer_id] = subscription

        # Register layer in WAITING_FOR_JOB state before starting subscription
        # This ensures PlotDataService has state for the layer when widgets are notified
        self._plot_data_service.layer_added(layer_id)

        subscription.start()  # May fire on_ready synchronously if workflows running

        # Always notify current config - sessions will poll PlotDataService for state
        cell = self._grids[grid_id].cells[cell_id]
        self._notify_cell_updated(grid_id, cell_id, cell)
        self._persist_to_store()

    def _on_all_jobs_ready(
        self,
        layer_id: LayerId,
        ready: SubscriptionReady,
    ) -> None:
        """
        Handle notification when all workflows for a layer are ready.

        Called by LayerSubscription when all data sources have running jobs.
        Sets up the data pipeline that computes plot state and stores it in
        PlotDataService. Sessions poll PlotDataService for updates.

        Parameters
        ----------
        layer_id
            ID of the layer to create plot for.
        ready
            SubscriptionReady containing keys_by_role for structured data access.
        """
        # Defensive check: layer may have been removed before callback fires
        if layer_id not in self._layer_to_cell:
            self._logger.warning(
                'Ignoring all-jobs-ready for removed layer_id=%s', layer_id
            )
            return

        config = self.get_layer_config(layer_id)

        # Cleanup old data subscription if this layer had one (e.g., workflow restart)
        if layer_id in self._data_subscriptions:
            self._data_service.unregister_subscriber(self._data_subscriptions[layer_id])
            del self._data_subscriptions[layer_id]

        plotter = self._create_and_register_plotter(layer_id, config)
        if plotter is None:
            return

        # Set up data pipeline - _run_compute will be called when data arrives
        try:
            subscriber = self._plotting_controller.setup_pipeline(
                keys_by_role=ready.keys_by_role,
                plot_name=config.plot_name,
                params=config.params,
                on_data=lambda data: self._run_compute(layer_id, plotter, data),
            )
            self._data_subscriptions[layer_id] = subscriber
        except Exception:
            error_msg = traceback.format_exc()
            self._logger.exception(
                'Failed to set up data pipeline for layer_id=%s', layer_id
            )
            self._plot_data_service.error_occurred(layer_id, error_msg)

    def _on_layer_job_stopped(self, layer_id: LayerId, job_number: JobNumber) -> None:
        """
        Handle workflow stopped notification for a layer.

        Called when a workflow job is stopped. Marks the layer as stopped
        and notifies the UI. The plot (if any) is preserved but marked as
        no longer receiving updates.

        Parameters
        ----------
        layer_id
            ID of the layer whose workflow was stopped.
        job_number
            Job number that was stopped.
        """
        # Defensive check: layer may have been removed before callback fires
        if layer_id not in self._layer_to_cell:
            self._logger.warning(
                'Ignoring workflow stopped for removed layer_id=%s', layer_id
            )
            return

        self._logger.info(
            'Workflow stopped for layer_id=%s, job_number=%s',
            layer_id,
            job_number,
        )

        # Transition to STOPPED state - UI will detect via polling
        self._plot_data_service.job_stopped(layer_id)

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

    def parse_raw_cell(self, cell_data: dict[str, Any]) -> PlotCell | None:
        """
        Parse a raw cell dict into a typed PlotCell.

        Use this to convert cells from templates or persisted configurations
        into typed objects that can be passed to :py:meth:`add_cell`.

        Supports two formats:
        - Legacy format: 'geometry' + 'config' (single layer)
        - New format: 'geometry' + 'layers' (multiple layers)

        Parameters
        ----------
        cell_data
            Cell configuration dict with 'geometry' and either 'config' or 'layers'.
            Each layer/config must contain: workflow_id, source_names, plot_name.
            Optional: output_name, params.

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

        # Handle both legacy 'config' and new 'layers' format
        if 'layers' in cell_data:
            raw_layers = cell_data['layers']
        elif 'config' in cell_data:
            # Legacy format: wrap single config in list
            raw_layers = [cell_data['config']]
        else:
            self._logger.warning('Cell has neither config nor layers, skipping')
            return None

        layers: list[Layer] = []
        for layer_data in raw_layers:
            layer = self._parse_raw_layer(layer_data)
            if layer is not None:
                layers.append(layer)

        if not layers:
            return None

        return PlotCell(geometry=geometry, layers=layers)

    def _parse_raw_layer(self, layer_data: dict[str, Any]) -> Layer | None:
        """
        Parse a raw layer dict into a typed Layer.

        Supports two formats for backward compatibility:
        - New format: 'data_sources' list containing workflow_id, source_names,
          output_name
        - Old format: 'workflow_id', 'source_names', 'output_name' at top level

        Parameters
        ----------
        layer_data
            Layer configuration dict. Must contain 'plot_name' and either
            'data_sources' (new format) or 'workflow_id' (old format).

        Returns
        -------
        :
            Parsed Layer with a new LayerId, or None if the plotter is unknown.
        """
        plot_name = layer_data['plot_name']

        # Validate params, skipping layers with unknown plotters
        params = self._validate_params(plot_name, layer_data.get('params', {}))
        if params is None:
            return None

        # Parse data sources: support dict, list, and legacy formats
        data_sources: dict[str, DataSourceConfig]
        if 'data_sources' in layer_data:
            raw_sources = layer_data['data_sources']
            if isinstance(raw_sources, dict):
                # New format: data_sources dict with role keys
                data_sources = {
                    role: DataSourceConfig(
                        workflow_id=WorkflowId.from_string(ds['workflow_id']),
                        source_names=ds['source_names'],
                        output_name=ds.get('output_name', 'result'),
                    )
                    for role, ds in raw_sources.items()
                }
            else:
                # Legacy format: data_sources list (first entry is primary)
                data_sources = (
                    {
                        PRIMARY: DataSourceConfig(
                            workflow_id=WorkflowId.from_string(
                                raw_sources[0]['workflow_id']
                            ),
                            source_names=raw_sources[0]['source_names'],
                            output_name=raw_sources[0].get('output_name', 'result'),
                        )
                    }
                    if raw_sources
                    else {}
                )
        elif 'workflow_id' in layer_data:
            # Legacy format: single workflow at top level
            data_sources = {
                PRIMARY: DataSourceConfig(
                    workflow_id=WorkflowId.from_string(layer_data['workflow_id']),
                    source_names=layer_data['source_names'],
                    output_name=layer_data.get('output_name', 'result'),
                )
            }
        else:
            # Fallback for templates missing workflow specification.
            # Note: This creates an invalid config - static overlays should use
            # the data_sources format with a synthetic workflow ID. This branch
            # exists for robustness but templates should always specify workflow_id
            # or data_sources.
            data_sources = {}

        config = PlotConfig(
            data_sources=data_sources,
            plot_name=plot_name,
            params=params,
        )

        return Layer(layer_id=LayerId(uuid4()), config=config)

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

    def serialize_grid(self, grid_id: GridId) -> dict[str, Any]:
        """
        Serialize a single grid configuration for export.

        The returned dict is suitable for saving as a YAML grid template
        that can be loaded later via :py:meth:`_parse_single_spec`.

        Parameters
        ----------
        grid_id
            ID of the grid to serialize.

        Returns
        -------
        :
            Grid configuration dict. UUIDs are not included as they are
            runtime identity handles with no cross-session significance.

        Raises
        ------
        KeyError
            If grid_id is not found.
        """
        grid = self._grids[grid_id]
        return {
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
                    'layers': [self._serialize_layer(layer) for layer in cell.layers],
                }
                for cell in grid.cells.values()
            ],
        }

    def _serialize_grids(self) -> list[dict[str, Any]]:
        """
        Serialize all grids to list for persistence.

        Returns
        -------
        :
            List of grid configurations. UUIDs are not persisted as they are
            runtime identity handles with no cross-session significance.
        """
        return [self.serialize_grid(grid_id) for grid_id in self._grids]

    def _serialize_layer(self, layer: Layer) -> dict[str, Any]:
        """Serialize a single layer to dict format."""
        config = layer.config
        return {
            'data_sources': {
                role: {
                    'workflow_id': str(ds.workflow_id),
                    'source_names': ds.source_names,
                    'output_name': ds.output_name,
                }
                for role, ds in config.data_sources.items()
            },
            'plot_name': config.plot_name,
            'params': config.params.model_dump(mode='json'),
        }

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
                    cell_id = self.add_cell(grid_id, cell.geometry)
                    for layer in cell.layers:
                        self.add_layer(cell_id, layer.config)

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
    ) -> None:
        """Notify subscribers that a cell config was added or updated."""
        for subscription in self._lifecycle_subscribers.values():
            if subscription.on_cell_updated:
                try:
                    subscription.on_cell_updated(
                        grid_id=grid_id,
                        cell_id=cell_id,
                        cell=cell,
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
