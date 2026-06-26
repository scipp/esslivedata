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
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NewType, Protocol
from uuid import UUID, uuid4

import pydantic
import structlog

from ess.livedata.config.grid_template import GridSpec
from ess.livedata.config.workflow_spec import (
    JobNumber,
    ResultKey,
    StreamRole,
    WorkflowId,
    WorkflowSpec,
)

from .config_store import ConfigStore
from .data_roles import PRIMARY
from .data_service import DataService
from .frame_clock import FrameClock
from .job_orchestrator import WorkflowCallbacks
from .layer_subscription import LayerSubscription, SubscriptionReady
from .plot_data_service import LayerId, PlotDataService
from .plot_params import WindowMode
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

    def get_workflow_registry(self) -> Mapping[WorkflowId, WorkflowSpec]:
        """Get the workflow registry containing all managed workflows."""
        ...

    def get_previous_job_number(self, workflow_id: WorkflowId) -> JobNumber | None:
        """Get the job_number of the most recently stopped job, if any."""
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

    This defines how to connect a layer to a workflow's user-facing output
    view. The backend pydantic field name (used in ``ResultKey``) is
    resolved at subscription time from the view name plus the current
    window mode.
    """

    workflow_id: WorkflowId
    source_names: list[str]
    view_name: str = 'result'


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
    supports_windowing: bool = True

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
    def view_name(self) -> str:
        """Output view name from the primary data source."""
        if PRIMARY not in self.data_sources:
            raise ValueError("Cannot access view_name: no primary data source")
        return self.data_sources[PRIMARY].view_name

    def is_static(self) -> bool:
        """Return True if this is a static overlay (no workflow subscription needed).

        Static overlays have only a primary data source with empty source_names.
        They use a synthetic workflow ID and store the user-defined overlay name
        in view_name.
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

    ``user_title`` is an optional user-defined cell title shown in the cell
    titlebar. When ``None`` the titlebar shows a title derived from the layers.
    """

    geometry: CellGeometry
    layers: list[Layer]
    user_title: str | None = None


@dataclass
class PlotGridConfig:
    """A plot grid tab configuration."""

    title: str = ""
    nrows: int = 3
    ncols: int = 3
    cells: dict[CellId, PlotCell] = field(default_factory=dict)
    enabled: bool = True


GridCreatedCallback = Callable[[GridId, PlotGridConfig], None]
GridRemovedCallback = Callable[[GridId], None]
GridUpdatedCallback = Callable[[GridId, PlotGridConfig], None]
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
    on_grid_updated: GridUpdatedCallback | None = None
    on_cell_updated: CellUpdatedCallback | None = None
    on_cell_removed: CellRemovedCallback | None = None


def _stream_role_for_mode(mode: WindowMode) -> StreamRole:
    """Map a window mode to the stream role its data is subscribed from."""
    return 'since_start' if mode is WindowMode.since_start else 'per_update'


def resolve_field_name(
    spec: WorkflowSpec,
    view_name: str,
    *,
    role: StreamRole = 'since_start',
) -> str:
    """Resolve a (view, role) pair to the backend pydantic field name.

    Falls back to ``view_name`` as a raw field name when no matching view
    is declared (lets unannotated reduction outputs work unchanged).
    """
    view = spec.get_output_view(view_name)
    return view.field_for(role) if view is not None else view_name


def _role_for_slot(slot: str, params: pydantic.BaseModel) -> StreamRole:
    """Return the stream role wanted by a data-source slot.

    The primary slot follows the user-selected window mode; correlation
    axes always want per-update data.
    """
    if slot != PRIMARY:
        return 'per_update'
    window = getattr(params, 'window', None)
    return _stream_role_for_mode(window.mode) if window is not None else 'per_update'


def _build_resolved_data_sources(
    config: PlotConfig,
    registry: Mapping[WorkflowId, WorkflowSpec],
) -> dict[str, DataSourceConfig]:
    """Build a copy of ``config.data_sources`` with view names resolved to fields.

    The returned mapping carries backend pydantic field names in
    ``view_name`` (which is then used as ``ResultKey.output_name``).
    """
    resolved: dict[str, DataSourceConfig] = {}
    for slot, ds in config.data_sources.items():
        spec = registry.get(ds.workflow_id)
        if spec is None:
            resolved[slot] = ds
            continue
        field_name = resolve_field_name(
            spec, ds.view_name, role=_role_for_slot(slot, config.params)
        )
        resolved[slot] = DataSourceConfig(
            workflow_id=ds.workflow_id,
            source_names=ds.source_names,
            view_name=field_name,
        )
    return resolved


def _resolve_supports_windowing(
    data_sources: dict[str, DataSourceConfig],
    registry: Mapping[WorkflowId, WorkflowSpec],
) -> bool:
    """Determine whether the primary view exposes window/latest modes.

    Returns ``False`` for cumulative-only views (no ``per_update`` stream),
    in which case only ``WindowMode.since_start`` is meaningful.
    """
    if PRIMARY not in data_sources:
        return True
    from .plotting_controller import output_view_supports_windowing

    primary = data_sources[PRIMARY]
    spec = registry.get(primary.workflow_id)
    if spec is None:
        return True
    return output_view_supports_windowing(spec, primary.view_name)


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
        frame_clock: FrameClock | None = None,
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
        frame_clock
            Shared counter advanced when a visible layer is recomputed, letting
            per-session poll loops coalesce synchronized plot flushes. A private
            instance is created if none is provided.
        """
        self._plotting_controller = plotting_controller
        self._job_orchestrator = job_orchestrator
        self._data_service = data_service
        self._instrument = instrument
        self._instrument_config = instrument_config
        self._config_store = config_store
        self._plot_data_service = plot_data_service
        self._frame_clock = frame_clock or FrameClock()
        self._logger = structlog.get_logger()

        self._grids: dict[GridId, PlotGridConfig] = {}
        self._cell_to_grid: dict[CellId, GridId] = {}
        self._layer_subscriptions: dict[LayerId, LayerSubscription] = {}
        self._layer_to_cell: dict[LayerId, CellId] = {}
        self._lifecycle_subscribers: dict[SubscriptionId, LifecycleSubscription] = {}
        self._data_subscriptions: dict[LayerId, Any] = {}  # DataServiceSubscriber
        self._layer_resolvers: dict[LayerId, Any] = {}  # LayerId -> TitleResolver
        # Per-grid compute buckets filled during a burst by ``_enqueue_compute``
        # and drained by ``flush_frames``. Touched only on the ingestion thread.
        self._frame_buckets: dict[GridId | None, list[tuple[LayerId, Any]]] = {}

        # Parse templates (requires plotter registry, so must be done here)
        self._templates = self._parse_grid_specs(list(raw_templates))

        # Load persisted configurations
        self._load_from_store()

    def frame_generation(self, grid_id: GridId | None) -> int:
        """Counter advanced once per completed data-burst frame for a grid.

        Returns 0 for ``None`` (e.g. a non-grid tab), so a session showing no
        grid never sees a frame advance. See :class:`FrameClock`.
        """
        if grid_id is None:
            return 0
        return self._frame_clock.generation(grid_id)

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

    def get_dim_title(self, dim: str) -> str:
        """Get display title for a coord/dim name (axis label).

        Falls back to the dim name itself if no instrument config is available
        or no title is defined for the dim.
        """
        if self._instrument_config is not None:
            return self._instrument_config.get_dim_title(dim)
        return dim

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

    def rename_grid(self, grid_id: GridId, new_title: str) -> None:
        """
        Rename a plot grid.

        Parameters
        ----------
        grid_id
            ID of the grid to rename.
        new_title
            New display title for the grid.
        """
        grid = self._grids[grid_id]
        grid.title = new_title
        self._persist_to_store()
        self._logger.info('Renamed grid %s to %s', grid_id, new_title)
        self._notify_grid_updated(grid_id)

    def move_grid(self, grid_id: GridId, delta: int) -> None:
        """
        Reorder a grid by moving it ``delta`` positions in the grid list.

        Negative delta moves up, positive moves down. No-op if the move
        would place the grid outside the list boundaries.

        Parameters
        ----------
        grid_id
            ID of the grid to move.
        delta
            Number of positions to move (negative=up, positive=down).
        """
        keys = list(self._grids.keys())
        current_index = keys.index(grid_id)
        new_index = current_index + delta
        if new_index < 0 or new_index >= len(keys):
            return

        # Rebuild dict in new order
        keys.insert(new_index, keys.pop(current_index))
        self._grids = {k: self._grids[k] for k in keys}
        self._persist_to_store()
        self._logger.info('Moved grid %s by %d positions', grid_id, delta)
        self._notify_grid_updated(grid_id)

    def set_grid_enabled(self, grid_id: GridId, *, enabled: bool) -> None:
        """
        Enable or disable a grid.

        Disabled grids remain in the configuration but are hidden from
        the tab bar.

        Parameters
        ----------
        grid_id
            ID of the grid to toggle.
        enabled
            Whether the grid should be visible.
        """
        grid = self._grids[grid_id]
        grid.enabled = enabled
        self._persist_to_store()
        self._logger.info('Set grid %s enabled=%s', grid_id, enabled)
        self._notify_grid_updated(grid_id)

    def replace_grid(
        self, grid_id: GridId, title: str, nrows: int, ncols: int
    ) -> GridId:
        """
        Replace a grid with a new empty grid at the same position.

        Tears down all cells/layers of the old grid, fires ``on_grid_removed``,
        then creates a new grid at the same position with the given dimensions
        and fires ``on_grid_created``. The new grid inherits the old grid's
        ``enabled`` state.

        Parameters
        ----------
        grid_id
            ID of the grid to replace. Must exist.
        title
            Display title for the new grid.
        nrows
            Number of rows in the new grid.
        ncols
            Number of columns in the new grid.

        Returns
        -------
        :
            ID of the newly created grid.

        Raises
        ------
        KeyError
            If ``grid_id`` does not exist.
        """
        old_grid = self._grids[grid_id]
        enabled = old_grid.enabled

        # Find position in ordered dict
        keys = list(self._grids.keys())
        position = keys.index(grid_id)

        # Clean up all cells/layers
        for cell_id, cell in list(old_grid.cells.items()):
            self._remove_cell_and_cleanup(grid_id, cell_id, cell)

        del self._grids[grid_id]
        self._notify_grid_removed(grid_id)

        # Create new grid and insert at the same position
        new_grid_id = GridId(uuid4())
        new_grid = PlotGridConfig(
            title=title, nrows=nrows, ncols=ncols, enabled=enabled
        )

        # Rebuild dict with new grid at the original position
        items = list(self._grids.items())
        items.insert(position, (new_grid_id, new_grid))
        self._grids = dict(items)

        self._persist_to_store()
        self._logger.info(
            'Replaced grid %s with %s (%s) at position %d',
            grid_id,
            new_grid_id,
            title,
            position,
        )
        self._notify_grid_created(new_grid_id)
        return new_grid_id

    def add_cell(
        self,
        grid_id: GridId,
        geometry: CellGeometry,
        user_title: str | None = None,
    ) -> CellId:
        """
        Add an empty cell to a grid.

        Use :py:meth:`add_layer` to add layers to the cell after creation.

        Parameters
        ----------
        grid_id
            ID of the grid to add the cell to.
        geometry
            Cell geometry (position and size in the grid).
        user_title
            Optional user-defined title for the cell.

        Returns
        -------
        :
            ID of the added cell.
        """
        cell_id = CellId(uuid4())
        cell = PlotCell(geometry=geometry, layers=[], user_title=user_title)
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

    def set_cell_title(self, cell_id: CellId, title: str | None) -> None:
        """
        Set or clear the user-defined title of a cell.

        Parameters
        ----------
        cell_id
            ID of the cell to rename.
        title
            New user-defined title, or ``None``/empty to clear it and fall back
            to the derived title.
        """
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]
        cell.user_title = title or None
        self._persist_to_store()
        self._logger.info('Set cell %s title to %r', cell_id, cell.user_title)
        self._notify_cell_updated(grid_id, cell_id, cell)

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
        self._refresh_resolvers_for_cell(cell_id)

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

        self._refresh_resolvers_for_cell(cell_id)
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
        self._refresh_resolvers_for_cell(cell_id)

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
        self._layer_resolvers.pop(layer_id, None)
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

    def _build_title_resolver(self, layer_id: LayerId) -> Any:
        """Build a TitleResolver for a layer, checking cell-level view uniqueness.

        When all non-static layers in the same cell share the same primary view,
        the view title is already shown on the Y-axis and is stripped from the
        legend label to reduce redundancy. The resolver maps backend pydantic
        field names (which appear in ``ResultKey.output_name``) back to their
        owning view's title so legends and axes show user-facing titles.
        """
        from .plots import TitleResolver, _identity

        config = self.get_layer_config(layer_id)
        registry = self._job_orchestrator.get_workflow_registry()
        spec = registry.get(config.workflow_id)

        cell_id = self._layer_to_cell[layer_id]
        cell = self.get_cell(cell_id)
        view_names = {
            layer.config.view_name
            for layer in cell.layers
            if not layer.config.is_static()
        }

        def field_to_view_title(field_name: str) -> str:
            if spec is None:
                return field_name
            for view in spec.get_output_views():
                if field_name in view.streams.values():
                    return view.title
            return field_name

        return TitleResolver(
            source=self.get_source_title,
            output=field_to_view_title if spec is not None else _identity,
            dim=self.get_dim_title,
            include_output_in_label=len(view_names) != 1,
        )

    def _refresh_resolvers_for_cell(self, cell_id: CellId) -> None:
        """Rebuild cached TitleResolvers for all layers in a cell.

        Called when cell composition changes (layer added/removed) to keep
        the cell-level ``include_output_in_label`` flag consistent across
        all layers.
        """
        cell = self.get_cell(cell_id)
        if cell is None:
            return
        for layer in cell.layers:
            if layer.layer_id in self._layer_resolvers:
                self._layer_resolvers[layer.layer_id] = self._build_title_resolver(
                    layer.layer_id
                )

    def _stash_pending(
        self,
        layer_id: LayerId,
        data: dict[str, dict[ResultKey, Any]],
    ) -> Any:
        """Submit input through the layer compute gate, returning a due task.

        Stashes input on the layer state machine. A build is due (a task is
        returned) only if at least one viewer holds an interest token;
        otherwise the latest input is retained and rebuilt on the next 0→1
        token transition (see ``activate_layer``).
        """
        if layer_id not in self._layer_to_cell:
            return None
        state = self._plot_data_service.get(layer_id)
        if state is None:
            return None
        title_resolver = self._layer_resolvers.get(layer_id)
        return state.stash_pending(data, title_resolver=title_resolver)

    def _grid_of_layer(self, layer_id: LayerId) -> GridId | None:
        return self._cell_to_grid.get(self._layer_to_cell[layer_id])

    def _run_compute(
        self,
        layer_id: LayerId,
        data: dict[str, dict[ResultKey, Any]],
    ) -> None:
        """Stash input and run a due build synchronously, committing its grid.

        The synchronous path for setup-time builds (e.g. static overlays added
        on the UI thread): compute and commit the grid inline so the layer is
        displayed without waiting for the next ingestion flush.
        """
        task = self._stash_pending(layer_id, data)
        if task is not None:
            self._dispatch_compute_task(layer_id, task)
            grid_id = self._grid_of_layer(layer_id)
            if grid_id is not None:
                self._frame_clock.commit(grid_id)

    def _enqueue_compute(
        self,
        layer_id: LayerId,
        data: dict[str, dict[ResultKey, Any]],
    ) -> None:
        """Stash input and bucket a due build per grid for ``flush_frames``.

        The ingestion-thread path for Kafka-delta builds. Deferring dispatch
        out of the inline ``DataService._notify`` lets ``flush_frames`` run a
        burst grid-by-grid and commit each grid the moment its layers finish,
        so a session waits only on its own tab's compute.
        """
        task = self._stash_pending(layer_id, data)
        if task is not None:
            grid_id = self._grid_of_layer(layer_id)
            self._frame_buckets.setdefault(grid_id, []).append((layer_id, task))

    def flush_frames(self) -> None:
        """Run each grid's bucketed builds, committing that grid as it finishes.

        Called on the ingestion thread once a burst has drained. Running
        grid-by-grid and committing per grid means a session showing one tab
        sees its frame the moment that grid's layers finish, rather than after
        every other visible tab's compute. A ``None`` grid (layer not mapped to
        a grid) is computed but never committed; nothing reads its generation.
        """
        buckets = self._frame_buckets
        self._frame_buckets = {}
        for grid_id, tasks in buckets.items():
            for layer_id, task in tasks:
                self._dispatch_compute_task(layer_id, task)
            if grid_id is not None:
                self._frame_clock.commit(grid_id)

    def activate_layer(self, layer_id: LayerId, token: object, active: bool) -> None:
        """Acquire or release a viewer interest token on a layer.

        On the 0→1 transition with pending input, the stashed build is run
        synchronously on the caller's thread (the polling thread). This keeps
        the same poll pass's component rebuild seeing fresh ``has_cached_state``.
        """
        state = self._plot_data_service.get(layer_id)
        if state is None:
            return
        task = state.set_active(token, active)
        if task is not None:
            self._dispatch_compute_task(layer_id, task)

    def _dispatch_compute_task(self, layer_id: LayerId, task: Any) -> None:
        """Run a flush task and transition the layer to READY or ERROR.

        Thread-agnostic: runs on whatever thread the caller is on — the bg
        ingestion thread when entered via ``flush_frames``, the polling thread
        when entered via ``activate_layer`` or the synchronous ``_run_compute``
        setup path. See ``LayerStateMachine`` for the gate's threading contract.
        """
        try:
            task.run()
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
                self._run_compute(layer_id, {})
            cell = self._grids[grid_id].cells[cell_id]
            self._notify_cell_updated(grid_id, cell_id, cell)
            self._persist_to_store()
            return

        # Unified path for all non-static layers (single or multi-source)
        def on_all_jobs_ready(ready: SubscriptionReady) -> None:
            self._on_all_jobs_ready(layer_id, ready)

        def on_any_job_stopped(job_number: JobNumber) -> None:
            self._on_layer_job_stopped(layer_id, job_number)

        resolved_data_sources = _build_resolved_data_sources(
            config, self._job_orchestrator.get_workflow_registry()
        )
        subscription = LayerSubscription(
            data_sources=resolved_data_sources,
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

        self._layer_resolvers[layer_id] = self._build_title_resolver(layer_id)

        # Set up data pipeline - _enqueue_compute buckets builds as data arrives
        try:
            subscriber = self._plotting_controller.setup_pipeline(
                keys_by_role=ready.keys_by_role,
                plot_name=config.plot_name,
                params=config.params,
                on_data=lambda data: self._enqueue_compute(layer_id, data),
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

        Parameters
        ----------
        cell_data
            Cell configuration dict with 'geometry' and 'layers'. Each layer
            must contain: data_sources (dict keyed by role), plot_name, and
            optionally params.

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

        layers: list[Layer] = []
        for layer_data in cell_data['layers']:
            layer = self._parse_raw_layer(layer_data)
            if layer is not None:
                layers.append(layer)

        if not layers:
            return None

        return PlotCell(
            geometry=geometry,
            layers=layers,
            user_title=cell_data.get('user_title'),
        )

    def _parse_raw_layer(self, layer_data: dict[str, Any]) -> Layer | None:
        """
        Parse a raw layer dict into a typed Layer.

        Parameters
        ----------
        layer_data
            Layer configuration dict. Must contain 'plot_name' and
            'data_sources' (dict keyed by role: primary, x_axis, ...).

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

        data_sources: dict[str, DataSourceConfig] = {
            role: DataSourceConfig(
                workflow_id=WorkflowId.from_string(ds['workflow_id']),
                source_names=ds['source_names'],
                view_name=ds['view_name'],
            )
            for role, ds in layer_data['data_sources'].items()
        }

        supports_windowing = _resolve_supports_windowing(
            data_sources, self._job_orchestrator.get_workflow_registry()
        )

        config = PlotConfig(
            data_sources=data_sources,
            plot_name=plot_name,
            params=params,
            supports_windowing=supports_windowing,
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
                enabled=raw.get('enabled', True),
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
        result: dict[str, Any] = {
            'title': grid.title,
            'nrows': grid.nrows,
            'ncols': grid.ncols,
            'cells': [self._serialize_cell(cell) for cell in grid.cells.values()],
        }
        if not grid.enabled:
            result['enabled'] = False
        return result

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

    def _serialize_cell(self, cell: PlotCell) -> dict[str, Any]:
        """Serialize a single cell to dict format."""
        result: dict[str, Any] = {
            'geometry': {
                'row': cell.geometry.row,
                'col': cell.geometry.col,
                'row_span': cell.geometry.row_span,
                'col_span': cell.geometry.col_span,
            },
            'layers': [self._serialize_layer(layer) for layer in cell.layers],
        }
        if cell.user_title is not None:
            result['user_title'] = cell.user_title
        return result

    def _serialize_layer(self, layer: Layer) -> dict[str, Any]:
        """Serialize a single layer to dict format."""
        config = layer.config
        return {
            'data_sources': {
                role: {
                    'workflow_id': str(ds.workflow_id),
                    'source_names': ds.source_names,
                    'view_name': ds.view_name,
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
                    cell_id = self.add_cell(
                        grid_id, cell.geometry, user_title=cell.user_title
                    )
                    for layer in cell.layers:
                        self.add_layer(cell_id, layer.config)
                if not spec.enabled:
                    self.set_grid_enabled(grid_id, enabled=False)

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

    def peek_grid(self, grid_id: GridId) -> PlotGridConfig | None:
        """
        Get a read-only view of a plot grid configuration without copying.

        Returns the internal grid object directly. Callers must not modify
        the returned object.

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
        on_grid_updated: GridUpdatedCallback | None = None,
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
        on_grid_updated
            Called when a grid is renamed, reordered, or enabled/disabled.
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
            on_grid_updated=on_grid_updated,
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

    def _notify_grid_updated(self, grid_id: GridId) -> None:
        """Notify subscribers that a grid was renamed, reordered, or toggled."""
        grid = self._grids[grid_id]
        for subscription in self._lifecycle_subscribers.values():
            if subscription.on_grid_updated:
                try:
                    subscription.on_grid_updated(grid_id, grid)
                except Exception:
                    self._logger.exception(
                        'Error in grid updated callback for grid %s', grid_id
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
