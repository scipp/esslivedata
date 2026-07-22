# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
PlotOrchestrator - Manages plot grid configurations and plot lifecycle.

Coordinates plot creation and management across multiple plot grids:
- Configuration staging and persistence
- Plot grid lifecycle (create, remove)
- Plot cell management (add, remove)
- Static plot creation at add-layer time; run-state driven via polling
- Grid template loading and parsing
"""

from __future__ import annotations

import copy
import threading
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NewType, Protocol
from uuid import UUID, uuid4

import pydantic
import structlog

from ess.livedata.config.grid_template import GridSpec
from ess.livedata.config.workflow_spec import (
    DataKey,
    JobNumber,
    StreamRole,
    WorkflowId,
    WorkflowSpec,
)

from .config_store import ConfigStore
from .data_roles import PRIMARY
from .data_service import DataService
from .frame_clock import FrameClock
from .plot_data_service import LayerId, PlotDataService
from .plot_params import TimeWindowMixin, TimeWindowMode
from .plotting_controller import PlottingController

if TYPE_CHECKING:
    from ess.livedata.config import Instrument

GridId = NewType('GridId', UUID)
CellId = NewType('CellId', UUID)


class JobOrchestratorProtocol(Protocol):
    """Protocol for JobOrchestrator interface needed by PlotOrchestrator."""

    def get_workflow_registry(self) -> Mapping[WorkflowId, WorkflowSpec]:
        """Get the workflow registry containing all managed workflows."""
        ...

    def get_active_job_number(self, workflow_id: WorkflowId) -> JobNumber | None:
        """Get the job_number of the currently active job, if any."""
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
    view. The backend pydantic field name (used in ``DataKey``) is
    resolved at subscription time from the view name plus the current
    window mode.
    """

    workflow_id: WorkflowId
    source_names: list[str]
    view_name: str = 'result'


@dataclass(frozen=True)
class ResolvedDataSource:
    """A data source with its output view resolved to a backend field name.

    Produced by :func:`_build_resolved_data_sources` from a
    :class:`DataSourceConfig`: ``output_name`` carries the backend pydantic
    field name selected for the current window mode, ready to key a
    :class:`DataKey`. Runtime-only — never persisted.
    """

    workflow_id: WorkflowId
    source_names: list[str]
    output_name: str


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


def _stream_role_for_mode(mode: TimeWindowMode) -> StreamRole:
    """Map a window mode to the stream role its data is subscribed from."""
    return 'since_start' if mode is TimeWindowMode.since_start else 'per_update'


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


def _stream_role_for_data_role(role: str, params: pydantic.BaseModel) -> StreamRole:
    """Return the stream role wanted by a data role.

    The primary role follows the user-selected window mode; correlation
    axes always want per-update data.
    """
    if role != PRIMARY:
        return 'per_update'
    window = params.time_window if isinstance(params, TimeWindowMixin) else None
    return _stream_role_for_mode(window.mode) if window is not None else 'per_update'


def _build_resolved_data_sources(
    config: PlotConfig,
    registry: Mapping[WorkflowId, WorkflowSpec],
) -> dict[str, ResolvedDataSource]:
    """Resolve ``config.data_sources`` view names to backend field names.

    Falls back to the view name verbatim when the data source's workflow is
    not in the registry (lets a layer whose workflow has not been seen yet
    still set up a pipeline, keyed by whatever name it was given).
    """
    resolved: dict[str, ResolvedDataSource] = {}
    for role, ds in config.data_sources.items():
        spec = registry.get(ds.workflow_id)
        output_name = (
            resolve_field_name(
                spec, ds.view_name, role=_stream_role_for_data_role(role, config.params)
            )
            if spec is not None
            else ds.view_name
        )
        resolved[role] = ResolvedDataSource(
            workflow_id=ds.workflow_id,
            source_names=ds.source_names,
            output_name=output_name,
        )
    return resolved


def _build_keys_by_role(
    data_sources: dict[str, ResolvedDataSource],
) -> dict[str, list[DataKey]]:
    """Build stable DataKeys grouped by role.

    Keys carry no job identity — they are fully determined by the plot
    config.
    """
    return {
        role: [
            DataKey(
                workflow_id=ds.workflow_id,
                source_name=sn,
                output_name=ds.output_name,
            )
            for sn in ds.source_names
        ]
        for role, ds in data_sources.items()
    }


def _resolve_supports_windowing(
    data_sources: dict[str, DataSourceConfig],
    registry: Mapping[WorkflowId, WorkflowSpec],
) -> bool:
    """Determine whether the primary view exposes window/latest modes.

    Returns ``False`` for cumulative-only views (no ``per_update`` stream),
    in which case only ``TimeWindowMode.since_start`` is meaningful.
    """
    if PRIMARY not in data_sources:
        return True
    from .plotting_controller import output_view_supports_windowing

    primary = data_sources[PRIMARY]
    spec = registry.get(primary.workflow_id)
    if spec is None:
        return True
    return output_view_supports_windowing(spec, primary.view_name)


@dataclass
class _LayerJobTracker:
    """Last-observed run-state of the workflows feeding a layer.

    ``sync_job_states`` compares the polled job numbers against
    ``job_numbers`` to detect generation changes (→ reset presentation) and
    stops (→ STOPPED). ``workflow_ids`` is deduplicated: a workflow shared by
    several roles contributes one entry.
    """

    workflow_ids: tuple[WorkflowId, ...]
    job_numbers: tuple[JobNumber | None, ...]


class PlotOrchestrator:
    """Manages plot grid configurations and plot lifecycle.

    Threading
    ---------
    The topology mappings (``_grids``, ``_cell_to_grid``, ``_layer_to_cell``)
    are shared across all browser sessions. They are mutated only on the single
    Tornado/Bokeh IOLoop thread (all UI callbacks and per-session polls run
    there): the IOLoop thread is the sole topology *writer*. The only other
    thread touching topology is the ``orchestrator-update`` thread, which
    *reads* it during ingestion (``_grid_of_layer``, ``_pull_and_build``,
    ``get_layer_config`` via ``_reset_layer_presentation``).

    ``_topology_lock`` (an ``RLock``) makes those cross-thread reads see a
    consistent multi-dict snapshot. Writers hold it only around the dict
    mutations themselves -- never across file I/O (``_persist_to_store``),
    pipeline setup, ``plotter.compute``, or ``DataService`` unregistration --
    so ingestion latency never couples to those. Deletes are leaf-first
    (``_layer_to_cell`` before ``_cell_to_grid`` before ``_grids``) and inserts
    root-first, so a reader that races a mutation fails at the first hop with a
    ``KeyError`` that all read paths tolerate. Lock order is
    ``_topology_lock`` before ``_dirty_lock``, never the reverse.

    Widgets do not receive pushed lifecycle notifications. Each mutator bumps
    ``_topology_version``; each session's poll compares it against its last
    seen value and reconciles on its own IOLoop tick and document lock (ADR
    0007). ``topology_version()`` needs no lock: the bump and every poll read
    happen on the same IOLoop thread, and an int read is atomic regardless.
    """

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
        self._layer_jobs: dict[LayerId, _LayerJobTracker] = {}
        self._layer_to_cell: dict[LayerId, CellId] = {}
        # Bumped by every topology mutator on the IOLoop thread; polled by each
        # session's widgets to detect grid/cell changes (see class docstring).
        self._topology_version: int = 0
        # Guards cross-thread multi-dict reads of the topology mappings; see the
        # class "Threading" docstring for the discipline and lock order.
        self._topology_lock = threading.RLock()
        self._data_subscriptions: dict[LayerId, Any] = {}  # DataSubscriber
        self._layer_resolvers: dict[LayerId, Any] = {}  # LayerId -> TitleResolver
        # Layers whose DataService keys changed since the last flush, bucketed
        # per grid. Filled by ``_mark_layer_dirty`` (any thread ending a
        # DataService transaction), drained by ``flush_frames``.
        self._dirty_layers: dict[GridId, set[LayerId]] = {}
        self._dirty_lock = threading.Lock()

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

    def topology_version(self) -> int:
        """Monotonic counter bumped on every grid/cell/layer topology change.

        Sessions poll this to detect creation, removal, reorder, rename,
        enable/disable, and cell/layer composition changes, then reconcile
        their widgets. It does *not* advance for pure data flow (frame
        flushes). See the class "Threading" docstring.
        """
        return self._topology_version

    def _bump_topology_version(self) -> None:
        """Advance the topology version. IOLoop-thread only."""
        self._topology_version += 1

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
        with self._topology_lock:
            self._grids[grid_id] = grid
        self._bump_topology_version()
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

            # Unsubscribe all layers and clean up mappings
            for cell_id, cell in list(grid.cells.items()):
                self._remove_cell_and_cleanup(grid_id, cell_id, cell)

            with self._topology_lock:
                del self._grids[grid_id]
            self._bump_topology_version()
            self._persist_to_store()
            self._logger.info('Removed plot grid %s (%s)', grid_id, title)

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
        if grid_id not in self._grids:
            return
        grid = self._grids[grid_id]
        grid.title = new_title
        self._persist_to_store()
        self._logger.info('Renamed grid %s to %s', grid_id, new_title)
        self._bump_topology_version()

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
        if grid_id not in self._grids:
            return
        keys = list(self._grids.keys())
        current_index = keys.index(grid_id)
        new_index = current_index + delta
        if new_index < 0 or new_index >= len(keys):
            return

        # Rebuild dict in new order
        keys.insert(new_index, keys.pop(current_index))
        with self._topology_lock:
            self._grids = {k: self._grids[k] for k in keys}
        self._bump_topology_version()
        self._persist_to_store()
        self._logger.info('Moved grid %s by %d positions', grid_id, delta)

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
        if grid_id not in self._grids:
            return
        grid = self._grids[grid_id]
        grid.enabled = enabled
        self._persist_to_store()
        self._logger.info('Set grid %s enabled=%s', grid_id, enabled)
        self._bump_topology_version()

    def replace_grid(
        self, grid_id: GridId, title: str, nrows: int, ncols: int
    ) -> GridId:
        """
        Replace a grid with a new empty grid at the same position.

        Tears down all cells/layers of the old grid, then creates a new grid at
        the same position with the given dimensions. The new grid inherits the
        old grid's ``enabled`` state. Bumps the topology version once; sessions
        reconcile the position change on their next poll.

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

        # Create new grid and insert at the same position
        new_grid_id = GridId(uuid4())
        new_grid = PlotGridConfig(
            title=title, nrows=nrows, ncols=ncols, enabled=enabled
        )

        with self._topology_lock:
            del self._grids[grid_id]
            # Rebuild dict with new grid at the original position
            items = list(self._grids.items())
            items.insert(position, (new_grid_id, new_grid))
            self._grids = dict(items)
        self._bump_topology_version()

        self._persist_to_store()
        self._logger.info(
            'Replaced grid %s with %s (%s) at position %d',
            grid_id,
            new_grid_id,
            title,
            position,
        )
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

        Raises
        ------
        KeyError
            If the grid does not exist (e.g. removed by another session).
        """
        if grid_id not in self._grids:
            raise KeyError(f'Grid {grid_id} no longer exists')
        cell_id = CellId(uuid4())
        cell = PlotCell(geometry=geometry, layers=[], user_title=user_title)
        grid = self._grids[grid_id]
        with self._topology_lock:
            grid.cells[cell_id] = cell
            self._cell_to_grid[cell_id] = grid_id
        self._bump_topology_version()
        return cell_id

    def remove_cell(self, cell_id: CellId) -> None:
        """
        Remove a plot cell (with all its layers) by its unique ID.

        Parameters
        ----------
        cell_id
            ID of the cell to remove.
        """
        if cell_id not in self._cell_to_grid:
            return
        grid_id = self._cell_to_grid[cell_id]
        grid = self._grids[grid_id]
        cell = grid.cells[cell_id]

        self._remove_cell_and_cleanup(grid_id, cell_id, cell)

        self._persist_to_store()
        self._bump_topology_version()

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
        if cell_id not in self._cell_to_grid:
            return
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]
        cell.user_title = title or None
        self._persist_to_store()
        self._logger.info('Set cell %s title to %r', cell_id, cell.user_title)
        self._bump_topology_version()

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
        with self._topology_lock:
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

        Raises
        ------
        KeyError
            If the cell does not exist (e.g. removed by another session).
        ValueError
            If the resulting cell would combine a non-overlayable layer (a table
            or a layout-mode plot) with any other layer.
        """
        if cell_id not in self._cell_to_grid:
            raise KeyError(f'Cell {cell_id} no longer exists')
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        self._check_overlayable_composition(
            [layer.config for layer in cell.layers] + [config]
        )

        layer_id = LayerId(uuid4())
        layer = Layer(layer_id=layer_id, config=config)
        with self._topology_lock:
            cell.layers.append(layer)
            self._layer_to_cell[layer_id] = cell_id
        self._setup_layer(layer)
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
        if layer_id not in self._layer_to_cell:
            return
        cell_id = self._layer_to_cell[layer_id]
        grid_id = self._cell_to_grid[cell_id]
        cell = self._grids[grid_id].cells[cell_id]

        # Find and remove the layer
        with self._topology_lock:
            cell.layers = [layer for layer in cell.layers if layer.layer_id != layer_id]

        # Unsubscribe and clean up layer state
        self._cleanup_layer(layer_id)

        # If no layers left, remove the entire cell
        if not cell.layers:
            self.remove_cell(cell_id)
            return

        self._refresh_resolvers_for_cell(cell_id)
        self._persist_to_store()
        self._bump_topology_version()

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

        Raises
        ------
        KeyError
            If the layer does not exist (e.g. removed by another session).
        """
        if layer_id not in self._layer_to_cell:
            raise KeyError(f'Layer {layer_id} no longer exists')
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

        self._check_overlayable_composition(
            [
                new_config if i == layer_index else layer.config
                for i, layer in enumerate(cell.layers)
            ]
        )

        # Fully clean up old layer (including cell mapping)
        self._cleanup_layer(layer_id, remove_from_cell_mapping=True)

        # Create new layer with new identity
        new_layer = Layer(layer_id=new_layer_id, config=new_config)
        with self._topology_lock:
            cell.layers[layer_index] = new_layer
            self._layer_to_cell[new_layer_id] = cell_id

        # Subscribe new layer to workflow
        self._setup_layer(new_layer)
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
            self._cleanup_layer(layer.layer_id)

        # Remove cell from grid and mapping
        grid = self._grids[grid_id]
        with self._topology_lock:
            del grid.cells[cell_id]
            del self._cell_to_grid[cell_id]

    def _cleanup_layer(
        self, layer_id: LayerId, *, remove_from_cell_mapping: bool = True
    ) -> None:
        """
        Tear down a layer's pipeline and clean up state.

        Layer removal is the sole owner of extractor cleanup: unregistering
        the data subscriber recomputes the buffers' retention requirements
        from the surviving subscribers.

        Parameters
        ----------
        layer_id
            ID of the layer to clean up.
        remove_from_cell_mapping
            If True, remove the layer from _layer_to_cell mapping.
            Set to False when the layer is being updated (still in the cell),
            True when removing the layer completely.
        """
        self._layer_jobs.pop(layer_id, None)
        # Clean up data subscription
        if layer_id in self._data_subscriptions:
            self._data_service.unregister_subscriber(self._data_subscriptions[layer_id])
            del self._data_subscriptions[layer_id]
        # Clean up from PlotDataService
        self._plot_data_service.remove(layer_id)
        self._layer_resolvers.pop(layer_id, None)
        if remove_from_cell_mapping:
            with self._topology_lock:
                self._layer_to_cell.pop(layer_id, None)

    def _check_overlayable_composition(self, configs: list[PlotConfig]) -> None:
        """Reject a multi-layer cell that contains a non-overlayable layer.

        Tables and layout-mode plotters produce elements that cannot be fused
        via ``hv.Overlay``, so they must be the sole layer in their cell.
        """
        if len(configs) > 1 and not all(
            self._plotting_controller.is_overlayable(config.plot_name, config.params)
            for config in configs
        ):
            raise ValueError(
                'A table or layout-mode plot cannot share a cell with other '
                'layers; place it in its own cell.'
            )

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
        field names (which appear in ``DataKey.output_name``) back to their
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

    def _mark_layer_dirty(self, layer_id: LayerId) -> None:
        """Record that a layer's input changed; ``flush_frames`` rebuilds it.

        The data-subscriber callback: runs once per update batch on the
        ingestion thread and does no extraction or compute. Dirty flags
        coalesce, so any number of updates between flushes costs one rebuild.

        ``DataService`` notifies from a snapshot of its subscriber list taken
        before the callbacks run, so the UI thread may remove this layer while
        the notification is in flight. The grid lookup therefore tolerates the
        mappings changing underneath it; dropping the mark is correct, since a
        removed layer has no state left for ``flush_frames`` to rebuild.
        """
        try:
            grid_id = self._grid_of_layer(layer_id)
        except KeyError:
            return
        with self._dirty_lock:
            self._dirty_layers.setdefault(grid_id, set()).add(layer_id)

    def _grid_of_layer(self, layer_id: LayerId) -> GridId:
        with self._topology_lock:
            return self._cell_to_grid[self._layer_to_cell[layer_id]]

    def flush_frames(self) -> None:
        """Rebuild each grid's dirty viewed layers, committing per grid.

        Called on the ingestion thread once a burst has drained. Each dirty
        layer with a viewer is rebuilt from a fresh DataService pull; running
        grid-by-grid and committing per grid means a session showing one tab
        sees its frame the moment that grid's layers finish, rather than after
        every other visible tab's compute.

        Dirty flags of layers without viewers are dropped: the 0→1 viewer
        transition rebuilds unconditionally from current DataService content
        (see ``activate_layer``), so nothing is lost.
        """
        with self._dirty_lock:
            buckets = self._dirty_layers
            self._dirty_layers = {}
        for grid_id, layer_ids in buckets.items():
            computed = False
            for layer_id in layer_ids:
                state = self._plot_data_service.get(layer_id)
                if state is None or not state.has_viewers:
                    continue
                computed |= self._pull_and_build(layer_id)
            if computed:
                self._frame_clock.commit(grid_id)

    def activate_layer(self, layer_id: LayerId, token: object, active: bool) -> None:
        """Acquire or release a viewer interest token on a layer.

        While no viewer holds a token, frame flushes skip the layer (no
        extraction or compute for hidden layers); buffering continues. On the
        0→1 transition the layer is rebuilt from a DataService pull,
        synchronously on the caller's thread (the polling thread). This keeps
        the same poll pass's component rebuild seeing fresh
        ``has_cached_state``.
        """
        state = self._plot_data_service.get(layer_id)
        if state is None:
            return
        if state.set_active(token, active):
            self._refresh_layer(layer_id)

    def _refresh_layer(self, layer_id: LayerId) -> None:
        """Rebuild a layer from current DataService content, committing its grid."""
        if not self._pull_and_build(layer_id):
            return
        try:
            grid_id = self._grid_of_layer(layer_id)
        except KeyError:
            # The UI thread removed the layer while we were building; a layer
            # without a cell has nothing to commit.
            return
        self._frame_clock.commit(grid_id)

    def _pull_and_build(self, layer_id: LayerId) -> bool:
        """Pull a layer's data through its extractors and rebuild its plot.

        Returns True if a build ran (also on a failed pull or build, which
        transitions the layer to ERROR and thus still warrants a frame
        commit), False if the layer is gone or its data is not ready.
        """
        with self._topology_lock:
            if layer_id not in self._layer_to_cell:
                return False
            subscriber = self._data_subscriptions.get(layer_id)
        if subscriber is None:
            return False
        state = self._plot_data_service.get(layer_id)
        if state is None or state.plotter is None:
            return False
        # ``state`` is the live state machine, so these two reads are not
        # atomic against a concurrent ``job_started`` on the UI thread.
        # Capturing version first rules out the harmful pairing (stale version
        # with new plotter): ``job_started`` bumps the version before swapping
        # the plotter, so an old version implies the swap had not yet run.
        # The converse pairing (new version, old plotter) is possible and
        # tolerated: the check in ``_compute_layer`` then passes and flips the
        # layer READY on the old plotter's result. The swap also replaced this
        # layer's subscription and re-marked it dirty, so the next flush
        # overwrites that frame. See #1060.
        version = state.version
        plotter = state.plotter
        try:
            assembled = subscriber.assemble(self._data_service.snapshot(subscriber))
        except Exception:
            error_msg = traceback.format_exc()
            self._logger.exception('Failed to extract data for layer_id=%s', layer_id)
            if self._layer_version(layer_id) == version:
                self._plot_data_service.error_occurred(layer_id, error_msg)
            return True
        if assembled is None:
            return False
        self._compute_layer(layer_id, plotter, version, assembled)
        return True

    def _compute_layer(
        self,
        layer_id: LayerId,
        plotter: Any,
        version: int,
        data: dict[str, dict[DataKey, Any]],
    ) -> None:
        """Run ``plotter.compute`` and transition the layer to READY or ERROR.

        ``plotter`` and ``version`` are the caller's pre-pull capture. If the
        layer's version moved on by compute end, the plotter was swapped (job
        restart on the UI thread) and both transitions are skipped: the stale
        result must neither flip the new plotter READY nor mark it ERROR. The
        replacement subscription re-marked the layer dirty, so the next flush
        rebuilds it consistently.

        Thread-agnostic: runs on whatever thread the caller is on — the bg
        ingestion thread when entered via ``flush_frames``, the polling thread
        when entered via ``activate_layer``, the UI thread for setup-time
        static-overlay builds.
        """
        try:
            plotter.compute(data, title_resolver=self._layer_resolvers.get(layer_id))
            if self._layer_version(layer_id) == version:
                self._plot_data_service.data_arrived(layer_id)
        except Exception:
            error_msg = traceback.format_exc()
            self._logger.exception('Failed to compute state for layer_id=%s', layer_id)
            if self._layer_version(layer_id) == version:
                self._plot_data_service.error_occurred(layer_id, error_msg)

    def _layer_version(self, layer_id: LayerId) -> int | None:
        state = self._plot_data_service.get(layer_id)
        return None if state is None else state.version

    def _setup_layer(self, layer: Layer) -> None:
        """
        Create a layer's plotter and data pipeline.

        Keys are computable from the plot config alone (stable DataKeys carry
        no job identity), so the pipeline is set up unconditionally at
        add-layer time — no waiting for a workflow job. The initial dirty
        mark pulls whatever the DataService retains under those keys, so a
        layer bound to a stopped workflow renders its last results.

        Branches based on layer type (determined by data sources):

        - **Static overlay** (single data source with empty source_names):
          computed once from params alone, no data pipeline.
        - **Dynamic layers** (single or multiple data sources): pipeline plus
          a run-state tracker consumed by ``sync_job_states``.

        Parameters
        ----------
        layer
            The layer to set up.
        """
        layer_id = layer.layer_id
        config = layer.config

        if config.is_static():
            # Static overlay: built unconditionally (no viewer gate): there is
            # no data subscription to pull from later, and the build is a
            # one-off.
            plotter = self._create_and_register_plotter(layer_id, config)
            state = self._plot_data_service.get(layer_id)
            if plotter is not None and state is not None:
                # Empty input: a static overlay computes from its params alone.
                self._compute_layer(layer_id, plotter, state.version, {})
                self._frame_clock.commit(self._grid_of_layer(layer_id))
            self._bump_topology_version()
            self._persist_to_store()
            return

        resolved_data_sources = _build_resolved_data_sources(
            config, self._job_orchestrator.get_workflow_registry()
        )
        workflow_ids = tuple(
            dict.fromkeys(ds.workflow_id for ds in resolved_data_sources.values())
        )
        self._layer_jobs[layer_id] = _LayerJobTracker(
            workflow_ids=workflow_ids,
            job_numbers=self._current_job_numbers(workflow_ids),
        )

        plotter = self._create_and_register_plotter(layer_id, config)
        if plotter is not None:
            self._layer_resolvers[layer_id] = self._build_title_resolver(layer_id)
            # Set up data pipeline - updates mark the layer dirty; flush_frames
            # pulls and rebuilds. The immediate mark covers retained data
            # already present, so the plot shows without waiting for a delta.
            try:
                subscriber = self._plotting_controller.setup_pipeline(
                    keys_by_role=_build_keys_by_role(resolved_data_sources),
                    plot_name=config.plot_name,
                    params=config.params,
                    on_update=lambda: self._mark_layer_dirty(layer_id),
                )
                self._data_subscriptions[layer_id] = subscriber
                self._mark_layer_dirty(layer_id)
            except Exception:
                error_msg = traceback.format_exc()
                self._logger.exception(
                    'Failed to set up data pipeline for layer_id=%s', layer_id
                )
                self._plot_data_service.error_occurred(layer_id, error_msg)
            else:
                if any(n is None for n in self._layer_jobs[layer_id].job_numbers):
                    self._plot_data_service.job_stopped(layer_id)

        # Sessions poll PlotDataService for state; the version bump makes them
        # reconcile the new cell/layer composition.
        self._bump_topology_version()
        self._persist_to_store()

    def _current_job_numbers(
        self, workflow_ids: tuple[WorkflowId, ...]
    ) -> tuple[JobNumber | None, ...]:
        return tuple(
            self._job_orchestrator.get_active_job_number(w) for w in workflow_ids
        )

    def sync_job_states(self) -> None:
        """Drive layer lifecycle state from polled workflow run-state.

        Called from the orchestrator update loop (before ``flush_frames``),
        replacing per-commit callbacks. For each layer, the active job
        numbers of its workflows are compared against the last observed:

        - changed and all running → a new generation was committed (its
          buffers were cleared at commit): reset the layer presentation so
          the stale frame is not shown.
        - changed and not all running → the workflow stopped: freeze the
          layer (retained data stays displayed).

        A commit immediately followed by a stop within one poll interval is
        observed as a stop, retaining the pre-commit frame instead of the
        cleared buffers' blank; accepted as unreachable in practice.
        """
        for layer_id, tracker in list(self._layer_jobs.items()):
            job_numbers = self._current_job_numbers(tracker.workflow_ids)
            if job_numbers == tracker.job_numbers:
                continue
            was_running = all(n is not None for n in tracker.job_numbers)
            tracker.job_numbers = job_numbers
            if all(n is not None for n in job_numbers):
                self._reset_layer_presentation(layer_id)
            elif was_running:
                self._plot_data_service.job_stopped(layer_id)

    def _reset_layer_presentation(self, layer_id: LayerId) -> None:
        """Give a layer a fresh plotter on generation change.

        The commit that started the new generation cleared the workflow's
        buffers; without a reset the layer would keep showing the previous
        generation's last frame. Replacing the plotter drives the layer to
        WAITING_FOR_DATA (blank until new data arrives) and deliberately
        resets plotter-internal accumulation such as autoscale ranges. The
        data pipeline is untouched: extractors and keys are
        generation-independent.
        """
        if layer_id not in self._data_subscriptions:
            # Setup failed for this layer: without a data subscription no data
            # can ever arrive, so a fresh plotter would drive the layer to a
            # permanently blank WAITING_FOR_DATA and hide the ERROR state.
            return
        try:
            config = self.get_layer_config(layer_id)
        except KeyError:
            # Layer removed on the UI thread while this poll pass was running.
            return
        self._create_and_register_plotter(layer_id, config)

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
