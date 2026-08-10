# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
PlotGridTabs - Tabbed interface for managing multiple plot grids.

Provides a Panel Tabs widget that displays multiple PlotGrid instances,
kept in sync with PlotOrchestrator by polling its topology version on each
session's own periodic tick (see ``PlotOrchestrator`` "Threading").

Framework-trap invariants
-------------------------
Each item below encodes a measured Panel/Bokeh failure mode. They are easy to
drop in a refactor and hard or impossible to cover with unit tests, so any
change to this module must re-check the list; the inline ``IMPORTANT:``
comments at the enforcement sites carry the details.

- **Lazy tabs**: ``dynamic=True`` on the Tabs widget is what keeps hidden
  grids from materializing Bokeh models; without it any content change
  freezes the UI for seconds (enforced in ``__init__``).
- **Batching**: every update touching more than one widget runs inside
  ``batched_update()`` / ``pn.io.hold()`` — the poll pass via
  ``SessionUpdater``, tab switches via ``_BatchedTabs``, tab rebuilds via
  ``_reconcile_grid_tabs``.
- **Modal container parenting**: the modal container is created once, lives
  at the top level wrapping the Tabs (never per tab — Panel components have
  one parent), and the outer Column is created once in ``__init__`` so the
  container's parent is stable across ``panel`` accesses.
- **Reveal ordering**: a revealed tab's cells are built *before* Panel
  materializes the tab's models (``_BatchedTabs.on_activate`` →
  ``_prebuild_revealed_grid``); building after costs a throwaway round of
  placeholder models and repaints.
- **Disposal ordering and completeness**: ``CellWidget.dispose()`` must
  break the Bokeh toolbar reference cycles and sever the rendered plots'
  pipe subscriptions, and must run *before* the replacement widget renders —
  holoviews severs all weak plot-refresh subscribers on the touched streams,
  so ordering carries the correctness (#1224, enforced in ``_build_cell``).
- **Wake-before-load guard**: ``SessionUpdater`` defers wake registration
  until the browser session is loaded; ticking earlier touches a document
  with no websocket.
- **Markup-pane reveal**: markup panes built after page load stay invisible
  without the ``LivedataDesign`` stylesheet-latch override (#1154,
  ``dashboard/design.py``); this module's freshness pills and time labels
  rely on it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import panel as pn
import structlog

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec

from ..batched_update import batched_update
from ..cell_plan import SessionView, cell_build_inputs, desired_cells
from ..notifications import show_error
from ..plot_data_service import PlotDataService
from ..plot_orchestrator import (
    CellGeometry,
    CellId,
    GridId,
    LayerId,
    PlotCell,
    PlotConfig,
    PlotGridConfig,
    PlotOrchestrator,
)
from ..plots import TimeBounds
from ..session_layer import SessionLayer
from ..session_updater import SessionUpdater
from .cell import CellDeps, CellWidget
from .cell_properties_modal import CellPropertiesModal
from .icons import get_icon_data_uri
from .modal_escape_closer import ModalEscapeCloser
from .plot_config_modal import PlotConfigModal
from .plot_grid import PlotGrid
from .plot_grid_manager import PlotGridManager
from .plot_popout import PlotPopoutManager
from .styles import Colors

logger = structlog.get_logger(__name__)

# Cadence for advancing the freshness pill while no new data arrives. Between
# data frames the pill only ages a stalled stream, which carries no new
# information, so we refresh it at most this often rather than every poll tick.
# Kept comfortably above the backend's ~1 s publish cadence so that for a
# healthy stream each data flush resets the timer before it fires: the stall
# path then never triggers and the pill updates once per frame, avoiding a
# beat between the flush and the timer that would double-update the age.
_FRESHNESS_STALL_INTERVAL_S = 2.0


class _PassStamps(NamedTuple):
    """Versioned inputs of one reconcile pass, compared for equality only.

    Read as one value at gate time (:meth:`PlotGridTabs._input_stamps`) and
    recorded field-by-field as the pass completes each concern; named fields
    keep the two sites from pairing values positionally.
    """

    topology_version: int | None
    layer_version: int
    active_grid_id: GridId | None
    generations: Mapping[GridId | None, int]


def _static_tab_stylesheet(icons: Sequence[str]) -> str:
    """CSS distinguishing the fixed leading tabs from the plot-grid tabs.

    Bokeh renders a tab label as plain text, so an icon cannot be part of the
    title; it is painted on a ``::before`` pseudo-element selected by tab
    position (the static tabs are always the first ones). ``mask-image`` rather
    than ``background-image`` makes the icon take the tab's ``currentColor``, so
    it follows the label through the active/inactive color change for free.

    Parameters
    ----------
    icons:
        Icon name per static tab, in tab order.
    """
    masks = '\n'.join(
        f"""
        .bk-tab:nth-child({position})::before {{
            mask-image: url("{uri}");
            -webkit-mask-image: url("{uri}");
        }}"""
        for position, uri in enumerate(
            (get_icon_data_uri(icon, color=None) for icon in icons), start=1
        )
    )
    return f"""
        .bk-tab:nth-child(-n+{len(icons)}) {{
            font-weight: bold;
        }}
        .bk-tab:nth-child(-n+{len(icons)})::before {{
            content: '';
            display: inline-block;
            width: 1.15em;
            height: 1.15em;
            margin-right: 0.4em;
            vertical-align: -0.22em;
            background-color: currentColor;
            mask-size: contain;
            mask-repeat: no-repeat;
            mask-position: center;
            -webkit-mask-size: contain;
            -webkit-mask-repeat: no-repeat;
            -webkit-mask-position: center;
        }}
        {masks}
        .bk-tab {{
            border-bottom: 1px solid {Colors.TAB_BORDER} !important;
        }}
        .bk-tab.bk-active {{
            background-color: {Colors.TAB_ACTIVE_BG} !important;
            border: 1px solid {Colors.TAB_BORDER} !important;
            border-bottom: none !important;
        }}
        """


class _BatchedTabs(pn.Tabs):
    """Tabs subclass that batches Bokeh model updates on tab switch.

    With ``dynamic=True``, switching tabs triggers a synchronous cascade
    (``_update_active`` → ``param.trigger('objects')`` → nested
    ``_apply_update`` calls) that independently serializes PATCH-DOC
    messages and recomputes the Bokeh model graph for each step.

    Wrapping the cascade in :func:`batched_update` collapses N
    dispatches/recomputes into 1 each.

    See https://github.com/holoviz/panel/issues/8461.

    ``on_activate`` is the hook for populating the tab about to be revealed. It
    runs *before* the cascade, which is the only point where a change to the
    tab's content still lands in the same materialization: Panel registers
    ``_update_active`` in its own constructor, so any ``active`` watcher added
    afterwards necessarily runs once the models already exist and can only
    patch them a second time.
    """

    # Declared so param treats the assignment as a known attribute rather than
    # warning about setting a non-parameter on a Parameterized.
    on_activate: Callable[[], None] | None = None

    def _update_active(self, *events) -> None:
        with batched_update():
            if self.on_activate is not None and any(
                event.name == 'dynamic' or (self.dynamic and event.name == 'active')
                for event in events
            ):
                self.on_activate()
            super()._update_active(*events)


class PlotGridTabs:
    """
    Tabbed widget for managing multiple plot grids.

    Displays static tabs for Workflows and Manage Plots,
    followed by one tab per plot grid.
    Each instance polls the orchestrator's topology version on its own periodic
    tick and reconciles tabs and cells independently, so cross-session changes
    are picked up under this session's document lock rather than pushed.

    Parameters
    ----------
    plot_orchestrator
        The orchestrator managing plot grid configurations. Templates are
        retrieved from the orchestrator via get_available_templates().
    workflow_registry
        Registry of available workflows and their specifications.
    plotting_controller
        Controller for determining available plotters from workflow specs.
    workflow_status_widget
        Widget for displaying workflow status and controls.
    system_status_widget
        Optional widget for displaying system status (sessions and backend workers).
    plot_data_service
        Shared service for plot data with version tracking.
    session_updater
        This session's updater for periodic callbacks.
    """

    def __init__(
        self,
        plot_orchestrator: PlotOrchestrator,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        plotting_controller,
        workflow_status_widget,
        system_status_widget=None,
        *,
        plot_data_service: PlotDataService,
        session_updater: SessionUpdater,
    ) -> None:
        self._orchestrator = plot_orchestrator
        self._workflow_registry = dict(workflow_registry)
        self._plotting_controller = plotting_controller
        self._plot_data_service = plot_data_service

        # Track grid widgets (insertion order determines tab position)
        self._grid_widgets: dict[GridId, PlotGrid] = {}

        # Per-session layer state: version tracking and optional render
        # components. Owned by the poll loop; read by CellWidgets when composing
        # plots (shared via CellDeps).
        self._session_layers: dict[LayerId, SessionLayer] = {}

        # Per-session cell widgets, keyed by CellId — the applied state the
        # differ compares plans against. Each widget owns its titlebar,
        # per-layer toolbars, content, autoscale controller, and the panes
        # updated in place by the poll loop (freshness pill, layer time
        # labels), and records the grid it lives in and the build inputs it
        # was constructed from. Rebuilt when the current inputs differ;
        # disposed when the cell leaves the topology.
        self._cells: dict[CellId, CellWidget] = {}

        # Topology-version stamp gating the tab reconcile.
        self._last_topology_version: int | None = None
        # Tab-level fingerprint of the topology (grid ids, titles, enabled, in
        # order). Cell/layer changes bump the topology version too, but must
        # not tear down and re-append the Tabs entries (Bokeh model churn and
        # flicker on the active tab); the rebuild runs only when this changes.
        self._tab_composition: tuple[tuple[GridId, str, bool], ...] | None = None
        # Set by the local grid-creation callback; the creating session's next
        # poll focuses this grid's tab. Never shared across sessions.
        self._pending_focus_grid_id: GridId | None = None

        # Gate state for coalescing plot-data flushes (see _poll_for_plot_updates).
        # The data push runs only when a new data-burst frame is ready or the
        # visible tab changed. Keyed by grid, because this session may render
        # more than one at a time: the visible grid, plus the grid of every
        # cell held live by a pop-out window. Empty forces a flush on the
        # first poll.
        self._last_generations: dict[GridId | None, int] = {}
        self._last_active_grid_id: GridId | None = None
        # Aggregate PlotDataService version observed by the last completed poll
        # pass. Per-layer lifecycle changes (plotter swap on job restart, the
        # STOPPED transition on job stop) bump no topology version and, for a
        # stop, produce no further frame, so this is their only cheap signal.
        # -1 forces a pass on the first poll.
        self._last_layer_version: int = -1
        # Wall-clock (monotonic) of the last freshness-pill refresh, throttling
        # the stall-aging path between data frames.
        self._last_freshness_update: float = 0.0

        # Shared dependencies handed to every CellWidget. Built once; the
        # callbacks route modal interactions back here (modal container lives
        # at this level). Created after the dependencies above are set.
        self._cell_deps = CellDeps(
            orchestrator=plot_orchestrator,
            workflow_registry=self._workflow_registry,
            plot_data_service=plot_data_service,
            session_layers=self._session_layers,
            on_edit_title=self._show_cell_properties_modal,
            on_reconfigure_layer=self._on_reconfigure_layer,
            on_popout=self._show_popout,
        )

        # Floating pop-out windows, one per cell at most. Kept out of
        # ``_current_modal``: that field gates the poll loop's data flush, and
        # a pop-out must keep updating.
        self._popouts = PlotPopoutManager(
            on_window_change=self._on_popout_window_change
        )

        # The manager reports locally-initiated grid creations so this session
        # (only) focuses the new tab on its next poll; other sessions merely
        # gain the tab.
        self._grid_manager = PlotGridManager(
            orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            on_local_grid_created=self._on_local_grid_created,
        )

        # Fixed tabs, in order, ahead of the variable number of grid tabs. Each
        # carries an icon so the two groups are distinguishable at a glance;
        # grid tabs deliberately carry none, making a bare label the mark of a
        # user-created grid.
        static_tabs = [('Workflows', 'workflow', workflow_status_widget.panel())]
        if system_status_widget is not None:
            static_tabs.append(
                ('System Status', 'activity', system_status_widget.panel())
            )
        static_tabs.append(('Manage Plots', 'layout-grid', self._grid_manager.panel))

        # Main tabs widget.
        # IMPORTANT: dynamic=True is critical for performance. Without it, Panel
        # renders ALL tabs simultaneously, causing severe UI lag (3-5 seconds) when
        # any tab content changes (e.g., updating grid preview). We even observe causes
        # of near-total UI freezes when there are many active plots. With dynamic=True,
        # only the visible tab is rendered; hidden tabs are rendered on-demand when
        # selected. _BatchedTabs wraps each tab switch in hold+freeze so the
        # resulting model-graph updates are dispatched in a single batch.
        self._tabs = _BatchedTabs(
            sizing_mode='stretch_both',
            dynamic=True,
            stylesheets=[_static_tab_stylesheet([icon for _, icon, _ in static_tabs])],
        )

        # Modal container for plot configuration
        # IMPORTANT: Use height=0 to ensure the modal is in the component tree
        # (required for rendering) but doesn't compete for vertical space.
        # The modal itself renders as an overlay when shown.
        # This container must be at the TOP LEVEL (wrapping Tabs), not inside
        # individual grid tabs. Panel components can only have one parent, so
        # adding the same container to multiple tabs would break rendering.
        self._modal_container = pn.Row(height=0, sizing_mode='stretch_width')
        self._current_modal: PlotConfigModal | CellPropertiesModal | None = None

        # Create main widget - tabs with zero-height modal container
        # IMPORTANT: Create the widget once in __init__ (not in the panel property)
        # to give modal_container a stable parent. If we created a new Column on
        # each .panel access, the modal_container would be reparented repeatedly,
        # breaking its connection to the component tree.
        self._widget = pn.Column(
            self._tabs,
            self._modal_container,
            self._popouts.container,
            ModalEscapeCloser(),
            sizing_mode='stretch_both',
        )

        for title, _, panel in static_tabs:
            self._tabs.append((title, panel))

        # Store static tabs count for use as offset in grid tab index calculations
        self._static_tabs_count = len(self._tabs)

        # Build grid tabs from current topology and record the version so the
        # first poll does not rebuild everything again (cells are populated on
        # the first poll). Mirrors WorkflowStatusWidget's init-to-current pattern.
        self._reconcile_topology()
        self._last_topology_version = self._orchestrator.topology_version()

        # Wired only now: during the construction above a tab switch cannot
        # happen, and the pass the hook runs expects a fully built instance.
        self._tabs.on_activate = self._prebuild_revealed_grid

        # Register the poll pass; the predicate lets wake ticks skip it (and
        # the batched hold+freeze) when nothing visible changed.
        session_updater.register_custom_handler(
            self._poll_for_plot_updates, has_work=self._has_pending_work
        )
        # A tab switch changes what the poll pass must flush but bumps no
        # shared version, so request an immediate in-session tick rather than
        # waiting for the housekeeping cadence.
        self._session_updater = session_updater
        self._tabs.param.watch(self._on_active_tab_changed, 'active')

        # Two-tier teardown, run on both the clean disconnect
        # (``on_session_destroyed``) and the stale-session reaper paths. Tier 2
        # (``sever``) releases shared orchestrator state and must run even when
        # the reaper's IOLoop tick never fires; tier 1 (``dispose_widgets``)
        # mutates Bokeh document state and is marshalled onto the session's
        # IOLoop. See issue #955 and ADR 0007.
        session_updater.register_cleanup_handler(self.sever)
        session_updater.register_document_teardown_handler(self.dispose_widgets)

    def _add_grid_tab(self, grid_id: GridId, grid_config: PlotGridConfig) -> None:
        """Add a new grid tab after the Manage tab."""

        # Create grid-specific callback using closure to capture grid_id
        def grid_callback(geometry: CellGeometry) -> None:
            self._on_plot_requested(grid_id, geometry)

        def occupied_geometries() -> tuple[CellGeometry, ...]:
            config = self._orchestrator.peek_grid(grid_id)
            if config is None:
                return ()
            return tuple(cell.geometry for cell in config.cells.values())

        # Create PlotGrid widget with grid-specific callback
        plot_grid = PlotGrid(
            nrows=grid_config.nrows,
            ncols=grid_config.ncols,
            plot_request_callback=grid_callback,
            occupied_geometries=occupied_geometries,
        )

        # Store widget reference
        self._grid_widgets[grid_id] = plot_grid

        # Append grid directly to tabs (Manage tab is always first at index 0)
        # NOTE: Do NOT wrap each grid with modal_container here. The modal
        # container is shared across all tabs and lives at the top level
        # (wrapping the entire Tabs widget). Cells are populated by the poll's
        # cell reconcile, not here.
        self._tabs.append((grid_config.title, plot_grid.panel))

    def _tabbed_grid_ids(self) -> list[GridId]:
        """GridIds that currently have a tab, in tab order.

        Disabled grids keep a widget in ``_grid_widgets`` (preserving state
        across re-enable) but have no tab. Positional lookups between a Bokeh
        tab index and a GridId must go through this list, never through
        ``_grid_widgets`` directly, or a disabled grid preceding an enabled one
        skews the mapping. Membership is read from the live tabs -- the grid
        panels actually appended -- so it stays correct even mid-removal, when
        the orchestrator has already dropped the grid being removed.
        """
        panel_to_grid = {
            id(grid.panel): grid_id for grid_id, grid in self._grid_widgets.items()
        }
        tabbed: list[GridId] = []
        for tab in self._tabs[self._static_tabs_count :]:
            grid_id = panel_to_grid.get(id(tab))
            if grid_id is not None:
                tabbed.append(grid_id)
        return tabbed

    def _get_active_grid_id(self) -> GridId | None:
        """Return the GridId of the currently visible grid tab, or None.

        The tab widget contains static tabs (Workflows, Manage Plots, ...)
        followed by dynamic grid tabs. Subtracting the static tab count from the
        active tab index maps to a grid position. When a static tab is selected
        the result is negative, which the bounds check rejects.

        Returns None while any modal is open: the modal overlay obscures the
        plots, so rendering would be wasted. Dirty flags are preserved and the
        first poll after the modal closes pushes the latest cached state.
        """
        if self._current_modal is not None:
            return None
        grid_idx = self._tabs.active - self._static_tabs_count
        tabbed = self._tabbed_grid_ids()
        if 0 <= grid_idx < len(tabbed):
            return tabbed[grid_idx]
        return None

    def _on_local_grid_created(self, grid_id: GridId) -> None:
        """Record a grid this session just created so its tab is focused.

        Called by ``PlotGridManager`` on the three local creation paths. The
        creating session's next poll activates the new grid's tab; other
        sessions gain the tab without a focus change. This is the only mechanism
        that moves ``tabs.active`` on grid creation -- there is no shared focus
        signal.
        """
        self._pending_focus_grid_id = grid_id

    def _reconcile_topology(self) -> None:
        """Rebuild grid tabs to match orchestrator topology, preserving focus.

        Runs on this session's thread (constructor or poll). The tab rebuild
        runs only when the tab-level composition (grid ids, titles, enabled,
        order) changed -- cell/layer edits bump the topology version too but
        must not churn the Tabs models. For a rebuild: captures the active tab
        by identity first (bypassing ``_get_active_grid_id``'s modal guard,
        which would misreport "no active grid" while a modal is open), rebuilds
        all grid tabs, then restores the active tab by identity. A pending
        local-creation focus overrides the restore. Cells are reconciled
        separately by the poll loop.
        """
        all_grids = self._orchestrator.get_all_grids()
        composition = tuple(
            (grid_id, config.title, config.enabled)
            for grid_id, config in all_grids.items()
        )
        if composition != self._tab_composition:
            # Capture the active tab by identity before the rebuild reorders
            # tabs.
            active_before = self._tabs.active
            static_active = active_before < self._static_tabs_count
            active_grid_id: GridId | None = None
            if not static_active:
                tabbed = self._tabbed_grid_ids()
                idx = active_before - self._static_tabs_count
                if 0 <= idx < len(tabbed):
                    active_grid_id = tabbed[idx]

            self._reconcile_grid_tabs(all_grids)

            # Restore the active tab by identity.
            if static_active:
                self._tabs.active = active_before
            elif active_grid_id is not None:
                tabbed = self._tabbed_grid_ids()
                if active_grid_id in tabbed:
                    self._tabs.active = self._static_tabs_count + tabbed.index(
                        active_grid_id
                    )
                # else: the active grid was removed; leave Bokeh's clamped
                # value.

            # Recorded only after the rebuild succeeded: an exception above
            # leaves the memo stale, so the retried pass rebuilds the tabs
            # instead of skipping a half-applied composition.
            self._tab_composition = composition

        # A locally-created grid overrides the restore and focuses its tab.
        # Cleared unconditionally: if the grid vanished again before this poll,
        # the focus intent is dead.
        if self._pending_focus_grid_id is not None:
            tabbed = self._tabbed_grid_ids()
            if self._pending_focus_grid_id in tabbed:
                self._tabs.active = self._static_tabs_count + tabbed.index(
                    self._pending_focus_grid_id
                )
            self._pending_focus_grid_id = None

    def _reconcile_grid_tabs(self, all_grids: dict[GridId, PlotGridConfig]) -> None:
        """Rebuild grid tabs to match orchestrator state (order, titles, enabled)."""
        # Rebuild _grid_widgets in orchestrator order, preserving existing widgets
        old_widgets = self._grid_widgets
        self._grid_widgets = {}

        with pn.io.hold():
            # Remove all existing grid tabs
            while len(self._tabs) > self._static_tabs_count:
                self._tabs.pop(self._static_tabs_count)

            for grid_id, grid_config in all_grids.items():
                if not grid_config.enabled:
                    # Disabled grids are hidden from tabs but we keep the widget
                    # reference so re-enabling doesn't lose state
                    if grid_id in old_widgets:
                        self._grid_widgets[grid_id] = old_widgets[grid_id]
                    continue

                if grid_id in old_widgets:
                    # Reuse existing widget, update tab title
                    plot_grid = old_widgets[grid_id]
                    self._grid_widgets[grid_id] = plot_grid
                    self._tabs.append((grid_config.title, plot_grid.panel))
                else:
                    # Grid was re-enabled or is new — create fresh tab
                    self._add_grid_tab(grid_id, grid_config)

    def _add_plot(
        self, grid_id: GridId, geometry: CellGeometry, plot_config: PlotConfig
    ) -> None:
        """Commit a configured plot to the orchestrator, surfacing races."""
        try:
            cell_id = self._orchestrator.add_cell(grid_id, geometry)
            self._orchestrator.add_layer(cell_id, plot_config)
        except KeyError:
            # The grid vanished while the modal was open (removed or
            # replaced by another session).
            show_error('Cannot add plot: the grid was removed.')
        except ValueError:
            # The position was free when the wizard opened and is occupied now.
            # ``PlotGrid`` reads occupancy from topology, so it no longer offers
            # a region topology already holds; what remains is two sessions in
            # the wizard on the same free region at once, which no occupancy
            # check can prevent -- whoever commits second lands here.
            show_error('Cannot add plot: that position is already occupied.')

    def _on_plot_requested(self, grid_id: GridId, geometry: CellGeometry) -> None:
        """
        Handle plot request from PlotGrid.

        Shows the PlotConfigModal to configure the plot, then adds it
        to the orchestrator on success.

        Parameters
        ----------
        grid_id
            ID of the grid where the plot was requested.
        geometry
            Cell geometry of the selected region.
        """
        self._show_config_modal(
            on_success=lambda cfg: self._add_plot(grid_id, geometry, cfg)
        )

    def _on_reconfigure_layer(self, layer_id: LayerId) -> None:
        """
        Handle layer reconfiguration request from gear button.

        Shows the PlotConfigModal with existing configuration, then updates
        the layer in the orchestrator on success.

        Parameters
        ----------
        layer_id
            ID of the layer to reconfigure.
        """

        def on_success(plot_config: PlotConfig) -> None:
            """Handle successful layer reconfiguration."""
            try:
                self._orchestrator.update_layer_config(layer_id, plot_config)
            except KeyError:
                # The layer vanished while the modal was open (removed by
                # another session).
                show_error('Cannot apply changes: the plot was removed.')
            except ValueError as e:
                show_error(str(e))

        try:
            current_config = self._orchestrator.get_layer_config(layer_id)
        except KeyError:
            # Gear click raced a removal in another session within the poll
            # window; the widget disappears on the next poll.
            show_error('The plot was removed.')
            return
        self._show_config_modal(on_success=on_success, initial_config=current_config)

    def _on_add_layer(self, cell_id: CellId) -> None:
        """
        Handle add layer request from the cell-properties modal.

        Shows the PlotConfigModal to configure the new layer, then adds it
        to the cell in the orchestrator on success.

        Parameters
        ----------
        cell_id
            ID of the cell to add the layer to.
        """

        def on_success(plot_config: PlotConfig) -> None:
            """Handle successful layer configuration."""
            try:
                self._orchestrator.add_layer(cell_id, plot_config)
            except KeyError:
                # The cell vanished while the modal was open (removed by
                # another session).
                show_error('Cannot add layer: the cell was removed.')
            except ValueError as e:
                show_error(str(e))

        self._show_config_modal(on_success=on_success)

    def _show_config_modal(
        self,
        *,
        on_success: Callable[[PlotConfig], None],
        initial_config: PlotConfig | None = None,
    ) -> None:
        """
        Show the plot configuration modal.

        Parameters
        ----------
        on_success
            Callback to invoke when configuration is successfully completed.
        initial_config
            Optional initial configuration for editing an existing plot.
        """

        def wrapped_on_success(plot_config: PlotConfig) -> None:
            """Wrap success callback to include cleanup."""
            on_success(plot_config)
            self._cleanup_modal()

        # Create and show modal
        self._current_modal = PlotConfigModal(
            workflow_registry=self._workflow_registry,
            plotting_controller=self._plotting_controller,
            success_callback=wrapped_on_success,
            cancel_callback=self._cleanup_modal,
            initial_config=initial_config,
            instrument_config=self._orchestrator.instrument_config,
        )

        # Add modal to container so it renders
        self._modal_container.clear()
        self._modal_container.append(self._current_modal.modal)
        self._current_modal.show()

    def _cleanup_modal(self) -> None:
        """Clean up modal state after completion or cancellation."""
        self._current_modal = None
        self._modal_container.clear()
        # Closing the modal un-hides the grid (``_get_active_grid_id`` reports
        # None while one is open), which is a visibility change like a tab
        # switch: no shared state moved, so request the tick explicitly.
        self._session_updater.request_tick(full=True)

    def _show_cell_properties_modal(
        self, cell_id: CellId, current_title: str, has_user_title: bool
    ) -> None:
        """
        Show the cell-properties modal (rename, add layer, remove layers).

        Parameters
        ----------
        cell_id
            ID of the cell to edit.
        current_title
            The currently displayed title (user-defined or derived).
        has_user_title
            Whether ``current_title`` is user-defined; if not, the input starts
            empty so the placeholder hints at the derived fallback.
        """
        self._current_modal = CellPropertiesModal(
            orchestrator=self._orchestrator,
            workflow_registry=self._workflow_registry,
            plotting_controller=self._plotting_controller,
            cell_id=cell_id,
            current_title=current_title,
            has_user_title=has_user_title,
            on_add_layer=self._on_add_layer,
            on_close=self._cleanup_modal,
        )
        self._modal_container.clear()
        self._modal_container.append(self._current_modal.modal)
        self._current_modal.show()

    def _show_popout(self, cell_id: CellId) -> None:
        """Open the cell's plot in a floating window (see ``plot_popout``).

        Needs no repair pass of its own: the extra pane simply subscribes to
        the layer pipes the cell already renders from, and the button that got
        us here lives in a titlebar, so the cell is on the visible tab and
        already live.
        """
        cell_widget = self._cells.get(cell_id)
        if cell_widget is not None:
            self._popouts.open(cell_id, cell_widget)

    def _on_popout_window_change(self, cell_id: CellId, status: str) -> None:
        """Handle a pop-out window's own controls: close, minimize, restore.

        None of these move shared state, so no version-gated predicate can see
        them; ask for a pass explicitly, exactly as a tab switch does. A
        restored window that did not get one would sit frozen at whatever it
        showed when it was minimized until the next unconditional full pass.
        """
        if status == 'closed':
            self._close_popout(cell_id)
        self._session_updater.request_tick(full=True)

    def _close_popout(self, cell_id: CellId) -> None:
        """Close a cell's pop-out window and re-render the cell behind it.

        Removing the window runs Panel's pane cleanup, which unsubscribes
        *every* plot on the cell's layer pipes -- the grid cell's included,
        because HoloViews' owner filter is defeated upstream (holoviews#6988,
        see :meth:`CellWidget.dispose`). Rebuilding the cell is what puts a
        live plot back in the grid; there is no finer-grained sever available.

        A cell with no widget, or one that left the topology, needs no repair:
        the sweep disposes it on this same pass.
        """
        if self._popouts.status_of(cell_id) is None:
            return
        self._popouts.close(cell_id)
        cell_widget = self._cells.get(cell_id)
        if cell_widget is None:
            return
        grid_config = self._orchestrator.peek_grid(cell_widget.grid_id)
        if grid_config is None or cell_id not in grid_config.cells:
            return
        self._insert_cell(cell_id, grid_config.cells[cell_id], cell_widget.grid_id)

    def _insert_cell(self, cell_id: CellId, cell: PlotCell, grid_id: GridId) -> None:
        """Build a cell's widget and place it in its grid."""
        widget = self._build_cell(cell_id, cell, grid_id)
        self._grid_widgets[grid_id].insert_widget_at(cell.geometry, widget.view)

    def _build_cell(
        self, cell_id: CellId, cell: PlotCell, grid_id: GridId
    ) -> CellWidget:
        """Build (or rebuild) the session widget for a cell.

        Samples the build inputs immediately before constructing the widget
        and records them on it: a lifecycle transition landing mid-build then
        shows up as an input difference on the next pass instead of being
        absorbed unrendered. Preserves the per-layer toolbar visibility toggle
        across rebuilds and disposes the displaced widget (breaking its
        autoscale reference cycle and severing its pipe subscriptions) before
        the replacement renders — ordering #1224 relies on.

        A pop-out window renders the displaced widget's pane, so it is rebuilt
        on the new widget rather than left behind. Reopening rather than
        closing keeps the window following its cell: a rebuild is triggered by
        things the user did elsewhere (a rename, a job restart), and having the
        floating view vanish in response would look like a crash. The status
        carries over, so a minimized window stays minimized -- and its cell
        asleep -- instead of popping open. Its position and size do not.
        """
        previous = self._cells.get(cell_id)
        toolbars_visible = previous.toolbars_shown if previous is not None else False
        popout_status = self._popouts.status_of(cell_id)
        self._popouts.close(cell_id)
        cell_widget = CellWidget(
            cell_id,
            cell,
            self._cell_deps,
            toolbars_visible=toolbars_visible,
            grid_id=grid_id,
            build_inputs=cell_build_inputs(cell, self._plot_data_service.get),
        )
        self._cells[cell_id] = cell_widget
        if previous is not None:
            previous.dispose()
        # After the sever above, never before: the reopened window's pane
        # subscribes to the same pipes the disposal clears.
        if popout_status is not None:
            self._popouts.open(cell_id, cell_widget, status=popout_status)
        return cell_widget

    def _prebuild_revealed_grid(self) -> None:
        """Populate a grid tab before its Bokeh models are materialized.

        Hidden grids defer their cell builds (see
        :meth:`_poll_for_plot_updates`), so at the moment a grid tab is
        revealed its ``PlotGrid`` still holds nothing but the empty-cell
        placeholders it was created with. Materializing that state and building
        the cells afterwards costs a throwaway round of placeholder models —
        constructed, serialized, laid out by the browser, then discarded — and
        shows the user a grid of "Click to add plot" buttons over positions that
        are in fact occupied.

        Running the pass here instead puts the cells in the grid first, so the
        materialization that follows is of the real content and the whole reveal
        is a single patch.

        The topology reconcile is skipped: it may append to or pop from the tabs
        we are currently inside a watcher of, and the tab structure is by
        definition current at this point. A topology change pending from another
        session is picked up by the tick :meth:`_on_active_tab_changed`
        requests.

        Failures are logged and swallowed: this runs on the tab-switch path,
        which nothing else guards, and letting the exception escape would abort
        the materialization that follows and leave the tab blank for the rest of
        the session. Swallowing it degrades to the deferred build -- the tick
        :meth:`_on_active_tab_changed` requests runs the same pass, guarded by
        :class:`SessionUpdater`.
        """
        grid_id = self._get_active_grid_id()
        if grid_id is None or grid_id == self._last_active_grid_id:
            return
        try:
            self._poll_for_plot_updates(reconcile_topology=False)
        except Exception:
            logger.exception("Failed to pre-build revealed grid %s", grid_id)

    def _on_active_tab_changed(self, event) -> None:
        # Full tick: a tab switch changes what is visible without changing any
        # shared state, so handlers that skipped work while their tab was
        # hidden (status badges, timing text) have nothing to report to their
        # predicates yet would render content frozen since the tab was last
        # shown.
        self._session_updater.request_tick(full=True)

    def _input_stamps(self) -> _PassStamps:
        """The pass's input versions and view, as one comparable stamp.

        The stamp components are exactly the values the pass records when it
        completes — an input added to the pass must join :class:`_PassStamps`,
        this read, and the end-of-pass recording, and nothing else.

        Two pass inputs are deliberately *not* stamped, so a change in them
        alone waits for the 5 s unconditional full pass: another session's
        viewer tokens (``has_viewers`` — bounds how long a hidden cell's
        pre-warm build lags the watcher's appearance) and a plotter gaining
        cached state without a lifecycle transition (``has_plot`` for STOPPED
        retained data — in practice revealed by the same pass that computes
        it). Both are cheap no-ops when nothing else changed.
        """
        active_grid_id = self._get_active_grid_id()
        return _PassStamps(
            topology_version=self._orchestrator.topology_version(),
            layer_version=self._plot_data_service.version,
            active_grid_id=active_grid_id,
            generations=self._live_generations(active_grid_id),
        )

    def _live_generations(
        self, active_grid_id: GridId | None
    ) -> dict[GridId | None, int]:
        """Frame generations of the grids this session currently renders.

        The visible grid, plus the grid of every cell whose pop-out window is
        on screen: a pop-out floats above whatever tab is shown, so its cell
        goes on rendering wherever it sits. Scoping generations per grid keeps
        data arriving for a grid nobody here is looking at from waking this
        session; with no pop-out showing, this is the visible grid alone and
        the gate is exactly what it was before pop-outs existed.

        The clock is per grid, so a pop-out buys wakes for every frame of the
        grid behind it -- committed whenever any session keeps any of that
        grid's cells computing -- not just for its own cell. Those extra
        passes flush nothing and are bounded by the frame rate; a per-cell
        clock is not worth carrying for that.
        """
        live_grids = {active_grid_id} | {
            widget.grid_id
            for cell_id in self._popouts.live_cells()
            if (widget := self._cells.get(cell_id)) is not None
        }
        return {
            grid_id: self._orchestrator.frame_generation(grid_id)
            for grid_id in live_grids
        }

    def _has_pending_work(self) -> bool:
        """Gated-tick gate: True when the next pass would do visible work.

        Work is pending exactly when the input stamps differ from those the
        last completed pass recorded, plus one genuinely time-based term: a
        stalled stream ages its freshness pill by wall clock, because the
        absence of events sends no wakes. With healthy data the per-frame
        flush resets the timer before the stall interval elapses and the term
        stays False.
        """
        recorded = _PassStamps(
            topology_version=self._last_topology_version,
            layer_version=self._last_layer_version,
            active_grid_id=self._last_active_grid_id,
            generations=self._last_generations,
        )
        if self._input_stamps() != recorded:
            return True
        return (
            time.monotonic() - self._last_freshness_update
            >= _FRESHNESS_STALL_INTERVAL_S
            and any(
                widget.grid_id == self._last_active_grid_id
                for widget in self._cells.values()
            )
        )

    def _poll_for_plot_updates(self, *, reconcile_topology: bool = True) -> None:
        """
        Reconcile this session's widgets with shared state and push plot data.

        Called from SessionUpdater's wake and housekeeping ticks (inside
        hold+freeze), gated by :meth:`_has_pending_work`. After the tab
        reconcile, the pass has a strict read → decide → act shape:

        1. *Tokens*: hold viewer tokens on the layers this session renders --
           the visible grid's, plus those of any cell held live by a pop-out
           window -- and release the rest. The first token on a layer computes
           its plot synchronously, so the plans computed next see fresh state
           and a reveal renders on its own pass.
        2. *Decide*: :func:`~..cell_plan.desired_cells` maps topology, layer
           snapshots, and this session's view to a target plan per cell —
           including whether this session should hold a built widget at all
           (``materialize``) and the inputs it must be built from.
        3. *Act*: cells that left the topology are disposed; materialized
           plans whose inputs differ from what the applied widget records are
           rebuilt. A cell that is desired but not materialized — a hidden
           grid nobody watches — is *deferred*: any number of input changes
           coalesce into zero builds (#1216), and the input comparison builds
           it exactly once on reveal or when a watcher appears. A failed build
           leaves the applied record unchanged, so the next tick retries.
        4. *Flush*: pending plot data is pushed to the figures this session
           renders, gated per grid on a new data-burst frame for *that* grid
           (or a tab switch), so one burst repaints in one frame and a frame
           for a pop-out's grid cannot push the visible grid's half-built
           burst. Hidden tabs have no materialized Bokeh models
           (``dynamic=True``); their presenters keep their dirty flag and the
           tab switch's own tick sends the latest cached state. Freshness
           pills age on flush, rebuild, or the wall-clock stall cadence, and
           only for the visible grid -- a pop-out shows the plot alone, with
           no pill to age.

        A hidden grid with no pop-out over it therefore does no display work
        at all between switches. Its layers are deactivated, so unless another
        session is viewing them they do not even compute, though their buffers
        keep filling. What the user sees on returning to the tab is the latest
        state, computed on the switch, not a five-second-old rendering. That
        reveal pass is :meth:`_prebuild_revealed_grid`, which runs ahead of the
        tab's materialization so the deferred cells are in the grid before its
        models are built.

        Parameters
        ----------
        reconcile_topology:
            Whether to rebuild the grid tabs on a topology-version change. False
            from the reveal hook, which is already inside the tab machinery; the
            version stays unrecorded so the next tick reconciles.
        """
        # Snapshot before reading any layer state, and record it only once the
        # pass completes (below). The ingestion thread bumps this counter while
        # the pass runs, so reading it at the end would also absorb transitions
        # that landed mid-pass and were never rendered, closing the gate on them
        # until the next unconditional full pass. Recording late likewise leaves
        # the gate armed when an exception escapes mid-pass, so the next tick
        # retries.
        layer_version = self._plot_data_service.version

        # Reconcile grid tabs only when the shared topology changed. Runs on
        # this session's thread and document lock, not pushed cross-session.
        # The stamp is recorded only after the reconcile succeeded, like every
        # other stamp: an exception leaves it unrecorded, so the next tick
        # retries instead of silently dropping the topology change.
        version = self._orchestrator.topology_version()
        if reconcile_topology and version != self._last_topology_version:
            self._reconcile_topology()
            self._grid_manager.on_topology_changed()
            self._last_topology_version = version

        active_grid_id = self._get_active_grid_id()

        # Live view of the topology for the grids this session has widgets
        # for; peeked, not copied — read-only by contract.
        grids = {}
        for grid_id in self._grid_widgets:
            grid_config = self._orchestrator.peek_grid(grid_id)
            if grid_config is not None:
                grids[grid_id] = grid_config

        # A disabled grid's layers lose their viewers below, so a pop-out over
        # one of its cells would float on frozen with no cue. Close it first,
        # before the live set is read; the cell widget itself is kept for a
        # re-enable.
        for grid_config in grids.values():
            if not grid_config.enabled:
                for cell_id in grid_config.cells:
                    self._close_popout(cell_id)

        # Flush a grid's plot data only when a new data-burst frame is ready
        # for *that* grid (so all its layers from one burst repaint in a
        # single frame, never staggered across poll ticks) or when it just
        # became the visible tab (newly shown cells must render their latest
        # data without waiting for the next frame). Staleness is per grid and
        # not one flag: a presenter holds pending data as soon as the
        # ingestion thread builds its layer, before the grid's frame commits,
        # so a shared flag would let a frame for a pop-out's grid push the
        # visible grid's half-built burst, or the reverse.
        live_cells = self._popouts.live_cells()
        generations = self._live_generations(active_grid_id)
        stale_grids = {
            grid_id
            for grid_id, generation in generations.items()
            if generation != self._last_generations.get(grid_id)
        }
        if active_grid_id != self._last_active_grid_id:
            stale_grids.add(active_grid_id)

        # 1 · Tokens. The 0→1 transition computes the layer synchronously so
        # the plans below see fresh snapshots and cached state on this same
        # pass. This is the one policy the pass applies before deciding —
        # tokens follow what this session renders — and it must stay in step
        # with desired_cells' rule; it cannot live there because the plans
        # must be computed from post-activation snapshots. Disabled grids'
        # layers release their token but keep their SessionLayer: the kept
        # widgets stay bound to its pipe, so deleting it would leave a
        # re-enabled grid rendering through dead components with no build
        # input changing to force a rebuild.
        seen_layer_ids: set[LayerId] = set()
        for grid_id, grid_config in grids.items():
            grid_visible = grid_config.enabled and grid_id == active_grid_id
            for cell_id, cell in grid_config.cells.items():
                # A cell whose pop-out is showing keeps computing while the
                # user works in another tab; the window renders it either way.
                # Minimized, it shows nothing and sleeps with its grid.
                is_active = grid_visible or cell_id in live_cells
                for layer in cell.layers:
                    layer_id = layer.layer_id
                    seen_layer_ids.add(layer_id)
                    if self._plot_data_service.get(layer_id) is None:
                        # Should not happen: layers are registered before
                        # widgets are notified. Skip but log for debugging.
                        logger.warning(
                            "Layer %s has no state in PlotDataService during poll",
                            layer_id,
                        )
                        continue
                    session_layer = self._session_layers.get(layer_id)
                    if session_layer is None:
                        session_layer = SessionLayer(layer_id=layer_id)
                        self._session_layers[layer_id] = session_layer
                    self._orchestrator.activate_layer(
                        layer_id, session_layer, is_active
                    )

        # Clean up orphaned session layers (removed from orchestrator).
        # Per-cell widget state (freshness/time panes, autoscale) is swept on
        # cell rebuild or removal, so only the global layer registry needs
        # explicit cleanup.
        for layer_id in list(self._session_layers.keys()):
            if layer_id not in seen_layer_ids:
                self._orchestrator.activate_layer(
                    layer_id, self._session_layers[layer_id], False
                )
                del self._session_layers[layer_id]

        # 2 · Decide. Runs after the token maintenance above so the viewed
        # set no longer counts this session's own released tokens; read once
        # rather than lock-per-layer.
        viewed = self._plot_data_service.viewed_layers()
        plans = desired_cells(
            grids,
            self._plot_data_service.get,
            SessionView(active_grid_id=active_grid_id, live_cell_ids=live_cells),
            viewed.__contains__,
        )

        # 3 · Act. First sweep cells that vanished from topology (cell or grid
        # removed). Membership is checked against the orchestrator directly,
        # not the ``grids`` view above, so the sweep stays correct even if a
        # reentrant tick observes ``_grid_widgets`` mid-rebuild. A cell on a
        # merely disabled grid still exists in topology and is kept so its
        # widget (and state) survives a re-enable. Remove the widget from its
        # grid (if the grid still exists) and dispose it.
        for cell_id in list(self._cells):
            cell_widget = self._cells[cell_id]
            grid_config = self._orchestrator.peek_grid(cell_widget.grid_id)
            if grid_config is not None and cell_id in grid_config.cells:
                continue
            del self._cells[cell_id]
            # The widget is disposed below, which severs the window's pane
            # along with the grid cell's; nothing is left to repair.
            self._popouts.close(cell_id)
            plot_grid = self._grid_widgets.get(cell_widget.grid_id)
            if plot_grid is not None:
                plot_grid.remove_widget_at(cell_widget.geometry)
            cell_widget.dispose()

        # Then build the plans whose inputs differ from what the applied
        # widget was built from. Built and inserted synchronously: the pass
        # holds the document lock, so a pn.state.execute here would run
        # inline anyway rather than defer. The build samples its inputs again
        # (see _build_cell), so a transition landing between plan and build is
        # recorded honestly. On a build failure the applied record is
        # unchanged and the armed gate retries next tick.
        rebuilt = False
        for cell_id, plan in plans.items():
            applied = self._cells.get(cell_id)
            if applied is not None and applied.build_inputs == plan.inputs:
                continue
            self._insert_cell(cell_id, grids[plan.grid_id].cells[cell_id], plan.grid_id)
            rebuilt = True

        # 4 · Flush pending data to the figures this session renders and
        # sample the per-layer time bounds driving the titlebar freshness pill
        # (merged) and the per-layer time-range panes. Bounds are read from
        # the plan snapshots, i.e. after activation: the 0→1 transition is
        # what computes a revealed layer's first frame, and with it the
        # bounds. Only the visible grid's cells show those panes -- a pop-out
        # renders the plot alone.
        active_cell_bounds: dict[CellId, dict[LayerId, TimeBounds | None]] = {}
        for cell_id, plan in plans.items():
            on_active_grid = plan.grid_id == active_grid_id
            if not (on_active_grid or cell_id in live_cells):
                continue
            flush = plan.grid_id in stale_grids
            per_layer: dict[LayerId, TimeBounds | None] = {}
            for layer_input in plan.inputs.layers:
                snapshot = layer_input.snapshot
                per_layer[layer_input.layer_id] = (
                    snapshot.plotter.time_bounds
                    if snapshot is not None and snapshot.plotter is not None
                    else None
                )
                if flush:
                    session_layer = self._session_layers.get(layer_input.layer_id)
                    if session_layer is not None:
                        session_layer.update_pipe()
            if on_active_grid:
                active_cell_bounds[cell_id] = per_layer

        # Refresh the freshness/lag indicator for active-grid cells. Runs after
        # rebuilds so cells recreated this poll update their fresh pane handle.
        # The pill refreshes in step with the data flush (so it reads the actual
        # lag of the frame just shown) and otherwise only on the slow stall
        # cadence -- ticking it every poll just animates aging with no new info.
        # The per-layer time/lag row only changes on a new frame, so it tracks
        # the data flush too.
        #
        # A rebuild is due as well: a rebuilt cell carries brand-new, blank
        # panes, so without this it would show no age and no time range until
        # the next frame or stall tick, however good the data behind it is.
        # Frames for a pop-out's grid do not count: they reach no pill, and
        # letting them reset the timer would keep deferring the aging of a
        # visible grid whose own stream stalled.
        now_mono = time.monotonic()
        stalled = now_mono - self._last_freshness_update
        active_flushed = active_grid_id in stale_grids
        freshness_due = (
            active_flushed or rebuilt or stalled >= _FRESHNESS_STALL_INTERVAL_S
        )
        # Only start the stall interval over when there was something to update:
        # a pass over a hidden or empty grid would otherwise consume the very
        # interval a rebuild landing just after it depends on.
        if freshness_due and active_cell_bounds:
            self._last_freshness_update = now_mono
        for cell_id, per_layer in active_cell_bounds.items():
            cell_widget = self._cells.get(cell_id)
            if cell_widget is None:
                continue
            if freshness_due:
                cell_widget.update_freshness(per_layer)
            if active_flushed or rebuilt:
                for layer_id, bounds in per_layer.items():
                    cell_widget.update_layer_time(layer_id, bounds)

        # Record gate state for the next poll. Every stale grid was flushed
        # above and every other one records the generation it already had, so
        # the read can be recorded wholesale -- a tab change piggybacking on an
        # unchanged generation cannot suppress the next genuine frame. (An
        # exception escaping the pass skips this, leaving the gate armed.)
        self._last_generations = generations
        self._last_active_grid_id = active_grid_id
        self._last_layer_version = layer_version

    def sever(self) -> None:
        """Release shared orchestrator state held by this session (tier 2).

        Releases the session's viewer/interest tokens so hidden layers stop
        computing. Touches only shared orchestrator state, not the Bokeh
        document, so it is safe to call from the background stale-session reaper
        thread and must run there regardless of whether tier-1 teardown
        (:meth:`dispose_widgets`) ever runs. Idempotent.

        There is no lifecycle subscription to unsubscribe: topology changes are
        observed by polling, which stops when the session's periodic callback
        stops.
        """
        for layer_id, session_layer in list(self._session_layers.items()):
            self._orchestrator.activate_layer(layer_id, session_layer, False)
        self._session_layers.clear()

    def dispose_widgets(self) -> None:
        """Dispose session-bound widgets (tier 1).

        Disposes cell widgets and closes any open pop-out windows, breaking
        Bokeh-tool reference cycles. Mutates Bokeh document state, so it must
        run on the session's IOLoop. Idempotent.
        """
        self._popouts.close_all()
        for cell_widget in self._cells.values():
            cell_widget.dispose()
        self._cells.clear()

    @property
    def panel(self) -> pn.Column:
        """Get the Panel viewable object for this widget."""
        return self._widget

    @property
    def active_tab_index(self) -> int:
        """Index of the currently active tab."""
        return self._tabs.active

    @property
    def tabs(self) -> pn.Tabs:
        """Get the Tabs widget containing grid tabs."""
        return self._tabs
