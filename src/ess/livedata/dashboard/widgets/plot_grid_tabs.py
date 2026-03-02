# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
PlotGridTabs - Tabbed interface for managing multiple plot grids.

Provides a Panel Tabs widget that displays multiple PlotGrid instances,
synchronized with PlotOrchestrator via lifecycle subscriptions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import holoviews as hv
import panel as pn
import structlog

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec

from ..data_roles import PRIMARY
from ..format_utils import extract_error_summary
from ..plot_data_service import LayerState, LayerStateMachine, PlotDataService
from ..plot_orchestrator import (
    CellGeometry,
    CellId,
    DataSourceConfig,
    GridId,
    LayerId,
    PlotCell,
    PlotConfig,
    PlotGridConfig,
    PlotOrchestrator,
    SubscriptionId,
)
from ..plot_params import PlotAspectType, StretchMode
from ..save_filename import build_save_filename_from_cell, make_save_filename_hook
from ..session_layer import SessionLayer
from ..session_updater import SessionUpdater
from .plot_config_modal import PlotConfigModal
from .plot_grid import GridCellStyles, PlotGrid
from .plot_grid_manager import PlotGridManager
from .plot_widgets import (
    create_cell_toolbar,
    get_plot_cell_display_info,
    get_workflow_display_info,
)

logger = structlog.get_logger(__name__)


def _get_sizing_mode(config: PlotConfig) -> str:
    """Extract Panel sizing_mode from plot configuration.

    Parameters
    ----------
    config:
        Plot configuration containing plotter params.

    Returns
    -------
    :
        Panel sizing_mode string ('stretch_both', 'stretch_width', or 'stretch_height').
    """
    params = config.params
    if hasattr(params, 'plot_aspect'):
        aspect = params.plot_aspect
        if aspect.aspect_type == PlotAspectType.free:
            return 'stretch_both'
        if aspect.stretch_mode == StretchMode.width:
            return 'stretch_width'
        return 'stretch_height'
    return 'stretch_both'


class PlotGridTabs:
    """
    Tabbed widget for managing multiple plot grids.

    Displays static tabs for Jobs, Workflows, and Manage Plots,
    followed by one tab per plot grid.
    Synchronizes with PlotOrchestrator via lifecycle subscriptions to support
    multiple linked instances.

    Parameters
    ----------
    plot_orchestrator
        The orchestrator managing plot grid configurations. Templates are
        retrieved from the orchestrator via get_available_templates().
    workflow_registry
        Registry of available workflows and their specifications.
    plotting_controller
        Controller for determining available plotters from workflow specs.
    job_status_widget
        Widget for displaying job status information.
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
        job_status_widget,
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

        # Per-session layer state: version tracking and optional render components.
        self._session_layers: dict[LayerId, SessionLayer] = {}

        # Determine number of static tabs for stylesheet
        static_tab_count = 4 if system_status_widget else 3

        # Build nth-child selectors for static tabs
        static_tab_selectors = ',\n                '.join(
            f'.bk-tab:nth-child({i})' for i in range(1, static_tab_count + 1)
        )

        # Main tabs widget.
        # IMPORTANT: dynamic=True is critical for performance. Without it, Panel
        # renders ALL tabs simultaneously, causing severe UI lag (3-5 seconds) when
        # any tab content changes (e.g., updating grid preview). We even observe causes
        # of near-total UI freezes when there are many active plots. With dynamic=True,
        # only the visible tab is rendered; hidden tabs are rendered on-demand when
        # selected. Downside: slight delay when switching to a tab.
        self._tabs = pn.Tabs(
            sizing_mode='stretch_both',
            dynamic=True,
            stylesheets=[
                f"""
                {static_tab_selectors} {{
                    font-weight: bold;
                }}
                .bk-tab {{
                    border-bottom: 1px solid #2c5aa0 !important;
                }}
                .bk-tab.bk-active {{
                    background-color: #e8f4f8 !important;
                    border: 1px solid #2c5aa0 !important;
                    border-bottom: none !important;
                }}
                """
            ],
        )

        # Modal container for plot configuration
        # IMPORTANT: Use height=0 to ensure the modal is in the component tree
        # (required for rendering) but doesn't compete for vertical space.
        # The modal itself renders as an overlay when shown.
        # This container must be at the TOP LEVEL (wrapping Tabs), not inside
        # individual grid tabs. Panel components can only have one parent, so
        # adding the same container to multiple tabs would break rendering.
        self._modal_container = pn.Row(height=0, sizing_mode='stretch_width')
        self._current_modal: PlotConfigModal | None = None

        # Create main widget - tabs with zero-height modal container
        # IMPORTANT: Create the widget once in __init__ (not in the panel property)
        # to give modal_container a stable parent. If we created a new Column on
        # each .panel access, the modal_container would be reparented repeatedly,
        # breaking its connection to the component tree.
        self._widget = pn.Column(
            self._tabs, self._modal_container, sizing_mode='stretch_both'
        )

        # Subscribe to lifecycle events
        self._subscription_id: SubscriptionId | None = (
            self._orchestrator.subscribe_to_lifecycle(
                on_grid_created=self._on_grid_created,
                on_grid_removed=self._on_grid_removed,
                on_cell_updated=self._on_cell_updated,
                on_cell_removed=self._on_cell_removed,
            )
        )

        # Add Jobs tab (always first)
        self._tabs.append(('Jobs', job_status_widget.panel()))

        # Add Workflows tab (always second)
        self._tabs.append(('Workflows', workflow_status_widget.panel()))

        # Add System Status tab (third, if widget provided)
        if system_status_widget is not None:
            self._tabs.append(('System Status', system_status_widget.panel()))

        # Add Manage tab (third or fourth depending on system_status_widget)
        self._grid_manager = PlotGridManager(
            orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
        )
        self._tabs.append(('Manage Plots', self._grid_manager.panel))

        # Store static tabs count for use as offset in grid tab index calculations
        self._static_tabs_count = len(self._tabs)

        # Initialize from existing grids
        for grid_id, grid_config in self._orchestrator.get_all_grids().items():
            self._add_grid_tab(grid_id, grid_config)

        # Register handler for periodic polling
        session_updater.register_custom_handler(self._poll_for_plot_updates)

    def _add_grid_tab(self, grid_id: GridId, grid_config: PlotGridConfig) -> None:
        """Add a new grid tab after the Manage tab."""

        # Create grid-specific callback using closure to capture grid_id
        def grid_callback(geometry: CellGeometry) -> None:
            self._on_plot_requested(grid_id, geometry)

        # Create PlotGrid widget with grid-specific callback
        plot_grid = PlotGrid(
            nrows=grid_config.nrows,
            ncols=grid_config.ncols,
            plot_request_callback=grid_callback,
        )

        # Store widget reference
        self._grid_widgets[grid_id] = plot_grid

        # Append grid directly to tabs (Manage tab is always first at index 0)
        # NOTE: Do NOT wrap each grid with modal_container here. The modal
        # container is shared across all tabs and lives at the top level
        # (wrapping the entire Tabs widget).
        self._tabs.append((grid_config.title, plot_grid.panel))

        # Populate with existing cells (important for late subscribers / new sessions)
        for cell_id, cell in grid_config.cells.items():
            # Notify about cell config - widget will query PlotDataService for state
            self._on_cell_updated(
                grid_id=grid_id,
                cell_id=cell_id,
                cell=cell,
            )

    def _remove_grid_tab(self, grid_id: GridId) -> None:
        """Remove a grid tab."""
        if grid_id not in self._grid_widgets:
            return

        tab_index = list(self._grid_widgets.keys()).index(grid_id)
        self._tabs.pop(self._static_tabs_count + tab_index)
        del self._grid_widgets[grid_id]

    def _get_active_grid_id(self) -> GridId | None:
        """Return the GridId of the currently visible grid tab, or None.

        The tab widget contains static tabs (Jobs, Workflows, Manage Plots, ...)
        followed by dynamic grid tabs. Subtracting the static tab count from the
        active tab index maps to a grid position. When a static tab is selected
        the result is negative, which the bounds check rejects.

        Returns None when a config modal is open: the modal overlay obscures
        the plots, so rendering would be wasted. Dirty flags are preserved and
        the first poll after the modal closes pushes the latest cached state.
        """
        if self._current_modal is not None:
            return None
        grid_idx = self._tabs.active - self._static_tabs_count
        grid_keys = list(self._grid_widgets.keys())
        if 0 <= grid_idx < len(grid_keys):
            return grid_keys[grid_idx]
        return None

    def _on_grid_created(self, grid_id: GridId, grid_config: PlotGridConfig) -> None:
        """Handle grid creation from orchestrator."""
        self._add_grid_tab(grid_id, grid_config)

        if grid_id in self._grid_widgets:
            tab_index = list(self._grid_widgets.keys()).index(grid_id)
            self._tabs.active = self._static_tabs_count + tab_index

    def _on_grid_removed(self, grid_id: GridId) -> None:
        """Handle grid removal from orchestrator."""
        self._remove_grid_tab(grid_id)

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

        def on_success(plot_config: PlotConfig) -> None:
            """Handle successful plot configuration."""
            cell_id = self._orchestrator.add_cell(grid_id, geometry)
            self._orchestrator.add_layer(cell_id, plot_config)

        self._show_config_modal(on_success=on_success)

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
            self._orchestrator.update_layer_config(layer_id, plot_config)

        current_config = self._orchestrator.get_layer_config(layer_id)
        self._show_config_modal(on_success=on_success, initial_config=current_config)

    def _on_add_layer(self, cell_id: CellId) -> None:
        """
        Handle add layer request from plus button.

        Shows the PlotConfigModal to configure the new layer, then adds it
        to the cell in the orchestrator on success.

        Parameters
        ----------
        cell_id
            ID of the cell to add the layer to.
        """

        def on_success(plot_config: PlotConfig) -> None:
            """Handle successful layer configuration."""
            self._orchestrator.add_layer(cell_id, plot_config)

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

    def _on_cell_updated(
        self,
        *,
        grid_id: GridId,
        cell_id: CellId,
        cell: PlotCell,
    ) -> None:
        """
        Handle cell update from orchestrator.

        Creates a cell widget with per-layer toolbars and either a placeholder
        or the composed plot, then inserts it into the grid.

        This is called when cell configuration changes (layer added/removed/updated).
        Layer runtime state (error, stopped, data) is queried from PlotDataService.

        Parameters
        ----------
        grid_id
            ID of the grid containing the cell.
        cell_id
            ID of the cell being updated.
        cell
            Plot cell configuration with all layers.
        """
        plot_grid = self._grid_widgets.get(grid_id)
        if plot_grid is None:
            return

        # Get session-local composed plot if data is available.
        # Note: When config changes, update_layer_config() creates a new layer_id.
        # The old layer_id is orphaned and cleaned up by update_pipes().
        # New layer_ids have no cache, so fresh components are created naturally.
        session_plot = self._get_session_composed_plot(cell)

        # Create widget with toolbars and content
        widget = self._create_cell_widget(cell_id, cell, session_plot)

        # Note: SessionLayer.create() already records state.version in
        # last_seen_version, so _poll_for_plot_updates won't trigger
        # redundant rebuilds.

        # Defer insertion for plots to allow Panel to update layout sizing.
        # When a workflow is already running with data, subscribing triggers
        # plot creation synchronously (in subscribe_to_workflow's immediate
        # callback path). This can cause the HoloViews pane to initialize with
        # collapsed/default size before the grid container is properly sized,
        # resulting in "glitched" rendering. Deferring to the next event loop
        # iteration allows Panel to process layout updates first.
        if session_plot is not None:
            pn.state.execute(
                lambda g=cell.geometry: plot_grid.insert_widget_at(g, widget)
            )
        else:
            # Status widgets can be inserted immediately
            plot_grid.insert_widget_at(cell.geometry, widget)

    def _on_cell_removed(self, grid_id: GridId, geometry: CellGeometry) -> None:
        """
        Handle cell removal from orchestrator.

        Removes the widget from the grid at the specified position.

        Parameters
        ----------
        grid_id
            ID of the grid containing the cell.
        geometry
            Cell geometry of the removed cell.
        """
        plot_grid = self._grid_widgets.get(grid_id)
        if plot_grid is None:
            return

        # Remove widget at explicit position
        plot_grid.remove_widget_at(geometry)

    def _get_layer_states(self, cell: PlotCell) -> dict[LayerId, LayerStateMachine]:
        """
        Get layer states from PlotDataService for all layers in a cell.

        Parameters
        ----------
        cell
            Plot cell with layers.

        Returns
        -------
        :
            Dict mapping layer IDs to their state.
        """
        result: dict[LayerId, LayerStateMachine] = {}
        for layer in cell.layers:
            state = self._plot_data_service.get(layer.layer_id)
            if state is None:
                raise RuntimeError(
                    f"Layer {layer.layer_id} has no state in PlotDataService. "
                    "This indicates a bug: layers should be registered before "
                    "widgets are notified."
                )
            result[layer.layer_id] = state
        return result

    def _create_cell_widget(
        self,
        cell_id: CellId,
        cell: PlotCell,
        plot: hv.DynamicMap | hv.Element | hv.Overlay | None,
    ) -> pn.Column:
        """
        Create a cell widget with per-layer toolbars and content area.

        The widget has a stable toolbar section (one toolbar per layer) and
        a content area that shows either a placeholder or the composed plot.

        Parameters
        ----------
        cell_id
            ID of the cell.
        cell
            Plot cell configuration with all layers.
        plot
            The composed plot, or None if no layers have data yet.

        Returns
        -------
        :
            Panel widget with toolbars and content.
        """
        # Get layer states from PlotDataService
        layer_states = self._get_layer_states(cell)

        # Create toolbars for all layers
        toolbars = self._create_layer_toolbars(cell_id, cell, layer_states)

        # Create content area (placeholder or plot)
        if plot is not None:
            content = self._create_plot_content(cell, plot)
            border = None
            bg_color = None
        else:
            content = self._create_placeholder_content(cell, layer_states)
            # Check if any layer has an error
            has_error = any(
                state.error_message is not None for state in layer_states.values()
            )
            if has_error:
                bg_color = '#ffe6e6'
                border = '2px solid #dc3545'
            else:
                bg_color = '#f8f9fa'
                border = '2px dashed #dee2e6'

        styles = {}
        if bg_color:
            styles['background-color'] = bg_color
        if border:
            styles['border'] = border

        return pn.Column(
            *toolbars,
            content,
            sizing_mode='stretch_both',
            styles=styles,
            margin=GridCellStyles.CELL_MARGIN,
        )

    def _get_available_overlays_for_layer(
        self, config: PlotConfig
    ) -> list[tuple[str, str, str]]:
        """
        Get available overlay suggestions for a layer based on its configuration.

        Parameters
        ----------
        config
            Layer's plot configuration.

        Returns
        -------
        :
            List of (output_name, plotter_name, plotter_title) tuples.
            Returns empty list if overlays are not applicable.
        """
        # Skip for static overlays (no workflow)
        if config.is_static():
            return []

        # Get workflow spec
        workflow_spec = self._workflow_registry.get(config.workflow_id)
        if workflow_spec is None:
            return []

        return self._plotting_controller.get_available_overlays(
            workflow_spec, config.plot_name
        )

    def _create_overlay_layer(
        self,
        cell_id: CellId,
        base_config: PlotConfig,
        output_name: str,
        plotter_name: str,
    ) -> None:
        """
        Create an overlay layer inheriting configuration from a base layer.

        Parameters
        ----------
        cell_id
            ID of the cell to add the overlay to.
        base_config
            Configuration of the base layer (e.g., image layer).
        output_name
            Name of the output for the overlay (e.g., 'rectangles_readback').
        plotter_name
            Name of the plotter to use for the overlay.
        """
        # Get default params for the plotter
        spec = self._plotting_controller.get_spec(plotter_name)
        params = spec.params() if spec.params else None

        # Create PlotConfig inheriting workflow/sources from base layer
        overlay_config = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=base_config.workflow_id,
                    source_names=list(base_config.source_names),
                    output_name=output_name,
                )
            },
            plot_name=plotter_name,
            params=params,
        )

        self._orchestrator.add_layer(cell_id, overlay_config)

    def _create_layer_toolbars(
        self,
        cell_id: CellId,
        cell: PlotCell,
        layer_states: dict[LayerId, LayerStateMachine],
    ) -> list[pn.Row]:
        """
        Create toolbars for all layers in a cell.

        Parameters
        ----------
        cell_id
            ID of the cell.
        cell
            Plot cell with layers.
        layer_states
            Per-layer runtime state from PlotDataService.

        Returns
        -------
        :
            List of toolbar widgets, one per layer.
        """
        # Collect existing plotter names to filter already-added overlays
        existing_plotter_names = {layer.config.plot_name for layer in cell.layers}

        toolbars = []
        for layer in cell.layers:
            layer_id = layer.layer_id
            config = layer.config
            state = layer_states[layer_id]

            # Get display info for this layer
            title, description = get_plot_cell_display_info(
                config,
                self._workflow_registry,
                get_source_title=self._orchestrator.get_source_title,
            )

            # Add state info to description using explicit state enum
            stopped = False
            match state.state:
                case LayerState.ERROR:
                    description = f"{description}\n\nError: {state.error_message}"
                case LayerState.STOPPED:
                    stopped = True
                    description = f"{description}\n\nStatus: Workflow ended"
                case LayerState.WAITING_FOR_DATA:
                    description = f"{description}\n\nStatus: Waiting for data..."
                case LayerState.WAITING_FOR_JOB:
                    description = f"{description}\n\nStatus: Waiting for workflow"
                case LayerState.READY:
                    pass  # No extra description for ready state

            # Get available overlays, excluding those already present in the cell
            available_overlays = [
                overlay
                for overlay in self._get_available_overlays_for_layer(config)
                if overlay[1]
                not in existing_plotter_names  # overlay[1] is plotter_name
            ]

            # Create callbacks that capture layer_id / cell_id / config
            def make_close_callback(lid: LayerId) -> Callable[[], None]:
                def on_close() -> None:
                    self._orchestrator.remove_layer(lid)

                return on_close

            def make_gear_callback(lid: LayerId) -> Callable[[], None]:
                def on_gear() -> None:
                    self._on_reconfigure_layer(lid)

                return on_gear

            def make_add_callback(cid: CellId) -> Callable[[], None]:
                def on_add() -> None:
                    self._on_add_layer(cid)

                return on_add

            def make_overlay_callback(
                cid: CellId, cfg: PlotConfig
            ) -> Callable[[str, str], None]:
                def on_overlay(output_name: str, plotter_name: str) -> None:
                    self._create_overlay_layer(cid, cfg, output_name, plotter_name)

                return on_overlay

            toolbar = create_cell_toolbar(
                on_gear_callback=make_gear_callback(layer_id),
                on_close_callback=make_close_callback(layer_id),
                on_add_callback=make_add_callback(cell_id),
                on_overlay_selected=make_overlay_callback(cell_id, config),
                available_overlays=available_overlays,
                title=title,
                description=description,
                stopped=stopped,
            )
            toolbars.append(toolbar)

        return toolbars

    def _create_placeholder_content(
        self,
        cell: PlotCell,
        layer_states: dict[LayerId, LayerStateMachine],
    ) -> pn.pane.Markdown:
        """
        Create placeholder content showing layer status.

        Parameters
        ----------
        cell
            Plot cell with layers.
        layer_states
            Per-layer runtime state from PlotDataService.

        Returns
        -------
        :
            Markdown pane showing status for all layers.
        """
        # Build status info for each layer
        status_lines = []
        for layer in cell.layers:
            config = layer.config
            state = layer_states[layer.layer_id]

            workflow_title, output_title = get_workflow_display_info(
                self._workflow_registry, config.workflow_id, config.output_name
            )

            # Determine status from explicit state enum
            match state.state:
                case LayerState.ERROR:
                    error_text = state.error_message or "Unknown error"
                    status = f"Error: {extract_error_summary(error_text)}"
                    text_color = '#dc3545'
                case LayerState.STOPPED:
                    status = "Workflow ended"
                    text_color = '#495057'
                case LayerState.WAITING_FOR_DATA:
                    status = "Waiting for data..."
                    text_color = '#6c757d'
                case LayerState.WAITING_FOR_JOB:
                    status = "Waiting for workflow..."
                    text_color = '#6c757d'
                case LayerState.READY:
                    # Defensive: READY should have displayable plot and not
                    # reach placeholder. Log if this happens.
                    logger.warning(
                        "Layer %s in READY state but showing placeholder",
                        layer.layer_id,
                    )
                    status = "Ready (loading...)"
                    text_color = '#6c757d'

            status_lines.append(
                f"**{workflow_title} → {output_title}**: "
                f"<span style='color: {text_color}'>{status}</span>"
            )

        content = "\n\n".join(status_lines)

        return pn.pane.Markdown(
            content,
            styles={'text-align': 'left', 'padding': '20px'},
        )

    def _create_plot_content(
        self,
        cell: PlotCell,
        plot: hv.DynamicMap | hv.Element | hv.Overlay,
    ) -> pn.pane.HoloViews:
        """
        Create plot content widget.

        Parameters
        ----------
        cell
            Plot cell with layers.
        plot
            The composed plot.

        Returns
        -------
        :
            HoloViews pane containing the plot.
        """
        # Use sizing mode from first layer (they should be consistent for overlay)
        if cell.layers:
            sizing_mode = _get_sizing_mode(cell.layers[0].config)
        else:
            sizing_mode = 'stretch_both'

        # Use .layout to preserve widgets for DynamicMaps with kdims.
        # When pn.pane.HoloViews wraps a DynamicMap with kdims, it generates
        # widgets. However, these widgets don't render when the pane is placed
        # in a Panel layout (Tabs, Column, etc.). The .layout property contains
        # both the plot and widgets, which renders correctly in layouts.
        # See: https://github.com/holoviz/panel/issues/5628
        #
        # CRITICAL: Use linked_axes=False to prevent unintended axis linking (#607)
        #
        # Problem: By default, Panel's HoloViews pane links axes across different
        # plots based on their axis labels (e.g., all plots with 'x' and 'y' axes
        # get linked). For detector panels in different grid cells, this is unwanted:
        # - Different detector panels have independent spatial coordinates
        # - Zooming one panel shouldn't affect others
        # - Each panel needs independent autoscaling
        #
        # Previous workarounds and why they failed:
        # - shared_axes=False (HoloViews): Breaks framewise autoscaling and other
        #   dynamic features that rely on shared axis infrastructure
        # - Wrapping in hv.Layout: Prevents multi-layer composition with hv.Overlay,
        #   which was needed for the layer system (#606)
        #
        # Solution: linked_axes=False on the Panel pane
        # - Disables Panel's cross-plot axis linking while preserving all HoloViews
        #   features (framewise options, autoscaling, dynamic updates)
        # - Allows proper multi-layer composition via hv.Overlay
        # - Each grid cell's plot remains independent
        plot_pane_wrapper = pn.pane.HoloViews(
            plot, sizing_mode=sizing_mode, linked_axes=False
        )
        return plot_pane_wrapper.layout

    def _get_session_composed_plot(
        self, cell: PlotCell
    ) -> hv.DynamicMap | hv.Element | None:
        """
        Get composed plot from session-local DynamicMaps or static elements.

        Ensures session components exist when data is available.
        Sets a descriptive SaveTool filename on the result so that
        browser "Save" downloads get a meaningful name.

        Parameters
        ----------
        cell
            The plot cell with layers.

        Returns
        -------
        :
            Composed plot from session DMaps/elements, or None if none available.
        """
        plots = []
        has_layout = False
        for layer in cell.layers:
            layer_id = layer.layer_id
            session_layer = self._session_layers.get(layer_id)
            if session_layer is None:
                continue

            # Ensure components exist if data is now available
            state = self._plot_data_service.get(layer_id)
            if state is not None:
                session_layer.ensure_components(state)
                if state.plotter is not None and isinstance(
                    state.plotter.get_cached_state(), hv.Layout
                ):
                    has_layout = True

            if session_layer.dmap is not None:
                plots.append(session_layer.dmap)

        if not plots:
            return None

        result: hv.DynamicMap | hv.Element
        if len(plots) == 1:
            result = plots[0]
        else:
            result = hv.Overlay(plots)

        # Skip hooks for Layouts — each sub-figure has its own SaveTool,
        # so a single cell-level filename is not meaningful.
        if not has_layout:
            filename = build_save_filename_from_cell(
                cell, self._workflow_registry, self._orchestrator.get_source_title
            )
            if filename is not None:
                hook = make_save_filename_hook(filename)
                result = result.opts(hooks=[hook])

        return result

    def _poll_for_plot_updates(self) -> None:
        """
        Poll PlotDataService for updates and set up new layers.

        Called from SessionUpdater's periodic callback. Single pass over all
        orchestrator layers to:
        - Push data updates to existing session pipes (active tab only)
        - Detect version changes requiring cell rebuilds
        - Create/update session layers as needed

        Only layers on the currently visible grid tab call ``update_pipe()``,
        since ``dynamic=True`` on Tabs means hidden tabs have no materialized
        Bokeh models. Skipped layers keep their dirty flag set; on tab switch
        the next poll cycle sends the latest cached state.

        Version-based change detection replaces callback-based updates for state
        changes (waiting/ready/stopped/error). Polling at ~100ms intervals is
        acceptable for config UI updates.
        """
        cells_to_rebuild: dict[CellId, tuple[PlotCell, PlotGrid]] = {}
        seen_layer_ids: set[LayerId] = set()
        active_grid_id = self._get_active_grid_id()

        for grid_id, plot_grid in self._grid_widgets.items():
            grid_config = self._orchestrator.peek_grid(grid_id)
            if grid_config is None:
                continue

            is_active = grid_id == active_grid_id

            for cell_id, cell in grid_config.cells.items():
                for layer in cell.layers:
                    layer_id = layer.layer_id
                    seen_layer_ids.add(layer_id)

                    state = self._plot_data_service.get(layer_id)
                    if state is None:
                        # Should not happen: layers are registered before widgets
                        # are notified. Skip this layer but log for debugging.
                        logger.warning(
                            "Layer %s has no state in PlotDataService during poll",
                            layer_id,
                        )
                        continue

                    # Get or create session layer for version tracking
                    session_layer = self._session_layers.get(layer_id)
                    if session_layer is None:
                        self._session_layers[layer_id] = SessionLayer(
                            layer_id=layer_id, last_seen_version=state.version
                        )
                        # New layer → rebuild cell
                        cells_to_rebuild[cell_id] = (cell, plot_grid)
                    else:
                        if is_active:
                            session_layer.update_pipe()

                        # Check for version changes (plotter changes increment version)
                        if state.version != session_layer.last_seen_version:
                            cells_to_rebuild[cell_id] = (cell, plot_grid)
                            session_layer.last_seen_version = state.version

        # Clean up orphaned session layers (removed from orchestrator)
        for layer_id in list(self._session_layers.keys()):
            if layer_id not in seen_layer_ids:
                del self._session_layers[layer_id]

        # Rebuild affected cells.
        # Defer insertion to allow Bokeh to process any pending model updates
        # from pipe.send() calls above. Without deferral, widget removal can
        # race with DynamicMap updates, causing KeyError when Panel tries to
        # access removed models.
        for cell_id, (cell, plot_grid) in cells_to_rebuild.items():
            session_plot = self._get_session_composed_plot(cell)
            widget = self._create_cell_widget(cell_id, cell, session_plot)
            pn.state.execute(
                lambda g=cell.geometry, w=widget, pg=plot_grid: pg.insert_widget_at(
                    g, w
                )
            )

    def shutdown(self) -> None:
        """Unsubscribe from lifecycle events and clean up session state."""
        if self._subscription_id is not None:
            self._orchestrator.unsubscribe_from_lifecycle(self._subscription_id)
            self._subscription_id = None
        self._session_layers.clear()
        self._grid_manager.shutdown()

    @property
    def panel(self) -> pn.Column:
        """Get the Panel viewable object for this widget."""
        return self._widget

    @property
    def tabs(self) -> pn.Tabs:
        """Get the Tabs widget containing grid tabs."""
        return self._tabs
