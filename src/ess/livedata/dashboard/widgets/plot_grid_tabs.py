# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
PlotGridTabs - Tabbed interface for managing multiple plot grids.

Provides a Panel Tabs widget that displays multiple PlotGrid instances,
synchronized with PlotOrchestrator via lifecycle subscriptions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import holoviews as hv
import panel as pn

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec

from ..plot_orchestrator import (
    CellGeometry,
    CellId,
    GridId,
    PlotCell,
    PlotConfig,
    PlotGridConfig,
    PlotOrchestrator,
    SubscriptionId,
)
from ..plot_params import PlotAspectType, StretchMode
from .plot_config_modal import PlotConfigModal
from .plot_grid import GridCellStyles, PlotGrid
from .plot_grid_manager import PlotGridManager
from .plot_widgets import (
    create_cell_toolbar,
    get_plot_cell_display_info,
    get_workflow_display_info,
)


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

    Displays a "Manage" tab (always first) for adding/removing grids,
    a "Jobs" tab for monitoring job status, followed by one tab per plot grid.
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
    """

    def __init__(
        self,
        plot_orchestrator: PlotOrchestrator,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        plotting_controller,
        job_status_widget,
    ) -> None:
        self._orchestrator = plot_orchestrator
        self._workflow_registry = dict(workflow_registry)
        self._plotting_controller = plotting_controller

        # Track grid widgets (insertion order determines tab position)
        self._grid_widgets: dict[GridId, PlotGrid] = {}

        # Main tabs widget.
        # IMPORTANT: dynamic=True is critical for performance. Without it, Panel
        # renders ALL tabs simultaneously, causing severe UI lag (3-5 seconds) when
        # any tab content changes (e.g., updating grid preview). We even observe causes
        # of near-total UI freezes when there are many active plots. With dynamic=True,
        # only the visible tab is rendered; hidden tabs are rendered on-demand when
        # selected. Downside: slight delay when switching to a tab.
        self._tabs = pn.Tabs(sizing_mode='stretch_both', dynamic=True)

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

        # Add Manage tab (always second)
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
            # Get current plot/error state from orchestrator
            # This ensures late subscribers (new sessions) see existing plots
            plot, error = self._orchestrator.get_cell_state(cell_id)
            self._on_cell_updated(
                grid_id=grid_id, cell_id=cell_id, cell=cell, plot=plot, error=error
            )

    def _remove_grid_tab(self, grid_id: GridId) -> None:
        """Remove a grid tab."""
        if grid_id not in self._grid_widgets:
            return

        tab_index = list(self._grid_widgets.keys()).index(grid_id)
        self._tabs.pop(self._static_tabs_count + tab_index)
        del self._grid_widgets[grid_id]

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
            plot_cell = PlotCell(geometry=geometry, config=plot_config)
            self._orchestrator.add_plot(grid_id, plot_cell)

        self._show_config_modal(on_success=on_success)

    def _on_reconfigure_plot(self, cell_id: CellId) -> None:
        """
        Handle plot reconfiguration request from gear button.

        Shows the PlotConfigModal with existing configuration, then updates
        the plot in the orchestrator on success.

        Parameters
        ----------
        cell_id
            ID of the cell to reconfigure.
        """

        def on_success(plot_config: PlotConfig) -> None:
            """Handle successful plot reconfiguration."""
            self._orchestrator.update_plot_config(cell_id, plot_config)

        current_config = self._orchestrator.get_plot_config(cell_id)
        self._show_config_modal(on_success=on_success, initial_config=current_config)

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
        plot: Any = None,
        error: str | None = None,
    ) -> None:
        """
        Handle cell update from orchestrator.

        Creates appropriate widget (placeholder, plot, or error) and inserts
        it into the grid at the specified position.

        Parameters
        ----------
        grid_id
            ID of the grid containing the cell.
        cell_id
            ID of the cell being updated.
        cell
            Plot cell configuration.
        plot
            The plot widget (HoloViews DynamicMap), or None if not yet available.
        error
            Error message if plot creation failed, or None.
        """
        plot_grid = self._grid_widgets.get(grid_id)
        if plot_grid is None:
            return

        # Create appropriate widget based on what's available
        if plot is not None:
            # Show actual plot
            widget = self._create_plot_widget(cell_id, cell, plot)
        else:
            # Show status widget (either waiting for data or error)
            widget = self._create_status_widget(cell_id, cell, error=error)

        # Defer insertion for plots to allow Panel to update layout sizing.
        # When a workflow is already running with data, subscribing triggers
        # plot creation synchronously (in subscribe_to_workflow's immediate
        # callback path). This can cause the HoloViews pane to initialize with
        # collapsed/default size before the grid container is properly sized,
        # resulting in "glitched" rendering. Deferring to the next event loop
        # iteration allows Panel to process layout updates first.
        if plot is not None:
            # Schedule insertion on next event loop iteration using pn.state.execute
            # This is more appropriate than add_periodic_callback for one-shot deferred
            # execution and ensures proper thread safety for Bokeh model updates.
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

    def _create_status_widget(
        self, cell_id: CellId, cell: PlotCell, error: str | None = None
    ) -> pn.Column:
        """
        Create a status widget for a cell without a plot.

        Shows either a placeholder (waiting for data) or an error message,
        depending on whether an error occurred during plot creation.

        Parameters
        ----------
        cell_id
            ID of the cell.
        cell
            Plot cell configuration.
        error
            Error message if plot creation failed, or None for placeholder.

        Returns
        -------
        :
            Panel widget showing status information.
        """
        config = cell.config

        # Get display info for toolbar
        toolbar_title, toolbar_description = get_plot_cell_display_info(
            config, self._workflow_registry
        )

        # Get workflow and output display titles for body content
        workflow_title, output_title = get_workflow_display_info(
            self._workflow_registry, config.workflow_id, config.output_name
        )

        # Build title from workflow and output (most prominent)
        body_title = f"### {workflow_title} - {output_title}"

        # Build info section: sources first, then status
        info_lines = [
            f"**Sources:** {', '.join(config.source_names)}",
        ]

        # Add status-specific line and determine styling
        if error is not None:
            info_lines.append(f"**Error:** {error}")
            text_color = '#dc3545'
            bg_color = '#ffe6e6'
            border = '2px solid #dc3545'
        else:
            info_lines.append("**Status:** Waiting for data...")
            text_color = '#6c757d'
            bg_color = '#f8f9fa'
            border = '2px dashed #dee2e6'

        content = f"{body_title}\n\n" + "\n\n".join(info_lines)

        # Create toolbar with title, description, and buttons
        def on_close() -> None:
            self._orchestrator.remove_plot(cell_id)

        def on_gear() -> None:
            self._on_reconfigure_plot(cell_id)

        toolbar = create_cell_toolbar(
            on_gear_callback=on_gear,
            on_close_callback=on_close,
            title=toolbar_title,
            description=toolbar_description,
        )

        status_widget = pn.Column(
            toolbar,
            pn.pane.Markdown(
                content,
                styles={
                    'text-align': 'left',
                    'color': text_color,
                    'padding': '20px',
                },
            ),
            sizing_mode='stretch_both',
            styles={
                'background-color': bg_color,
                'border': border,
            },
            margin=GridCellStyles.CELL_MARGIN,
        )
        return status_widget

    def _create_plot_widget(
        self,
        cell_id: CellId,
        cell: PlotCell,
        plot: hv.DynamicMap | hv.Layout,
    ) -> pn.Column:
        """
        Create a widget containing the actual plot.

        Parameters
        ----------
        cell_id
            ID of the cell.
        cell
            Plot cell configuration.
        plot
            HoloViews plot object.

        Returns
        -------
        :
            Panel widget containing the plot.
        """

        # Get display info for toolbar
        title, description = get_plot_cell_display_info(
            cell.config, self._workflow_registry
        )

        # Create toolbar with title, description, and buttons
        def on_close() -> None:
            self._orchestrator.remove_plot(cell_id)

        def on_gear() -> None:
            self._on_reconfigure_plot(cell_id)

        toolbar = create_cell_toolbar(
            on_gear_callback=on_gear,
            on_close_callback=on_close,
            title=title,
            description=description,
        )

        # Use .layout to preserve widgets for DynamicMaps with kdims.
        # When pn.pane.HoloViews wraps a DynamicMap with kdims, it generates
        # widgets. However, these widgets don't render when the pane is placed
        # in a Panel layout (Tabs, Column, etc.). The .layout property contains
        # both the plot and widgets, which renders correctly in layouts.
        # See: https://github.com/holoviz/panel/issues/5628
        sizing_mode = _get_sizing_mode(cell.config)
        plot_pane_wrapper = pn.pane.HoloViews(plot, sizing_mode=sizing_mode)
        plot_pane = plot_pane_wrapper.layout

        return pn.Column(
            toolbar,
            plot_pane,
            sizing_mode='stretch_both',
            margin=GridCellStyles.CELL_MARGIN,
        )

    def shutdown(self) -> None:
        """Unsubscribe from lifecycle events and shutdown manager."""
        if self._subscription_id is not None:
            self._orchestrator.unsubscribe_from_lifecycle(self._subscription_id)
            self._subscription_id = None
        self._grid_manager.shutdown()

    @property
    def panel(self) -> pn.Column:
        """Get the Panel viewable object for this widget."""
        return self._widget

    @property
    def tabs(self) -> pn.Tabs:
        """Get the Tabs widget containing grid tabs."""
        return self._tabs
