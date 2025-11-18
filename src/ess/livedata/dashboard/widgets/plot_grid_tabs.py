# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
PlotGridTabs - Tabbed interface for managing multiple plot grids.

Provides a Panel Tabs widget that displays multiple PlotGrid instances,
synchronized with PlotOrchestrator via lifecycle subscriptions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import holoviews as hv
import panel as pn

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec

from ..plot_orchestrator import (
    CellId,
    GridId,
    PlotCell,
    PlotConfig,
    PlotGridConfig,
    PlotOrchestrator,
    SubscriptionId,
)
from .plot_grid import PlotGrid, _create_close_button
from .plot_grid_manager import PlotGridManager
from .simple_plot_config_modal import PlotConfigResult, SimplePlotConfigModal


class PlotGridTabs:
    """
    Tabbed widget for managing multiple plot grids.

    Displays a "Manage" tab (always first) for adding/removing grids,
    followed by one tab per plot grid. Synchronizes with PlotOrchestrator
    via lifecycle subscriptions to support multiple linked instances.

    Parameters
    ----------
    plot_orchestrator
        The orchestrator managing plot grid configurations.
    workflow_registry
        Registry of available workflows and their specifications.
    plotting_controller
        Controller for determining available plotters from workflow specs.
    """

    def __init__(
        self,
        plot_orchestrator: PlotOrchestrator,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        plotting_controller,
    ) -> None:
        self._orchestrator = plot_orchestrator
        self._workflow_registry = dict(workflow_registry)
        self._plotting_controller = plotting_controller

        # Track grid widgets and tab indices
        self._grid_widgets: dict[GridId, PlotGrid] = {}
        self._grid_to_tab_index: dict[GridId, int] = {}

        # Track cells added to each grid for lifecycle management
        self._grid_cells: dict[GridId, dict[CellId, tuple[int, int, int, int]]] = {}

        # Main tabs widget
        self._tabs = pn.Tabs(sizing_mode='stretch_both')

        # Modal container for plot configuration
        # Using pn.Row with height=0 ensures the modal is part of the component tree
        # but doesn't compete for vertical space. The modal renders as an overlay.
        self._modal_container = pn.Row(height=0, sizing_mode='stretch_width')
        self._current_modal: SimplePlotConfigModal | None = None
        self._modal_grid_id: GridId | None = None

        # Subscribe to lifecycle events
        self._subscription_id: SubscriptionId | None = (
            self._orchestrator.subscribe_to_lifecycle(
                on_grid_created=self._on_grid_created,
                on_grid_removed=self._on_grid_removed,
                on_cell_updated=self._on_cell_updated,
                on_cell_removed=self._on_cell_removed,
            )
        )

        # Add Manage tab (always first)
        self._grid_manager = PlotGridManager(orchestrator=plot_orchestrator)
        self._tabs.append(('Manage', self._grid_manager.panel))

        # Initialize from existing grids
        for grid_id, grid_config in self._orchestrator.get_all_grids().items():
            self._add_grid_tab(grid_id, grid_config)

    def _add_grid_tab(self, grid_id: GridId, grid_config: PlotGridConfig) -> None:
        """Add a new grid tab after the Manage tab."""

        # Create grid-specific callback using closure to capture grid_id
        def grid_callback() -> None:
            self._on_plot_requested(grid_id)

        # Create PlotGrid widget with grid-specific callback
        plot_grid = PlotGrid(
            nrows=grid_config.nrows,
            ncols=grid_config.ncols,
            plot_request_callback=grid_callback,
        )

        # Store widget reference
        self._grid_widgets[grid_id] = plot_grid

        # Initialize cell tracking for this grid
        self._grid_cells[grid_id] = {}

        # Wrap PlotGrid with modal container for consistent layout
        grid_with_modal = pn.Column(
            plot_grid.panel,
            self._modal_container,
            sizing_mode='stretch_both',
        )

        # Append at the end (Manage tab is always first at index 0)
        tab_index = len(self._tabs)
        self._tabs.append((grid_config.title, grid_with_modal))
        self._grid_to_tab_index[grid_id] = tab_index

        # Populate with existing cells (important for late subscribers / new sessions)
        for cell in grid_config.cells.values():
            # Create placeholder widget for each existing cell
            # (actual plots will be created when workflows commit)
            self._on_cell_updated(grid_id, cell, plot=None, error=None)

    def _remove_grid_tab(self, grid_id: GridId) -> None:
        """Remove a grid tab and update indices."""
        if grid_id not in self._grid_to_tab_index:
            return

        tab_index = self._grid_to_tab_index[grid_id]

        # Remove the tab using pop()
        self._tabs.pop(tab_index)

        # Clean up tracking
        del self._grid_widgets[grid_id]
        del self._grid_to_tab_index[grid_id]
        if grid_id in self._grid_cells:
            del self._grid_cells[grid_id]

        # Update indices for all grids that came after the removed one
        for gid, idx in list(self._grid_to_tab_index.items()):
            if idx > tab_index:
                self._grid_to_tab_index[gid] = idx - 1

    def _on_grid_created(self, grid_id: GridId, grid_config: PlotGridConfig) -> None:
        """Handle grid creation from orchestrator."""
        self._add_grid_tab(grid_id, grid_config)

        # Auto-switch to the new tab
        if grid_id in self._grid_to_tab_index:
            self._tabs.active = self._grid_to_tab_index[grid_id]

    def _on_grid_removed(self, grid_id: GridId) -> None:
        """Handle grid removal from orchestrator."""
        self._remove_grid_tab(grid_id)

    def _on_plot_requested(self, grid_id: GridId) -> None:
        """
        Handle plot request from PlotGrid.

        Shows the SimplePlotConfigModal to configure the plot, then adds it
        to the orchestrator on success.

        Parameters
        ----------
        grid_id
            ID of the grid where the plot was requested.
        """
        # Store which grid this modal is for
        self._modal_grid_id = grid_id

        # Create and show modal
        self._current_modal = SimplePlotConfigModal(
            workflow_registry=self._workflow_registry,
            plotting_controller=self._plotting_controller,
            success_callback=self._on_plot_configured,
            cancel_callback=self._on_modal_cancelled,
        )

        # Add modal to container so it renders
        self._modal_container.clear()
        self._modal_container.append(self._current_modal.modal)
        self._current_modal.show()

    def _on_plot_configured(self, result: PlotConfigResult) -> None:
        """
        Handle successful plot configuration from modal.

        Creates PlotConfig with hardcoded plot_name and params, gets the
        pending selection from PlotGrid, creates PlotCell, and adds to
        orchestrator.

        Parameters
        ----------
        result
            Configuration result from the modal.
        """
        if self._modal_grid_id is None:
            return

        grid_id = self._modal_grid_id
        plot_grid = self._grid_widgets.get(grid_id)

        if plot_grid is None:
            return

        # Get pending selection from PlotGrid
        pending = plot_grid._pending_selection
        if pending is None:
            return

        row, col, row_span, col_span = pending

        # Create PlotConfig using selected plotter and configured parameters
        # Convert params to dict if it's a Pydantic model
        import pydantic

        params_dict = (
            result.params.model_dump()
            if isinstance(result.params, pydantic.BaseModel)
            else result.params
        )

        plot_config = PlotConfig(
            workflow_id=result.workflow_id,
            output_name=result.output_name,
            source_names=result.source_names,
            plot_name=result.plot_name,
            params=params_dict,
        )

        # Create PlotCell
        plot_cell = PlotCell(
            row=row,
            col=col,
            row_span=row_span,
            col_span=col_span,
            config=plot_config,
        )

        # Clear modal references before adding plot
        self._current_modal = None
        self._modal_grid_id = None

        # Clear pending selection in PlotGrid
        plot_grid.cancel_pending_selection()

        # Add to orchestrator (will trigger lifecycle callbacks for all sessions)
        self._orchestrator.add_plot(grid_id, plot_cell)

    def _on_modal_cancelled(self) -> None:
        """Handle modal cancellation."""
        if self._modal_grid_id is not None:
            plot_grid = self._grid_widgets.get(self._modal_grid_id)
            if plot_grid is not None:
                # Cancel pending selection in PlotGrid
                plot_grid.cancel_pending_selection()

        # Clear modal references
        self._current_modal = None
        self._modal_grid_id = None

    def _on_cell_updated(
        self, grid_id: GridId, cell: PlotCell, plot: Any, error: str | None
    ) -> None:
        """
        Handle cell update from orchestrator.

        Creates appropriate widget (placeholder, plot, or error) and inserts
        it into the grid at the specified position.

        Parameters
        ----------
        grid_id
            ID of the grid containing the cell.
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
            widget = self._create_plot_widget(grid_id, cell, plot)
        else:
            # Show status widget (either waiting for data or error)
            widget = self._create_status_widget(grid_id, cell, error=error)

        # Insert widget at explicit position
        plot_grid.insert_widget_at(
            cell.row, cell.col, cell.row_span, cell.col_span, widget
        )

    def _on_cell_removed(self, grid_id: GridId, cell: PlotCell) -> None:
        """
        Handle cell removal from orchestrator.

        Removes the widget from the grid at the specified position.

        Parameters
        ----------
        grid_id
            ID of the grid containing the cell.
        cell
            Plot cell that was removed.
        """
        plot_grid = self._grid_widgets.get(grid_id)
        if plot_grid is None:
            return

        # Remove widget at explicit position
        plot_grid.remove_widget_at(cell.row, cell.col, cell.row_span, cell.col_span)

    def _create_status_widget(
        self, grid_id: GridId, cell: PlotCell, error: str | None = None
    ) -> pn.Column:
        """
        Create a status widget for a cell without a plot.

        Shows either a placeholder (waiting for data) or an error message,
        depending on whether an error occurred during plot creation.

        Parameters
        ----------
        grid_id
            ID of the grid containing this cell.
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

        # Get workflow spec for display information
        workflow_spec = self._workflow_registry.get(config.workflow_id)
        workflow_title = (
            workflow_spec.title if workflow_spec else str(config.workflow_id)
        )

        # Get output title from spec if available
        output_title = config.output_name
        if workflow_spec and workflow_spec.outputs:
            output_fields = workflow_spec.outputs.model_fields
            if config.output_name in output_fields:
                field_info = output_fields[config.output_name]
                output_title = field_info.title or config.output_name

        # Build common info section
        info_lines = [
            f"**Workflow:** {workflow_title}",
            f"**Output:** {output_title}",
            f"**Sources:** {', '.join(config.source_names)}",
        ]

        # Add status-specific line and determine styling
        if error is not None:
            title = "### Plot Creation Error"
            info_lines.append(f"**Error:** {error}")
            text_color = '#dc3545'
            bg_color = '#ffe6e6'
            border = '2px solid #dc3545'
        else:
            title = "### Waiting for data..."
            text_color = '#6c757d'
            bg_color = '#f8f9fa'
            border = '2px dashed #dee2e6'

        content = f"{title}\n\n" + "\n\n".join(info_lines)

        # Create close button
        def on_close() -> None:
            # Look up cell_id from orchestrator state
            for cell_id, stored_cell in self._orchestrator._grids[
                grid_id
            ].cells.items():
                if stored_cell is cell:
                    self._orchestrator.remove_plot(cell_id)
                    return

        close_button = _create_close_button(on_close)

        status_widget = pn.Column(
            close_button,
            pn.pane.Markdown(
                content,
                styles={
                    'text-align': 'center',
                    'color': text_color,
                    'padding': '20px',
                },
            ),
            sizing_mode='stretch_both',
            styles={
                'background-color': bg_color,
                'border': border,
                'position': 'relative',
            },
        )
        return status_widget

    def _create_plot_widget(
        self, grid_id: GridId, cell: PlotCell, plot: hv.DynamicMap | hv.Layout
    ) -> pn.Column:
        """
        Create a widget containing the actual plot.

        Parameters
        ----------
        grid_id
            ID of the grid containing this cell.
        cell
            Plot cell configuration.
        plot
            HoloViews plot object.

        Returns
        -------
        :
            Panel widget containing the plot.
        """

        # Create close button
        def on_close() -> None:
            # Look up cell_id from orchestrator state
            for cell_id, stored_cell in self._orchestrator._grids[
                grid_id
            ].cells.items():
                if stored_cell is cell:
                    self._orchestrator.remove_plot(cell_id)
                    return

        close_button = _create_close_button(on_close)

        plot_pane = pn.pane.HoloViews(plot, sizing_mode='stretch_both')
        return pn.Column(
            close_button,
            plot_pane,
            sizing_mode='stretch_both',
            styles={'position': 'relative'},
        )

    def shutdown(self) -> None:
        """Unsubscribe from lifecycle events and shutdown manager."""
        if self._subscription_id is not None:
            self._orchestrator.unsubscribe_from_lifecycle(self._subscription_id)
            self._subscription_id = None
        self._grid_manager.shutdown()

    @property
    def panel(self) -> pn.Tabs:
        """Get the Panel viewable object for this widget."""
        return self._tabs
