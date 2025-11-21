# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
PlotGridTabs - Tabbed interface for managing multiple plot grids.

Provides a Panel Tabs widget that displays multiple PlotGrid instances,
synchronized with PlotOrchestrator via lifecycle subscriptions.
"""

from __future__ import annotations

import panel as pn

from ..plot_orchestrator import GridId, PlotGridConfig, PlotOrchestrator, SubscriptionId
from .plot_grid import PlotGrid
from .plot_grid_manager import PlotGridManager


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
    """

    def __init__(self, plot_orchestrator: PlotOrchestrator) -> None:
        self._orchestrator = plot_orchestrator

        # Track grid widgets and tab indices
        self._grid_widgets: dict[GridId, PlotGrid] = {}
        self._grid_to_tab_index: dict[GridId, int] = {}

        # Main tabs widget
        self._tabs = pn.Tabs(sizing_mode='stretch_both')

        # Subscribe to lifecycle events
        self._subscription_id: SubscriptionId | None = (
            self._orchestrator.subscribe_to_lifecycle(
                on_grid_created=self._on_grid_created,
                on_grid_removed=self._on_grid_removed,
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
        # Create PlotGrid widget
        plot_grid = PlotGrid(
            nrows=grid_config.nrows,
            ncols=grid_config.ncols,
            plot_request_callback=self._on_plot_requested,
        )

        # Store widget reference
        self._grid_widgets[grid_id] = plot_grid

        # Append at the end (Manage tab is always first at index 0)
        tab_index = len(self._tabs)
        self._tabs.append((grid_config.title, plot_grid.panel))
        self._grid_to_tab_index[grid_id] = tab_index

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

    def _on_plot_requested(self) -> None:
        """Handle plot request from PlotGrid (not yet implemented)."""
        if pn.state.notifications is not None:
            pn.state.notifications.info(
                'Plot management not yet implemented', duration=3000
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
