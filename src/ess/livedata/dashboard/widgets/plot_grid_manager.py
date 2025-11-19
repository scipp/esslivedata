# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
PlotGridManager - Widget for managing plot grid configurations.

Provides UI for adding and removing plot grids through PlotOrchestrator.
"""

from __future__ import annotations

import panel as pn

from ..plot_orchestrator import GridId, PlotGridConfig, PlotOrchestrator, SubscriptionId


class PlotGridManager:
    """
    Widget for managing plot grid configurations.

    Provides a form to add new grids and a list of existing grids with
    remove buttons. Automatically updates when grids are added or removed
    through the orchestrator.

    Parameters
    ----------
    orchestrator
        The orchestrator managing plot grid configurations.
    """

    def __init__(self, orchestrator: PlotOrchestrator) -> None:
        self._orchestrator = orchestrator

        # Input fields for new grid
        self._title_input = pn.widgets.TextInput(
            name='Grid Title', value='New Grid', placeholder='Enter grid title'
        )
        self._nrows_input = pn.widgets.IntInput(name='Rows', value=3, start=2, end=6)
        self._ncols_input = pn.widgets.IntInput(name='Columns', value=3, start=2, end=6)

        # Add grid button
        self._add_button = pn.widgets.Button(name='Add Grid', button_type='primary')
        self._add_button.on_click(self._on_add_grid)

        # Grid list container
        self._grid_list = pn.Column(sizing_mode='stretch_width')

        # Subscribe to orchestrator updates
        self._subscription_id: SubscriptionId | None = (
            self._orchestrator.subscribe_to_lifecycle(
                on_grid_created=self._on_grid_created,
                on_grid_removed=self._on_grid_removed,
            )
        )

        # Initialize grid list
        self._update_grid_list()

        # Main widget layout
        self._widget = pn.Column(
            pn.pane.Markdown('## Add New Grid'),
            self._title_input,
            pn.Row(self._nrows_input, self._ncols_input),
            self._add_button,
            pn.layout.Divider(),
            pn.pane.Markdown('## Existing Grids'),
            self._grid_list,
            sizing_mode='stretch_width',
        )

    def _on_add_grid(self, event) -> None:
        """Handle add grid button click."""
        self._orchestrator.add_grid(
            title=self._title_input.value,
            nrows=self._nrows_input.value,
            ncols=self._ncols_input.value,
        )
        # Reset inputs
        self._title_input.value = 'New Grid'
        self._nrows_input.value = 3
        self._ncols_input.value = 3

    def _on_grid_created(self, grid_id: GridId, grid_config: PlotGridConfig) -> None:
        """Handle grid creation from orchestrator."""
        self._update_grid_list()

    def _on_grid_removed(self, grid_id: GridId) -> None:
        """Handle grid removal from orchestrator."""
        self._update_grid_list()

    def _update_grid_list(self) -> None:
        """Update the grid list display."""
        self._grid_list.clear()
        for grid_id, grid_config in self._orchestrator.get_all_grids().items():
            # Action buttons
            move_up_button = pn.widgets.Button(
                name='↑',
                button_type='default',
                width=40,
                disabled=True,
                description='Reordering support coming soon',
            )
            move_down_button = pn.widgets.Button(
                name='↓',
                button_type='default',
                width=40,
                disabled=True,
                description='Reordering support coming soon',
            )
            remove_button = pn.widgets.Button(
                name='Remove',
                button_type='danger',
                width=80,
            )

            # Capture grid_id in closure
            def make_remove_handler(gid: GridId):
                def handler(event):
                    self._orchestrator.remove_grid(gid)

                return handler

            remove_button.on_click(make_remove_handler(grid_id))

            grid_row = pn.Row(
                pn.pane.Str(
                    f'{grid_config.title} ({grid_config.nrows}x{grid_config.ncols})',
                    styles={'flex-grow': '1'},
                ),
                move_up_button,
                move_down_button,
                remove_button,
                sizing_mode='stretch_width',
            )
            self._grid_list.append(grid_row)

    def shutdown(self) -> None:
        """Unsubscribe from lifecycle events."""
        if self._subscription_id is not None:
            self._orchestrator.unsubscribe_from_lifecycle(self._subscription_id)
            self._subscription_id = None

    @property
    def panel(self) -> pn.Column:
        """Get the Panel viewable object for this widget."""
        return self._widget
