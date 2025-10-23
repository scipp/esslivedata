# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import holoviews as hv
import panel as pn


class PlotGrid:
    """
    A grid widget for displaying multiple plots in a customizable layout.

    The PlotGrid allows users to select rectangular regions by clicking cells
    and insert HoloViews plots into those regions. Each plot can be removed
    via a close button.

    Parameters
    ----------
    nrows:
        Number of rows in the grid.
    ncols:
        Number of columns in the grid.
    plot_request_callback:
        Callback invoked when a region is selected. Should return a
        HoloViews DynamicMap to insert into the grid.
    """

    def __init__(
        self,
        nrows: int,
        ncols: int,
        plot_request_callback: Callable[[], hv.DynamicMap],
    ) -> None:
        self._nrows = nrows
        self._ncols = ncols
        self._plot_request_callback = plot_request_callback

        # State tracking
        self._occupied_cells: dict[tuple[int, int, int, int], pn.Column] = {}
        self._first_click: tuple[int, int] | None = None
        self._highlighted_cell: pn.pane.HTML | None = None

        # Create the grid
        self._grid = pn.GridSpec(sizing_mode='stretch_both', name='PlotGrid')

        # Initialize empty cells
        self._initialize_empty_cells()

        # Setup keyboard event handling for ESC key
        self._setup_keyboard_handler()

    def _initialize_empty_cells(self) -> None:
        """Populate the grid with empty clickable cells."""
        for row in range(self._nrows):
            for col in range(self._ncols):
                self._grid[row, col] = self._create_empty_cell(row, col)

    def _create_empty_cell(
        self, row: int, col: int, highlighted: bool = False
    ) -> pn.Column:
        """Create an empty cell with placeholder text and click handler."""
        border_color = '#007bff' if highlighted else '#dee2e6'
        border_width = 3 if highlighted else 1
        border_style = 'dashed' if highlighted else 'solid'
        background_color = '#e7f3ff' if highlighted else '#f8f9fa'

        # Create a button that fills the cell
        button = pn.widgets.Button(
            name='Click to add plot',
            sizing_mode='stretch_both',
            button_type='light',
            styles={
                'background-color': background_color,
                'border': f'{border_width}px {border_style} {border_color}',
                'color': '#6c757d',
                'font-size': '14px',
                'min-height': '100px',
            },
            margin=2,
        )

        # Attach click handler
        def on_click(event: Any) -> None:
            self._on_cell_click(row, col)

        button.on_click(on_click)

        # Wrap in Column to allow for future expansion
        return pn.Column(button, sizing_mode='stretch_both', margin=0)

    def _on_cell_click(self, row: int, col: int) -> None:
        """Handle cell click for region selection."""
        # Check if cell is occupied
        if self._is_cell_occupied(row, col):
            self._show_error('Cannot select a cell that already contains a plot')
            return

        if self._first_click is None:
            # First click - start selection
            self._first_click = (row, col)
            self._highlight_cell(row, col)
        else:
            # Second click - complete selection
            r1, c1 = self._first_click
            r2, c2 = row, col

            # Normalize to get top-left and bottom-right corners
            row_start = min(r1, r2)
            row_end = max(r1, r2)
            col_start = min(c1, c2)
            col_end = max(c1, c2)

            # Check if the entire region is available
            if not self._is_region_available(row_start, col_start, row_end, col_end):
                self._show_error(
                    'Cannot select a region that overlaps with existing plots'
                )
                self._clear_selection()
                return

            # Calculate span
            row_span = row_end - row_start + 1
            col_span = col_end - col_start + 1

            # Store selection for plot insertion
            self._pending_selection = (row_start, col_start, row_span, col_span)

            # Clear selection highlight
            self._clear_selection()

            # Request plot from callback
            try:
                plot = self._plot_request_callback()
                self._insert_plot(plot)
            except Exception as e:
                self._show_error(f'Error creating plot: {e}')
                self._pending_selection = None

    def _is_cell_occupied(self, row: int, col: int) -> bool:
        """Check if a specific cell is occupied by a plot."""
        for r, c, r_span, c_span in self._occupied_cells:
            if r <= row < r + r_span and c <= col < c + c_span:
                return True
        return False

    def _is_region_available(
        self, row_start: int, col_start: int, row_end: int, col_end: int
    ) -> bool:
        """Check if an entire region is available for plot insertion."""
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                if self._is_cell_occupied(row, col):
                    return False
        return True

    def _highlight_cell(self, row: int, col: int) -> None:
        """Highlight a cell to indicate selection in progress."""
        # Replace the cell with a highlighted version
        self._highlighted_cell = (row, col)
        self._grid[row, col] = self._create_empty_cell(row, col, highlighted=True)

    def _clear_selection(self) -> None:
        """Clear the current selection state."""
        if self._first_click is not None and self._highlighted_cell is not None:
            row, col = self._first_click
            if not self._is_cell_occupied(row, col):
                self._grid[row, col] = self._create_empty_cell(row, col)

        self._first_click = None
        self._highlighted_cell = None

    def _insert_plot(self, plot: hv.DynamicMap) -> None:
        """Insert a plot into the grid at the pending selection."""
        if not hasattr(self, '_pending_selection') or self._pending_selection is None:
            return

        row, col, row_span, col_span = self._pending_selection

        # Create plot pane using the .layout pattern for DynamicMaps
        plot_pane_wrapper = pn.pane.HoloViews(plot, sizing_mode='stretch_both')
        plot_pane = plot_pane_wrapper.layout

        # Create close button
        close_button = pn.widgets.Button(
            name='\u00d7',  # multiplication sign
            width=30,
            height=30,
            button_type='danger',
            sizing_mode='fixed',
            margin=(5, 5),
        )

        def on_close(event: Any) -> None:
            self._remove_plot(row, col, row_span, col_span)

        close_button.on_click(on_close)

        # Create container with close button positioned at top-right
        container = pn.Column(
            pn.Row(
                pn.Spacer(),
                close_button,
                sizing_mode='stretch_width',
                margin=(0, 0, 0, 0),
            ),
            plot_pane,
            sizing_mode='stretch_both',
            margin=2,
        )

        # Insert into grid
        self._grid[row : row + row_span, col : col + col_span] = container

        # Track occupation
        self._occupied_cells[(row, col, row_span, col_span)] = container

        # Clear pending selection
        self._pending_selection = None

    def _remove_plot(self, row: int, col: int, row_span: int, col_span: int) -> None:
        """Remove a plot from the grid and restore empty cells."""
        # Remove from tracking
        key = (row, col, row_span, col_span)
        if key in self._occupied_cells:
            del self._occupied_cells[key]

        # Restore empty cells
        for r in range(row, row + row_span):
            for c in range(col, col + col_span):
                self._grid[r, c] = self._create_empty_cell(r, c)

    def _show_error(self, message: str) -> None:
        """Display a temporary error notification."""
        if pn.state.notifications is not None:
            pn.state.notifications.error(message, duration=3000)

    def _setup_keyboard_handler(self) -> None:
        """Setup keyboard event handler for ESC key."""
        # Panel doesn't have built-in ESC key handling for custom widgets
        # This would require JavaScript integration which is complex
        # For now, we'll document that clicking outside the grid cancels selection
        # A future enhancement could add proper keyboard support
        pass

    @property
    def panel(self) -> pn.viewable.Viewable:
        """Get the Panel viewable object for this widget."""
        return self._grid
