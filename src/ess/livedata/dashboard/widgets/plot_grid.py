# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import holoviews as hv
import panel as pn


@dataclass(frozen=True)
class _CellStyles:
    """Styling constants for PlotGrid cells."""

    # Colors
    PRIMARY_BLUE = '#007bff'
    LIGHT_GRAY = '#dee2e6'
    LIGHT_RED = '#ffe6e6'
    LIGHT_BLUE = '#e7f3ff'
    VERY_LIGHT_GRAY = '#f8f9fa'
    MEDIUM_GRAY = '#6c757d'
    MUTED_GRAY = '#adb5bd'
    DANGER_RED = '#dc3545'

    # Dimensions
    CELL_MIN_HEIGHT_PX = 100
    CELL_BORDER_WIDTH_NORMAL = 1
    CELL_BORDER_WIDTH_HIGHLIGHTED = 3
    CELL_MARGIN = 2
    CLOSE_BUTTON_SIZE = 40
    CLOSE_BUTTON_TOP_OFFSET = '5px'
    CLOSE_BUTTON_RIGHT_OFFSET = '5px'
    CLOSE_BUTTON_Z_INDEX = '1000'

    # Typography
    FONT_SIZE_LARGE = '24px'
    FONT_SIZE_CLOSE_BUTTON = '20px'


def _normalize_region(r1: int, c1: int, r2: int, c2: int) -> tuple[int, int, int, int]:
    """
    Normalize region coordinates to (row_start, col_start, row_end, col_end).

    Parameters
    ----------
    r1:
        First row coordinate.
    c1:
        First column coordinate.
    r2:
        Second row coordinate.
    c2:
        Second column coordinate.

    Returns
    -------
    :
        Tuple of (row_start, col_start, row_end, col_end) where
        row_start <= row_end and col_start <= col_end.
    """
    return min(r1, r2), min(c1, c2), max(r1, r2), max(c1, c2)


def _calculate_region_span(
    row_start: int, row_end: int, col_start: int, col_end: int
) -> tuple[int, int]:
    """
    Calculate the span dimensions of a region.

    Parameters
    ----------
    row_start:
        Starting row (inclusive).
    row_end:
        Ending row (inclusive).
    col_start:
        Starting column (inclusive).
    col_end:
        Ending column (inclusive).

    Returns
    -------
    :
        Tuple of (row_span, col_span).
    """
    return row_end - row_start + 1, col_end - col_start + 1


def _format_region_label(row_span: int, col_span: int) -> str:
    """
    Format a label describing region dimensions.

    Parameters
    ----------
    row_span:
        Number of rows in the region.
    col_span:
        Number of columns in the region.

    Returns
    -------
    :
        Formatted label string like "Click for 2x3 plot".
    """
    return f'Click for {row_span}x{col_span} plot'


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
        Callback invoked when a region is selected. This callback will be
        called asynchronously and should not return a value. The plot should
        be inserted later via `insert_plot_deferred()`.
    """

    def __init__(
        self,
        nrows: int,
        ncols: int,
        plot_request_callback: Callable[[], None],
    ) -> None:
        self._nrows = nrows
        self._ncols = ncols
        self._plot_request_callback = plot_request_callback

        # State tracking
        self._occupied_cells: dict[tuple[int, int, int, int], pn.Column] = {}
        self._first_click: tuple[int, int] | None = None
        self._highlighted_cell: pn.pane.HTML | None = None
        self._pending_selection: tuple[int, int, int, int] | None = None

        # Create the grid
        self._grid = pn.GridSpec(
            sizing_mode='stretch_both', name='PlotGrid', min_height=600
        )

        # Initialize empty cells
        self._initialize_empty_cells()

    def _initialize_empty_cells(self) -> None:
        """Populate the grid with empty clickable cells."""
        with pn.io.hold():
            for row in range(self._nrows):
                for col in range(self._ncols):
                    self._grid[row, col] = self._create_empty_cell(row, col)

    def _create_empty_cell(
        self,
        row: int,
        col: int,
        highlighted: bool = False,
        disabled: bool = False,
        label: str | None = None,
        large_font: bool = False,
    ) -> pn.Column:
        """Create an empty cell with placeholder text and click handler."""
        border_color = (
            _CellStyles.PRIMARY_BLUE if highlighted else _CellStyles.LIGHT_GRAY
        )
        border_width = (
            _CellStyles.CELL_BORDER_WIDTH_HIGHLIGHTED
            if highlighted
            else _CellStyles.CELL_BORDER_WIDTH_NORMAL
        )
        border_style = 'dashed' if highlighted else 'solid'

        if disabled:
            background_color = _CellStyles.LIGHT_RED
            text_color = _CellStyles.MUTED_GRAY
        elif highlighted:
            background_color = _CellStyles.LIGHT_BLUE
            text_color = _CellStyles.MEDIUM_GRAY
        else:
            background_color = _CellStyles.VERY_LIGHT_GRAY
            text_color = _CellStyles.MEDIUM_GRAY

        # Determine button label
        if label is None:
            label = '' if disabled else 'Click to add plot'

        # Font size - larger during selection process
        # Use stylesheets to target the button element directly
        if large_font:
            stylesheets = [
                f"""
                button {{
                    font-size: {_CellStyles.FONT_SIZE_LARGE};
                    font-weight: bold;
                }}
                """
            ]
        else:
            stylesheets = []

        # Create a button that fills the cell
        button = pn.widgets.Button(
            name=label,
            sizing_mode='stretch_both',
            button_type='light',
            disabled=disabled,
            styles={
                'background-color': background_color,
                'border': f'{border_width}px {border_style} {border_color}',
                'color': text_color,
                'min-height': f'{_CellStyles.CELL_MIN_HEIGHT_PX}px',
            },
            stylesheets=stylesheets,
            margin=_CellStyles.CELL_MARGIN,
        )

        # Attach click handler (even if disabled, for consistency)
        def on_click(event: Any) -> None:
            if not disabled:
                self._on_cell_click(row, col)

        button.on_click(on_click)

        # Wrap in Column to allow for future expansion
        return pn.Column(button, sizing_mode='stretch_both', margin=0)

    def _on_cell_click(self, row: int, col: int) -> None:
        """Handle cell click for region selection."""
        if self._first_click is None:
            # First click - start selection
            self._first_click = (row, col)
            self._refresh_all_cells()
        else:
            # Second click - complete selection
            r1, c1 = self._first_click
            r2, c2 = row, col

            # Normalize to get top-left and bottom-right corners
            row_start, col_start, row_end, col_end = _normalize_region(r1, c1, r2, c2)

            # Calculate span
            row_span, col_span = _calculate_region_span(
                row_start, row_end, col_start, col_end
            )

            # Store selection for plot insertion
            self._pending_selection = (row_start, col_start, row_span, col_span)

            # Clear selection highlight
            self._clear_selection()

            # Request plot from callback (async, no return value)
            self._plot_request_callback()

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

    def _refresh_all_cells(self) -> None:
        """Refresh all empty cells based on current selection state."""
        with pn.io.hold():
            for row in range(self._nrows):
                for col in range(self._ncols):
                    if not self._is_cell_occupied(row, col):
                        # Delete the old cell first to avoid overlap warnings
                        try:
                            del self._grid[row, col]
                        except (KeyError, IndexError):
                            # Cell might not exist yet (during initialization)
                            pass
                        self._grid[row, col] = self._get_cell_for_state(row, col)

    def _get_cell_for_state(self, row: int, col: int) -> pn.Column:
        """Get the appropriate cell widget based on current selection state."""
        if self._first_click is None:
            # No selection in progress
            return self._create_empty_cell(row, col)

        r1, c1 = self._first_click

        if row == r1 and col == c1:
            # This is the first clicked cell - highlight it
            return self._create_empty_cell(
                row,
                col,
                highlighted=True,
                label='Click again for 1x1 plot',
                large_font=True,
            )

        # Check if this cell would create a valid region
        row_start, col_start, row_end, col_end = _normalize_region(r1, c1, row, col)

        # Check if region is valid
        is_valid = self._is_region_available(row_start, col_start, row_end, col_end)

        if not is_valid:
            # Disable this cell
            return self._create_empty_cell(row, col, disabled=True, large_font=True)

        # Calculate dimensions and format label
        row_span, col_span = _calculate_region_span(
            row_start, row_end, col_start, col_end
        )
        label = _format_region_label(row_span, col_span)

        return self._create_empty_cell(row, col, label=label, large_font=True)

    def _clear_selection(self) -> None:
        """Clear the current selection state."""
        self._first_click = None
        self._highlighted_cell = None
        self._refresh_all_cells()

    def _insert_plot(self, plot: hv.DynamicMap) -> None:
        """Insert a plot into the grid at the pending selection."""
        if self._pending_selection is None:
            return

        row, col, row_span, col_span = self._pending_selection

        # Create plot pane using the .layout pattern for DynamicMaps
        plot_pane_wrapper = pn.pane.HoloViews(plot, sizing_mode='stretch_both')
        plot_pane = plot_pane_wrapper.layout

        # Create close button with stylesheets for proper styling override
        close_button = pn.widgets.Button(
            name='\u00d7',  # "X" multiplication sign
            width=_CellStyles.CLOSE_BUTTON_SIZE,
            height=_CellStyles.CLOSE_BUTTON_SIZE,
            button_type='light',
            sizing_mode='fixed',
            margin=(_CellStyles.CELL_MARGIN, _CellStyles.CELL_MARGIN),
            styles={
                'position': 'absolute',
                'top': _CellStyles.CLOSE_BUTTON_TOP_OFFSET,
                'right': _CellStyles.CLOSE_BUTTON_RIGHT_OFFSET,
                'z-index': _CellStyles.CLOSE_BUTTON_Z_INDEX,
            },
            stylesheets=[
                f"""
                button {{
                    background-color: transparent !important;
                    border: none !important;
                    color: {_CellStyles.DANGER_RED} !important;
                    font-weight: bold !important;
                    font-size: {_CellStyles.FONT_SIZE_CLOSE_BUTTON} !important;
                    padding: 0 !important;
                }}
                button:hover {{
                    background-color: rgba(220, 53, 69, 0.1) !important;
                }}
                """
            ],
        )

        def on_close(event: Any) -> None:
            self._remove_plot(row, col, row_span, col_span)

        close_button.on_click(on_close)

        container = pn.Column(
            close_button,
            plot_pane,
            sizing_mode='stretch_both',
            margin=2,
            styles={'position': 'relative'},
        )

        with pn.io.hold():
            # Delete existing cells in the region to avoid overlap warnings
            for r in range(row, row + row_span):
                for c in range(col, col + col_span):
                    try:
                        del self._grid[r, c]
                    except (KeyError, IndexError):
                        pass

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
        with pn.io.hold():
            for r in range(row, row + row_span):
                for c in range(col, col + col_span):
                    self._grid[r, c] = self._create_empty_cell(r, c)

    def _show_error(self, message: str) -> None:
        """Display a temporary error notification."""
        if pn.state.notifications is not None:
            pn.state.notifications.error(message, duration=3000)

    def insert_plot_deferred(self, plot: hv.DynamicMap) -> None:
        """
        Complete plot insertion after async workflow.

        This method should be called after the plot request callback completes
        successfully. It inserts the plot at the pending selection location
        and clears the in-flight state.

        Parameters
        ----------
        plot:
            The HoloViews DynamicMap to insert into the grid.
        """
        if self._pending_selection is None:
            self._show_error('No pending selection to insert plot into')
            return

        self._insert_plot(plot)

    def cancel_pending_selection(self) -> None:
        """
        Abort the current plot creation workflow and reset state.

        This method should be called when the plot request callback is cancelled
        or fails. It clears the pending selection.
        """
        self._pending_selection = None

    @property
    def panel(self) -> pn.viewable.Viewable:
        """Get the Panel viewable object for this widget."""
        return self._grid
