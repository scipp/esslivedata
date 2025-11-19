# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
    TOOL_BUTTON_SIZE = 28
    TOOL_BUTTON_TOP_OFFSET = '5px'
    CLOSE_BUTTON_RIGHT_OFFSET = '5px'
    TOOL_BUTTON_Z_INDEX = '1000'

    # Typography
    FONT_SIZE_LARGE = '24px'
    TOOL_BUTTON_FONT_SIZE = '20px'


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


def _create_tool_button_stylesheet(button_color: str, hover_color: str) -> list[str]:
    """
    Create a stylesheet for tool buttons (close, gear, etc.).

    Parameters
    ----------
    button_color:
        Color for the button icon.
    hover_color:
        RGBA color for the hover background.

    Returns
    -------
    :
        List containing the stylesheet string.
    """
    return [
        f"""
        button {{
            background-color: transparent !important;
            border: none !important;
            color: {button_color} !important;
            font-weight: bold !important;
            font-size: {_CellStyles.TOOL_BUTTON_FONT_SIZE} !important;
            padding: 0 !important;
            margin: 0 !important;
            line-height: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            height: 100% !important;
            width: 100% !important;
        }}
        button:hover {{
            background-color: {hover_color} !important;
        }}
        """
    ]


def _create_close_button(on_close_callback: Callable[[], None]) -> pn.widgets.Button:
    """
    Create a styled close button for plot cells.

    Parameters
    ----------
    on_close_callback:
        Callback function to invoke when the button is clicked.

    Returns
    -------
    :
        Panel Button widget styled as a close button.
    """
    close_button = pn.widgets.Button(
        name='\u00d7',  # "X" multiplication sign
        width=_CellStyles.TOOL_BUTTON_SIZE,
        height=_CellStyles.TOOL_BUTTON_SIZE,
        button_type='light',
        sizing_mode='fixed',
        margin=(_CellStyles.CELL_MARGIN, _CellStyles.CELL_MARGIN),
        styles={
            'position': 'absolute',
            'top': _CellStyles.TOOL_BUTTON_TOP_OFFSET,
            'right': _CellStyles.CLOSE_BUTTON_RIGHT_OFFSET,
            'z-index': _CellStyles.TOOL_BUTTON_Z_INDEX,
        },
        stylesheets=_create_tool_button_stylesheet(
            _CellStyles.DANGER_RED, 'rgba(220, 53, 69, 0.1)'
        ),
    )
    close_button.on_click(lambda _: on_close_callback())
    return close_button


def _create_gear_button(on_gear_callback: Callable[[], None]) -> pn.widgets.Button:
    """
    Create a styled gear button for plot cells (configuration/settings).

    Parameters
    ----------
    on_gear_callback:
        Callback function to invoke when the button is clicked.

    Returns
    -------
    :
        Panel Button widget styled as a gear button.
    """
    gear_button = pn.widgets.Button(
        name='\u2699',  # Gear symbol
        width=_CellStyles.TOOL_BUTTON_SIZE,
        height=_CellStyles.TOOL_BUTTON_SIZE,
        button_type='light',
        sizing_mode='fixed',
        margin=(_CellStyles.CELL_MARGIN, _CellStyles.CELL_MARGIN),
        styles={
            'position': 'absolute',
            'top': _CellStyles.TOOL_BUTTON_TOP_OFFSET,
            # Position to the left of the close button (which is at right offset)
            'right': f'{_CellStyles.TOOL_BUTTON_SIZE + 10}px',
            'z-index': _CellStyles.TOOL_BUTTON_Z_INDEX,
        },
        stylesheets=_create_tool_button_stylesheet(
            _CellStyles.PRIMARY_BLUE, 'rgba(0, 123, 255, 0.1)'
        ),
    )
    gear_button.on_click(lambda _: on_gear_callback())
    return gear_button


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
        Callback invoked when a region is selected with the region coordinates
        (row, col, row_span, col_span). This callback will be called
        asynchronously and should not return a value.
    """

    def __init__(
        self,
        nrows: int,
        ncols: int,
        plot_request_callback: Callable[[int, int, int, int], None],
    ) -> None:
        self._nrows = nrows
        self._ncols = ncols
        self._plot_request_callback = plot_request_callback

        # State tracking
        self._occupied_cells: dict[tuple[int, int, int, int], pn.Column] = {}
        self._first_click: tuple[int, int] | None = None
        self._highlighted_cell: pn.pane.HTML | None = None

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

            # Clear selection highlight
            self._clear_selection()

            # Request plot from callback, passing region coordinates
            self._plot_request_callback(row_start, col_start, row_span, col_span)

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
                        except (KeyError, IndexError, TypeError):
                            # Cell might not exist yet or grid state issue
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

    def _delete_cells_in_region(
        self, row: int, col: int, row_span: int, col_span: int
    ) -> None:
        """Delete all cells in the specified region from the grid."""
        for r in range(row, row + row_span):
            for c in range(col, col + col_span):
                try:
                    del self._grid[r, c]
                except (KeyError, IndexError, TypeError):
                    # Cell might not exist (spanned widgets) or grid state issue
                    pass

    def _insert_widget_into_grid(
        self, row: int, col: int, row_span: int, col_span: int, widget: Any
    ) -> None:
        """Insert a widget into the grid at the specified region."""
        with pn.io.hold():
            self._delete_cells_in_region(row, col, row_span, col_span)
            self._grid[row : row + row_span, col : col + col_span] = widget

    def _restore_empty_cells_in_region(
        self, row: int, col: int, row_span: int, col_span: int
    ) -> None:
        """Restore empty cells in the specified region."""
        with pn.io.hold():
            self._delete_cells_in_region(row, col, row_span, col_span)
            for r in range(row, row + row_span):
                for c in range(col, col + col_span):
                    self._grid[r, c] = self._create_empty_cell(r, c)

    def _show_error(self, message: str) -> None:
        """Display a temporary error notification."""
        if pn.state.notifications is not None:
            pn.state.notifications.error(message, duration=3000)

    def insert_widget_at(
        self, row: int, col: int, row_span: int, col_span: int, widget: Any
    ) -> None:
        """
        Insert a widget at an explicit position (for orchestrator-driven updates).

        This method is used for multi-user synchronized plot management via the
        orchestrator pattern. Unlike insert_plot_deferred(), it doesn't rely on
        pending selection state and can be called from lifecycle callbacks.

        Parameters
        ----------
        row:
            Starting row position (0-indexed).
        col:
            Starting column position (0-indexed).
        row_span:
            Number of rows the widget should span.
        col_span:
            Number of columns the widget should span.
        widget:
            Panel widget or viewable to insert.
        """
        # Validate position is within grid bounds
        if row < 0 or row >= self._nrows or col < 0 or col >= self._ncols:
            self._show_error(f'Invalid position: ({row}, {col})')
            return

        if row + row_span > self._nrows or col + col_span > self._ncols:
            self._show_error(
                f'Widget extends beyond grid: ({row}, {col}) + '
                f'span ({row_span}, {col_span})'
            )
            return

        # Remove any existing widget at this exact position
        key = (row, col, row_span, col_span)
        if key in self._occupied_cells:
            del self._occupied_cells[key]

        self._insert_widget_into_grid(row, col, row_span, col_span, widget)

        # Track occupation
        self._occupied_cells[key] = widget

    def remove_widget_at(
        self, row: int, col: int, row_span: int, col_span: int
    ) -> None:
        """
        Remove a widget at an explicit position (for orchestrator-driven updates).

        This method is used for multi-user synchronized plot management via the
        orchestrator pattern.

        Parameters
        ----------
        row:
            Starting row position (0-indexed).
        col:
            Starting column position (0-indexed).
        row_span:
            Number of rows the widget spans.
        col_span:
            Number of columns the widget spans.
        """
        # Remove from tracking
        key = (row, col, row_span, col_span)
        if key in self._occupied_cells:
            del self._occupied_cells[key]

        # Restore empty cells
        self._restore_empty_cells_in_region(row, col, row_span, col_span)

    @property
    def panel(self) -> pn.viewable.Viewable:
        """Get the Panel viewable object for this widget."""
        return self._grid
