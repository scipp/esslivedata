# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import panel as pn

from ..notifications import show_error
from ..plot_orchestrator import CellGeometry


@dataclass(frozen=True)
class GridCellStyles:
    """Styling constants for PlotGrid cells."""

    # Colors
    PRIMARY_BLUE = '#007bff'
    LIGHT_GRAY = '#dee2e6'
    LIGHT_RED = '#ffe6e6'
    LIGHT_BLUE = '#e7f3ff'
    VERY_LIGHT_GRAY = '#f8f9fa'
    MEDIUM_GRAY = '#6c757d'
    MUTED_GRAY = '#adb5bd'

    # Dimensions
    CELL_MIN_HEIGHT_PX = 100
    CELL_BORDER_WIDTH_NORMAL = 1
    CELL_BORDER_WIDTH_HIGHLIGHTED = 3
    CELL_MARGIN = 2

    # Typography
    FONT_SIZE_LARGE = '24px'


@dataclass
class CellAppearance:
    """Computed appearance properties for a grid cell."""

    label: str
    disabled: bool
    styles: dict[str, str]
    stylesheets: list[str]


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


def _compute_cell_appearance(
    highlighted: bool = False,
    disabled: bool = False,
    label: str | None = None,
    large_font: bool = False,
) -> CellAppearance:
    """
    Compute the appearance properties for a cell based on its state.

    Parameters
    ----------
    highlighted:
        Whether the cell is highlighted (first click selection).
    disabled:
        Whether the cell is disabled (would create invalid region).
    label:
        Custom label text. If None, uses default based on state.
    large_font:
        Whether to use large font (during selection process).

    Returns
    -------
    :
        CellAppearance with all computed styling properties.
    """
    border_color = (
        GridCellStyles.PRIMARY_BLUE if highlighted else GridCellStyles.LIGHT_GRAY
    )
    border_width = (
        GridCellStyles.CELL_BORDER_WIDTH_HIGHLIGHTED
        if highlighted
        else GridCellStyles.CELL_BORDER_WIDTH_NORMAL
    )
    border_style = 'dashed' if highlighted else 'solid'

    if disabled:
        background_color = GridCellStyles.LIGHT_RED
        text_color = GridCellStyles.MUTED_GRAY
    elif highlighted:
        background_color = GridCellStyles.LIGHT_BLUE
        text_color = GridCellStyles.MEDIUM_GRAY
    else:
        background_color = GridCellStyles.VERY_LIGHT_GRAY
        text_color = GridCellStyles.MEDIUM_GRAY

    if label is None:
        label = '' if disabled else 'Click to add plot'

    if large_font:
        stylesheets = [
            f"""
            button {{
                font-size: {GridCellStyles.FONT_SIZE_LARGE};
                font-weight: bold;
            }}
            """
        ]
    else:
        stylesheets = []

    styles = {
        'background-color': background_color,
        'border': f'{border_width}px {border_style} {border_color}',
        'color': text_color,
        'min-height': f'{GridCellStyles.CELL_MIN_HEIGHT_PX}px',
    }

    return CellAppearance(
        label=label,
        disabled=disabled,
        styles=styles,
        stylesheets=stylesheets,
    )


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
        Callback invoked when a region is selected with the cell geometry.
        This callback will be called asynchronously and should not return a value.
    """

    def __init__(
        self,
        nrows: int,
        ncols: int,
        plot_request_callback: Callable[[CellGeometry], None],
    ) -> None:
        self._nrows = nrows
        self._ncols = ncols
        self._plot_request_callback = plot_request_callback

        # State tracking
        self._occupied_cells: dict[CellGeometry, pn.Column] = {}
        self._first_click: tuple[int, int] | None = None
        self._highlighted_cell: pn.pane.HTML | None = None

        # Button references for in-place updates (row, col) -> Button
        self._cell_buttons: dict[tuple[int, int], pn.widgets.Button] = {}

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
        appearance = _compute_cell_appearance(
            highlighted=highlighted,
            disabled=disabled,
            label=label,
            large_font=large_font,
        )

        # Create a button that fills the cell
        button = pn.widgets.Button(
            name=appearance.label,
            sizing_mode='stretch_both',
            button_type='light',
            disabled=appearance.disabled,
            styles=appearance.styles,
            stylesheets=appearance.stylesheets,
            margin=GridCellStyles.CELL_MARGIN,
        )

        # Store reference for in-place updates
        self._cell_buttons[(row, col)] = button

        # Attach click handler - checks disabled state dynamically
        def on_click(_: Any) -> None:
            if not button.disabled:
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

            # Request plot from callback, passing cell geometry
            geometry = CellGeometry(
                row=row_start, col=col_start, row_span=row_span, col_span=col_span
            )
            self._plot_request_callback(geometry)

    def _is_cell_occupied(self, row: int, col: int) -> bool:
        """Check if a specific cell is occupied by a plot."""
        for geometry in self._occupied_cells:
            if (
                geometry.row <= row < geometry.row + geometry.row_span
                and geometry.col <= col < geometry.col + geometry.col_span
            ):
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
                        self._update_cell_in_place(row, col)

    def _get_cell_appearance_for_state(self, row: int, col: int) -> CellAppearance:
        """Compute the appearance for a cell based on current selection state."""
        if self._first_click is None:
            # No selection in progress
            return _compute_cell_appearance()

        r1, c1 = self._first_click

        if row == r1 and col == c1:
            # This is the first clicked cell - highlight it
            return _compute_cell_appearance(
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
            return _compute_cell_appearance(disabled=True, large_font=True)

        # Calculate dimensions and format label
        row_span, col_span = _calculate_region_span(
            row_start, row_end, col_start, col_end
        )
        label = _format_region_label(row_span, col_span)

        return _compute_cell_appearance(label=label, large_font=True)

    def _update_cell_in_place(self, row: int, col: int) -> None:
        """Update an existing cell's appearance without recreating it."""
        button = self._cell_buttons.get((row, col))
        if button is None:
            return

        appearance = self._get_cell_appearance_for_state(row, col)

        # Update button properties in-place
        button.name = appearance.label
        button.disabled = appearance.disabled
        button.styles = appearance.styles
        button.stylesheets = appearance.stylesheets

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
                # Clean up button reference
                self._cell_buttons.pop((r, c), None)
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

    def insert_widget_at(self, geometry: CellGeometry, widget: Any) -> None:
        """
        Insert a widget at an explicit position (for orchestrator-driven updates).

        This method is used for multi-user synchronized plot management via the
        orchestrator pattern and can be called from lifecycle callbacks.

        Parameters
        ----------
        geometry:
            Cell geometry specifying position and span.
        widget:
            Panel widget or viewable to insert.
        """
        # Validate position is within grid bounds
        if (
            geometry.row < 0
            or geometry.row >= self._nrows
            or geometry.col < 0
            or geometry.col >= self._ncols
        ):
            show_error(f'Invalid position: ({geometry.row}, {geometry.col})')
            return

        if (
            geometry.row + geometry.row_span > self._nrows
            or geometry.col + geometry.col_span > self._ncols
        ):
            show_error(
                f'Widget extends beyond grid: ({geometry.row}, {geometry.col}) + '
                f'span ({geometry.row_span}, {geometry.col_span})'
            )
            return

        # Remove any existing widget at this exact position
        if geometry in self._occupied_cells:
            del self._occupied_cells[geometry]

        self._insert_widget_into_grid(
            geometry.row, geometry.col, geometry.row_span, geometry.col_span, widget
        )

        # Track occupation
        self._occupied_cells[geometry] = widget

    def remove_widget_at(self, geometry: CellGeometry) -> None:
        """
        Remove a widget at an explicit position (for orchestrator-driven updates).

        This method is used for multi-user synchronized plot management via the
        orchestrator pattern.

        Parameters
        ----------
        geometry:
            Cell geometry specifying position and span.
        """
        # Remove from tracking
        if geometry in self._occupied_cells:
            del self._occupied_cells[geometry]

        # Restore empty cells
        self._restore_empty_cells_in_region(
            geometry.row, geometry.col, geometry.row_span, geometry.col_span
        )

    @property
    def panel(self) -> pn.viewable.Viewable:
        """Get the Panel viewable object for this widget."""
        return self._grid
