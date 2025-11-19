# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import holoviews as hv
import numpy as np
import panel as pn
import pytest

from ess.livedata.dashboard.widgets.plot_grid import PlotGrid


class FakeCallback:
    """Fake callback for testing plot requests."""

    def __init__(self, side_effect: Exception | None = None) -> None:
        self.call_count = 0
        self.calls: list = []
        self._side_effect = side_effect

    def __call__(self, *args, **kwargs) -> None:
        self.call_count += 1
        self.calls.append((args, kwargs))
        if self._side_effect is not None:
            raise self._side_effect

    def reset(self) -> None:
        self.call_count = 0
        self.calls.clear()

    def assert_called_once(self) -> None:
        assert self.call_count == 1, f"Expected 1 call, got {self.call_count}"

    def assert_not_called(self) -> None:
        assert self.call_count == 0, f"Expected 0 calls, got {self.call_count}"


@pytest.fixture
def mock_plot() -> hv.DynamicMap:
    """Create a mock HoloViews DynamicMap for testing."""

    def create_curve(x_range):
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.sin(x)
        return hv.Curve((x, y))

    return hv.DynamicMap(create_curve, kdims=['x_range']).redim.range(x_range=(0, 10))


@pytest.fixture
def mock_callback() -> FakeCallback:
    """Create a fake callback for plot requests."""
    return FakeCallback()


def get_cell_button(grid: PlotGrid, row: int, col: int) -> pn.widgets.Button | None:
    """
    Get the empty cell button widget from a grid cell.

    Returns None if cell is not a simple empty cell (e.g., contains a plot).
    """
    try:
        cell = grid.panel[row, col]  # type: ignore[index]
        if isinstance(cell, pn.Column) and len(cell) > 0:
            first_item = cell[0]
            if isinstance(first_item, pn.widgets.Button):
                # Check if this is the close button (multiplication sign character)
                if first_item.name == '\u00d7':
                    # This is a plot cell with a close button
                    return None
                # This is an empty cell button
                return first_item
    except (KeyError, IndexError):
        pass
    return None


def simulate_click(grid: PlotGrid, row: int, col: int) -> None:
    """Simulate a user clicking on a grid cell by triggering button's click event.

    This simulates a standard left-click interaction with the button.
    """
    button = get_cell_button(grid, row, col)
    if button is None:
        msg = f"Cannot click cell ({row}, {col}): no clickable button found"
        raise ValueError(msg)
    if button.disabled:  # type: ignore[truthy-bool]
        msg = f"Cannot click cell ({row}, {col}): button is disabled"
        raise ValueError(msg)
    # Trigger the click event by incrementing clicks parameter
    button.param.trigger('clicks')


def is_cell_occupied(grid: PlotGrid, row: int, col: int) -> bool:
    """
    Check if a cell contains a plot (observable behavior).

    A cell is considered occupied if it doesn't have a simple button widget.
    """
    return get_cell_button(grid, row, col) is None


def find_close_button(grid: PlotGrid, row: int, col: int) -> pn.widgets.Button | None:
    """Find the close button within a plot cell."""
    try:
        cell = grid.panel[row, col]  # type: ignore[index]
        if isinstance(cell, pn.Column):
            for item in cell:
                if isinstance(item, pn.widgets.Button) and item.name == '\u00d7':
                    return item
    except (KeyError, IndexError):
        pass
    return None


def count_occupied_cells(grid: PlotGrid) -> int:
    """Count how many cell positions contain plots."""
    count = 0
    # We need to know grid dimensions - we can infer from the panel
    # Panel GridSpec doesn't expose nrows/ncols directly, but we can check
    # We'll need to iterate over what we expect based on initialization
    # This is a bit tricky without accessing private attributes
    # For now, let's just try reasonable ranges
    for row in range(10):  # Assume max 10 rows
        for col in range(10):  # Assume max 10 cols
            if is_cell_occupied(grid, row, col):
                count += 1
    return count


def insert_plot_at_pending_selection(grid: PlotGrid, plot: hv.DynamicMap) -> None:
    """
    Insert a plot at the pending selection position.

    This mimics the workflow used by PlotGridTabs: reading the pending
    selection and calling insert_widget_at with a plot widget.

    Parameters
    ----------
    grid:
        The PlotGrid instance.
    plot:
        The HoloViews DynamicMap to insert.
    """
    pending = grid._pending_selection
    if pending is None:
        return

    row, col, row_span, col_span = pending

    # Create plot widget similar to PlotGridTabs._create_plot_widget
    # Use .layout to preserve widgets for DynamicMaps with kdims
    plot_pane_wrapper = pn.pane.HoloViews(plot, sizing_mode='stretch_both')
    plot_pane = plot_pane_wrapper.layout

    # Create close button using the helper from plot_grid module
    from ess.livedata.dashboard.widgets.plot_grid import _create_close_button

    def on_close() -> None:
        grid.remove_widget_at(row, col, row_span, col_span)

    close_button = _create_close_button(on_close)

    widget = pn.Column(
        close_button,
        plot_pane,
        sizing_mode='stretch_both',
        styles={'position': 'relative'},
    )

    # Clear pending selection before inserting (matches PlotGridTabs behavior)
    grid.cancel_pending_selection()

    # Insert widget at the position
    grid.insert_widget_at(row, col, row_span, col_span, widget)


class TestPlotGridInitialization:
    def test_grid_has_panel_property(self, mock_callback: FakeCallback) -> None:
        grid = PlotGrid(nrows=2, ncols=2, plot_request_callback=mock_callback)
        assert grid.panel is not None
        assert isinstance(grid.panel, pn.GridSpec)

    def test_grid_starts_with_empty_clickable_cells(
        self, mock_callback: FakeCallback
    ) -> None:
        grid = PlotGrid(nrows=2, ncols=2, plot_request_callback=mock_callback)

        # All cells should have clickable buttons
        for row in range(2):
            for col in range(2):
                button = get_cell_button(grid, row, col)
                assert button is not None, f"Cell ({row}, {col}) should have a button"
                assert (
                    not button.disabled  # type: ignore[truthy-bool]
                ), f"Cell ({row}, {col}) should be enabled"


class TestCellSelection:
    def test_single_cell_selection_triggers_callback(
        self, mock_callback: FakeCallback, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        # First click should not trigger callback
        simulate_click(grid, 1, 1)
        mock_callback.assert_not_called()

        # Second click on same cell should trigger callback
        simulate_click(grid, 1, 1)
        mock_callback.assert_called_once()

        # Complete the deferred insertion
        insert_plot_at_pending_selection(grid, mock_plot)

        # Cell should now contain a plot
        assert is_cell_occupied(grid, 1, 1)

    def test_rectangular_region_selection(
        self, mock_callback: FakeCallback, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=4, ncols=4, plot_request_callback=mock_callback)

        # Click two corners of a region
        simulate_click(grid, 0, 0)
        simulate_click(grid, 1, 2)

        mock_callback.assert_called_once()

        # Complete the deferred insertion
        insert_plot_at_pending_selection(grid, mock_plot)

        # All cells in the 2x3 region should be occupied
        assert is_cell_occupied(grid, 0, 0)
        assert is_cell_occupied(grid, 0, 1)
        assert is_cell_occupied(grid, 0, 2)
        assert is_cell_occupied(grid, 1, 0)
        assert is_cell_occupied(grid, 1, 1)
        assert is_cell_occupied(grid, 1, 2)

        # Cells outside region should be empty
        assert not is_cell_occupied(grid, 2, 0)
        assert not is_cell_occupied(grid, 0, 3)

    def test_selection_works_regardless_of_click_order(
        self, mock_callback: FakeCallback, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=4, ncols=4, plot_request_callback=mock_callback)

        # Click bottom-right first, then top-left
        simulate_click(grid, 2, 2)
        simulate_click(grid, 1, 1)

        insert_plot_at_pending_selection(grid, mock_plot)

        # Should still create a 2x2 region
        assert is_cell_occupied(grid, 1, 1)
        assert is_cell_occupied(grid, 1, 2)
        assert is_cell_occupied(grid, 2, 1)
        assert is_cell_occupied(grid, 2, 2)

    def test_first_click_changes_cell_appearance(
        self, mock_callback: FakeCallback
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        # Get initial button state
        button_before = get_cell_button(grid, 0, 0)
        initial_label = button_before.name if button_before else None

        # Click the cell
        simulate_click(grid, 0, 0)

        # Button should now show different label
        button_after = get_cell_button(grid, 0, 0)
        new_label = button_after.name if button_after else None

        assert initial_label != new_label
        assert new_label is not None
        assert '1x1' in new_label  # type: ignore[operator]


class TestPlotInsertion:
    def test_multiple_plots_can_be_inserted(
        self, mock_callback: FakeCallback, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        # Insert first plot
        simulate_click(grid, 0, 0)
        simulate_click(grid, 0, 0)
        insert_plot_at_pending_selection(grid, mock_plot)

        # Insert second plot
        simulate_click(grid, 2, 2)
        simulate_click(grid, 2, 2)
        insert_plot_at_pending_selection(grid, mock_plot)

        # Both cells should be occupied
        assert is_cell_occupied(grid, 0, 0)
        assert is_cell_occupied(grid, 2, 2)

    def test_inserted_plot_has_close_button(
        self, mock_callback: FakeCallback, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        simulate_click(grid, 1, 1)
        simulate_click(grid, 1, 1)
        insert_plot_at_pending_selection(grid, mock_plot)

        # Should be able to find a close button
        close_button = find_close_button(grid, 1, 1)
        assert close_button is not None


class TestPlotRemoval:
    def test_clicking_close_button_removes_plot(
        self, mock_callback: FakeCallback, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        # Insert plot
        simulate_click(grid, 0, 0)
        simulate_click(grid, 1, 1)
        insert_plot_at_pending_selection(grid, mock_plot)

        # Verify plot is there
        assert is_cell_occupied(grid, 0, 0)

        # Click close button
        close_button = find_close_button(grid, 0, 0)
        assert close_button is not None
        close_button.param.trigger('clicks')

        # Cells should now be empty and clickable again
        assert not is_cell_occupied(grid, 0, 0)
        assert not is_cell_occupied(grid, 1, 1)
        assert get_cell_button(grid, 0, 0) is not None
        assert get_cell_button(grid, 1, 1) is not None

    def test_removed_cells_become_selectable_again(
        self, mock_callback: FakeCallback, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        # Insert and remove plot
        simulate_click(grid, 1, 1)
        simulate_click(grid, 1, 1)
        insert_plot_at_pending_selection(grid, mock_plot)

        close_button = find_close_button(grid, 1, 1)
        assert close_button is not None
        close_button.param.trigger('clicks')

        mock_callback.reset()

        # Should be able to select the cell again
        simulate_click(grid, 1, 1)
        simulate_click(grid, 1, 1)

        assert mock_callback.call_count == 1

        insert_plot_at_pending_selection(grid, mock_plot)
        assert is_cell_occupied(grid, 1, 1)


class TestOverlapPrevention:
    def test_cannot_select_overlapping_region(
        self, mock_callback: FakeCallback, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=4, ncols=4, plot_request_callback=mock_callback)

        # Insert plot at (1, 1) to (2, 2)
        simulate_click(grid, 1, 1)
        simulate_click(grid, 2, 2)
        insert_plot_at_pending_selection(grid, mock_plot)

        # Start new selection at (0, 0)
        simulate_click(grid, 0, 0)

        # Cell (1, 1) should now be disabled (since it would overlap)
        button = get_cell_button(grid, 1, 1)
        # Button should be None because that cell is occupied
        assert button is None

    def test_non_overlapping_regions_can_be_selected(
        self, mock_callback: FakeCallback, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=4, ncols=4, plot_request_callback=mock_callback)

        # Insert plot at (1, 1) to (2, 2)
        simulate_click(grid, 1, 1)
        simulate_click(grid, 2, 2)
        insert_plot_at_pending_selection(grid, mock_plot)

        mock_callback.reset()

        # Should be able to select non-overlapping regions
        simulate_click(grid, 0, 0)
        simulate_click(grid, 0, 0)
        mock_callback.assert_called_once()

        insert_plot_at_pending_selection(grid, mock_plot)
        assert is_cell_occupied(grid, 0, 0)


class TestSelectionCancellation:
    def test_cancel_pending_selection_clears_state(
        self, mock_callback: FakeCallback
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        # Start a selection
        simulate_click(grid, 0, 0)
        simulate_click(grid, 1, 1)

        # Cancel it
        grid.cancel_pending_selection()

        # Should be able to start a new selection
        mock_callback.reset()
        simulate_click(grid, 2, 2)
        simulate_click(grid, 2, 2)
        mock_callback.assert_called_once()


class TestErrorHandling:
    def test_insert_without_pending_selection_shows_error(
        self, mock_callback: FakeCallback, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        # Try to insert without making a selection
        # This should handle gracefully (no crash)
        insert_plot_at_pending_selection(grid, mock_plot)

        # No cells should be occupied
        assert not is_cell_occupied(grid, 0, 0)
        assert not is_cell_occupied(grid, 1, 1)

    def test_callback_error_prevents_plot_insertion(self) -> None:
        error_callback = FakeCallback(side_effect=ValueError('Test error'))
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=error_callback)

        simulate_click(grid, 0, 0)

        # Second click raises error, but grid should handle it
        with pytest.raises(ValueError, match='Test error'):
            simulate_click(grid, 0, 0)

        # Grid should still be in a usable state
        # We never called insert_plot_at_pending_selection, so cell
        # should still be empty
        assert not is_cell_occupied(grid, 0, 0)
