# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from unittest.mock import MagicMock

import holoviews as hv
import numpy as np
import pytest

from ess.livedata.dashboard.widgets.plot_grid import PlotGrid


@pytest.fixture
def mock_plot() -> hv.DynamicMap:
    """Create a mock HoloViews DynamicMap for testing."""

    def create_curve(x_range):
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.sin(x)
        return hv.Curve((x, y))

    return hv.DynamicMap(create_curve, kdims=['x_range']).redim.range(x_range=(0, 10))


@pytest.fixture
def mock_callback(mock_plot: hv.DynamicMap) -> MagicMock:
    """Create a mock callback that returns a plot."""
    callback = MagicMock(return_value=mock_plot)
    return callback


class TestPlotGridInitialization:
    def test_grid_created_with_correct_dimensions(
        self, mock_callback: MagicMock
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=4, plot_request_callback=mock_callback)
        assert grid._nrows == 3
        assert grid._ncols == 4

    def test_grid_has_panel_property(self, mock_callback: MagicMock) -> None:
        grid = PlotGrid(nrows=2, ncols=2, plot_request_callback=mock_callback)
        assert grid.panel is not None
        # GridSpec is a Panel viewable
        assert grid.panel is grid._grid

    def test_grid_starts_empty(self, mock_callback: MagicMock) -> None:
        grid = PlotGrid(nrows=2, ncols=2, plot_request_callback=mock_callback)
        assert len(grid._occupied_cells) == 0


class TestCellSelection:
    def test_single_cell_selection(
        self, mock_callback: MagicMock, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        # Simulate clicking the same cell twice
        grid._on_cell_click(1, 1)
        assert grid._first_click == (1, 1)
        assert grid._highlighted_cell == (1, 1)

        grid._on_cell_click(1, 1)

        # Callback should be invoked
        mock_callback.assert_called_once()

        # Plot should be inserted
        assert len(grid._occupied_cells) == 1
        assert (1, 1, 1, 1) in grid._occupied_cells

    def test_rectangular_region_selection(
        self, mock_callback: MagicMock, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=4, ncols=4, plot_request_callback=mock_callback)

        # Click two corners of a 2x3 region
        grid._on_cell_click(0, 0)
        grid._on_cell_click(1, 2)

        mock_callback.assert_called_once()

        # Should create a 2x3 region starting at (0, 0)
        assert (0, 0, 2, 3) in grid._occupied_cells

    def test_selection_normalized_to_top_left(
        self, mock_callback: MagicMock, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=4, ncols=4, plot_request_callback=mock_callback)

        # Click bottom-right first, then top-left
        grid._on_cell_click(2, 2)
        grid._on_cell_click(1, 1)

        # Should still create region with top-left as starting point
        assert (1, 1, 2, 2) in grid._occupied_cells

    def test_first_click_highlights_cell(self, mock_callback: MagicMock) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        grid._on_cell_click(0, 0)

        assert grid._first_click == (0, 0)
        assert grid._highlighted_cell == (0, 0)

    def test_selection_cleared_after_insertion(
        self, mock_callback: MagicMock, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        grid._on_cell_click(0, 0)
        grid._on_cell_click(1, 1)

        assert grid._first_click is None
        assert grid._highlighted_cell is None


class TestOccupancyChecking:
    def test_cannot_select_occupied_cell(
        self, mock_callback: MagicMock, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        # Insert a plot at (0, 0)
        grid._on_cell_click(0, 0)
        grid._on_cell_click(0, 0)

        mock_callback.reset_mock()

        # Try to select the same cell again
        grid._on_cell_click(0, 0)

        # Should not trigger callback
        mock_callback.assert_not_called()
        assert grid._first_click is None

    def test_cannot_select_region_overlapping_plot(
        self, mock_callback: MagicMock, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=4, ncols=4, plot_request_callback=mock_callback)

        # Insert a 2x2 plot at (1, 1)
        grid._on_cell_click(1, 1)
        grid._on_cell_click(2, 2)

        mock_callback.reset_mock()

        # Try to select a region that overlaps
        grid._on_cell_click(0, 0)
        grid._on_cell_click(2, 2)

        # Should not insert new plot
        mock_callback.assert_not_called()
        assert len(grid._occupied_cells) == 1

    def test_is_cell_occupied_detects_cells_within_span(
        self, mock_callback: MagicMock, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=4, ncols=4, plot_request_callback=mock_callback)

        # Insert a 2x2 plot
        grid._on_cell_click(1, 1)
        grid._on_cell_click(2, 2)

        # All cells within the span should be occupied
        assert grid._is_cell_occupied(1, 1)
        assert grid._is_cell_occupied(1, 2)
        assert grid._is_cell_occupied(2, 1)
        assert grid._is_cell_occupied(2, 2)

        # Cells outside should not be occupied
        assert not grid._is_cell_occupied(0, 0)
        assert not grid._is_cell_occupied(3, 3)


class TestPlotInsertion:
    def test_plot_inserted_at_correct_position(
        self, mock_callback: MagicMock, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        grid._on_cell_click(1, 2)
        grid._on_cell_click(1, 2)

        # Check the plot is tracked
        assert (1, 2, 1, 1) in grid._occupied_cells

    def test_callback_invoked_on_complete_selection(
        self, mock_callback: MagicMock
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        grid._on_cell_click(0, 0)
        mock_callback.assert_not_called()

        grid._on_cell_click(1, 1)
        mock_callback.assert_called_once()

    def test_multiple_plots_can_be_inserted(
        self, mock_callback: MagicMock, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        # Insert first plot
        grid._on_cell_click(0, 0)
        grid._on_cell_click(0, 0)

        # Insert second plot
        grid._on_cell_click(2, 2)
        grid._on_cell_click(2, 2)

        assert len(grid._occupied_cells) == 2
        assert (0, 0, 1, 1) in grid._occupied_cells
        assert (2, 2, 1, 1) in grid._occupied_cells


class TestPlotRemoval:
    def test_remove_plot_clears_cells(
        self, mock_callback: MagicMock, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        # Insert plot
        grid._on_cell_click(0, 0)
        grid._on_cell_click(1, 1)

        # Remove plot
        grid._remove_plot(0, 0, 2, 2)

        assert len(grid._occupied_cells) == 0
        assert not grid._is_cell_occupied(0, 0)
        assert not grid._is_cell_occupied(1, 1)

    def test_removed_cells_become_selectable_again(
        self, mock_callback: MagicMock, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=mock_callback)

        # Insert and remove plot
        grid._on_cell_click(1, 1)
        grid._on_cell_click(1, 1)
        grid._remove_plot(1, 1, 1, 1)

        mock_callback.reset_mock()

        # Should be able to select the cell again
        grid._on_cell_click(1, 1)
        grid._on_cell_click(1, 1)

        assert mock_callback.call_count == 1
        assert (1, 1, 1, 1) in grid._occupied_cells


class TestRegionAvailability:
    def test_is_region_available_for_empty_area(self, mock_callback: MagicMock) -> None:
        grid = PlotGrid(nrows=4, ncols=4, plot_request_callback=mock_callback)

        assert grid._is_region_available(0, 0, 2, 2)
        assert grid._is_region_available(1, 1, 3, 3)

    def test_is_region_available_detects_overlap(
        self, mock_callback: MagicMock, mock_plot: hv.DynamicMap
    ) -> None:
        grid = PlotGrid(nrows=4, ncols=4, plot_request_callback=mock_callback)

        # Insert plot at (1, 1) to (2, 2)
        grid._on_cell_click(1, 1)
        grid._on_cell_click(2, 2)

        # Overlapping region should not be available
        assert not grid._is_region_available(0, 0, 2, 2)
        assert not grid._is_region_available(1, 1, 3, 3)

        # Non-overlapping regions should be available
        assert grid._is_region_available(0, 0, 0, 0)
        assert grid._is_region_available(3, 3, 3, 3)


class TestCallbackErrors:
    def test_callback_error_does_not_insert_plot(
        self, mock_plot: hv.DynamicMap
    ) -> None:
        error_callback = MagicMock(side_effect=ValueError('Test error'))
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=error_callback)

        grid._on_cell_click(0, 0)
        grid._on_cell_click(0, 0)

        # No plot should be inserted
        assert len(grid._occupied_cells) == 0

    def test_callback_error_clears_selection(self) -> None:
        error_callback = MagicMock(side_effect=ValueError('Test error'))
        grid = PlotGrid(nrows=3, ncols=3, plot_request_callback=error_callback)

        grid._on_cell_click(0, 0)
        grid._on_cell_click(0, 0)

        # Selection should be cleared even on error
        assert grid._first_click is None
        has_pending = hasattr(grid, '_pending_selection')
        assert not has_pending or grid._pending_selection is None
