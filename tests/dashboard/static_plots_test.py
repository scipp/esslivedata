# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for static plotters (rectangles, vlines, hlines)."""

import holoviews as hv
import pytest

# Load holoviews extension for tests that create plots with options
hv.extension('bokeh')

from ess.livedata.dashboard.static_plots import (  # noqa: E402
    HLinesParams,
    LinesCoordinates,
    LinesPlotter,
    RectanglesCoordinates,
    RectanglesParams,
    RectanglesPlotter,
    VLinesParams,
)


class TestRectanglesCoordinates:
    """Tests for RectanglesCoordinates validation."""

    def test_simple_format(self):
        """Simple format without outer brackets is accepted."""
        coords = RectanglesCoordinates(coordinates="[0,0,10,10], [20,20,30,30]")
        assert coords.parse() == [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)]

    def test_single_rectangle_simple_format(self):
        """Single rectangle in simple format."""
        coords = RectanglesCoordinates(coordinates="[0,0,10,10]")
        assert coords.parse() == [(0.0, 0.0, 10.0, 10.0)]

    def test_multiple_rectangles(self):
        """Multiple rectangles with space variations are accepted."""
        coords = RectanglesCoordinates(coordinates="[0, 0, 10, 10], [20, 20, 30, 30]")
        assert coords.parse() == [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)]

    def test_empty_string_rejected(self):
        """Empty string is rejected."""
        with pytest.raises(ValueError, match="At least one rectangle is required"):
            RectanglesCoordinates(coordinates="")

    def test_empty_brackets_rejected(self):
        """Empty brackets is rejected."""
        with pytest.raises(
            ValueError, match="1 validation error for RectanglesCoordinates"
        ):
            RectanglesCoordinates(coordinates="[]")

    def test_invalid_format(self):
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid format"):
            RectanglesCoordinates(coordinates="not valid")

    def test_wrong_coordinate_count(self):
        """Wrong number of coordinates per rectangle raises ValueError."""
        with pytest.raises(ValueError, match="expected 4 coordinates"):
            RectanglesCoordinates(coordinates="[0, 0, 10]")

    def test_non_numeric_values(self):
        """Non-numeric values raise ValueError."""
        with pytest.raises(ValueError, match="must be a number"):
            RectanglesCoordinates(coordinates='[0, 0, "ten", 10]')


class TestRectanglesPlotter:
    """Tests for RectanglesPlotter."""

    def test_create_static_plot_with_rectangles(self):
        """Creates hv.Rectangles with data."""
        params = RectanglesParams(
            geometry=RectanglesCoordinates(coordinates="[0, 0, 10, 10]")
        )
        plotter = RectanglesPlotter.from_params(params)
        plot = plotter.create_static_plot()
        assert isinstance(plot, hv.Rectangles)


class TestLinesCoordinates:
    """Tests for LinesCoordinates validation."""

    def test_simple_format(self):
        """Simple comma-separated format is accepted."""
        coords = LinesCoordinates(positions="10, 20, 30")
        assert coords.parse() == [10.0, 20.0, 30.0]

    def test_positions_with_spaces(self):
        """Positions with space variations are accepted."""
        coords = LinesCoordinates(positions="10,  20,   30")
        assert coords.parse() == [10.0, 20.0, 30.0]

    def test_empty_string_rejected(self):
        """Empty string is rejected."""
        with pytest.raises(ValueError, match="At least one position is required"):
            LinesCoordinates(positions="")

    def test_empty_brackets_rejected(self):
        """Empty brackets is rejected."""
        with pytest.raises(ValueError, match="must be a number"):
            LinesCoordinates(positions="[]")

    def test_invalid_number(self):
        """Invalid number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid number"):
            LinesCoordinates(positions="10, not_a_number, 30")

    def test_non_numeric_values_json(self):
        """Non-numeric values in JSON format raise ValueError."""
        with pytest.raises(ValueError, match="must be a number"):
            LinesCoordinates(positions='[1, "two", 3]')


class TestLinesPlotter:
    """Tests for LinesPlotter."""

    def test_vlines_creates_vlines_element(self):
        """LinesPlotter.vlines creates hv.VLines."""
        params = VLinesParams(geometry=LinesCoordinates(positions="10, 20"))
        plotter = LinesPlotter.vlines(params)
        plot = plotter.create_static_plot()
        assert isinstance(plot, hv.VLines)

    def test_hlines_creates_hlines_element(self):
        """LinesPlotter.hlines creates hv.HLines."""
        params = HLinesParams(geometry=LinesCoordinates(positions="10, 20"))
        plotter = LinesPlotter.hlines(params)
        plot = plotter.create_static_plot()
        assert isinstance(plot, hv.HLines)


class TestPlotterRegistration:
    """Tests that static plotters are properly registered."""

    def test_static_plotters_registered(self):
        """Static plotters are registered in the registry."""
        from ess.livedata.dashboard.plotter_registry import plotter_registry

        static_plotters = plotter_registry.get_static_plotters()
        assert 'rectangles' in static_plotters
        assert 'vlines' in static_plotters
        assert 'hlines' in static_plotters

    def test_static_plotters_have_correct_category(self):
        """Static plotters have STATIC category."""
        from ess.livedata.dashboard.plotter_registry import (
            PlotterCategory,
            plotter_registry,
        )

        for name in ['rectangles', 'vlines', 'hlines']:
            spec = plotter_registry.get_spec(name)
            assert spec.category == PlotterCategory.STATIC

    def test_static_plotters_not_in_data_plotters(self):
        """Static plotters are not returned by get_specs (DATA only)."""
        from ess.livedata.dashboard.plotter_registry import plotter_registry

        data_plotters = plotter_registry.get_specs()
        assert 'rectangles' not in data_plotters
        assert 'vlines' not in data_plotters
        assert 'hlines' not in data_plotters
