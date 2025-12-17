# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for static plotters (rectangles, vlines, hlines)."""

import holoviews as hv
import pytest

# Load holoviews extension for tests that create plots with options
hv.extension('bokeh')

from ess.livedata.dashboard.static_plots import (  # noqa: E402
    HLinesCoordinates,
    HLinesParams,
    HLinesPlotter,
    RectanglesCoordinates,
    RectanglesParams,
    RectanglesPlotter,
    VLinesCoordinates,
    VLinesParams,
    VLinesPlotter,
)


class TestRectanglesCoordinates:
    """Tests for RectanglesCoordinates validation."""

    def test_valid_coordinates(self):
        """Valid rectangle coordinates are accepted."""
        coords = RectanglesCoordinates(coordinates="[[0, 0, 10, 10], [20, 20, 30, 30]]")
        assert coords.parse() == [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)]

    def test_empty_coordinates(self):
        """Empty list is valid."""
        coords = RectanglesCoordinates(coordinates="[]")
        assert coords.parse() == []

    def test_invalid_json(self):
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid format"):
            RectanglesCoordinates(coordinates="not json")

    def test_wrong_coordinate_count(self):
        """Wrong number of coordinates per rectangle raises ValueError."""
        with pytest.raises(ValueError, match="expected 4 coordinates"):
            RectanglesCoordinates(coordinates="[[0, 0, 10]]")

    def test_non_numeric_values(self):
        """Non-numeric values raise ValueError."""
        with pytest.raises(ValueError, match="must be a number"):
            RectanglesCoordinates(coordinates='[[0, 0, "ten", 10]]')

    def test_not_a_list(self):
        """Non-list top-level raises ValueError."""
        with pytest.raises(ValueError, match="Must be a list"):
            RectanglesCoordinates(coordinates='{"x": 1}')


class TestRectanglesPlotter:
    """Tests for RectanglesPlotter."""

    def test_create_static_plot_with_rectangles(self):
        """Creates hv.Rectangles with data."""
        params = RectanglesParams(
            geometry=RectanglesCoordinates(coordinates="[[0, 0, 10, 10]]")
        )
        plotter = RectanglesPlotter.from_params(params)
        plot = plotter.create_static_plot()
        assert isinstance(plot, hv.Rectangles)

    def test_create_static_plot_empty(self):
        """Creates empty hv.Rectangles when no coordinates."""
        params = RectanglesParams(geometry=RectanglesCoordinates(coordinates="[]"))
        plotter = RectanglesPlotter.from_params(params)
        plot = plotter.create_static_plot()
        assert isinstance(plot, hv.Rectangles)


class TestVLinesCoordinates:
    """Tests for VLinesCoordinates validation."""

    def test_valid_positions(self):
        """Valid positions are accepted."""
        coords = VLinesCoordinates(positions="[10, 20, 30]")
        assert coords.parse() == [10.0, 20.0, 30.0]

    def test_empty_positions(self):
        """Empty list is valid."""
        coords = VLinesCoordinates(positions="[]")
        assert coords.parse() == []

    def test_invalid_json(self):
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid format"):
            VLinesCoordinates(positions="not json")

    def test_non_numeric_values(self):
        """Non-numeric values raise ValueError."""
        with pytest.raises(ValueError, match="must be a number"):
            VLinesCoordinates(positions='[1, "two", 3]')


class TestVLinesPlotter:
    """Tests for VLinesPlotter."""

    def test_create_static_plot_with_lines(self):
        """Creates hv.VLines with data."""
        params = VLinesParams(geometry=VLinesCoordinates(positions="[10, 20]"))
        plotter = VLinesPlotter.from_params(params)
        plot = plotter.create_static_plot()
        assert isinstance(plot, hv.VLines)

    def test_create_static_plot_empty(self):
        """Creates empty hv.VLines when no positions."""
        params = VLinesParams(geometry=VLinesCoordinates(positions="[]"))
        plotter = VLinesPlotter.from_params(params)
        plot = plotter.create_static_plot()
        assert isinstance(plot, hv.VLines)


class TestHLinesCoordinates:
    """Tests for HLinesCoordinates validation."""

    def test_valid_positions(self):
        """Valid positions are accepted."""
        coords = HLinesCoordinates(positions="[10, 20, 30]")
        assert coords.parse() == [10.0, 20.0, 30.0]

    def test_empty_positions(self):
        """Empty list is valid."""
        coords = HLinesCoordinates(positions="[]")
        assert coords.parse() == []


class TestHLinesPlotter:
    """Tests for HLinesPlotter."""

    def test_create_static_plot_with_lines(self):
        """Creates hv.HLines with data."""
        params = HLinesParams(geometry=HLinesCoordinates(positions="[10, 20]"))
        plotter = HLinesPlotter.from_params(params)
        plot = plotter.create_static_plot()
        assert isinstance(plot, hv.HLines)

    def test_create_static_plot_empty(self):
        """Creates empty hv.HLines when no positions."""
        params = HLinesParams(geometry=HLinesCoordinates(positions="[]"))
        plotter = HLinesPlotter.from_params(params)
        plot = plotter.create_static_plot()
        assert isinstance(plot, hv.HLines)


class TestPlotterRegistration:
    """Tests that static plotters are properly registered."""

    def test_static_plotters_registered(self):
        """Static plotters are registered in the registry."""
        from ess.livedata.dashboard.plotting import plotter_registry

        static_plotters = plotter_registry.get_static_plotters()
        assert 'rectangles' in static_plotters
        assert 'vlines' in static_plotters
        assert 'hlines' in static_plotters

    def test_static_plotters_have_correct_category(self):
        """Static plotters have STATIC category."""
        from ess.livedata.dashboard.plotting import PlotterCategory, plotter_registry

        for name in ['rectangles', 'vlines', 'hlines']:
            spec = plotter_registry.get_spec(name)
            assert spec.category == PlotterCategory.STATIC

    def test_static_plotters_not_in_data_plotters(self):
        """Static plotters are not returned by get_specs (DATA only)."""
        from ess.livedata.dashboard.plotting import plotter_registry

        data_plotters = plotter_registry.get_specs()
        assert 'rectangles' not in data_plotters
        assert 'vlines' not in data_plotters
        assert 'hlines' not in data_plotters
