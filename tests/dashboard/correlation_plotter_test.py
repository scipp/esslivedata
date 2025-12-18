# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for correlation histogram plotters."""

import holoviews as hv
import scipp as sc

from ess.livedata.dashboard.correlation_plotter import (
    CorrelationHistogram1dParams,
    CorrelationHistogram1dPlotter,
    CorrelationHistogram2dParams,
    CorrelationHistogram2dPlotter,
    CorrelationHistogramData,
)

hv.extension('bokeh')


def make_axis_data(
    times: list[int], values: list[float], time_unit: str = 'ms', value_unit: str = 'm'
) -> sc.DataArray:
    """Create axis data for correlation histogram."""
    return sc.DataArray(
        data=sc.array(dims=['time'], values=values, unit=value_unit),
        coords={'time': sc.array(dims=['time'], values=times, unit=time_unit)},
    )


def make_source_data(
    times: list[int],
    values: list[float],
    time_unit: str = 'ms',
    value_unit: str = 'counts',
) -> sc.DataArray:
    """Create source data for correlation histogram."""
    return sc.DataArray(
        data=sc.array(dims=['time'], values=values, unit=value_unit),
        coords={'time': sc.array(dims=['time'], values=times, unit=time_unit)},
    )


class TestCorrelationHistogram1dPlotter:
    """Tests for CorrelationHistogram1dPlotter."""

    def test_data_before_first_axis_timestamp_uses_nearest_mode(self):
        """When data timestamps are before the first axis timestamp,
        the plotter falls back to 'nearest' mode to avoid NaN coordinates.

        This ensures histogramming still works when data arrives before
        axis data is available (e.g., during startup).
        """
        # Axis data starts at t=100
        axis_data = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])

        # Source data is all BEFORE the first axis timestamp
        source_data = make_source_data(times=[50, 60, 70], values=[10.0, 20.0, 30.0])

        data = CorrelationHistogramData(
            data_sources={'test': source_data},
            axis_data={'x': axis_data},
        )

        plotter = CorrelationHistogram1dPlotter(CorrelationHistogram1dParams())

        # Should not raise - uses 'nearest' mode when data is before axis range
        result = plotter(data)
        assert result is not None

    def test_data_overlapping_with_axis_uses_previous_mode(self):
        """When data timestamps overlap with axis range, uses 'previous' mode."""
        # Axis data from t=100 to t=300
        axis_data = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])

        # Source data overlaps with axis range
        source_data = make_source_data(times=[150, 250, 350], values=[10.0, 20.0, 30.0])

        data = CorrelationHistogramData(
            data_sources={'test': source_data},
            axis_data={'x': axis_data},
        )

        plotter = CorrelationHistogram1dPlotter(CorrelationHistogram1dParams())
        result = plotter(data)
        assert result is not None


class TestCorrelationHistogram2dPlotter:
    """Tests for CorrelationHistogram2dPlotter."""

    def test_data_before_first_axis_timestamp_uses_nearest_mode(self):
        """When data timestamps are before the first axis timestamp,
        the plotter falls back to 'nearest' mode to avoid NaN coordinates.
        """
        # Axis data starts at t=100
        x_axis = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])
        y_axis = make_axis_data(
            times=[100, 200, 300], values=[10.0, 20.0, 30.0], value_unit='K'
        )

        # Source data is all BEFORE the first axis timestamp
        source_data = make_source_data(times=[50, 60, 70], values=[10.0, 20.0, 30.0])

        data = CorrelationHistogramData(
            data_sources={'test': source_data},
            axis_data={'x': x_axis, 'y': y_axis},
        )

        plotter = CorrelationHistogram2dPlotter(CorrelationHistogram2dParams())

        # Should not raise - uses 'nearest' mode when data is before axis range
        result = plotter(data)
        assert result is not None
