# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for correlation histogram plotters."""

import holoviews as hv
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.correlation_plotter import (
    PRIMARY,
    X_AXIS,
    Y_AXIS,
    Bin1dParams,
    Bin2dParams,
    CorrelationHistogram1dParams,
    CorrelationHistogram1dPlotter,
    CorrelationHistogram2dParams,
    CorrelationHistogram2dPlotter,
)

hv.extension('bokeh')


def _make_result_key(source_name: str) -> ResultKey:
    """Create a ResultKey for testing with the given source name."""
    return ResultKey(
        workflow_id=WorkflowId(
            instrument='test', namespace='test', name='test', version=1
        ),
        job_id=JobId(source_name=source_name, job_number=1),
        output_name='result',
    )


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

        # Structured data by role
        data = {
            PRIMARY: {_make_result_key('detector'): source_data},
            X_AXIS: {_make_result_key('position'): axis_data},
        }

        # Configure plotter to use 'position' as x-axis source
        params = CorrelationHistogram1dParams(
            bins=Bin1dParams(x_axis_source='position')
        )
        plotter = CorrelationHistogram1dPlotter(params)

        # Should not raise - uses 'nearest' mode when data is before axis range
        result = plotter(data)
        assert result is not None

    def test_data_overlapping_with_axis_uses_previous_mode(self):
        """When data timestamps overlap with axis range, uses 'previous' mode."""
        # Axis data from t=100 to t=300
        axis_data = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])

        # Source data overlaps with axis range
        source_data = make_source_data(times=[150, 250, 350], values=[10.0, 20.0, 30.0])

        # Structured data by role
        data = {
            PRIMARY: {_make_result_key('detector'): source_data},
            X_AXIS: {_make_result_key('position'): axis_data},
        }

        params = CorrelationHistogram1dParams(
            bins=Bin1dParams(x_axis_source='position')
        )
        plotter = CorrelationHistogram1dPlotter(params)
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

        # Structured data by role
        data = {
            PRIMARY: {_make_result_key('detector'): source_data},
            X_AXIS: {_make_result_key('position'): x_axis},
            Y_AXIS: {_make_result_key('temperature'): y_axis},
        }

        # Configure plotter to use 'position' and 'temperature' as axis sources
        params = CorrelationHistogram2dParams(
            bins=Bin2dParams(x_axis_source='position', y_axis_source='temperature')
        )
        plotter = CorrelationHistogram2dPlotter(params)

        # Should not raise - uses 'nearest' mode when data is before axis range
        result = plotter(data)
        assert result is not None
