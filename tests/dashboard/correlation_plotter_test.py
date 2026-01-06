# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for correlation histogram plotters."""

import holoviews as hv
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.correlation_plotter import (
    PRIMARY,
    X_AXIS,
    Y_AXIS,
    AxisSpec,
    Bin1dParams,
    Bin2dParams,
    CorrelationHistogram1dParams,
    CorrelationHistogram1dPlotter,
    CorrelationHistogram2dParams,
    CorrelationHistogram2dPlotter,
    CorrelationHistogramPlotter,
    _make_lookup,
)
from ess.livedata.dashboard.plot_params import PlotScaleParams
from ess.livedata.dashboard.plots import LinePlotter

hv.extension('bokeh')


def _make_line_renderer() -> LinePlotter:
    """Create a LinePlotter for testing."""
    return LinePlotter(scale_opts=PlotScaleParams(), as_histogram=True)


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


class TestMakeLookup:
    """Tests for the _make_lookup helper function."""

    def test_uses_previous_mode_when_data_overlaps_axis(self):
        """When data timestamps overlap with axis range, uses 'previous' mode."""
        axis_data = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])
        data_max_time = sc.scalar(250, unit='ms')

        lookup = _make_lookup(axis_data, data_max_time)

        # 'previous' mode: for time=150, should get value at time=100 (1.0)
        result = lookup[sc.scalar(150, unit='ms')]
        assert result.value == 1.0

    def test_uses_nearest_mode_when_data_before_axis(self):
        """When all data timestamps are before the first axis timestamp,
        falls back to 'nearest' mode to avoid NaN coordinates.
        """
        axis_data = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])
        data_max_time = sc.scalar(50, unit='ms')  # Before first axis timestamp

        lookup = _make_lookup(axis_data, data_max_time)

        # 'nearest' mode: for time=50, should get value at nearest time=100 (1.0)
        result = lookup[sc.scalar(50, unit='ms')]
        assert result.value == 1.0

    def test_handles_axis_data_with_variances(self):
        """Lookup should work when axis data has variances."""
        axis_data = sc.DataArray(
            data=sc.array(
                dims=['time'],
                values=[1.0, 2.0, 3.0],
                variances=[0.1, 0.1, 0.1],
                unit='m',
            ),
            coords={'time': sc.array(dims=['time'], values=[100, 200, 300], unit='ms')},
        )
        data_max_time = sc.scalar(250, unit='ms')

        lookup = _make_lookup(axis_data, data_max_time)

        result = lookup[sc.scalar(150, unit='ms')]
        assert result.value == 1.0


class TestCorrelationHistogramPlotter:
    """Tests for the base CorrelationHistogramPlotter class."""

    def test_raises_when_primary_data_missing(self):
        """Should raise ValueError when no primary data is provided."""
        axes = [AxisSpec(role=X_AXIS, name='x', bins=10)]
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=_make_line_renderer()
        )

        data = {
            PRIMARY: {},
            X_AXIS: {_make_result_key('position'): make_axis_data([100], [1.0])},
        }

        with pytest.raises(ValueError, match="at least one data source"):
            plotter(data)

    def test_raises_when_axis_data_missing(self):
        """Should raise ValueError when required axis data is missing."""
        axes = [AxisSpec(role=X_AXIS, name='x', bins=10)]
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=_make_line_renderer()
        )

        data = {
            PRIMARY: {_make_result_key('detector'): make_source_data([50], [10.0])},
            X_AXIS: {},  # Missing axis data
        }

        with pytest.raises(ValueError, match=f"role '{X_AXIS}'"):
            plotter(data)

    def test_works_with_single_axis(self):
        """Should work with a single axis (1D histogram)."""
        axis_data = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])
        source_data = make_source_data(times=[150, 250], values=[10.0, 20.0])

        axes = [AxisSpec(role=X_AXIS, name='position', bins=10)]
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=_make_line_renderer()
        )

        data = {
            PRIMARY: {_make_result_key('detector'): source_data},
            X_AXIS: {_make_result_key('position'): axis_data},
        }

        result = plotter(data)
        assert result is not None

    def test_works_with_multiple_axes(self):
        """Should work with multiple axes (2D histogram)."""
        x_axis = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])
        y_axis = make_axis_data(
            times=[100, 200, 300],
            values=[10.0, 20.0, 30.0],
            value_unit='K',
        )
        source_data = make_source_data(times=[150, 250], values=[10.0, 20.0])

        axes = [
            AxisSpec(role=X_AXIS, name='position', bins=5),
            AxisSpec(role=Y_AXIS, name='temperature', bins=5),
        ]
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=_make_line_renderer()
        )

        data = {
            PRIMARY: {_make_result_key('detector'): source_data},
            X_AXIS: {_make_result_key('position'): x_axis},
            Y_AXIS: {_make_result_key('temperature'): y_axis},
        }

        result = plotter(data)
        assert result is not None

    def test_handles_data_before_axis_range(self):
        """When data timestamps are before the first axis timestamp,
        the plotter falls back to 'nearest' mode to avoid NaN coordinates.
        """
        # Axis data starts at t=100
        axis_data = make_axis_data(times=[100, 200, 300], values=[1.0, 2.0, 3.0])
        # Source data is all BEFORE the first axis timestamp
        source_data = make_source_data(times=[50, 60, 70], values=[10.0, 20.0, 30.0])

        axes = [AxisSpec(role=X_AXIS, name='position', bins=10)]
        plotter = CorrelationHistogramPlotter(
            axes=axes, normalize=False, renderer=_make_line_renderer()
        )

        data = {
            PRIMARY: {_make_result_key('detector'): source_data},
            X_AXIS: {_make_result_key('position'): axis_data},
        }

        # Should not raise
        result = plotter(data)
        assert result is not None


class TestCorrelationHistogram1dPlotter:
    """Tests for CorrelationHistogram1dPlotter wrapper."""

    def test_creates_correct_axis_spec(self):
        """Verifies 1D plotter creates correct axis configuration."""
        params = CorrelationHistogram1dParams(
            bins=Bin1dParams(x_axis_source='position', x_bins=50)
        )
        plotter = CorrelationHistogram1dPlotter(params)

        assert len(plotter._axes) == 1
        assert plotter._axes[0].role == X_AXIS
        assert plotter._axes[0].name == 'position'
        assert plotter._axes[0].bins == 50

    def test_from_params_factory(self):
        """Verifies from_params factory method works."""
        params = CorrelationHistogram1dParams()
        plotter = CorrelationHistogram1dPlotter.from_params(params)
        assert isinstance(plotter, CorrelationHistogram1dPlotter)


class TestCorrelationHistogram2dPlotter:
    """Tests for CorrelationHistogram2dPlotter wrapper."""

    def test_creates_correct_axis_specs(self):
        """Verifies 2D plotter creates correct axis configuration."""
        params = CorrelationHistogram2dParams(
            bins=Bin2dParams(
                x_axis_source='position',
                x_bins=20,
                y_axis_source='temperature',
                y_bins=30,
            )
        )
        plotter = CorrelationHistogram2dPlotter(params)

        assert len(plotter._axes) == 2
        assert plotter._axes[0].role == Y_AXIS
        assert plotter._axes[0].name == 'temperature'
        assert plotter._axes[0].bins == 30
        assert plotter._axes[1].role == X_AXIS
        assert plotter._axes[1].name == 'position'
        assert plotter._axes[1].bins == 20

    def test_from_params_factory(self):
        """Verifies from_params factory method works."""
        params = CorrelationHistogram2dParams()
        plotter = CorrelationHistogram2dPlotter.from_params(params)
        assert isinstance(plotter, CorrelationHistogram2dPlotter)
