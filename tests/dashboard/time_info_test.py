# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Tests for time-interval display in plot titles.

These tests verify that the time information shown in plot titles is factual,
i.e., it reflects the actual time range of the data being displayed.

The tests simulate the end-to-end flow: buffered data → extractor → plotter.
Extractors are responsible for setting correct start_time/end_time coords
based on the actual data range, not stale coords from buffer references.
"""

import time
import uuid

import holoviews as hv
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard import plots
from ess.livedata.dashboard.extractors import (
    FullHistoryExtractor,
    WindowAggregatingExtractor,
)
from ess.livedata.dashboard.plot_params import PlotParams1d

hv.extension('bokeh')


@pytest.fixture
def workflow_id():
    return WorkflowId(
        instrument='test_instrument',
        namespace='test_namespace',
        name='test_workflow',
        version=1,
    )


@pytest.fixture
def data_key(workflow_id):
    job_id = JobId(source_name='test_source', job_number=uuid.uuid4())
    return ResultKey(workflow_id=workflow_id, job_id=job_id, output_name='test_result')


def _extract_title(result) -> str:
    """Extract title from a HoloViews plot result."""
    opts = hv.Store.lookup_options('bokeh', result, 'plot').kwargs
    return opts.get('title', '')


def _parse_time_range_from_title(title: str) -> tuple[str, str] | None:
    """
    Parse start and end time from a title string.

    Expected format: "HH:MM:SS.s - HH:MM:SS.s (Lag: X.Xs)"
    Returns (start_time_str, end_time_str) or None if not found.
    """
    if ' - ' not in title or 'Lag:' not in title:
        return None
    time_part = title.split(' (Lag:')[0]
    parts = time_part.split(' - ')
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def _extract_lag_seconds(title: str) -> float:
    """Extract lag in seconds from title string."""
    lag_match = title.split('Lag:')[1].strip().rstrip('s)')
    return float(lag_match)


class TestTimeseriesTimeInfo:
    """
    Tests for time info in timeseries plots (full history extraction).

    These tests simulate the end-to-end flow through FullHistoryExtractor,
    which computes scalar start_time/end_time from min/max of 1-D coords.
    """

    def test_time_info_reflects_actual_time_dimension_range(self, data_key):
        """Time info should reflect the actual time range of the data."""
        now_ns = time.time_ns()

        # Buffered data with 1-D start_time/end_time coords (as TemporalBuffer produces)
        time_values = [now_ns - int(i * 1e9) for i in range(5, 0, -1)]
        frame_duration_ns = int(1e9)
        start_time_values = time_values
        end_time_values = [t + frame_duration_ns for t in time_values]

        buffered_data = sc.DataArray(
            data=sc.array(dims=['time', 'x'], values=[[1.0, 2.0]] * 5),
            coords={
                'time': sc.array(dims=['time'], values=time_values, unit='ns'),
                'x': sc.array(dims=['x'], values=[0.0, 1.0]),
                'start_time': sc.array(
                    dims=['time'], values=start_time_values, unit='ns'
                ),
                'end_time': sc.array(dims=['time'], values=end_time_values, unit='ns'),
            },
        )

        # Simulate extraction (this is what happens in the real data flow)
        extractor = FullHistoryExtractor()
        extracted_data = extractor.extract(buffered_data)

        plotter = plots.LinePlotter.from_params(PlotParams1d())
        plotter.compute({data_key: extracted_data})
        result = plotter.get_cached_state()

        title = _extract_title(result)
        assert 'Lag:' in title
        lag_seconds = _extract_lag_seconds(title)

        # Lag should reflect actual data (last end_time ~0s ago)
        assert lag_seconds < 5.0, f"Lag {lag_seconds}s should be ~0s"

    def test_time_info_with_datetime64_time_coord(self, data_key):
        """Time info should work with datetime64 time coordinates."""
        now_ns = time.time_ns()
        time_values = [now_ns - int(i * 1e9) for i in range(5, 0, -1)]
        time_var = sc.epoch(unit='ns') + sc.array(
            dims=['time'], values=time_values, unit='ns'
        )

        # Create with 1-D start_time/end_time coords
        frame_duration_ns = int(1e9)
        start_time_values = time_values
        end_time_values = [t + frame_duration_ns for t in time_values]

        buffered_data = sc.DataArray(
            data=sc.array(dims=['time', 'x'], values=[[1.0, 2.0]] * 5),
            coords={
                'time': time_var,
                'x': sc.array(dims=['x'], values=[0.0, 1.0]),
                'start_time': sc.array(
                    dims=['time'], values=start_time_values, unit='ns'
                ),
                'end_time': sc.array(dims=['time'], values=end_time_values, unit='ns'),
            },
        )

        # Simulate extraction
        extractor = FullHistoryExtractor()
        extracted_data = extractor.extract(buffered_data)

        plotter = plots.LinePlotter.from_params(PlotParams1d())
        plotter.compute({data_key: extracted_data})
        result = plotter.get_cached_state()

        title = _extract_title(result)
        # Extractor computes min/max of start_time/end_time
        assert 'Lag:' in title
        lag_seconds = _extract_lag_seconds(title)
        assert lag_seconds < 5.0


class TestWindowedTimeInfo:
    """
    Tests for time info in windowed plots (aggregation over time window).

    These tests simulate the end-to-end flow through WindowAggregatingExtractor,
    which computes min/max of start_time/end_time from the windowed 1-D coords.
    """

    def test_aggregated_data_reflects_actual_window_time_range(self, data_key):
        """Aggregated data should show the time range of the aggregation window."""
        now_ns = time.time_ns()

        # Buffered data with 1-D start_time/end_time coords (as TemporalBuffer produces)
        time_values = [now_ns - int(i * 1e9) for i in range(10, 0, -1)]
        frame_duration_ns = int(1e9)
        start_time_values = time_values
        end_time_values = [t + frame_duration_ns for t in time_values]

        buffered_data = sc.DataArray(
            data=sc.array(dims=['time', 'x'], values=[[1.0, 2.0, 3.0]] * 10),
            coords={
                'time': sc.array(dims=['time'], values=time_values, unit='ns'),
                'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0]),
                'start_time': sc.array(
                    dims=['time'], values=start_time_values, unit='ns'
                ),
                'end_time': sc.array(dims=['time'], values=end_time_values, unit='ns'),
            },
        )

        # Simulate extraction with 3-second window
        extractor = WindowAggregatingExtractor(window_duration_seconds=3.0)
        extracted_data = extractor.extract(buffered_data)

        plotter = plots.LinePlotter.from_params(PlotParams1d())
        plotter.compute({data_key: extracted_data})
        result = plotter.get_cached_state()

        title = _extract_title(result)
        assert 'Lag:' in title
        lag_seconds = _extract_lag_seconds(title)

        # Lag should reflect actual window end (~0s)
        assert lag_seconds < 5.0, f"Lag {lag_seconds}s should be ~0s"


class TestTimeInfoBaseline:
    """Tests verifying time info works when coords are accurate."""

    def test_correct_lag_with_accurate_coords(self, data_key):
        """Time info should be correct when coords accurately reflect the data."""
        now_ns = time.time_ns()
        start_time = now_ns - int(2e9)
        end_time = now_ns - int(1e9)

        data = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
            coords={
                'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0]),
                'start_time': sc.scalar(start_time, unit='ns'),
                'end_time': sc.scalar(end_time, unit='ns'),
            },
        )

        plotter = plots.LinePlotter.from_params(PlotParams1d())
        plotter.compute({data_key: data})
        result = plotter.get_cached_state()

        title = _extract_title(result)
        assert 'Lag:' in title
        lag_seconds = _extract_lag_seconds(title)
        assert 0.5 < lag_seconds < 3.0


class TestTimeInfoEdgeCases:
    """Tests for edge cases in time info handling."""

    def test_no_time_info_without_time_coords_or_dimension(self, data_key):
        """No time info should be shown when there are no time coordinates."""
        data = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
            coords={'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0])},
        )

        plotter = plots.LinePlotter.from_params(PlotParams1d())
        plotter.compute({data_key: data})
        result = plotter.get_cached_state()

        title = _extract_title(result)
        assert 'Lag:' not in title

    def test_time_dimension_without_coords_does_not_crash(self, data_key):
        """Data with 'time' dimension but no scalar time coords should not crash."""
        now_ns = time.time_ns()
        time_values = [now_ns - int(i * 1e9) for i in range(5, 0, -1)]

        data = sc.DataArray(
            data=sc.array(dims=['time', 'x'], values=[[1.0, 2.0]] * 5),
            coords={
                'time': sc.array(dims=['time'], values=time_values, unit='ns'),
                'x': sc.array(dims=['x'], values=[0.0, 1.0]),
            },
        )

        plotter = plots.LinePlotter.from_params(PlotParams1d())
        plotter.compute({data_key: data})  # Should not raise
        result = plotter.get_cached_state()
        assert isinstance(_extract_title(result), str)
