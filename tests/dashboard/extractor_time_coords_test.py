# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Tests for time coordinate handling in extractors.

Extractors are the naming boundary: on the wire a value's upper bound is its
``time``, and every extractor turns the covered range into the scalar
``(start_time, end_time)`` pair the plotters read. The pair must reflect the
actual range of the extracted data, not stale coordinates from the buffer
reference, and must be pristine wall-clock time even where ``time`` itself is
shifted for display.
"""

import time

import scipp as sc

from ess.livedata.dashboard.extractors import (
    FullHistoryExtractor,
    LatestValueExtractor,
    WindowAggregatingExtractor,
)
from ess.livedata.dashboard.time_utils import get_local_timezone_offset_ns


def _make_buffered_data(
    num_points: int = 5,
    time_span_seconds: float = 5.0,
) -> sc.DataArray:
    """
    Create data simulating what a TemporalBuffer holds for a window output.

    Each buffered slice covers ``[start_time, time]``: ``time`` is the instant
    the window closed, one second after it opened. Both are 1-D, one value per
    slice. The two are deliberately distinct, so that a test cannot pass by
    reading the wrong one.
    """
    now_ns = time.time_ns()

    # Window close times: spanning the last `time_span_seconds`
    time_step_ns = int(time_span_seconds * 1e9 / num_points)
    time_values = [
        now_ns - (num_points - 1 - i) * time_step_ns for i in range(num_points)
    ]
    frame_duration_ns = int(1e9)
    start_time_values = [t - frame_duration_ns for t in time_values]

    return sc.DataArray(
        data=sc.array(dims=['time', 'x'], values=[[1.0, 2.0]] * num_points),
        coords={
            'time': sc.array(dims=['time'], values=time_values, unit='ns'),
            'x': sc.array(dims=['x'], values=[0.0, 1.0]),
            'start_time': sc.array(dims=['time'], values=start_time_values, unit='ns'),
        },
    )


class TestFullHistoryExtractor:
    """Tests for FullHistoryExtractor time coordinate handling."""

    def test_sets_start_time_from_min_of_1d_coord(self):
        """start_time should be min of the 1-D start_time coord."""
        data = _make_buffered_data(num_points=5, time_span_seconds=5.0)
        extractor = FullHistoryExtractor()

        result = extractor.extract(data)

        # Result should have scalar start_time = min of input 1-D start_time
        expected_start = data.coords['start_time'][0]
        assert 'start_time' in result.coords
        assert result.coords['start_time'].ndim == 0  # Scalar
        assert sc.identical(result.coords['start_time'], expected_start)

    def test_sets_end_time_from_last_of_1d_time_coord(self):
        """end_time is minted from the wire's upper bound, the 'time' coord."""
        data = _make_buffered_data(num_points=5, time_span_seconds=5.0)
        extractor = FullHistoryExtractor()

        result = extractor.extract(data)

        assert result.coords['end_time'].ndim == 0  # Scalar
        assert sc.identical(result.coords['end_time'], data.coords['time'][-1])

    def test_end_time_is_pristine_wall_clock_despite_the_display_shift(self):
        """The bounds are provenance, so the timezone shift must not reach them.

        ``time`` is shifted into the local timezone for Bokeh's axis; taking the
        bounds afterwards would misreport them by the UTC offset.
        """
        data = _make_buffered_data(num_points=5, time_span_seconds=5.0)

        result = FullHistoryExtractor().extract(data)

        assert sc.identical(result.coords['end_time'], data.coords['time'][-1])
        assert sc.identical(result.coords['start_time'], data.coords['start_time'][0])

    def test_time_axis_survives_as_the_1d_dim_coord(self):
        """Minting the scalar bounds must not overwrite the plotted x axis.

        ``assign_coords`` replaces a 1-D dim coord with a scalar silently, so a
        bound accidentally emitted under the name ``time`` would kill the axis
        with no error anywhere.
        """
        data = _make_buffered_data(num_points=5, time_span_seconds=5.0)

        result = FullHistoryExtractor().extract(data)

        assert result.coords['time'].ndim == 1
        assert result.coords['time'].sizes['time'] == 5

    def test_plotted_axis_is_the_window_close_not_its_start(self):
        """A window output's wall-clock x is its right edge.

        The axis is the wire's ``time``, offset into the local timezone for
        Bokeh and nothing else — in particular not the interval's ``start_time``.
        """
        data = _make_buffered_data(num_points=5, time_span_seconds=5.0)
        origin = sc.epoch(unit='ns') + get_local_timezone_offset_ns().to_scipp().to(
            unit='ns', dtype='int64'
        )

        result = FullHistoryExtractor().extract(data)

        assert sc.identical(result.coords['time'], origin + data.coords['time'])

    def test_end_time_reflects_actual_data_range(self):
        """end_time should reflect the actual data, giving recent lag."""
        now_ns = time.time_ns()
        data = _make_buffered_data(num_points=5, time_span_seconds=5.0)
        extractor = FullHistoryExtractor()

        result = extractor.extract(data)

        # end_time should be recent (within last few seconds)
        end_time_ns = result.coords['end_time'].value
        lag_seconds = (now_ns - end_time_ns) / 1e9
        assert lag_seconds < 5.0, f"end_time lag {lag_seconds}s should be ~0-1s"

    def test_handles_datetime64_time_coord(self):
        """Should work with datetime64 time coordinates and 1-D time bounds."""
        now_ns = time.time_ns()
        time_values = [now_ns - int(i * 1e9) for i in range(5, 0, -1)]
        time_var = sc.epoch(unit='ns') + sc.array(
            dims=['time'], values=time_values, unit='ns'
        )

        frame_duration_ns = int(1e9)
        start_time_values = [t - frame_duration_ns for t in time_values]

        data = sc.DataArray(
            data=sc.array(dims=['time', 'x'], values=[[1.0, 2.0]] * 5),
            coords={
                'time': time_var,
                'x': sc.array(dims=['x'], values=[0.0, 1.0]),
                'start_time': sc.array(
                    dims=['time'], values=start_time_values, unit='ns'
                ),
            },
        )
        extractor = FullHistoryExtractor()

        result = extractor.extract(data)

        assert 'start_time' in result.coords
        assert 'end_time' in result.coords
        assert result.coords['start_time'].ndim == 0
        assert result.coords['end_time'].ndim == 0


class TestWindowAggregatingExtractor:
    """Tests for WindowAggregatingExtractor time coordinate handling."""

    def test_sets_time_coords_from_actual_window(self):
        """Time coords should reflect the actual window that was aggregated."""
        now_ns = time.time_ns()
        data = _make_buffered_data(num_points=10, time_span_seconds=10.0)
        # Window of 3 seconds should select roughly the last 3 data points
        extractor = WindowAggregatingExtractor(window_duration_seconds=3.0)

        result = extractor.extract(data)

        assert 'start_time' in result.coords
        assert 'end_time' in result.coords
        assert result.coords['start_time'].ndim == 0  # Scalar
        assert result.coords['end_time'].ndim == 0  # Scalar

        # end_time should be recent (within last second or so)
        end_time_ns = result.coords['end_time'].value
        lag_seconds = (now_ns - end_time_ns) / 1e9
        assert lag_seconds < 5.0, f"end_time lag {lag_seconds}s should be ~0s"

        # start_time should be roughly 3 seconds before end_time
        # Note: with 10 points over 10s (1s apart), a 3s window captures ~3 points,
        # so the span from first to last end_time is ~2-3s
        start_time_ns = result.coords['start_time'].value
        window_duration = (end_time_ns - start_time_ns) / 1e9
        assert 1.5 <= window_duration <= 5.0, (
            f"Window duration {window_duration}s should be ~2-3s"
        )

    def test_end_time_reflects_actual_window(self):
        """end_time should reflect the actual window, giving recent lag."""
        now_ns = time.time_ns()
        data = _make_buffered_data(num_points=10, time_span_seconds=10.0)
        extractor = WindowAggregatingExtractor(window_duration_seconds=3.0)

        result = extractor.extract(data)

        # end_time should be recent
        end_time_ns = result.coords['end_time'].value
        lag_seconds = (now_ns - end_time_ns) / 1e9
        assert lag_seconds < 5.0, f"end_time lag {lag_seconds}s should be ~0-1s"


class TestLatestValueExtractor:
    """Tests for LatestValueExtractor time coordinate handling."""

    def test_sets_time_coords_from_latest_slice(self):
        """Time coords should be extracted from the latest slice's 1-D coords."""
        now_ns = time.time_ns()
        data = _make_buffered_data(num_points=5, time_span_seconds=5.0)
        extractor = LatestValueExtractor()

        result = extractor.extract(data)

        # Should have scalar coords from the last slice
        assert 'start_time' in result.coords
        assert 'end_time' in result.coords
        assert result.coords['start_time'].ndim == 0
        assert result.coords['end_time'].ndim == 0

        # end_time should be recent
        end_time_ns = result.coords['end_time'].value
        lag_seconds = (now_ns - end_time_ns) / 1e9
        assert lag_seconds < 5.0, f"end_time lag {lag_seconds}s should be ~0s"

    def test_extracts_last_value_from_1d_coords(self):
        """Should extract the last slice's bounds, end_time minted from 'time'."""
        data = _make_buffered_data(num_points=5, time_span_seconds=5.0)
        extractor = LatestValueExtractor()

        result = extractor.extract(data)

        assert sc.identical(result.coords['start_time'], data.coords['start_time'][-1])
        assert sc.identical(result.coords['end_time'], data.coords['time'][-1])

    def test_mints_end_time_for_an_unbuffered_message(self):
        """A single message reaches this extractor via SingleValueBuffer.

        It carries the wire's ``(start_time, time)`` and no time dim, so the
        plotters would see no ``end_time`` at all without the boundary applying
        here too.
        """
        message = sc.DataArray(
            sc.scalar(5.0, unit='counts'),
            coords={
                'start_time': sc.scalar(1000, unit='ns'),
                'time': sc.scalar(3000, unit='ns'),
            },
        )

        result = LatestValueExtractor().extract(message)

        assert sc.identical(result.coords['start_time'], sc.scalar(1000, unit='ns'))
        assert sc.identical(result.coords['end_time'], sc.scalar(3000, unit='ns'))
