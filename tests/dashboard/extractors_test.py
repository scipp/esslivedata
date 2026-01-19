# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import pytest
import scipp as sc

from ess.livedata.dashboard.extractors import (
    FullHistoryExtractor,
    LatestValueExtractor,
    UpdateExtractor,
    WindowAggregatingExtractor,
)
from ess.livedata.dashboard.plot_params import WindowAggregation


class TestLatestValueExtractor:
    """Tests for LatestValueExtractor."""

    def test_get_required_timespan_returns_zero(self):
        """Latest value extractor requires zero history."""
        extractor = LatestValueExtractor()
        assert extractor.get_required_timespan() == 0.0

    def test_extract_latest_value_from_concatenated_data(self):
        """Extract the latest value from data with concat dimension."""
        extractor = LatestValueExtractor(concat_dim='time')

        # Create data with time dimension
        data = sc.DataArray(
            sc.array(
                dims=['time', 'x'], values=[[1, 2], [3, 4], [5, 6]], unit='counts'
            ),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0, 2.0], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Should extract only the last time slice
        assert 'time' not in result.dims
        assert sc.identical(result, data['time', -1])

    def test_extract_from_data_without_concat_dimension(self):
        """Extract from data that doesn't have concat dimension (single frame)."""
        extractor = LatestValueExtractor(concat_dim='time')

        # Create data without time dimension
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1, 2, 3], unit='counts'),
            coords={'x': sc.arange('x', 3, unit='m')},
        )

        result = extractor.extract(data)

        # Should return data as-is
        assert sc.identical(result, data)

    def test_extract_with_custom_concat_dim(self):
        """Test extraction with custom concat dimension name."""
        extractor = LatestValueExtractor(concat_dim='event')

        data = sc.DataArray(
            sc.array(dims=['event', 'x'], values=[[1, 2], [3, 4]], unit='counts'),
            coords={'event': sc.arange('event', 2)},
        )

        result = extractor.extract(data)

        assert 'event' not in result.dims
        assert sc.identical(result, data['event', -1])

    def test_extract_scalar_data(self):
        """Extract from scalar data."""
        extractor = LatestValueExtractor()
        data = sc.scalar(42.0, unit='counts')

        result = extractor.extract(data)

        assert sc.identical(result, data)


class TestFullHistoryExtractor:
    """Tests for FullHistoryExtractor."""

    def test_get_required_timespan_returns_infinity(self):
        """Full history extractor requires infinite timespan."""
        extractor = FullHistoryExtractor()
        assert extractor.get_required_timespan() == float('inf')

    def test_extract_returns_all_data(self):
        """Extract returns complete buffer history."""
        extractor = FullHistoryExtractor()

        data = sc.DataArray(
            sc.array(
                dims=['time', 'x'], values=[[1, 2], [3, 4], [5, 6]], unit='counts'
            ),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0, 2.0], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Should return all data unchanged
        assert sc.identical(result, data)

    def test_extract_with_multidimensional_data(self):
        """Extract with complex multidimensional data."""
        extractor = FullHistoryExtractor()

        data = sc.DataArray(
            sc.array(
                dims=['time', 'y', 'x'],
                values=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                unit='counts',
            ),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s'),
                'y': sc.arange('y', 2),
                'x': sc.arange('x', 2),
            },
        )

        result = extractor.extract(data)

        assert sc.identical(result, data)

    def test_extract_converts_int64_nanoseconds_to_datetime64(self):
        """Int64 nanosecond timestamps are converted to datetime64 for plotting."""
        extractor = FullHistoryExtractor()

        # Create data with int64 ns timestamps (as might come from Kafka)
        ns_epoch = int(1.733e18)  # ~Dec 2024
        data = sc.DataArray(
            sc.array(dims=['time'], values=[1.0, 2.0, 3.0], unit='K'),
            coords={
                'time': sc.array(
                    dims=['time'],
                    values=[ns_epoch, ns_epoch + int(1e9), ns_epoch + int(2e9)],
                    unit='ns',
                    dtype='int64',
                ),
            },
        )

        result = extractor.extract(data)

        # Time coordinate should be converted to datetime64
        assert result.coords['time'].dtype == sc.DType.datetime64
        # Data values should be unchanged
        assert sc.identical(result.data, data.data)

    def test_extract_applies_local_timezone_offset(self):
        """Datetime conversion applies local timezone offset for display."""
        from ess.livedata.dashboard.time_utils import get_local_timezone_offset_ns

        extractor = FullHistoryExtractor()

        # Create data with int64 ns timestamps
        ns_epoch = int(1.733e18)  # ~Dec 2024
        data = sc.DataArray(
            sc.array(dims=['time'], values=[1.0, 2.0], unit='K'),
            coords={
                'time': sc.array(
                    dims=['time'],
                    values=[ns_epoch, ns_epoch + int(1e9)],
                    unit='ns',
                    dtype='int64',
                ),
            },
        )

        result = extractor.extract(data)

        # The datetime should be shifted by the local timezone offset
        tz_offset_ns = get_local_timezone_offset_ns()
        expected_first = sc.epoch(unit='ns') + sc.scalar(
            ns_epoch + tz_offset_ns, unit='ns', dtype='int64'
        )
        assert result.coords['time'][0] == expected_first

    def test_extract_preserves_datetime64_time_coord(self):
        """Datetime64 time coordinates pass through unchanged."""
        extractor = FullHistoryExtractor()

        # Create data with datetime64 time coordinate
        epoch = sc.epoch(unit='ns')
        ns_epoch = int(1.733e18)
        time_coord = epoch + sc.array(
            dims=['time'],
            values=[ns_epoch, ns_epoch + int(1e9)],
            unit='ns',
            dtype='int64',
        )
        data = sc.DataArray(
            sc.array(dims=['time'], values=[1.0, 2.0], unit='K'),
            coords={'time': time_coord},
        )

        result = extractor.extract(data)

        assert sc.identical(result, data)

    def test_extract_with_custom_concat_dim(self):
        """Test extraction with custom concat dimension name."""
        extractor = FullHistoryExtractor(concat_dim='event')

        # Create data with int64 ns timestamps on 'event' dimension
        ns_epoch = int(1.733e18)
        data = sc.DataArray(
            sc.array(dims=['event'], values=[1.0, 2.0], unit='K'),
            coords={
                'event': sc.array(
                    dims=['event'],
                    values=[ns_epoch, ns_epoch + int(1e9)],
                    unit='ns',
                    dtype='int64',
                ),
            },
        )

        result = extractor.extract(data)

        # Event coordinate should be converted to datetime64
        assert result.coords['event'].dtype == sc.DType.datetime64


class TestWindowAggregatingExtractor:
    """Tests for WindowAggregatingExtractor."""

    def test_get_required_timespan(self):
        """Test that get_required_timespan returns the window duration."""
        extractor = WindowAggregatingExtractor(window_duration_seconds=5.0)
        assert extractor.get_required_timespan() == 5.0

    def test_extract_window_and_aggregate_with_nansum(self):
        """Extract a window of data and aggregate using nansum."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=1.5,
            aggregation=WindowAggregation.nansum,
            concat_dim='time',
        )

        # Create data spanning 3 seconds
        data = sc.DataArray(
            sc.array(
                dims=['time', 'x'], values=[[1, 2], [3, 4], [5, 6]], unit='counts'
            ),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0, 2.0], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Window is [2.0 - 1.5, 2.0] = [0.5, 2.0], should include times 1.0 and 2.0
        assert 'time' not in result.dims
        # nansum of [3, 4] and [5, 6] = [8, 10]
        expected = sc.DataArray(
            sc.array(dims=['x'], values=[8, 10], unit='counts'),
            coords={'x': sc.arange('x', 2, unit='m')},
        )
        assert sc.allclose(result.data, expected.data)

    def test_extract_window_and_aggregate_with_nanmean(self):
        """Extract a window of data and aggregate using nanmean."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=1.5,
            aggregation=WindowAggregation.nanmean,
            concat_dim='time',
        )

        data = sc.DataArray(
            sc.array(
                dims=['time', 'x'], values=[[1, 2], [3, 4], [5, 6]], unit='counts'
            ),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0, 2.0], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Window includes times 1.0 and 2.0
        # nanmean of [3, 4] and [5, 6] = [4, 5]
        assert sc.allclose(
            result.data, sc.array(dims=['x'], values=[4.0, 5.0], unit='counts')
        )

    def test_extract_window_and_aggregate_with_sum(self):
        """Extract a window of data and aggregate using sum."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=2.0,
            aggregation=WindowAggregation.sum,
            concat_dim='time',
        )

        data = sc.DataArray(
            sc.array(dims=['time', 'x'], values=[[1, 2], [3, 4]], unit='counts'),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Window includes all data
        # sum of [1, 2] and [3, 4] = [4, 6]
        assert sc.allclose(
            result.data, sc.array(dims=['x'], values=[4, 6], unit='counts')
        )

    def test_extract_window_and_aggregate_with_mean(self):
        """Extract a window of data and aggregate using mean."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=2.0,
            aggregation=WindowAggregation.mean,
            concat_dim='time',
        )

        data = sc.DataArray(
            sc.array(dims=['time', 'x'], values=[[2, 4], [4, 6]], unit='m'),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # mean of [2, 4] and [4, 6] = [3, 5]
        assert sc.allclose(
            result.data, sc.array(dims=['x'], values=[3.0, 5.0], unit='m')
        )

    def test_auto_aggregation_with_counts_uses_nansum(self):
        """Test that auto aggregation uses nansum for counts unit."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=2.0,
            aggregation=WindowAggregation.auto,
            concat_dim='time',
        )

        data = sc.DataArray(
            sc.array(dims=['time', 'x'], values=[[1, 2], [3, 4]], unit='counts'),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Should use nansum for counts: [1, 2] + [3, 4] = [4, 6]
        assert sc.allclose(
            result.data, sc.array(dims=['x'], values=[4, 6], unit='counts')
        )

    def test_auto_aggregation_with_non_counts_uses_nanmean(self):
        """Test that auto aggregation uses nanmean for non-counts unit."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=2.0,
            aggregation=WindowAggregation.auto,
            concat_dim='time',
        )

        data = sc.DataArray(
            sc.array(dims=['time', 'x'], values=[[2, 4], [4, 6]], unit='m'),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Should use nanmean for non-counts: mean([2, 4], [4, 6]) = [3, 5]
        assert sc.allclose(
            result.data, sc.array(dims=['x'], values=[3.0, 5.0], unit='m')
        )

    def test_extract_is_consistent_across_calls(self):
        """Test that extraction produces consistent results across multiple calls."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=1.0,
            aggregation=WindowAggregation.nansum,
        )

        data = sc.DataArray(
            sc.array(dims=['time', 'x'], values=[[1, 2]], unit='counts'),
            coords={'time': sc.array(dims=['time'], values=[0.0], unit='s')},
        )

        # Extract twice and verify results are identical
        result1 = extractor.extract(data)
        result2 = extractor.extract(data)

        assert sc.identical(result1, result2)

    def test_extract_with_different_time_units(self):
        """Test extraction with time in milliseconds."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=1.5,  # 1.5 seconds = 1500 ms
            aggregation=WindowAggregation.nansum,
            concat_dim='time',
        )

        # Create data with time in milliseconds
        data = sc.DataArray(
            sc.array(
                dims=['time', 'x'], values=[[1, 2], [3, 4], [5, 6]], unit='counts'
            ),
            coords={
                'time': sc.array(dims=['time'], values=[0, 1000, 2000], unit='ms'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Window is [2000 - 1500, 2000] = [500, 2000] ms
        # Should include times 1000 and 2000
        assert sc.allclose(
            result.data, sc.array(dims=['x'], values=[8, 10], unit='counts')
        )

    def test_extract_with_custom_concat_dim(self):
        """Test extraction with custom concat dimension name."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=0.5,
            aggregation=WindowAggregation.nansum,
            concat_dim='event',
        )

        data = sc.DataArray(
            sc.array(dims=['event', 'x'], values=[[1, 2], [3, 4]], unit='counts'),
            coords={
                'event': sc.array(dims=['event'], values=[0.0, 0.3], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Window is [0.3 - 0.5, 0.3] but bounded by data, includes all
        assert sc.allclose(
            result.data, sc.array(dims=['x'], values=[4, 6], unit='counts')
        )

    def test_all_aggregation_methods_work(self):
        """Test that all valid aggregation methods complete without error."""
        data = sc.DataArray(
            sc.array(dims=['time', 'x'], values=[[1, 2], [3, 4]], unit='counts'),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        # Test all aggregation methods
        for agg in [
            WindowAggregation.sum,
            WindowAggregation.nansum,
            WindowAggregation.mean,
            WindowAggregation.nanmean,
            WindowAggregation.auto,
        ]:
            extractor = WindowAggregatingExtractor(
                window_duration_seconds=2.0, aggregation=agg, concat_dim='time'
            )
            result = extractor.extract(data)
            # Verify extraction succeeds and time dimension is removed
            assert 'time' not in result.dims

    def test_extract_narrow_window(self):
        """Test extraction with very narrow window (may include only last point)."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=0.1,
            aggregation=WindowAggregation.nansum,
            concat_dim='time',
        )

        data = sc.DataArray(
            sc.array(
                dims=['time', 'x'], values=[[1, 2], [3, 4], [5, 6]], unit='counts'
            ),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0, 2.0], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Window is [2.0 - 0.1, 2.0] = [1.9, 2.0], should only include time 2.0
        assert sc.allclose(
            result.data, sc.array(dims=['x'], values=[5, 6], unit='counts')
        )

    def test_handles_timing_jitter_at_window_start(self):
        """Test that timing noise near window boundary doesn't include extra frames."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=5.0,
            aggregation=WindowAggregation.nansum,
            concat_dim='time',
        )

        # Regular 1 Hz data with timing jitter on first frame
        # Conceptually frames at t=[0, 1, 2, 3, 4, 5] but first has noise
        data = sc.DataArray(
            sc.array(
                dims=['time', 'x'],
                values=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
                unit='counts',
            ),
            coords={
                'time': sc.array(
                    dims=['time'], values=[0.0001, 1.0, 2.0, 3.0, 4.0, 5.0], unit='s'
                ),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Window (5-5, 5] = (0, 5] excludes frame at 0.0001 (using exclusive bound)
        # Should include 5 frames [1, 2, 3, 4, 5], not all 6
        expected_sum = sc.array(dims=['x'], values=[35, 40], unit='counts')
        assert sc.allclose(result.data, expected_sum)

    def test_handles_timing_jitter_at_window_end(self):
        """Test that timing noise on latest frame doesn't affect frame count."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=5.0,
            aggregation=WindowAggregation.nansum,
            concat_dim='time',
        )

        # Regular 1 Hz data with timing jitter on last frame
        data = sc.DataArray(
            sc.array(
                dims=['time', 'x'],
                values=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
                unit='counts',
            ),
            coords={
                'time': sc.array(
                    dims=['time'], values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0001], unit='s'
                ),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Window (5.0001-5, 5.0001] = (0.0001, 5.0001]
        # Should include 5 frames [1, 2, 3, 4, 5.0001]
        expected_sum = sc.array(dims=['x'], values=[35, 40], unit='counts')
        assert sc.allclose(result.data, expected_sum)

    def test_consistent_frame_count_with_perfect_timing(self):
        """Test baseline: perfect timing gives expected frame count."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=5.0,
            aggregation=WindowAggregation.nansum,
            concat_dim='time',
        )

        # Perfect 1 Hz data at exactly [0, 1, 2, 3, 4, 5]
        data = sc.DataArray(
            sc.array(
                dims=['time', 'x'],
                values=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
                unit='counts',
            ),
            coords={
                'time': sc.array(
                    dims=['time'], values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], unit='s'
                ),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Window (0, 5] excludes frame at exactly 0 (exclusive bound)
        # Should include 5 frames [1, 2, 3, 4, 5]
        expected_sum = sc.array(dims=['x'], values=[35, 40], unit='counts')
        assert sc.allclose(result.data, expected_sum)

    def test_extract_with_datetime64_time_coordinate(self):
        """Test extraction when time coordinate is datetime64 instead of float."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=2.0,
            aggregation=WindowAggregation.nansum,
            concat_dim='time',
        )

        # Create data with datetime64 time coordinate
        import numpy as np

        base_time = np.datetime64('2025-01-15T10:00:00', 'ns')
        times = base_time + np.array([0, 1, 2, 3], dtype='timedelta64[s]')

        data = sc.DataArray(
            sc.array(
                dims=['time', 'x'],
                values=[[1, 2], [3, 4], [5, 6], [7, 8]],
                unit='counts',
            ),
            coords={
                'time': sc.array(dims=['time'], values=times, unit='ns'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        result = extractor.extract(data)

        # Window cutoff is latest - 2s + 0.5*median_interval = 3s - 2s + 0.5s = 1.5s
        # Should include times >= 1.5s: times at 2s and 3s
        # nansum of [5, 6] and [7, 8] = [12, 14]
        expected_sum = sc.array(dims=['x'], values=[12, 14], unit='counts')
        assert sc.allclose(result.data, expected_sum)

    def test_extract_with_datetime64_single_frame(self):
        """Test extraction with datetime64 time coordinate and single frame."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=1.0,
            aggregation=WindowAggregation.nansum,
            concat_dim='time',
        )

        # Single frame with datetime64
        import numpy as np

        data = sc.DataArray(
            sc.array(dims=['time', 'x'], values=[[5, 6]], unit='counts'),
            coords={
                'time': sc.array(
                    dims=['time'],
                    values=[np.datetime64('2025-01-15T10:00:00', 'ns')],
                    unit='ns',
                ),
                'x': sc.arange('x', 2, unit='m'),
            },
        )

        # This exercises the single-frame path:
        # cutoff_time = latest_time - self._duration
        result = extractor.extract(data)

        # Should return the single frame
        expected = sc.array(dims=['x'], values=[5, 6], unit='counts')
        assert sc.allclose(result.data, expected)


class TestUpdateExtractorInterface:
    """Tests for UpdateExtractor abstract interface."""

    def test_update_extractor_is_abstract(self):
        """Test that UpdateExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            UpdateExtractor()  # type: ignore[abstract]

    def test_concrete_extractors_implement_interface(self):
        """Test that all concrete extractors implement the UpdateExtractor interface."""
        extractors = [
            LatestValueExtractor(),
            FullHistoryExtractor(),
            WindowAggregatingExtractor(window_duration_seconds=1.0),
        ]

        for extractor in extractors:
            assert isinstance(extractor, UpdateExtractor)
            # Check that required methods are implemented
            assert hasattr(extractor, 'extract')
            assert hasattr(extractor, 'get_required_timespan')
            assert callable(extractor.extract)
            assert callable(extractor.get_required_timespan)
