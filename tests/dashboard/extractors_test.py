# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from unittest.mock import Mock

import pytest
import scipp as sc

from ess.livedata.dashboard.extractors import (
    FullHistoryExtractor,
    LatestValueExtractor,
    UpdateExtractor,
    WindowAggregatingExtractor,
    create_extractors_from_params,
)
from ess.livedata.dashboard.plot_params import (
    WindowAggregation,
    WindowMode,
    WindowParams,
)


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


class TestCreateExtractorsFromParams:
    """Tests for create_extractors_from_params factory function."""

    def test_fallback_to_latest_value_when_no_params(self):
        """Test fallback to LatestValueExtractor when no window params provided."""
        keys = ['key1', 'key2']

        extractors = create_extractors_from_params(keys=keys, window=None, spec=None)

        assert len(extractors) == 2
        assert all(isinstance(ext, LatestValueExtractor) for ext in extractors.values())
        assert set(extractors.keys()) == {'key1', 'key2'}

    def test_create_latest_value_extractors_with_window_mode_latest(self):
        """Test creation of LatestValueExtractor when window mode is 'latest'."""
        keys = ['key1']
        window = WindowParams(mode=WindowMode.latest)

        extractors = create_extractors_from_params(keys=keys, window=window, spec=None)

        assert len(extractors) == 1
        assert isinstance(extractors['key1'], LatestValueExtractor)

    def test_create_window_aggregating_extractors_with_window_mode_window(self):
        """Test creation of WindowAggregatingExtractor when window mode is 'window'."""
        keys = ['key1', 'key2']
        window = WindowParams(
            mode=WindowMode.window,
            window_duration_seconds=5.0,
            aggregation=WindowAggregation.nansum,
        )

        extractors = create_extractors_from_params(keys=keys, window=window, spec=None)

        assert len(extractors) == 2
        assert all(
            isinstance(ext, WindowAggregatingExtractor) for ext in extractors.values()
        )

        # Verify behavior through public interface
        extractor = extractors['key1']
        assert extractor.get_required_timespan() == 5.0

    def test_spec_required_extractor_overrides_window_params(self):
        """Test that plotter spec's required extractor overrides window params."""
        keys = ['key1', 'key2']
        window = WindowParams(mode=WindowMode.latest)

        # Create mock spec with required extractor
        spec = Mock()
        spec.data_requirements.required_extractor = FullHistoryExtractor

        extractors = create_extractors_from_params(keys=keys, window=window, spec=spec)

        # Should use FullHistoryExtractor despite window params
        assert len(extractors) == 2
        assert all(isinstance(ext, FullHistoryExtractor) for ext in extractors.values())

    def test_spec_with_no_required_extractor_uses_window_params(self):
        """Test that window params are used when spec has no required extractor."""
        keys = ['key1']
        window = WindowParams(mode=WindowMode.window, window_duration_seconds=3.0)

        # Create mock spec without required extractor
        spec = Mock()
        spec.data_requirements.required_extractor = None

        extractors = create_extractors_from_params(keys=keys, window=window, spec=spec)

        assert isinstance(extractors['key1'], WindowAggregatingExtractor)
        assert extractors['key1'].get_required_timespan() == 3.0

    def test_creates_extractors_for_all_keys(self):
        """Test that extractors are created for all provided keys."""
        keys = ['result1', 'result2', 'result3']
        window = WindowParams(mode=WindowMode.latest)

        extractors = create_extractors_from_params(keys=keys, window=window, spec=None)

        assert len(extractors) == 3
        assert set(extractors.keys()) == {'result1', 'result2', 'result3'}
        assert all(isinstance(ext, LatestValueExtractor) for ext in extractors.values())

    def test_empty_keys_returns_empty_dict(self):
        """Test that empty keys list returns empty extractors dict."""
        keys = []
        window = WindowParams(mode=WindowMode.latest)

        extractors = create_extractors_from_params(keys=keys, window=window, spec=None)

        assert extractors == {}

    def test_window_aggregation_parameters_passed_correctly(self):
        """Test that window aggregation parameters result in correct behavior."""
        keys = ['key1']
        window = WindowParams(
            mode=WindowMode.window,
            window_duration_seconds=10.5,
            aggregation=WindowAggregation.mean,
        )

        extractors = create_extractors_from_params(keys=keys, window=window, spec=None)

        extractor = extractors['key1']
        assert isinstance(extractor, WindowAggregatingExtractor)
        # Verify timespan through public interface
        assert extractor.get_required_timespan() == 10.5

        # Verify aggregation behavior by extracting data
        data = sc.DataArray(
            sc.array(dims=['time', 'x'], values=[[2, 4], [4, 6]], unit='m'),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )
        result = extractor.extract(data)
        # Mean of [2, 4] and [4, 6] = [3, 5], verifying mean aggregation was used
        assert sc.allclose(
            result.data, sc.array(dims=['x'], values=[3.0, 5.0], unit='m')
        )


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
