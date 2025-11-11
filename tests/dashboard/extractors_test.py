# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import pytest
import scipp as sc

from ess.livedata.dashboard.buffer_strategy import BufferFactory
from ess.livedata.dashboard.extractors import (
    FullHistoryExtractor,
    LatestValueExtractor,
    WindowAggregatingExtractor,
    WindowExtractor,
)


@pytest.fixture
def buffer_factory() -> BufferFactory:
    """Create a buffer factory for testing."""
    return BufferFactory()


class TestLatestValueExtractor:
    """Tests for LatestValueExtractor."""

    def test_extract_latest_scalar(self, buffer_factory: BufferFactory):
        """Test extracting latest value from scalar data."""
        extractor = LatestValueExtractor()
        buffer = buffer_factory.create_buffer(10, max_size=1)
        buffer.append(10)
        buffer.append(20)
        buffer.append(30)

        result = extractor.extract(buffer)
        assert result == 30

    def test_extract_latest_from_list(self, buffer_factory: BufferFactory):
        """Test extracting latest value from list buffer with batched data."""
        extractor = LatestValueExtractor()
        buffer = buffer_factory.create_buffer([1, 2, 3], max_size=1)
        buffer.append([1, 2, 3])
        buffer.append([4, 5, 6])

        result = extractor.extract(buffer)
        # For list buffers in single_value_mode with batched data,
        # extract_latest_frame extracts the last element from the batch
        assert result == 6

    def test_extract_latest_from_scipp_dataarray(self, buffer_factory: BufferFactory):
        """Test extracting and unwrapping latest value from scipp DataArray."""
        extractor = LatestValueExtractor()
        data1 = sc.DataArray(
            sc.arange('time', 3, unit='counts'),
            coords={'time': sc.arange('time', 3, unit='s')},
        )
        buffer = buffer_factory.create_buffer(data1, max_size=3)
        buffer.append(data1)

        # Add second value
        data2 = sc.DataArray(
            sc.arange('time', 3, 6, unit='counts'),
            coords={'time': sc.arange('time', 3, 6, unit='s')},
        )
        buffer.append(data2)

        result = extractor.extract(buffer)

        # Result should be unwrapped (scalar, no time dimension)
        assert result.ndim == 0
        assert result.value == 5  # Last value from second append

    def test_get_required_size(self):
        """Test that LatestValueExtractor requires size 1."""
        extractor = LatestValueExtractor()
        assert extractor.get_required_size() == 1

    def test_extract_empty_buffer_returns_none(self, buffer_factory: BufferFactory):
        """Test that extracting from empty buffer returns None."""
        extractor = LatestValueExtractor()
        buffer = buffer_factory.create_buffer(10, max_size=1)

        result = extractor.extract(buffer)
        assert result is None


class TestWindowExtractor:
    """Tests for WindowExtractor."""

    def test_window_size_property(self):
        """Test window_size property."""
        extractor = WindowExtractor(5)
        assert extractor.window_size == 5

    def test_get_required_size(self):
        """Test that WindowExtractor requires size equal to window size."""
        extractor = WindowExtractor(10)
        assert extractor.get_required_size() == 10

    def test_extract_window_from_list(self, buffer_factory: BufferFactory):
        """Test extracting window from list buffer."""
        extractor = WindowExtractor(2)
        buffer = buffer_factory.create_buffer(0, max_size=2)
        buffer.append(10)
        buffer.append(20)
        buffer.append(30)

        result = extractor.extract(buffer)
        assert result == [20, 30]

    def test_extract_window_from_scipp(self, buffer_factory: BufferFactory):
        """Test extracting window from scipp buffer."""
        extractor = WindowExtractor(3)
        data = sc.arange('time', 5, unit='counts')

        buffer = buffer_factory.create_buffer(data[0:1], max_size=3)
        for i in range(5):
            buffer.append(data[i : i + 1])

        result = extractor.extract(buffer)
        assert result.sizes['time'] == 3

    def test_extract_window_larger_than_buffer(self, buffer_factory: BufferFactory):
        """Test extracting window larger than current buffer contents."""
        extractor = WindowExtractor(10)
        buffer = buffer_factory.create_buffer(0, max_size=10)
        buffer.append(10)
        buffer.append(20)

        result = extractor.extract(buffer)
        # Should still work, returning available data
        assert len(result) == 2

    def test_different_window_sizes(self, buffer_factory: BufferFactory):
        """Test extractors with different window sizes."""
        buffer = buffer_factory.create_buffer(0, max_size=10)
        for i in range(10):
            buffer.append(i)

        # Extract window of 3
        extractor3 = WindowExtractor(3)
        result3 = extractor3.extract(buffer)
        assert result3 == [7, 8, 9]

        # Extract window of 5
        extractor5 = WindowExtractor(5)
        result5 = extractor5.extract(buffer)
        assert result5 == [5, 6, 7, 8, 9]


class TestFullHistoryExtractor:
    """Tests for FullHistoryExtractor."""

    def test_get_required_size(self):
        """Test that FullHistoryExtractor requires large buffer size."""
        extractor = FullHistoryExtractor()
        assert extractor.get_required_size() == 10000

    def test_extract_all_data(self, buffer_factory: BufferFactory):
        """Test extracting all data from buffer."""
        extractor = FullHistoryExtractor()
        buffer = buffer_factory.create_buffer(0, max_size=10000)

        values = [10, 20, 30, 40, 50]
        for val in values:
            buffer.append(val)

        result = extractor.extract(buffer)
        assert result == values

    def test_extract_all_from_scipp(self, buffer_factory: BufferFactory):
        """Test extracting all scipp data."""
        extractor = FullHistoryExtractor()
        data = sc.arange('time', 5, unit='counts')

        buffer = buffer_factory.create_buffer(data[0:1], max_size=10000)
        for i in range(5):
            buffer.append(data[i : i + 1])

        result = extractor.extract(buffer)
        assert result.sizes['time'] == 5

    def test_default_max_size(self):
        """Test default max size constant."""
        assert FullHistoryExtractor.DEFAULT_MAX_SIZE == 10000


class TestExtractorIntegration:
    """Integration tests for extractors with different data types."""

    def test_multiple_extractors_same_buffer(self, buffer_factory: BufferFactory):
        """Test using multiple extractors on the same buffer."""
        buffer = buffer_factory.create_buffer(0, max_size=10)

        values = list(range(10))
        for val in values:
            buffer.append(val)

        latest = LatestValueExtractor()
        window = WindowExtractor(3)
        history = FullHistoryExtractor()

        assert latest.extract(buffer) == 9
        assert window.extract(buffer) == [7, 8, 9]
        assert history.extract(buffer) == values

    def test_extractors_with_custom_concat_dim(self, buffer_factory: BufferFactory):
        """Test LatestValueExtractor with custom concat dimension."""
        # The buffer uses 'time' as the concat dimension internally
        # The extractor delegates unwrapping to the buffer implementation
        extractor = LatestValueExtractor()
        data = sc.arange('time', 3, unit='counts')

        buffer = buffer_factory.create_buffer(data[0:1], max_size=3)
        buffer.append(data[0:1])
        buffer.append(data[1:2])
        buffer.append(data[2:3])

        result = extractor.extract(buffer)
        # Should unwrap 'time' dimension and return scalar
        assert result.ndim == 0

    def test_extractor_with_non_concat_data(self, buffer_factory: BufferFactory):
        """Test extractor with data that doesn't have concat dimension."""
        extractor = LatestValueExtractor()
        # Create data without 'time' dimension
        data = sc.scalar(42, unit='counts')

        buffer = buffer_factory.create_buffer(data, max_size=1)
        buffer.append(data)

        result = extractor.extract(buffer)
        # Result should be the scalar value
        assert isinstance(result, sc.Variable) or result == data


class TestWindowAggregatingExtractor:
    """Tests for WindowAggregatingExtractor."""

    def test_get_required_size(self):
        """Test that WindowAggregatingExtractor estimates required buffer size."""
        # 1.0 second at 20 Hz headroom = 20 frames minimum
        extractor = WindowAggregatingExtractor(window_duration_seconds=1.0)
        assert extractor.get_required_size() == max(100, int(1.0 * 20))

    def test_sum_aggregation_scipp(self, buffer_factory: BufferFactory):
        """Test sum aggregation over time dimension."""
        # Duration that gives exactly 3 frames at 14 Hz: 3/14 ≈ 0.214 seconds
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=3 / 14, aggregation='sum'
        )

        # Create 2D data with time dimension
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.arange('x', 3, unit='m')},
        )

        buffer = buffer_factory.create_buffer(data, max_size=3)
        buffer.append(data)
        buffer.append(data * 2)
        buffer.append(data * 3)

        result = extractor.extract(buffer)

        # Result should be summed over time (no time dimension)
        assert 'time' not in result.dims
        # Sum: [1,2,3] + [2,4,6] + [3,6,9] = [6,12,18]
        assert sc.allclose(result.data, sc.array(dims=['x'], values=[6.0, 12.0, 18.0]))

    def test_mean_aggregation_scipp(self, buffer_factory: BufferFactory):
        """Test mean aggregation over time dimension."""
        # Duration that gives exactly 3 frames at 14 Hz
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=3 / 14, aggregation='mean'
        )

        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.arange('x', 3, unit='m')},
        )

        buffer = buffer_factory.create_buffer(data, max_size=3)
        buffer.append(data)
        buffer.append(data * 2)
        buffer.append(data * 4)

        result = extractor.extract(buffer)

        # Mean: ([1,2,3] + [2,4,6] + [4,8,12]) / 3 = [7,14,21] / 3
        expected = sc.array(dims=['x'], values=[7.0 / 3, 14.0 / 3, 21.0 / 3])
        assert sc.allclose(result.data, expected)

    def test_last_aggregation_scipp(self, buffer_factory: BufferFactory):
        """Test last aggregation (returns last frame)."""
        # Duration that gives exactly 3 frames at 14 Hz
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=3 / 14, aggregation='last'
        )

        data1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.arange('x', 3, unit='m')},
        )
        data2 = sc.DataArray(
            sc.array(dims=['x'], values=[4.0, 5.0, 6.0], unit='counts'),
            coords={'x': sc.arange('x', 3, unit='m')},
        )

        buffer = buffer_factory.create_buffer(data1, max_size=3)
        buffer.append(data1)
        buffer.append(data2)

        result = extractor.extract(buffer)

        # Should return the last frame
        assert 'time' not in result.dims
        assert sc.allclose(result.data, sc.array(dims=['x'], values=[4.0, 5.0, 6.0]))

    def test_max_aggregation_scipp(self, buffer_factory: BufferFactory):
        """Test max aggregation over time dimension."""
        # Duration that gives exactly 3 frames at 14 Hz
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=3 / 14, aggregation='max'
        )

        data1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 5.0, 2.0], unit='counts'),
            coords={'x': sc.arange('x', 3, unit='m')},
        )
        data2 = sc.DataArray(
            sc.array(dims=['x'], values=[3.0, 2.0, 4.0], unit='counts'),
            coords={'x': sc.arange('x', 3, unit='m')},
        )

        buffer = buffer_factory.create_buffer(data1, max_size=3)
        buffer.append(data1)
        buffer.append(data2)

        result = extractor.extract(buffer)

        # Max of [1,5,2] and [3,2,4] = [3,5,4]
        assert sc.allclose(result.data, sc.array(dims=['x'], values=[3.0, 5.0, 4.0]))

    def test_extract_empty_buffer_returns_none(self, buffer_factory: BufferFactory):
        """Test that extracting from empty buffer returns None."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=3 / 14, aggregation='sum'
        )
        buffer = buffer_factory.create_buffer(sc.scalar(1.0), max_size=3)

        result = extractor.extract(buffer)
        assert result is None

    def test_extract_non_scipp_data_returns_as_is(self, buffer_factory: BufferFactory):
        """Test that non-scipp data without dims is returned as-is."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=3 / 14, aggregation='sum'
        )
        buffer = buffer_factory.create_buffer(42, max_size=3)
        buffer.append(42)

        result = extractor.extract(buffer)
        # Should return the raw data since it doesn't have dims
        assert result == [42]

    def test_invalid_aggregation_raises_error(self, buffer_factory: BufferFactory):
        """Test that invalid aggregation method raises error."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=2 / 14, aggregation='invalid'
        )

        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0], unit='counts'),
        )
        buffer = buffer_factory.create_buffer(data, max_size=2)
        buffer.append(data)

        with pytest.raises(ValueError, match="Unknown aggregation method"):
            extractor.extract(buffer)
