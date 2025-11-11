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
)
from ess.livedata.dashboard.temporal_requirements import (
    CompleteHistory,
    LatestFrame,
    TimeWindow,
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

    def test_get_temporal_requirement(self):
        """Test that LatestValueExtractor returns LatestFrame requirement."""
        extractor = LatestValueExtractor()
        requirement = extractor.get_temporal_requirement()
        assert isinstance(requirement, LatestFrame)

    def test_extract_empty_buffer_returns_none(self, buffer_factory: BufferFactory):
        """Test that extracting from empty buffer returns None."""
        extractor = LatestValueExtractor()
        buffer = buffer_factory.create_buffer(10, max_size=1)

        result = extractor.extract(buffer)
        assert result is None


class TestFullHistoryExtractor:
    """Tests for FullHistoryExtractor."""

    def test_get_temporal_requirement(self):
        """Test that FullHistoryExtractor returns CompleteHistory requirement."""
        extractor = FullHistoryExtractor()
        requirement = extractor.get_temporal_requirement()
        assert isinstance(requirement, CompleteHistory)

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

    def test_complete_history_max_frames(self):
        """Test CompleteHistory max frames constant."""
        assert CompleteHistory.MAX_FRAMES == 10000


class TestExtractorIntegration:
    """Integration tests for extractors with different data types."""

    def test_multiple_extractors_same_buffer(self, buffer_factory: BufferFactory):
        """Test using multiple extractors on the same buffer."""
        buffer = buffer_factory.create_buffer(0, max_size=10)

        values = list(range(10))
        for val in values:
            buffer.append(val)

        latest = LatestValueExtractor()
        history = FullHistoryExtractor()

        assert latest.extract(buffer) == 9
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

    def test_get_temporal_requirement(self):
        """Test that WindowAggregatingExtractor returns TimeWindow requirement."""
        extractor = WindowAggregatingExtractor(window_duration_seconds=5.0)
        requirement = extractor.get_temporal_requirement()
        assert isinstance(requirement, TimeWindow)
        assert requirement.duration_seconds == 5.0

    def test_sum_aggregation_scipp(self, buffer_factory: BufferFactory):
        """Test sum aggregation over time dimension."""
        # Create frames with realistic timestamps (spaced ~71ms apart at 14 Hz)
        t0 = 0  # Start at time=0
        dt_ns = int(1e9 / 14)  # ~71.4 ms in nanoseconds

        data1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={
                'x': sc.arange('x', 3, unit='m'),
                'time': sc.array(dims=[], values=t0, unit='ns', dtype='int64'),
            },
        )
        data2 = sc.DataArray(
            sc.array(dims=['x'], values=[2.0, 4.0, 6.0], unit='counts'),
            coords={
                'x': sc.arange('x', 3, unit='m'),
                'time': sc.array(dims=[], values=t0 + dt_ns, unit='ns', dtype='int64'),
            },
        )
        data3 = sc.DataArray(
            sc.array(dims=['x'], values=[3.0, 6.0, 9.0], unit='counts'),
            coords={
                'x': sc.arange('x', 3, unit='m'),
                'time': sc.array(
                    dims=[], values=t0 + 2 * dt_ns, unit='ns', dtype='int64'
                ),
            },
        )

        buffer = buffer_factory.create_buffer(data1, max_size=10)
        buffer.append(data1)
        buffer.append(data2)
        buffer.append(data3)

        # Extract window of 0.2 seconds (should get all 3 frames at 14 Hz)
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=0.2, aggregation='sum'
        )
        result = extractor.extract(buffer)

        # Result should be summed over time (no time dimension)
        assert 'time' not in result.dims
        # Sum: [1,2,3] + [2,4,6] + [3,6,9] = [6,12,18]
        assert sc.allclose(result.data, sc.array(dims=['x'], values=[6.0, 12.0, 18.0]))

    def test_mean_aggregation_scipp(self, buffer_factory: BufferFactory):
        """Test mean aggregation over time dimension."""
        t0 = 0
        dt_ns = int(1e9 / 14)

        data1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={
                'x': sc.arange('x', 3, unit='m'),
                'time': sc.array(dims=[], values=t0, unit='ns', dtype='int64'),
            },
        )
        data2 = sc.DataArray(
            sc.array(dims=['x'], values=[2.0, 4.0, 6.0], unit='counts'),
            coords={
                'x': sc.arange('x', 3, unit='m'),
                'time': sc.array(dims=[], values=t0 + dt_ns, unit='ns', dtype='int64'),
            },
        )
        data3 = sc.DataArray(
            sc.array(dims=['x'], values=[4.0, 8.0, 12.0], unit='counts'),
            coords={
                'x': sc.arange('x', 3, unit='m'),
                'time': sc.array(
                    dims=[], values=t0 + 2 * dt_ns, unit='ns', dtype='int64'
                ),
            },
        )

        buffer = buffer_factory.create_buffer(data1, max_size=10)
        buffer.append(data1)
        buffer.append(data2)
        buffer.append(data3)

        extractor = WindowAggregatingExtractor(
            window_duration_seconds=0.2, aggregation='mean'
        )
        result = extractor.extract(buffer)

        # Mean: ([1,2,3] + [2,4,6] + [4,8,12]) / 3 = [7,14,21] / 3
        expected = sc.array(dims=['x'], values=[7.0 / 3, 14.0 / 3, 21.0 / 3])
        assert sc.allclose(result.data, expected)

    def test_last_aggregation_scipp(self, buffer_factory: BufferFactory):
        """Test last aggregation (returns last frame)."""
        t0 = 0
        dt_ns = int(1e9 / 14)

        data1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={
                'x': sc.arange('x', 3, unit='m'),
                'time': sc.array(dims=[], values=t0, unit='ns', dtype='int64'),
            },
        )
        data2 = sc.DataArray(
            sc.array(dims=['x'], values=[4.0, 5.0, 6.0], unit='counts'),
            coords={
                'x': sc.arange('x', 3, unit='m'),
                'time': sc.array(dims=[], values=t0 + dt_ns, unit='ns', dtype='int64'),
            },
        )

        buffer = buffer_factory.create_buffer(data1, max_size=10)
        buffer.append(data1)
        buffer.append(data2)

        extractor = WindowAggregatingExtractor(
            window_duration_seconds=0.2, aggregation='last'
        )
        result = extractor.extract(buffer)

        # Should return the last frame
        assert 'time' not in result.dims
        assert sc.allclose(result.data, sc.array(dims=['x'], values=[4.0, 5.0, 6.0]))

    def test_max_aggregation_scipp(self, buffer_factory: BufferFactory):
        """Test max aggregation over time dimension."""
        t0 = 0
        dt_ns = int(1e9 / 14)

        data1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 5.0, 2.0], unit='counts'),
            coords={
                'x': sc.arange('x', 3, unit='m'),
                'time': sc.array(dims=[], values=t0, unit='ns', dtype='int64'),
            },
        )
        data2 = sc.DataArray(
            sc.array(dims=['x'], values=[3.0, 2.0, 4.0], unit='counts'),
            coords={
                'x': sc.arange('x', 3, unit='m'),
                'time': sc.array(dims=[], values=t0 + dt_ns, unit='ns', dtype='int64'),
            },
        )

        buffer = buffer_factory.create_buffer(data1, max_size=10)
        buffer.append(data1)
        buffer.append(data2)

        extractor = WindowAggregatingExtractor(
            window_duration_seconds=0.2, aggregation='max'
        )
        result = extractor.extract(buffer)

        # Max of [1,5,2] and [3,2,4] = [3,5,4]
        assert sc.allclose(result.data, sc.array(dims=['x'], values=[3.0, 5.0, 4.0]))

    def test_extract_empty_buffer_returns_none(self, buffer_factory: BufferFactory):
        """Test that extracting from empty buffer returns None."""
        data = sc.DataArray(
            sc.scalar(1.0, unit='counts'),
            coords={'time': sc.array(dims=[], values=0, unit='ns', dtype='int64')},
        )
        buffer = buffer_factory.create_buffer(data, max_size=10)

        extractor = WindowAggregatingExtractor(
            window_duration_seconds=0.2, aggregation='sum'
        )
        result = extractor.extract(buffer)
        assert result is None

    def test_extract_non_scipp_data_raises_error(self, buffer_factory: BufferFactory):
        """Test that non-scipp data raises NotImplementedError for window extraction."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=0.2, aggregation='sum'
        )
        buffer = buffer_factory.create_buffer(42, max_size=10)
        buffer.append(42)

        # ListBuffer doesn't support time-based windowing
        with pytest.raises(NotImplementedError, match="Time-based windowing"):
            extractor.extract(buffer)

    def test_invalid_aggregation_raises_error(self, buffer_factory: BufferFactory):
        """Test that invalid aggregation method raises error."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=0.2, aggregation='invalid'
        )

        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0], unit='counts'),
            coords={
                'x': sc.arange('x', 1, unit='m'),
                'time': sc.array(dims=[], values=0, unit='ns', dtype='int64'),
            },
        )
        buffer = buffer_factory.create_buffer(data, max_size=10)
        buffer.append(data)

        with pytest.raises(ValueError, match="Unknown aggregation method"):
            extractor.extract(buffer)

    def test_extract_without_time_coord_raises_error(
        self, buffer_factory: BufferFactory
    ):
        """Test that data without time coordinate raises error."""
        extractor = WindowAggregatingExtractor(
            window_duration_seconds=0.2, aggregation='sum'
        )

        # Data without time coordinate
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'),
            coords={'x': sc.arange('x', 2, unit='m')},
        )
        buffer = buffer_factory.create_buffer(data, max_size=10)
        buffer.append(data)

        with pytest.raises(ValueError, match="no 'time' coordinate"):
            extractor.extract(buffer)
