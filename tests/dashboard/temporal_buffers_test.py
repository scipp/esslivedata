# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import pytest
import scipp as sc

from ess.livedata.dashboard.temporal_buffers import (
    SingleValueBuffer,
    TemporalBuffer,
    VariableBuffer,
)


class TestSingleValueBuffer:
    """Tests for SingleValueBuffer."""

    def test_add_and_get_scalar(self):
        """Test adding and retrieving a scalar value."""
        buffer = SingleValueBuffer()
        data = sc.scalar(42, unit='counts')

        buffer.add(data)
        result = buffer.get()

        assert result == data

    def test_add_replaces_previous_value(self):
        """Test that add replaces the previous value."""
        buffer = SingleValueBuffer()
        data1 = sc.scalar(10, unit='counts')
        data2 = sc.scalar(20, unit='counts')

        buffer.add(data1)
        buffer.add(data2)
        result = buffer.get()

        assert result == data2

    def test_get_empty_buffer_returns_none(self):
        """Test that get returns None for empty buffer."""
        buffer = SingleValueBuffer()
        assert buffer.get() is None

    def test_clear_removes_value(self):
        """Test that clear removes the stored value."""
        buffer = SingleValueBuffer()
        data = sc.scalar(42, unit='counts')

        buffer.add(data)
        buffer.clear()
        result = buffer.get()

        assert result is None

    def test_set_required_timespan(self):
        """
        Test that set_required_timespan can be called (no-op).
        """
        buffer = SingleValueBuffer()
        buffer.set_required_timespan(10.0)
        # No assertion - just verify it doesn't error

    def test_set_max_memory(self):
        """Test that set_max_memory can be called (no-op for SingleValueBuffer)."""
        buffer = SingleValueBuffer()
        buffer.set_max_memory(1000)
        # No assertion - just verify it doesn't error

    def test_add_dataarray_with_dimensions(self):
        """Test adding a DataArray with dimensions."""
        buffer = SingleValueBuffer()
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.arange('x', 3, unit='m')},
        )

        buffer.add(data)
        result = buffer.get()

        assert sc.identical(result, data)


class TestTemporalBuffer:
    """Tests for TemporalBuffer."""

    def test_add_single_slice_without_time_dim(self):
        """Test adding a single slice without time dimension."""
        buffer = TemporalBuffer()
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={
                'x': sc.arange('x', 3, unit='m'),
                'time': sc.scalar(0.0, unit='s'),
            },
        )

        buffer.add(data)
        result = buffer.get()

        assert result is not None
        assert 'time' in result.dims
        assert result.sizes['time'] == 1

    def test_add_thick_slice_with_time_dim(self):
        """Test adding a thick slice with time dimension."""
        buffer = TemporalBuffer()
        data = sc.DataArray(
            sc.array(
                dims=['time', 'x'], values=[[1.0, 2.0], [3.0, 4.0]], unit='counts'
            ),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s'),
            },
        )

        buffer.add(data)
        result = buffer.get()

        assert result is not None
        assert 'time' in result.dims
        assert result.sizes['time'] == 2

    def test_add_multiple_single_slices(self):
        """Test concatenating multiple single slices."""
        buffer = TemporalBuffer()

        for i in range(3):
            data = sc.DataArray(
                sc.array(dims=['x'], values=[float(i)] * 2, unit='counts'),
                coords={
                    'x': sc.arange('x', 2, unit='m'),
                    'time': sc.scalar(float(i), unit='s'),
                },
            )
            buffer.add(data)

        result = buffer.get()
        assert result is not None
        assert result.sizes['time'] == 3

    def test_add_multiple_thick_slices(self):
        """Test concatenating multiple thick slices."""
        buffer = TemporalBuffer()

        # Add first thick slice
        data1 = sc.DataArray(
            sc.array(
                dims=['time', 'x'], values=[[1.0, 2.0], [3.0, 4.0]], unit='counts'
            ),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s'),
            },
        )
        buffer.add(data1)

        # Add second thick slice
        data2 = sc.DataArray(
            sc.array(
                dims=['time', 'x'], values=[[5.0, 6.0], [7.0, 8.0]], unit='counts'
            ),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.array(dims=['time'], values=[2.0, 3.0], unit='s'),
            },
        )
        buffer.add(data2)

        result = buffer.get()
        assert result is not None
        assert result.sizes['time'] == 4

    def test_add_mixed_single_and_thick_slices(self):
        """Test concatenating mixed single and thick slices."""
        buffer = TemporalBuffer()

        # Add single slice
        data1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.scalar(0.0, unit='s'),
            },
        )
        buffer.add(data1)

        # Add thick slice
        data2 = sc.DataArray(
            sc.array(
                dims=['time', 'x'], values=[[3.0, 4.0], [5.0, 6.0]], unit='counts'
            ),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.array(dims=['time'], values=[1.0, 2.0], unit='s'),
            },
        )
        buffer.add(data2)

        result = buffer.get()
        assert result is not None
        assert result.sizes['time'] == 3

    def test_add_without_time_coord_raises_error(self):
        """Test that adding data without time coordinate raises ValueError."""
        buffer = TemporalBuffer()
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.arange('x', 3, unit='m')},
        )

        with pytest.raises(ValueError, match="requires data with 'time' coordinate"):
            buffer.add(data)

    def test_get_empty_buffer_returns_none(self):
        """Test that get returns None for empty buffer."""
        buffer = TemporalBuffer()
        assert buffer.get() is None

    def test_clear_removes_all_data(self):
        """Test that clear removes all buffered data."""
        buffer = TemporalBuffer()
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.scalar(0.0, unit='s'),
            },
        )

        buffer.add(data)
        buffer.clear()
        result = buffer.get()

        assert result is None

    def test_set_required_timespan(self):
        """Test that set_required_timespan stores the value."""
        buffer = TemporalBuffer()
        buffer.set_required_timespan(5.0)
        assert buffer._required_timespan == 5.0

    def test_set_max_memory(self):
        """Test that set_max_memory stores the value."""
        buffer = TemporalBuffer()
        buffer.set_max_memory(10000)
        assert buffer._max_memory == 10000

    def test_max_memory_limits_capacity(self):
        """Test that max_memory limits buffer capacity."""
        buffer = TemporalBuffer()
        # Set memory limit before adding data
        buffer.set_max_memory(100)  # 100 bytes

        # Add initial data (float64 = 8 bytes per element, 2 elements = 16 bytes)
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.scalar(0.0, unit='s'),
            },
        )
        buffer.add(data)

        # Buffer capacity should be limited by memory: 100 bytes / 16 bytes = 6
        assert buffer._data_buffer.max_capacity == 6

    def test_timespan_trimming_on_capacity_failure(self):
        """Test that old data is trimmed when capacity is reached."""
        buffer = TemporalBuffer()
        buffer.set_required_timespan(5.0)  # Keep last 5 seconds
        buffer.set_max_memory(100)  # Small memory limit to trigger trimming

        # Add data at t=0
        data1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.scalar(0.0, unit='s'),
            },
        )
        buffer.add(data1)
        initial_capacity = buffer._data_buffer.max_capacity

        # Fill buffer close to capacity with data at t=1, 2, 3, 4
        for t in range(1, initial_capacity):
            data = sc.DataArray(
                sc.array(dims=['x'], values=[float(t), float(t)], unit='counts'),
                coords={
                    'x': sc.arange('x', 2, unit='m'),
                    'time': sc.scalar(float(t), unit='s'),
                },
            )
            buffer.add(data)

        # Add data at t=10 (outside timespan from t=0-4)
        # This should trigger trimming of old data
        data_new = sc.DataArray(
            sc.array(dims=['x'], values=[10.0, 10.0], unit='counts'),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.scalar(10.0, unit='s'),
            },
        )
        buffer.add(data_new)

        result = buffer.get()
        # Only data from t >= 5.0 should remain (t=10 - 5.0)
        # So only t=10 should be in buffer (since we only added up to t=capacity-1)
        assert result.coords['time'].values[0] >= 5.0

    def test_no_trimming_when_capacity_available(self):
        """Test that trimming doesn't occur when there's available capacity."""
        buffer = TemporalBuffer()
        buffer.set_required_timespan(2.0)  # Keep last 2 seconds

        # Add data at t=0, 1, 2, 3, 4, 5
        for t in range(6):
            data = sc.DataArray(
                sc.array(dims=['x'], values=[float(t), float(t)], unit='counts'),
                coords={
                    'x': sc.arange('x', 2, unit='m'),
                    'time': sc.scalar(float(t), unit='s'),
                },
            )
            buffer.add(data)

        result = buffer.get()
        # With default large capacity (10000), no trimming should occur
        # All 6 time points should be present despite timespan=2.0
        assert result.sizes['time'] == 6
        assert result.coords['time'].values[0] == 0.0

    def test_trim_drops_all_old_data(self):
        """Test trimming when all buffered data is older than timespan."""
        buffer = TemporalBuffer()
        buffer.set_required_timespan(1.0)
        buffer.set_max_memory(50)  # Very small to trigger trim quickly

        # Add data at t=0
        data1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.scalar(0.0, unit='s'),
            },
        )
        buffer.add(data1)

        # Fill to capacity
        capacity = buffer._data_buffer.max_capacity
        for t in range(1, capacity):
            data = sc.DataArray(
                sc.array(dims=['x'], values=[float(t), float(t)], unit='counts'),
                coords={
                    'x': sc.arange('x', 2, unit='m'),
                    'time': sc.scalar(float(t), unit='s'),
                },
            )
            buffer.add(data)

        # Add data far in future, all previous data should be dropped
        data_future = sc.DataArray(
            sc.array(dims=['x'], values=[99.0, 99.0], unit='counts'),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.scalar(100.0, unit='s'),
            },
        )
        buffer.add(data_future)

        result = buffer.get()
        # Only data >= 99.0 should remain (100 - 1.0 timespan)
        assert result.coords['time'].values[0] >= 99.0

    def test_capacity_exceeded_even_after_trimming_raises(self):
        """Test that ValueError is raised if data exceeds capacity even after trim."""
        buffer = TemporalBuffer()
        buffer.set_required_timespan(1.0)
        buffer.set_max_memory(20)  # Very small capacity (~ 1 element)

        # Add first data point
        data1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.scalar(0.0, unit='s'),
            },
        )
        buffer.add(data1)

        # Try to add thick slice that exceeds capacity
        large_data = sc.DataArray(
            sc.array(
                dims=['time', 'x'],
                values=[[i, i] for i in range(10)],
                unit='counts',
            ),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.array(dims=['time'], values=list(range(10)), unit='s'),
            },
        )

        with pytest.raises(ValueError, match="exceeds buffer capacity even after"):
            buffer.add(large_data)

    def test_timespan_zero_trims_all_old_data_on_overflow(self):
        """Test that timespan=0.0 trims all data to make room for new data."""
        buffer = TemporalBuffer()
        buffer.set_required_timespan(0.0)  # Keep only latest value
        buffer.set_max_memory(100)  # Small memory limit to force overflow

        # Add first data point
        data1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.scalar(0.0, unit='s'),
            },
        )
        buffer.add(data1)
        initial_capacity = buffer._data_buffer.max_capacity

        # Fill buffer to capacity
        for t in range(1, initial_capacity):
            data = sc.DataArray(
                sc.array(dims=['x'], values=[float(t), float(t)], unit='counts'),
                coords={
                    'x': sc.arange('x', 2, unit='m'),
                    'time': sc.scalar(float(t), unit='s'),
                },
            )
            buffer.add(data)

        # Buffer is now full, verify it has all data
        result = buffer.get()
        assert result.sizes['time'] == initial_capacity

        # Add one more data point - should trigger trimming
        # With timespan=0.0, should drop ALL old data to make room
        data_new = sc.DataArray(
            sc.array(dims=['x'], values=[999.0, 999.0], unit='counts'),
            coords={
                'x': sc.arange('x', 2, unit='m'),
                'time': sc.scalar(999.0, unit='s'),
            },
        )
        buffer.add(data_new)  # Should not raise

        # Should only have the latest value
        result = buffer.get()
        assert result.sizes['time'] == 1
        assert result.coords['time'].values[0] == 999.0
        assert result['time', 0].values[0] == 999.0


class TestVariableBuffer:
    """Tests for VariableBuffer."""

    def test_init_with_single_slice(self):
        """Test initialization with single slice (no concat_dim)."""
        data = sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts')
        buffer = VariableBuffer(data=data, max_capacity=10, concat_dim='time')

        assert buffer.size == 1
        assert buffer.max_capacity == 10
        result = buffer.get()
        assert result.sizes['time'] == 1
        assert sc.identical(result['time', 0], data)

    def test_init_with_thick_slice(self):
        """Test initialization with thick slice (has concat_dim)."""
        data = sc.array(
            dims=['time', 'x'], values=[[1.0, 2.0], [3.0, 4.0]], unit='counts'
        )
        buffer = VariableBuffer(data=data, max_capacity=10, concat_dim='time')

        assert buffer.size == 2
        assert buffer.max_capacity == 10
        result = buffer.get()
        assert result.sizes['time'] == 2
        assert sc.identical(result, data)

    def test_dimension_ordering_single_slice_makes_concat_dim_outer(self):
        """Test that concat_dim becomes outer dimension for single slice."""
        # 2D image without time dimension
        image = sc.array(dims=['y', 'x'], values=[[1, 2], [3, 4]], unit='counts')
        buffer = VariableBuffer(data=image, max_capacity=10, concat_dim='time')

        # Buffer should have dims: time, y, x (time is outer)
        assert list(buffer._buffer.dims) == ['time', 'y', 'x']

    def test_dimension_ordering_thick_slice_preserves_order(self):
        """Test that existing dimension order is preserved for thick slice."""
        # Data with time already in the middle
        data = sc.array(
            dims=['y', 'time', 'x'], values=[[[1, 2], [3, 4]]], unit='counts'
        )
        buffer = VariableBuffer(data=data, max_capacity=10, concat_dim='time')

        # Buffer should preserve dimension order
        assert list(buffer._buffer.dims) == ['y', 'time', 'x']

    def test_append_single_slice(self):
        """Test appending single slices."""
        data = sc.array(dims=['x'], values=[1.0, 2.0], unit='counts')
        buffer = VariableBuffer(data=data, max_capacity=10, concat_dim='time')

        # Append more slices
        data2 = sc.array(dims=['x'], values=[3.0, 4.0], unit='counts')
        data3 = sc.array(dims=['x'], values=[5.0, 6.0], unit='counts')

        assert buffer.append(data2)
        assert buffer.append(data3)

        assert buffer.size == 3
        result = buffer.get()
        assert result.sizes['time'] == 3
        assert sc.identical(result['time', 0], data)
        assert sc.identical(result['time', 1], data2)
        assert sc.identical(result['time', 2], data3)

    def test_append_thick_slice(self):
        """Test appending thick slices."""
        data = sc.array(dims=['time', 'x'], values=[[1.0, 2.0]], unit='counts')
        buffer = VariableBuffer(data=data, max_capacity=10, concat_dim='time')

        # Append thick slice
        data2 = sc.array(
            dims=['time', 'x'], values=[[3.0, 4.0], [5.0, 6.0]], unit='counts'
        )
        assert buffer.append(data2)

        assert buffer.size == 3
        result = buffer.get()
        assert result.sizes['time'] == 3

    def test_capacity_expansion(self):
        """Test that buffer capacity expands as needed."""
        data = sc.array(dims=['x'], values=[1.0], unit='counts')
        buffer = VariableBuffer(data=data, max_capacity=100, concat_dim='time')

        initial_capacity = buffer.capacity
        assert initial_capacity == 16  # min(16, max_capacity)

        # Append until we exceed initial capacity
        for i in range(20):
            assert buffer.append(sc.array(dims=['x'], values=[float(i)], unit='counts'))

        # Capacity should have expanded
        assert buffer.capacity > initial_capacity
        assert buffer.size == 21

    def test_large_append_requires_multiple_expansions(self):
        """Test appending data much larger than current capacity (bug fix)."""
        data = sc.array(dims=['x'], values=[1.0], unit='counts')
        buffer = VariableBuffer(data=data, max_capacity=200, concat_dim='time')

        assert buffer.capacity == 16

        # Append 100 elements at once (requires multiple doublings: 16->32->64->128)
        large_data = sc.array(
            dims=['time', 'x'], values=[[float(i)] for i in range(100)], unit='counts'
        )
        assert buffer.append(large_data)

        assert buffer.size == 101
        assert buffer.capacity >= 101
        result = buffer.get()
        assert result.sizes['time'] == 101

    def test_append_exceeding_max_capacity_fails(self):
        """Test that append fails when exceeding max_capacity."""
        data = sc.array(dims=['x'], values=[1.0], unit='counts')
        buffer = VariableBuffer(data=data, max_capacity=5, concat_dim='time')

        # Fill to max_capacity
        for i in range(4):
            assert buffer.append(sc.array(dims=['x'], values=[float(i)], unit='counts'))

        assert buffer.size == 5

        # Next append should fail
        assert not buffer.append(sc.array(dims=['x'], values=[99.0], unit='counts'))
        assert buffer.size == 5  # Size unchanged

    def test_init_exceeding_max_capacity_raises(self):
        """Test that initialization with data exceeding max_capacity raises."""
        data = sc.array(dims=['time', 'x'], values=[[1.0], [2.0], [3.0]], unit='counts')

        with pytest.raises(ValueError, match="exceeds max_capacity"):
            VariableBuffer(data=data, max_capacity=2, concat_dim='time')

    def test_get_returns_valid_data_only(self):
        """Test that get returns only valid data, not full buffer capacity."""
        data = sc.array(dims=['x'], values=[1.0], unit='counts')
        buffer = VariableBuffer(data=data, max_capacity=100, concat_dim='time')

        # Capacity is 16, but size is 1
        assert buffer.capacity == 16
        assert buffer.size == 1

        result = buffer.get()
        assert result.sizes['time'] == 1  # Not 16

    def test_drop_from_start(self):
        """Test dropping data from the start."""
        data = sc.array(
            dims=['time', 'x'],
            values=[[1.0], [2.0], [3.0], [4.0], [5.0]],
            unit='counts',
        )
        buffer = VariableBuffer(data=data, max_capacity=10, concat_dim='time')

        # Drop first 2 elements
        buffer.drop(2)

        assert buffer.size == 3
        result = buffer.get()
        assert result.sizes['time'] == 3
        assert result.values[0, 0] == 3.0
        assert result.values[2, 0] == 5.0

    def test_drop_all(self):
        """Test dropping all data."""
        data = sc.array(dims=['time', 'x'], values=[[1.0], [2.0]], unit='counts')
        buffer = VariableBuffer(data=data, max_capacity=10, concat_dim='time')

        buffer.drop(5)  # Drop more than size

        assert buffer.size == 0

    def test_drop_zero_does_nothing(self):
        """Test that dropping zero elements does nothing."""
        data = sc.array(dims=['time', 'x'], values=[[1.0], [2.0]], unit='counts')
        buffer = VariableBuffer(data=data, max_capacity=10, concat_dim='time')

        buffer.drop(0)

        assert buffer.size == 2

    def test_drop_negative_does_nothing(self):
        """Test that dropping negative index does nothing."""
        data = sc.array(dims=['time', 'x'], values=[[1.0], [2.0]], unit='counts')
        buffer = VariableBuffer(data=data, max_capacity=10, concat_dim='time')

        buffer.drop(-1)

        assert buffer.size == 2

    def test_multidimensional_data(self):
        """Test with multidimensional data (images)."""
        # 2D image
        image1 = sc.array(dims=['y', 'x'], values=[[1, 2], [3, 4]], unit='counts')
        buffer = VariableBuffer(data=image1, max_capacity=10, concat_dim='time')

        image2 = sc.array(dims=['y', 'x'], values=[[5, 6], [7, 8]], unit='counts')
        buffer.append(image2)

        result = buffer.get()
        assert result.sizes == {'time': 2, 'y': 2, 'x': 2}
        assert result.values[0, 0, 0] == 1
        assert result.values[1, 1, 1] == 8

    def test_properties(self):
        """Test buffer properties."""
        data = sc.array(dims=['time', 'x'], values=[[1.0], [2.0]], unit='counts')
        buffer = VariableBuffer(data=data, max_capacity=50, concat_dim='time')

        assert buffer.size == 2
        assert buffer.max_capacity == 50
        assert buffer.capacity == 16  # Initial allocation

    def test_custom_concat_dim(self):
        """Test using a custom concat dimension."""
        data = sc.array(dims=['x'], values=[1.0, 2.0], unit='counts')
        buffer = VariableBuffer(data=data, max_capacity=10, concat_dim='event')

        assert buffer.size == 1
        result = buffer.get()
        assert 'event' in result.dims
        assert result.sizes['event'] == 1

    def test_scalar_to_1d(self):
        """Test stacking scalars into 1D array."""
        scalar = sc.scalar(42.0, unit='counts')
        buffer = VariableBuffer(data=scalar, max_capacity=10, concat_dim='time')

        buffer.append(sc.scalar(43.0, unit='counts'))
        buffer.append(sc.scalar(44.0, unit='counts'))

        result = buffer.get()
        assert result.sizes == {'time': 3}
        assert list(result.values) == [42.0, 43.0, 44.0]
