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


# Fixtures for common test data
@pytest.fixture
def single_slice_2element():
    """Create a single time slice with 2 x-elements."""
    return sc.DataArray(
        sc.array(dims=['x'], values=[1.0, 2.0], unit='counts'),
        coords={
            'x': sc.arange('x', 2, unit='m'),
            'time': sc.scalar(0.0, unit='s'),
        },
    )


@pytest.fixture
def thick_slice_2x2():
    """Create a thick slice with 2 time points and 2 x-elements."""
    return sc.DataArray(
        sc.array(dims=['time', 'x'], values=[[1.0, 2.0], [3.0, 4.0]], unit='counts'),
        coords={
            'x': sc.arange('x', 2, unit='m'),
            'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s'),
        },
    )


# Helper functions for creating test data
def make_single_slice(x_values, time_value, time_unit='s'):
    """Create a single time slice DataArray."""
    return sc.DataArray(
        sc.array(dims=['x'], values=x_values, unit='counts'),
        coords={
            'x': sc.arange('x', len(x_values), unit='m'),
            'time': sc.scalar(time_value, unit=time_unit),
        },
    )


def make_thick_slice(x_size, time_values, time_unit='s'):
    """Create a thick slice DataArray with multiple time points."""
    n_times = len(time_values)
    return sc.DataArray(
        sc.array(
            dims=['time', 'x'],
            values=[[float(i)] * x_size for i in range(n_times)],
            unit='counts',
        ),
        coords={
            'x': sc.arange('x', x_size, unit='m'),
            'time': sc.array(dims=['time'], values=time_values, unit=time_unit),
        },
    )


def assert_buffer_has_time_data(buffer, expected_size):
    """Assert buffer contains time-dimensioned data of expected size."""
    result = buffer.get()
    assert result is not None
    assert 'time' in result.dims
    assert result.sizes['time'] == expected_size


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

    def test_set_required_timespan_does_not_error(self):
        """Test that set_required_timespan can be called without error."""
        buffer = SingleValueBuffer()
        buffer.set_required_timespan(10.0)  # Should not raise

    def test_set_max_memory_does_not_error(self):
        """Test that set_max_memory can be called without error."""
        buffer = SingleValueBuffer()
        buffer.set_max_memory(1000)  # Should not raise

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

    @pytest.mark.parametrize(
        ('data_creator', 'expected_time_size'),
        [
            (lambda: make_single_slice([1.0, 2.0, 3.0], 0.0), 1),
            (lambda: make_thick_slice(2, [0.0, 1.0]), 2),
        ],
        ids=['single_slice', 'thick_slice'],
    )
    def test_add_data_creates_time_dimension(self, data_creator, expected_time_size):
        """Test that adding data creates buffer with time dimension."""
        buffer = TemporalBuffer()
        buffer.add(data_creator())
        assert_buffer_has_time_data(buffer, expected_time_size)

    def test_add_multiple_single_slices(self):
        """Test concatenating multiple single slices."""
        buffer = TemporalBuffer()

        for i in range(3):
            buffer.add(make_single_slice([float(i)] * 2, float(i)))

        assert_buffer_has_time_data(buffer, 3)

    def test_add_multiple_thick_slices(self):
        """Test concatenating multiple thick slices."""
        buffer = TemporalBuffer()
        buffer.add(make_thick_slice(2, [0.0, 1.0]))
        buffer.add(make_thick_slice(2, [2.0, 3.0]))
        assert_buffer_has_time_data(buffer, 4)

    def test_add_mixed_single_and_thick_slices(self):
        """Test concatenating mixed single and thick slices."""
        buffer = TemporalBuffer()
        buffer.add(make_single_slice([1.0, 2.0], 0.0))
        buffer.add(make_thick_slice(2, [1.0, 2.0]))
        assert_buffer_has_time_data(buffer, 3)

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
        buffer.add(make_single_slice([1.0, 2.0], 0.0))
        buffer.clear()
        assert buffer.get() is None

    def test_timespan_trimming_on_capacity_failure(self):
        """Test that old data is trimmed when capacity is reached."""
        buffer = TemporalBuffer()
        buffer.set_required_timespan(5.0)  # Keep last 5 seconds
        buffer.set_max_memory(100)  # Small memory limit to trigger trimming

        # Add data at t=0
        buffer.add(make_single_slice([1.0, 2.0], 0.0))
        initial_capacity = buffer._data_buffer.max_capacity

        # Fill buffer close to capacity with data at t=1, 2, 3, 4
        for t in range(1, initial_capacity):
            buffer.add(make_single_slice([float(t), float(t)], float(t)))

        # Add data at t=10 (outside timespan from t=0-4)
        # This should trigger trimming of old data
        buffer.add(make_single_slice([10.0, 10.0], 10.0))

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
            buffer.add(make_single_slice([float(t), float(t)], float(t)))

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
        buffer.add(make_single_slice([1.0, 2.0], 0.0))

        # Fill to capacity
        capacity = buffer._data_buffer.max_capacity
        for t in range(1, capacity):
            buffer.add(make_single_slice([float(t), float(t)], float(t)))

        # Add data far in future, all previous data should be dropped
        buffer.add(make_single_slice([99.0, 99.0], 100.0))

        result = buffer.get()
        # Only data >= 99.0 should remain (100 - 1.0 timespan)
        assert result.coords['time'].values[0] >= 99.0

    def test_capacity_exceeded_even_after_trimming_raises(self):
        """Test that ValueError is raised if data exceeds capacity even after trim."""
        buffer = TemporalBuffer()
        buffer.set_required_timespan(1.0)
        buffer.set_max_memory(20)  # Very small capacity (~ 1 element)

        # Add first data point
        buffer.add(make_single_slice([1.0, 2.0], 0.0))

        # Try to add thick slice that exceeds capacity
        large_data = make_thick_slice(2, list(range(10)))

        with pytest.raises(ValueError, match="exceeds buffer capacity even after"):
            buffer.add(large_data)

    def test_timespan_trimming_with_nanosecond_time_coords(self):
        """Test trimming works when time coordinates use nanoseconds.

        Regression test for https://github.com/scipp/esslivedata/issues/711
        where production detector data uses unit='ns' but the trim computation
        hardcoded unit='s', causing a UnitError.
        """
        buffer = TemporalBuffer()
        buffer.set_required_timespan(5.0)  # 5 seconds
        buffer.set_max_memory(100)  # Small to trigger trimming

        ns_per_s = int(1e9)
        # Add data at t=0 ns
        buffer.add(make_single_slice([1.0, 2.0], 0, time_unit='ns'))
        initial_capacity = buffer._data_buffer.max_capacity

        # Fill buffer to capacity
        for t in range(1, initial_capacity):
            buffer.add(make_single_slice([float(t)] * 2, t * ns_per_s, time_unit='ns'))

        # Add data at t=10s (in ns). Should trigger trimming and keep t >= 5s.
        buffer.add(make_single_slice([10.0, 10.0], 10 * ns_per_s, time_unit='ns'))

        result = buffer.get()
        assert result.coords['time'].values[0] >= 5 * ns_per_s

    def test_timespan_trimming_with_nanosecond_thick_slices(self):
        """Test trimming with nanosecond thick slices."""
        buffer = TemporalBuffer()
        buffer.set_required_timespan(2.0)  # 2 seconds
        buffer.set_max_memory(100)

        ns_per_s = int(1e9)
        # Add thick slice at t=0, 1s
        buffer.add(make_thick_slice(2, [0, 1 * ns_per_s], time_unit='ns'))
        initial_capacity = buffer._data_buffer.max_capacity

        # Fill to capacity
        t = 2
        while buffer._data_buffer.size < initial_capacity:
            buffer.add(make_single_slice([float(t)] * 2, t * ns_per_s, time_unit='ns'))
            t += 1

        # Trigger trimming with data far in the future
        buffer.add(make_single_slice([99.0, 99.0], 100 * ns_per_s, time_unit='ns'))

        result = buffer.get()
        assert result.coords['time'].values[0] >= 98 * ns_per_s

    def test_timespan_zero_trims_all_old_data_on_overflow(self):
        """Test that timespan=0.0 trims all data to make room for new data."""
        buffer = TemporalBuffer()
        buffer.set_required_timespan(0.0)  # Keep only latest value
        buffer.set_max_memory(100)  # Small memory limit to force overflow

        # Add first data point
        buffer.add(make_single_slice([1.0, 2.0], 0.0))
        initial_capacity = buffer._data_buffer.max_capacity

        # Fill buffer to capacity
        for t in range(1, initial_capacity):
            buffer.add(make_single_slice([float(t), float(t)], float(t)))

        # Buffer is now full, verify it has all data
        result = buffer.get()
        assert result.sizes['time'] == initial_capacity

        # Add one more data point - should trigger trimming
        # With timespan=0.0, should drop ALL old data to make room
        buffer.add(make_single_slice([999.0, 999.0], 999.0))

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

    def test_init_with_thick_slice(self, thick_slice_2x2):
        """Test initialization with thick slice (has concat_dim)."""
        data = thick_slice_2x2.data  # Extract the raw array
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

    def test_append_with_variances_preserves_variances_on_expansion(self):
        """Test that variances are preserved when buffer expands.

        Regression test for bug where expanding the buffer would lose variances
        because sc.empty() wasn't called with with_variances flag.
        """
        # Create data with variances
        data = sc.array(
            dims=['x'], values=[1.0, 2.0], variances=[0.1, 0.2], unit='counts'
        )
        buffer = VariableBuffer(data=data, max_capacity=100, concat_dim='time')

        # Verify initial data has variances
        assert buffer.get().variances is not None

        # Append enough single slices to trigger multiple buffer expansions
        # Initial capacity is 16, so we need > 16 appends
        for i in range(20):
            new_data = sc.array(
                dims=['x'],
                values=[float(i + 3), float(i + 4)],
                variances=[0.3 + i * 0.01, 0.4 + i * 0.01],
                unit='counts',
            )
            assert buffer.append(new_data), f"Failed to append slice {i}"

        # Verify final result still has variances after expansion
        result = buffer.get()
        assert result.variances is not None
        assert result.sizes['time'] == 21
        # Verify variances were preserved from at least one slice
        assert result.variances[0, 0] > 0
