# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for unified Buffer using TDD.

Tests Buffer against simple VariableBuffer implementation to verify
the storage logic is correct and agnostic to the underlying buffer type.
"""

import scipp as sc

from ess.livedata.dashboard.buffer import Buffer
from ess.livedata.dashboard.buffer_strategy import VariableBuffer


class TestBufferStorageWithVariableBuffer:
    """Test Buffer with simple Variable buffers."""

    def test_empty_buffer(self):
        """Test that empty buffer returns None."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        assert storage.get_all() is None

    def test_append_single_element(self):
        """Test appending a single element."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.array(dims=['time'], values=[42], dtype='int64')
        storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 1
        assert result.values[0] == 42

    def test_append_multiple_elements(self):
        """Test appending multiple elements."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data1 = sc.array(dims=['time'], values=[1, 2, 3], dtype='int64')
        data2 = sc.array(dims=['time'], values=[4, 5], dtype='int64')

        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 5
        assert list(result.values) == [1, 2, 3, 4, 5]

    def test_growth_phase_doubles_capacity(self):
        """Test that capacity doubles during growth phase."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=20, buffer_impl=buffer_impl, initial_capacity=2)

        # Add data progressively to trigger doubling
        for i in range(10):
            data = sc.array(dims=['time'], values=[i], dtype='int64')
            storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 10
        assert list(result.values) == list(range(10))

    def test_sliding_window_maintains_max_size(self):
        """Test that sliding window keeps only last max_size elements."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(
            max_size=5,
            buffer_impl=buffer_impl,
            initial_capacity=2,
            overallocation_factor=2.0,
        )

        # Add more than max_size
        for i in range(10):
            data = sc.array(dims=['time'], values=[i], dtype='int64')
            storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 5
        # Should keep last 5 elements: [5, 6, 7, 8, 9]
        assert list(result.values) == [5, 6, 7, 8, 9]

    def test_overallocation_factor_controls_capacity(self):
        """Test that overallocation_factor affects when shifting occurs."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(
            max_size=4,
            buffer_impl=buffer_impl,
            initial_capacity=2,
            overallocation_factor=3.0,  # Max capacity = 12
        )

        # Fill to 8 elements (< 12, so no shift yet)
        for i in range(8):
            data = sc.array(dims=['time'], values=[i], dtype='int64')
            storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 8  # No trimming yet

    def test_shift_on_overflow_no_regrow_cycles(self):
        """Test that shift doesn't trigger repeated regrow cycles."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(
            max_size=4,
            buffer_impl=buffer_impl,
            initial_capacity=2,
            overallocation_factor=2.0,
        )

        # Keep adding - should stabilize with shifts, not regrow each time
        for i in range(20):
            data = sc.array(dims=['time'], values=[i], dtype='int64')
            storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 4
        assert list(result.values) == [16, 17, 18, 19]

    def test_clear(self):
        """Test clearing storage."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.array(dims=['time'], values=[1, 2, 3], dtype='int64')
        storage.append(data)
        assert storage.get_all() is not None

        storage.clear()
        assert storage.get_all() is None

    def test_multidimensional_variable(self):
        """Test with multidimensional Variable."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # 2D data: time x x
        data1 = sc.array(dims=['time', 'x'], values=[[1, 2], [3, 4]], dtype='int64')
        data2 = sc.array(dims=['time', 'x'], values=[[5, 6]], dtype='int64')

        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert result.sizes['x'] == 2
        assert result.values[0, 0] == 1
        assert result.values[2, 1] == 6

    def test_0d_scalar_to_1d_timeseries(self):
        """Test stacking 0D scalars into 1D timeseries."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # Add 0D scalars (no dimensions)
        scalar1 = sc.scalar(42, dtype='int64')
        scalar2 = sc.scalar(43, dtype='int64')
        scalar3 = sc.scalar(44, dtype='int64')

        storage.append(scalar1)
        storage.append(scalar2)
        storage.append(scalar3)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert list(result.values) == [42, 43, 44]

    def test_1d_array_to_2d_stack(self):
        """Test stacking 1D arrays into 2D."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # Add 1D arrays (no time dimension)
        data1 = sc.array(dims=['x'], values=[1, 2, 3], dtype='int64')
        data2 = sc.array(dims=['x'], values=[4, 5, 6], dtype='int64')

        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 2
        assert result.sizes['x'] == 3
        assert list(result.values[0]) == [1, 2, 3]
        assert list(result.values[1]) == [4, 5, 6]

    def test_2d_images_to_3d_stack(self):
        """Test stacking 2D images into 3D."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # Add 2D images (no time dimension)
        image1 = sc.array(dims=['y', 'x'], values=[[1, 2], [3, 4]], dtype='int64')
        image2 = sc.array(dims=['y', 'x'], values=[[5, 6], [7, 8]], dtype='int64')

        storage.append(image1)
        storage.append(image2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 2
        assert result.sizes['y'] == 2
        assert result.sizes['x'] == 2
        assert result.values[0, 0, 0] == 1
        assert result.values[1, 1, 1] == 8


class TestBufferSingleValueMode:
    """Test Buffer with max_size=1 (single-value mode optimization)."""

    def test_single_value_mode_append_replaces(self):
        """Test that max_size=1 replaces value on each append."""
        from ess.livedata.dashboard.extractors import LatestValueExtractor

        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=1, buffer_impl=buffer_impl, initial_capacity=5)
        extractor = LatestValueExtractor()

        # First append
        data1 = sc.array(dims=['time'], values=[42], dtype='int64')
        storage.append(data1)

        result = extractor.extract(storage.get_all())
        assert result is not None
        assert result.value == 42

        # Second append should replace
        data2 = sc.array(dims=['time'], values=[99], dtype='int64')
        storage.append(data2)

        result = extractor.extract(storage.get_all())
        assert result is not None
        assert result.value == 99

    def test_single_value_mode_extracts_latest_from_batch(self):
        """Test that extractor extracts latest value from batched data in storage."""
        from ess.livedata.dashboard.extractors import LatestValueExtractor

        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=1, buffer_impl=buffer_impl, initial_capacity=5)
        extractor = LatestValueExtractor()

        # Append batch - extractor extracts last value
        data = sc.array(dims=['time'], values=[1, 2, 3, 4, 5], dtype='int64')
        storage.append(data)

        result = extractor.extract(storage.get_all())
        assert result is not None
        assert result.value == 5

    def test_single_value_mode_handles_scalar_data(self):
        """Test that max_size=1 handles 0D scalar data."""
        from ess.livedata.dashboard.extractors import LatestValueExtractor

        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=1, buffer_impl=buffer_impl, initial_capacity=5)
        extractor = LatestValueExtractor()

        # Append scalar (no time dimension)
        scalar = sc.scalar(42.0, dtype='float64')
        storage.append(scalar)

        result = extractor.extract(storage.get_all())
        assert result is not None
        assert result.value == 42.0

    def test_single_value_mode_clear(self):
        """Test clearing single-value mode."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=1, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.scalar(42, dtype='int64')
        storage.append(data)
        assert storage.get_all() is not None

        storage.clear()
        assert storage.get_all() is None


class TestBufferGetWindow:
    """Test Buffer.get_window() method."""

    def test_get_window_full(self):
        """Test get_window with size equal to buffer content."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.array(dims=['time'], values=[1, 2, 3, 4, 5], dtype='int64')
        storage.append(data)

        result = storage.get_window(size=5)
        assert result is not None
        assert result.sizes['time'] == 5
        assert list(result.values) == [1, 2, 3, 4, 5]

    def test_get_window_partial(self):
        """Test get_window with size smaller than buffer content."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.array(dims=['time'], values=[1, 2, 3, 4, 5], dtype='int64')
        storage.append(data)

        result = storage.get_window(size=3)
        assert result is not None
        assert result.sizes['time'] == 3
        # Should get last 3 elements
        assert list(result.values) == [3, 4, 5]

    def test_get_window_larger_than_content(self):
        """Test get_window with size larger than buffer content."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.array(dims=['time'], values=[1, 2, 3], dtype='int64')
        storage.append(data)

        result = storage.get_window(size=10)
        assert result is not None
        assert result.sizes['time'] == 3
        # Should return all available data
        assert list(result.values) == [1, 2, 3]

    def test_get_window_none_returns_all(self):
        """Test get_window(None) returns entire buffer."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.array(dims=['time'], values=[1, 2, 3, 4], dtype='int64')
        storage.append(data)

        result = storage.get_window(size=None)
        assert result is not None
        assert result.sizes['time'] == 4
        assert list(result.values) == [1, 2, 3, 4]

    def test_get_window_empty_buffer(self):
        """Test get_window on empty buffer."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        result = storage.get_window(size=5)
        assert result is None

    def test_get_window_single_value_mode(self):
        """Test get_window in single-value mode."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=1, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.scalar(42, dtype='int64')
        storage.append(data)

        result = storage.get_window(size=1)
        assert result is not None
        assert result.value == 42


class TestBufferGetLatest:
    """Test Buffer.get_latest() method."""

    def test_get_latest_from_buffer(self):
        """Test get_latest returns most recent value without concat dim."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.array(dims=['time'], values=[1, 2, 3, 4, 5], dtype='int64')
        storage.append(data)

        result = storage.get_latest()
        assert result is not None
        # Should be unwrapped (no time dimension)
        assert 'time' not in result.dims
        assert result.value == 5

    def test_get_latest_empty_buffer(self):
        """Test get_latest on empty buffer."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        result = storage.get_latest()
        assert result is None

    def test_get_latest_multidimensional(self):
        """Test get_latest with multidimensional data."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # Add 2D data: time x x
        data = sc.array(
            dims=['time', 'x'], values=[[1, 2], [3, 4], [5, 6]], dtype='int64'
        )
        storage.append(data)

        result = storage.get_latest()
        assert result is not None
        # Should have x dimension but not time
        assert 'time' not in result.dims
        assert 'x' in result.dims
        assert list(result.values) == [5, 6]

    def test_get_latest_single_value_mode(self):
        """Test get_latest in single-value mode."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=1, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.scalar(42, dtype='int64')
        storage.append(data)

        result = storage.get_latest()
        assert result is not None
        assert result.value == 42

    def test_get_latest_after_multiple_appends(self):
        """Test get_latest always returns most recent value."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data1 = sc.array(dims=['time'], values=[1, 2, 3], dtype='int64')
        storage.append(data1)

        data2 = sc.array(dims=['time'], values=[4, 5], dtype='int64')
        storage.append(data2)

        result = storage.get_latest()
        assert result is not None
        assert result.value == 5


class TestBufferSetMaxSize:
    """Test Buffer.set_max_size() method."""

    def test_set_max_size_grow(self):
        """Test growing max_size."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=5, buffer_impl=buffer_impl, initial_capacity=2)

        # Fill to max_size
        for i in range(10):
            data = sc.array(dims=['time'], values=[i], dtype='int64')
            storage.append(data)

        # Should have last 5
        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 5
        assert list(result.values) == [5, 6, 7, 8, 9]

        # Grow max_size
        storage.set_max_size(10)

        # Add more data
        for i in range(10, 15):
            data = sc.array(dims=['time'], values=[i], dtype='int64')
            storage.append(data)

        # Should now have last 10
        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 10
        assert list(result.values) == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    def test_set_max_size_no_shrink(self):
        """Test that set_max_size smaller than current is ignored."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.array(dims=['time'], values=[1, 2, 3, 4, 5], dtype='int64')
        storage.append(data)

        # Try to shrink - should be ignored
        storage.set_max_size(3)

        # Should still have all 5 elements
        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 5

    def test_set_max_size_transition_from_single_value_mode(self):
        """Test critical transition from max_size=1 to max_size>1."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=1, buffer_impl=buffer_impl, initial_capacity=5)

        # Append in single-value mode
        data = sc.scalar(42, dtype='int64')
        storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.value == 42

        # Transition to buffer mode
        storage.set_max_size(10)

        # Add more data
        data2 = sc.array(dims=['time'], values=[99, 100], dtype='int64')
        storage.append(data2)

        # Should have original value plus new data
        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert list(result.values) == [42, 99, 100]

    def test_set_max_size_transition_preserves_value(self):
        """Test that 1→N transition preserves the existing value correctly."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=1, buffer_impl=buffer_impl, initial_capacity=5)

        # Append scalar in single-value mode
        scalar = sc.scalar(7.5, dtype='float64')
        storage.append(scalar)

        # Transition to buffer mode
        storage.set_max_size(5)

        # Verify value is preserved
        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 1
        assert result.values[0] == 7.5

    def test_set_max_size_transition_from_empty_single_value(self):
        """Test transition from empty single-value mode."""
        buffer_impl = VariableBuffer(concat_dim='time')
        storage = Buffer(max_size=1, buffer_impl=buffer_impl, initial_capacity=5)

        # Don't append anything
        assert storage.get_all() is None

        # Transition to buffer mode
        storage.set_max_size(10)

        # Should still be empty
        assert storage.get_all() is None

        # Add data
        data = sc.array(dims=['time'], values=[1, 2, 3], dtype='int64')
        storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert list(result.values) == [1, 2, 3]
