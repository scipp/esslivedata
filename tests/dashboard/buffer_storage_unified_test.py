# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for unified Buffer using TDD.

Tests Buffer against simple VariableBuffer implementation to verify
the storage logic is correct and agnostic to the underlying buffer type.
"""

import scipp as sc

from ess.livedata.dashboard.buffer_strategy import Buffer, VariableBuffer


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
