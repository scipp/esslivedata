# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import pytest
import scipp as sc

from ess.livedata.dashboard.temporal_buffers import SingleValueBuffer, TemporalBuffer


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
