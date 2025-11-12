# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for DataArrayBuffer using TDD.

Tests DataArrayBuffer implementation against BufferInterface protocol to verify
it correctly handles DataArray's complexity (coords, masks).
"""

import scipp as sc

from ess.livedata.dashboard.buffer import Buffer
from ess.livedata.dashboard.buffer_strategy import DataArrayBuffer


class TestDataArrayBuffer:
    """Test DataArrayBuffer implementation."""

    def test_empty_buffer(self):
        """Test that empty buffer returns None."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        assert storage.get_all() is None

    def test_append_single_element(self):
        """Test appending a single element."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[42], dtype='int64'),
            coords={'time': sc.array(dims=['time'], values=[0], dtype='int64')},
        )
        storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 1
        assert result.data.values[0] == 42
        assert result.coords['time'].values[0] == 0

    def test_append_multiple_elements(self):
        """Test appending multiple elements."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data1 = sc.DataArray(
            data=sc.array(dims=['time'], values=[1, 2, 3], dtype='int64'),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2], dtype='int64')},
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['time'], values=[4, 5], dtype='int64'),
            coords={'time': sc.array(dims=['time'], values=[3, 4], dtype='int64')},
        )

        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 5
        assert list(result.data.values) == [1, 2, 3, 4, 5]
        assert list(result.coords['time'].values) == [0, 1, 2, 3, 4]

    def test_non_concat_coord_preserved(self):
        """Test that non-concat-dimension coordinates are preserved."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # 2D data with x coordinate that doesn't depend on time
        data1 = sc.DataArray(
            data=sc.array(dims=['time', 'x'], values=[[1, 2], [3, 4]], dtype='int64'),
            coords={
                'time': sc.array(dims=['time'], values=[0, 1], dtype='int64'),
                'x': sc.array(dims=['x'], values=[10, 20], dtype='int64'),
            },
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['time', 'x'], values=[[5, 6]], dtype='int64'),
            coords={
                'time': sc.array(dims=['time'], values=[2], dtype='int64'),
                'x': sc.array(dims=['x'], values=[10, 20], dtype='int64'),
            },
        )

        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert result.sizes['x'] == 2
        assert list(result.coords['x'].values) == [10, 20]
        assert result.data.values[0, 0] == 1
        assert result.data.values[2, 1] == 6

    def test_concat_dependent_coord_handled(self):
        """Test coordinates that depend on concat dimension."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # Data with a coordinate that varies along time
        data1 = sc.DataArray(
            data=sc.array(dims=['time'], values=[1, 2], dtype='int64'),
            coords={
                'time': sc.array(dims=['time'], values=[0, 1], dtype='int64'),
                'temperature': sc.array(
                    dims=['time'], values=[273.0, 274.0], dtype='float64'
                ),
            },
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['time'], values=[3], dtype='int64'),
            coords={
                'time': sc.array(dims=['time'], values=[2], dtype='int64'),
                'temperature': sc.array(dims=['time'], values=[275.0], dtype='float64'),
            },
        )

        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert list(result.coords['time'].values) == [0, 1, 2]
        assert list(result.coords['temperature'].values) == [273.0, 274.0, 275.0]

    def test_masks_preserved(self):
        """Test that masks are preserved."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data1 = sc.DataArray(
            data=sc.array(dims=['time'], values=[1, 2], dtype='int64'),
            coords={'time': sc.array(dims=['time'], values=[0, 1], dtype='int64')},
            masks={'bad': sc.array(dims=['time'], values=[False, True], dtype=bool)},
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['time'], values=[3], dtype='int64'),
            coords={'time': sc.array(dims=['time'], values=[2], dtype='int64')},
            masks={'bad': sc.array(dims=['time'], values=[False], dtype=bool)},
        )

        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert 'bad' in result.masks
        assert list(result.masks['bad'].values) == [False, True, False]

    def test_sliding_window_maintains_max_size(self):
        """Test that sliding window keeps only last max_size elements."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(
            max_size=5,
            buffer_impl=buffer_impl,
            initial_capacity=2,
            overallocation_factor=2.0,
        )

        # Add more than max_size
        for i in range(10):
            data = sc.DataArray(
                data=sc.array(dims=['time'], values=[i], dtype='int64'),
                coords={'time': sc.array(dims=['time'], values=[i], dtype='int64')},
            )
            storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 5
        # Should keep last 5 elements: [5, 6, 7, 8, 9]
        assert list(result.data.values) == [5, 6, 7, 8, 9]
        assert list(result.coords['time'].values) == [5, 6, 7, 8, 9]

    def test_multidimensional_data(self):
        """Test with multidimensional DataArray."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # 2D data: time x x
        data1 = sc.DataArray(
            data=sc.array(dims=['time', 'x'], values=[[1, 2], [3, 4]], dtype='int64'),
            coords={
                'time': sc.array(dims=['time'], values=[0, 1], dtype='int64'),
                'x': sc.array(dims=['x'], values=[10, 20], dtype='int64'),
            },
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['time', 'x'], values=[[5, 6]], dtype='int64'),
            coords={
                'time': sc.array(dims=['time'], values=[2], dtype='int64'),
                'x': sc.array(dims=['x'], values=[10, 20], dtype='int64'),
            },
        )

        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert result.sizes['x'] == 2
        assert result.data.values[0, 0] == 1
        assert result.data.values[2, 1] == 6

    def test_clear(self):
        """Test clearing storage."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1, 2, 3], dtype='int64'),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2], dtype='int64')},
        )
        storage.append(data)
        assert storage.get_all() is not None

        storage.clear()
        assert storage.get_all() is None

    def test_growth_phase_doubles_capacity(self):
        """Test that capacity doubles during growth phase."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=20, buffer_impl=buffer_impl, initial_capacity=2)

        # Add data progressively to trigger doubling
        for i in range(10):
            data = sc.DataArray(
                data=sc.array(dims=['time'], values=[i], dtype='int64'),
                coords={'time': sc.array(dims=['time'], values=[i], dtype='int64')},
            )
            storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 10
        assert list(result.data.values) == list(range(10))

    def test_2d_coordinate_along_time_and_x(self):
        """Test 2D coordinate that depends on both time and x."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # Data with 2D coordinate (time, x)
        data1 = sc.DataArray(
            data=sc.array(dims=['time', 'x'], values=[[1, 2], [3, 4]], dtype='int64'),
            coords={
                'time': sc.array(dims=['time'], values=[0, 1], dtype='int64'),
                'x': sc.array(dims=['x'], values=[10, 20], dtype='int64'),
                'detector_id': sc.array(
                    dims=['time', 'x'], values=[[100, 101], [102, 103]], dtype='int64'
                ),
            },
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['time', 'x'], values=[[5, 6]], dtype='int64'),
            coords={
                'time': sc.array(dims=['time'], values=[2], dtype='int64'),
                'x': sc.array(dims=['x'], values=[10, 20], dtype='int64'),
                'detector_id': sc.array(
                    dims=['time', 'x'], values=[[104, 105]], dtype='int64'
                ),
            },
        )

        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert result.sizes['x'] == 2
        assert result.coords['detector_id'].values[0, 0] == 100
        assert result.coords['detector_id'].values[2, 1] == 105

    def test_get_size(self):
        """Test get_size method."""
        buffer_impl = DataArrayBuffer(concat_dim='time')

        data = sc.DataArray(
            data=sc.array(dims=['time', 'x'], values=[[1, 2], [3, 4]], dtype='int64'),
            coords={'time': sc.array(dims=['time'], values=[0, 1], dtype='int64')},
        )

        assert buffer_impl.get_size(data) == 2

    def test_multiple_masks(self):
        """Test handling multiple masks."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data1 = sc.DataArray(
            data=sc.array(dims=['time'], values=[1, 2], dtype='int64'),
            coords={'time': sc.array(dims=['time'], values=[0, 1], dtype='int64')},
            masks={
                'bad': sc.array(dims=['time'], values=[False, True], dtype=bool),
                'saturated': sc.array(dims=['time'], values=[True, False], dtype=bool),
            },
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['time'], values=[3], dtype='int64'),
            coords={'time': sc.array(dims=['time'], values=[2], dtype='int64')},
            masks={
                'bad': sc.array(dims=['time'], values=[False], dtype=bool),
                'saturated': sc.array(dims=['time'], values=[False], dtype=bool),
            },
        )

        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert 'bad' in result.masks
        assert 'saturated' in result.masks
        assert list(result.masks['bad'].values) == [False, True, False]
        assert list(result.masks['saturated'].values) == [True, False, False]

    def test_empty_dataarray_appends(self):
        """Test appending DataArrays with zero elements."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # Start with a non-empty append
        data1 = sc.DataArray(
            data=sc.array(dims=['time'], values=[1, 2], dtype='int64'),
            coords={'time': sc.array(dims=['time'], values=[0, 1], dtype='int64')},
        )
        storage.append(data1)

        # Append empty array (edge case)
        data2 = sc.DataArray(
            data=sc.array(dims=['time'], values=[], dtype='int64'),
            coords={'time': sc.array(dims=['time'], values=[], dtype='int64')},
        )
        storage.append(data2)

        # Append more data
        data3 = sc.DataArray(
            data=sc.array(dims=['time'], values=[3], dtype='int64'),
            coords={'time': sc.array(dims=['time'], values=[2], dtype='int64')},
        )
        storage.append(data3)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert list(result.data.values) == [1, 2, 3]

    def test_shift_on_overflow_preserves_coords_and_masks(self):
        """Test that shift preserves coordinates and masks correctly."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(
            max_size=3,
            buffer_impl=buffer_impl,
            initial_capacity=2,
            overallocation_factor=2.0,
        )

        # Add data with coords and masks
        for i in range(6):
            data = sc.DataArray(
                data=sc.array(dims=['time'], values=[i * 10], dtype='int64'),
                coords={
                    'time': sc.array(dims=['time'], values=[i], dtype='int64'),
                    'temp': sc.array(dims=['time'], values=[i * 1.5], dtype='float64'),
                },
                masks={'bad': sc.array(dims=['time'], values=[i % 2 == 0], dtype=bool)},
            )
            storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        # Should have last 3 elements
        assert list(result.data.values) == [30, 40, 50]
        assert list(result.coords['time'].values) == [3, 4, 5]
        assert list(result.coords['temp'].values) == [4.5, 6.0, 7.5]
        assert list(result.masks['bad'].values) == [False, True, False]

    def test_allocate_with_different_concat_dim(self):
        """Test buffer with non-default concat dimension."""
        buffer_impl = DataArrayBuffer(concat_dim='event')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        data1 = sc.DataArray(
            data=sc.array(dims=['event', 'x'], values=[[1, 2], [3, 4]], dtype='int64'),
            coords={
                'event': sc.array(dims=['event'], values=[0, 1], dtype='int64'),
                'x': sc.array(dims=['x'], values=[10, 20], dtype='int64'),
            },
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['event', 'x'], values=[[5, 6]], dtype='int64'),
            coords={
                'event': sc.array(dims=['event'], values=[2], dtype='int64'),
                'x': sc.array(dims=['x'], values=[10, 20], dtype='int64'),
            },
        )

        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['event'] == 3
        assert result.sizes['x'] == 2
        assert list(result.data.values.flatten()) == [1, 2, 3, 4, 5, 6]

    def test_0d_scalar_to_1d_timeseries(self):
        """Test stacking 0D scalar DataArrays into 1D timeseries."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # Add 0D scalars (no dimensions)
        scalar1 = sc.DataArray(sc.scalar(42.0, dtype='float64'))
        scalar2 = sc.DataArray(sc.scalar(43.0, dtype='float64'))
        scalar3 = sc.DataArray(sc.scalar(44.0, dtype='float64'))

        storage.append(scalar1)
        storage.append(scalar2)
        storage.append(scalar3)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert list(result.data.values) == [42.0, 43.0, 44.0]

    def test_1d_array_to_2d_stack(self):
        """Test stacking 1D DataArrays into 2D."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # Add 1D arrays (no time dimension)
        data1 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1, 2, 3], dtype='int64'),
            coords={'x': sc.array(dims=['x'], values=[10, 20, 30], dtype='int64')},
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['x'], values=[4, 5, 6], dtype='int64'),
            coords={'x': sc.array(dims=['x'], values=[10, 20, 30], dtype='int64')},
        )

        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 2
        assert result.sizes['x'] == 3
        assert list(result.data.values[0]) == [1, 2, 3]
        assert list(result.data.values[1]) == [4, 5, 6]
        assert list(result.coords['x'].values) == [10, 20, 30]

    def test_2d_images_to_3d_stack(self):
        """Test stacking 2D image DataArrays into 3D."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # Add 2D images (no time dimension)
        image1 = sc.DataArray(
            data=sc.array(dims=['y', 'x'], values=[[1, 2], [3, 4]], dtype='int64'),
            coords={
                'y': sc.array(dims=['y'], values=[0, 1], dtype='int64'),
                'x': sc.array(dims=['x'], values=[0, 1], dtype='int64'),
            },
        )
        image2 = sc.DataArray(
            data=sc.array(dims=['y', 'x'], values=[[5, 6], [7, 8]], dtype='int64'),
            coords={
                'y': sc.array(dims=['y'], values=[0, 1], dtype='int64'),
                'x': sc.array(dims=['x'], values=[0, 1], dtype='int64'),
            },
        )

        storage.append(image1)
        storage.append(image2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 2
        assert result.sizes['y'] == 2
        assert result.sizes['x'] == 2
        assert result.data.values[0, 0, 0] == 1
        assert result.data.values[1, 1, 1] == 8

    def test_2d_images_to_3d_stack_with_masks(self):
        """Test stacking 2D images with masks into 3D."""
        buffer_impl = DataArrayBuffer(concat_dim='time')
        storage = Buffer(max_size=10, buffer_impl=buffer_impl, initial_capacity=5)

        # Add 2D images with masks
        image1 = sc.DataArray(
            data=sc.array(dims=['y', 'x'], values=[[1, 2], [3, 4]], dtype='int64'),
            coords={'x': sc.array(dims=['x'], values=[0, 1], dtype='int64')},
            masks={
                'bad': sc.array(
                    dims=['y', 'x'],
                    values=[[False, True], [False, False]],
                    dtype=bool,
                )
            },
        )
        image2 = sc.DataArray(
            data=sc.array(dims=['y', 'x'], values=[[5, 6], [7, 8]], dtype='int64'),
            coords={'x': sc.array(dims=['x'], values=[0, 1], dtype='int64')},
            masks={
                'bad': sc.array(
                    dims=['y', 'x'],
                    values=[[True, False], [False, False]],
                    dtype=bool,
                )
            },
        )

        storage.append(image1)
        storage.append(image2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 2
        assert result.sizes['y'] == 2
        assert result.sizes['x'] == 2
        assert result.masks['bad'].values[0, 0, 1]
        assert result.masks['bad'].values[1, 0, 0]
