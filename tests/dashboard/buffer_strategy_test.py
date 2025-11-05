# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for buffer storage strategies.

Tests assume that non-concat-dimension coordinates are constant across all
appended data (only the concat dimension changes).
"""

import pytest
import scipp as sc

from ess.livedata.dashboard.buffer_strategy import (
    GrowingStorage,
    SlidingWindowStorage,
)


@pytest.fixture
def simple_batch1() -> sc.DataArray:
    """Create a simple 1D DataArray batch."""
    return sc.DataArray(
        data=sc.array(dims=['time'], values=[1, 2, 3]),
        coords={'time': sc.array(dims=['time'], values=[0, 1, 2])},
    )


@pytest.fixture
def simple_batch2() -> sc.DataArray:
    """Create a second batch with different time coords (constant non-concat coords)."""
    return sc.DataArray(
        data=sc.array(dims=['time'], values=[4, 5]),
        coords={'time': sc.array(dims=['time'], values=[3, 4])},
    )


@pytest.fixture
def multi_dim_batch1() -> sc.DataArray:
    """Create a 2D DataArray batch with time and x dimensions."""
    return sc.DataArray(
        data=sc.array(
            dims=['time', 'x'],
            values=[[1, 2], [3, 4]],
        ),
        coords={
            'time': sc.array(dims=['time'], values=[0, 1]),
            'x': sc.array(dims=['x'], values=[10, 20]),  # Constant across batches
        },
    )


@pytest.fixture
def multi_dim_batch2() -> sc.DataArray:
    """Create a second 2D batch (same x coords, different time)."""
    return sc.DataArray(
        data=sc.array(
            dims=['time', 'x'],
            values=[[5, 6], [7, 8]],
        ),
        coords={
            'time': sc.array(dims=['time'], values=[2, 3]),
            'x': sc.array(dims=['x'], values=[10, 20]),  # Same as batch1
        },
    )


@pytest.fixture
def data_with_mask1() -> sc.DataArray:
    """Create a DataArray with a mask."""
    data = sc.DataArray(
        data=sc.array(dims=['time'], values=[1, 2, 3]),
        coords={'time': sc.array(dims=['time'], values=[0, 1, 2])},
    )
    data.masks['bad'] = sc.array(dims=['time'], values=[False, True, False])
    return data


@pytest.fixture
def data_with_mask2() -> sc.DataArray:
    """Create a second DataArray with a mask (same structure, different time)."""
    data = sc.DataArray(
        data=sc.array(dims=['time'], values=[4, 5]),
        coords={'time': sc.array(dims=['time'], values=[3, 4])},
    )
    data.masks['bad'] = sc.array(dims=['time'], values=[True, False])
    return data


class TestSlidingWindowStorage:
    """Tests for SlidingWindowStorage."""

    def test_initialization(self):
        """Test storage initialization."""
        storage = SlidingWindowStorage(max_size=10)
        assert storage.get_all() is None
        assert storage.estimate_memory() == 0

    def test_append_single_batch(self, simple_batch1):
        """Test appending a single batch."""
        storage = SlidingWindowStorage(max_size=10)
        storage.append(simple_batch1)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert sc.allclose(result.coords['time'], simple_batch1.coords['time'])

    def test_append_multiple_batches(self, simple_batch1, simple_batch2):
        """Test appending multiple batches sequentially."""
        storage = SlidingWindowStorage(max_size=10)

        storage.append(simple_batch1)
        storage.append(simple_batch2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 5
        # Verify concatenation: first batch (3 elements) + second batch (2 elements)
        expected_times = sc.concat(
            [simple_batch1.coords['time'], simple_batch2.coords['time']], dim='time'
        )
        assert sc.allclose(result.coords['time'], expected_times)

    def test_sliding_window_trims_old_data(self, simple_batch1, simple_batch2):
        """Test that sliding window keeps only the most recent max_size elements."""
        storage = SlidingWindowStorage(max_size=3)

        # Append batch1 (3 elements) + batch2 (2 elements) = 5 total
        storage.append(simple_batch1)  # time: [0, 1, 2]
        storage.append(simple_batch2)  # time: [3, 4]

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3  # Only keeps last 3
        # Should keep time values [2, 3, 4] (the last 3 added)
        assert result.coords['time'].values[0] == 2
        assert result.coords['time'].values[1] == 3
        assert result.coords['time'].values[2] == 4

    def test_append_with_multiple_dimensions(self, multi_dim_batch1, multi_dim_batch2):
        """Test appending data with multiple dimensions."""
        storage = SlidingWindowStorage(max_size=10, concat_dim='time')
        storage.append(multi_dim_batch1)
        storage.append(multi_dim_batch2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 4
        assert result.sizes['x'] == 2
        # Verify x coordinate is preserved (constant across batches)
        assert sc.allclose(result.coords['x'], multi_dim_batch1.coords['x'])

    def test_append_with_mask(self, data_with_mask1, data_with_mask2):
        """Test appending data with masks."""
        storage = SlidingWindowStorage(max_size=10)
        storage.append(data_with_mask1)
        storage.append(data_with_mask2)

        result = storage.get_all()
        assert result is not None
        assert 'bad' in result.masks
        # Verify masks are concatenated correctly
        expected_mask = sc.concat(
            [data_with_mask1.masks['bad'], data_with_mask2.masks['bad']], dim='time'
        )
        assert sc.all(result.masks['bad'] == expected_mask).value

    def test_clear(self, simple_batch1):
        """Test clearing storage."""
        storage = SlidingWindowStorage(max_size=10)
        storage.append(simple_batch1)
        assert storage.get_all() is not None

        storage.clear()
        assert storage.get_all() is None
        assert storage.estimate_memory() == 0

    def test_memory_estimation(self, simple_batch1):
        """Test memory estimation."""
        storage = SlidingWindowStorage(max_size=10)
        initial_memory = storage.estimate_memory()
        assert initial_memory == 0

        storage.append(simple_batch1)
        memory_after_append = storage.estimate_memory()
        assert memory_after_append > 0

    def test_invalid_max_size(self):
        """Test that invalid max_size raises error."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            SlidingWindowStorage(max_size=0)

        with pytest.raises(ValueError, match="max_size must be positive"):
            SlidingWindowStorage(max_size=-1)

    def test_missing_concat_dimension(self):
        """Test that appending data without concat dimension raises error."""
        storage = SlidingWindowStorage(max_size=10, concat_dim='time')
        data = sc.DataArray(data=sc.array(dims=['x'], values=[1, 2, 3]))

        with pytest.raises(ValueError, match="Data must have 'time' dimension"):
            storage.append(data)

    def test_custom_concat_dimension(self):
        """Test using a custom concat dimension."""
        data1 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1, 2, 3]),
            coords={'x': sc.array(dims=['x'], values=[0, 1, 2])},
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['x'], values=[4, 5, 6]),
            coords={'x': sc.array(dims=['x'], values=[0, 1, 2])},
        )

        storage = SlidingWindowStorage(max_size=10, concat_dim='x')
        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['x'] == 6

    def test_window_trimming_multiple_small_batches(self):
        """Test that buffer correctly trims when adding many small batches."""
        storage = SlidingWindowStorage(max_size=4)

        # Add 6 single-element batches
        for i in range(6):
            data = sc.DataArray(
                data=sc.array(dims=['time'], values=[i * 10]),
                coords={'time': sc.array(dims=['time'], values=[i])},
            )
            storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 4
        # Should keep last 4 elements (indices 2, 3, 4, 5)
        assert sc.allclose(
            result.coords['time'],
            sc.array(dims=['time'], values=[2, 3, 4, 5]),
        )

    def test_large_batch_exceeding_max_size(self):
        """Test appending a batch larger than max_size."""
        storage = SlidingWindowStorage(max_size=3)

        # Append batch with 5 elements
        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1, 2, 3, 4, 5]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2, 3, 4])},
        )
        storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        # Should keep the last 3 elements
        assert sc.allclose(
            result.coords['time'],
            sc.array(dims=['time'], values=[2, 3, 4]),
        )


class TestGrowingStorage:
    """Tests for GrowingStorage."""

    def test_initialization(self):
        """Test storage initialization."""
        storage = GrowingStorage(initial_size=5, max_size=100)
        assert storage.get_all() is None
        assert storage.estimate_memory() == 0

    def test_append_single_batch(self, simple_batch1):
        """Test appending a single batch."""
        storage = GrowingStorage(initial_size=5, max_size=100)
        storage.append(simple_batch1)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 3
        assert sc.allclose(result.coords['time'], simple_batch1.coords['time'])

    def test_append_multiple_batches(self, simple_batch1, simple_batch2):
        """Test appending multiple batches sequentially."""
        storage = GrowingStorage(initial_size=5, max_size=100)

        storage.append(simple_batch1)
        storage.append(simple_batch2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 5
        # Verify concatenation
        expected_times = sc.concat(
            [simple_batch1.coords['time'], simple_batch2.coords['time']], dim='time'
        )
        assert sc.allclose(result.coords['time'], expected_times)

    def test_grows_with_many_appends(self):
        """Test that storage grows capacity as needed."""
        storage = GrowingStorage(initial_size=2, max_size=100)

        # Append enough data to exceed initial capacity
        for i in range(10):
            data = sc.DataArray(
                data=sc.array(dims=['time'], values=[i]),
                coords={'time': sc.array(dims=['time'], values=[i])},
            )
            storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 10

    def test_respects_max_size(self):
        """Test that storage doesn't exceed max_size."""
        storage = GrowingStorage(initial_size=2, max_size=5)

        # Append more than max_size
        for i in range(10):
            data = sc.DataArray(
                data=sc.array(dims=['time'], values=[i]),
                coords={'time': sc.array(dims=['time'], values=[i])},
            )
            storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 5
        # Should keep last 5 elements (indices 5-9)
        assert result.coords['time'].values[0] == 5
        assert result.coords['time'].values[4] == 9

    def test_append_with_multiple_dimensions(self, multi_dim_batch1, multi_dim_batch2):
        """Test appending data with multiple dimensions."""
        storage = GrowingStorage(initial_size=5, max_size=100)
        storage.append(multi_dim_batch1)
        storage.append(multi_dim_batch2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 4
        assert result.sizes['x'] == 2
        # Verify x coordinate is preserved
        assert sc.allclose(result.coords['x'], multi_dim_batch1.coords['x'])

    def test_append_with_mask(self, data_with_mask1, data_with_mask2):
        """Test appending data with masks."""
        storage = GrowingStorage(initial_size=5, max_size=100)
        storage.append(data_with_mask1)
        storage.append(data_with_mask2)

        result = storage.get_all()
        assert result is not None
        assert 'bad' in result.masks
        # Verify masks are concatenated correctly
        expected_mask = sc.concat(
            [data_with_mask1.masks['bad'], data_with_mask2.masks['bad']], dim='time'
        )
        assert sc.all(result.masks['bad'] == expected_mask).value

    def test_clear(self, simple_batch1):
        """Test clearing storage."""
        storage = GrowingStorage(initial_size=5, max_size=100)
        storage.append(simple_batch1)
        assert storage.get_all() is not None

        storage.clear()
        assert storage.get_all() is None
        assert storage.estimate_memory() == 0

    def test_memory_estimation(self, simple_batch1):
        """Test memory estimation."""
        storage = GrowingStorage(initial_size=5, max_size=100)
        initial_memory = storage.estimate_memory()
        assert initial_memory == 0

        storage.append(simple_batch1)
        memory_after_append = storage.estimate_memory()
        assert memory_after_append > 0

    def test_invalid_initial_size(self):
        """Test that invalid initial_size raises error."""
        with pytest.raises(
            ValueError, match="initial_size and max_size must be positive"
        ):
            GrowingStorage(initial_size=0, max_size=100)

        with pytest.raises(
            ValueError, match="initial_size and max_size must be positive"
        ):
            GrowingStorage(initial_size=-1, max_size=100)

    def test_invalid_max_size(self):
        """Test that invalid max_size raises error."""
        with pytest.raises(
            ValueError, match="initial_size and max_size must be positive"
        ):
            GrowingStorage(initial_size=5, max_size=0)

    def test_initial_size_exceeds_max(self):
        """Test that initial_size cannot exceed max_size."""
        with pytest.raises(ValueError, match="initial_size cannot exceed max_size"):
            GrowingStorage(initial_size=100, max_size=50)

    def test_missing_concat_dimension(self):
        """Test that appending data without concat dimension raises error."""
        storage = GrowingStorage(initial_size=5, max_size=100, concat_dim='time')
        data = sc.DataArray(data=sc.array(dims=['x'], values=[1, 2, 3]))

        with pytest.raises(ValueError, match="Data must have 'time' dimension"):
            storage.append(data)

    def test_custom_concat_dimension(self):
        """Test using a custom concat dimension."""
        data1 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1, 2, 3]),
            coords={'x': sc.array(dims=['x'], values=[0, 1, 2])},
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['x'], values=[4, 5, 6]),
            coords={'x': sc.array(dims=['x'], values=[0, 1, 2])},
        )

        storage = GrowingStorage(initial_size=5, max_size=100, concat_dim='x')
        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['x'] == 6

    def test_large_batch_exceeding_max_size(self):
        """Test appending a batch larger than max_size."""
        storage = GrowingStorage(initial_size=2, max_size=5)

        # Create batch with size 7 (larger than max_size of 5)
        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1, 2, 3, 4, 5, 6, 7]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2, 3, 4, 5, 6])},
        )
        storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 5
        # Should keep last 5 elements
        assert sc.allclose(
            result.coords['time'],
            sc.array(dims=['time'], values=[2, 3, 4, 5, 6]),
        )

    def test_sequential_growth(self):
        """Test that storage grows correctly with sequential appends."""
        storage = GrowingStorage(initial_size=3, max_size=50)

        # Append data in multiple rounds
        for batch_num in range(3):
            for i in range(5):
                data = sc.DataArray(
                    data=sc.array(dims=['time'], values=[batch_num * 5 + i]),
                    coords={
                        'time': sc.array(dims=['time'], values=[batch_num * 5 + i])
                    },
                )
                storage.append(data)

        result = storage.get_all()
        assert result is not None
        assert result.sizes['time'] == 15

    def test_complex_data_with_multiple_coords(self):
        """Test with data that has multiple coordinates."""
        data1 = sc.DataArray(
            data=sc.array(dims=['time'], values=[1, 2, 3]),
            coords={
                'time': sc.array(dims=['time'], values=[0, 1, 2]),
                'timestamp': sc.array(dims=['time'], values=[100, 101, 102]),
            },
        )
        data2 = sc.DataArray(
            data=sc.array(dims=['time'], values=[4, 5]),
            coords={
                'time': sc.array(dims=['time'], values=[3, 4]),
                'timestamp': sc.array(dims=['time'], values=[103, 104]),
            },
        )

        storage = GrowingStorage(initial_size=5, max_size=100)
        storage.append(data1)
        storage.append(data2)

        result = storage.get_all()
        assert result is not None
        assert 'time' in result.coords
        assert 'timestamp' in result.coords
        # Verify concatenation
        expected_times = sc.concat(
            [data1.coords['time'], data2.coords['time']], dim='time'
        )
        assert sc.allclose(result.coords['time'], expected_times)


class TestStorageStrategyComparison:
    """Tests comparing behavior of different storage strategies."""

    def test_both_preserve_data_values(self, simple_batch1, simple_batch2):
        """Test that both strategies preserve data values."""
        sliding = SlidingWindowStorage(max_size=10)
        growing = GrowingStorage(initial_size=5, max_size=100)

        sliding.append(simple_batch1)
        sliding.append(simple_batch2)
        growing.append(simple_batch1)
        growing.append(simple_batch2)

        sliding_result = sliding.get_all()
        growing_result = growing.get_all()
        assert sc.allclose(sliding_result.data, growing_result.data)
        assert sc.allclose(sliding_result.coords['time'], growing_result.coords['time'])

    def test_both_preserve_coordinates(self, multi_dim_batch1, multi_dim_batch2):
        """Test that both strategies preserve all coordinates."""
        sliding = SlidingWindowStorage(max_size=10)
        growing = GrowingStorage(initial_size=5, max_size=100)

        sliding.append(multi_dim_batch1)
        sliding.append(multi_dim_batch2)
        growing.append(multi_dim_batch1)
        growing.append(multi_dim_batch2)

        sliding_result = sliding.get_all()
        growing_result = growing.get_all()

        assert 'time' in sliding_result.coords
        assert 'x' in sliding_result.coords
        assert sc.allclose(sliding_result.coords['time'], growing_result.coords['time'])
        assert sc.allclose(sliding_result.coords['x'], growing_result.coords['x'])

    def test_both_preserve_masks(self, data_with_mask1, data_with_mask2):
        """Test that both strategies preserve masks."""
        sliding = SlidingWindowStorage(max_size=10)
        growing = GrowingStorage(initial_size=5, max_size=100)

        sliding.append(data_with_mask1)
        sliding.append(data_with_mask2)
        growing.append(data_with_mask1)
        growing.append(data_with_mask2)

        sliding_result = sliding.get_all()
        growing_result = growing.get_all()

        assert 'bad' in sliding_result.masks
        assert 'bad' in growing_result.masks
        assert sc.all(sliding_result.masks['bad'] == growing_result.masks['bad']).value

    def test_both_handle_trimming_correctly(self):
        """Test that both strategies trim data correctly when exceeding max_size."""
        max_size = 4
        sliding = SlidingWindowStorage(max_size=max_size)
        growing = GrowingStorage(initial_size=2, max_size=max_size)

        # Append 6 single-element batches
        for i in range(6):
            data = sc.DataArray(
                data=sc.array(dims=['time'], values=[i * 10]),
                coords={'time': sc.array(dims=['time'], values=[i])},
            )
            sliding.append(data)
            growing.append(data)

        sliding_result = sliding.get_all()
        growing_result = growing.get_all()

        # Both should keep only last 4 elements
        assert sliding_result.sizes['time'] == max_size
        assert growing_result.sizes['time'] == max_size
        assert sc.allclose(sliding_result.data, growing_result.data)
        assert sc.allclose(sliding_result.coords['time'], growing_result.coords['time'])
