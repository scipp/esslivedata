# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.livedata.dashboard.buffer_strategy import (
    FixedSizeCircularBuffer,
    GrowingBuffer,
    TimeWindowBuffer,
)


class TestTimeWindowBuffer:
    def test_append_and_get(self):
        buffer = TimeWindowBuffer(time_window=sc.scalar(10, unit='s'))

        # Create time series data
        data1 = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0]),
            coords={'time': sc.array(dims=['time'], values=[0.0, 1.0, 2.0], unit='s')},
        )
        buffer.append(data1)

        result = buffer.get_buffer()
        assert result is not None
        assert sc.identical(result, data1)

    def test_time_window_eviction(self):
        buffer = TimeWindowBuffer(time_window=sc.scalar(5, unit='s'))

        # Add data at t=0-2
        data1 = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0]),
            coords={'time': sc.array(dims=['time'], values=[0.0, 1.0, 2.0], unit='s')},
        )
        buffer.append(data1)

        # Add data at t=8-10 (should evict t=0-2)
        data2 = sc.DataArray(
            data=sc.array(dims=['time'], values=[4.0, 5.0, 6.0]),
            coords={'time': sc.array(dims=['time'], values=[8.0, 9.0, 10.0], unit='s')},
        )
        buffer.append(data2)

        result = buffer.get_buffer()
        assert result is not None
        # Only data within 5s of latest time (10s) should remain
        # That means data >= 5s, so only data2 should remain
        assert len(result) == 3
        assert sc.identical(
            result.data, sc.array(dims=['time'], values=[4.0, 5.0, 6.0])
        )

    def test_get_window(self):
        buffer = TimeWindowBuffer(time_window=sc.scalar(100, unit='s'))

        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0, 4.0, 5.0]),
            coords={
                'time': sc.array(
                    dims=['time'], values=[0.0, 1.0, 2.0, 3.0, 4.0], unit='s'
                )
            },
        )
        buffer.append(data)

        # Get last 3 elements
        window = buffer.get_window(size=3)
        assert window is not None
        assert len(window) == 3
        assert sc.identical(
            window.data, sc.array(dims=['time'], values=[3.0, 4.0, 5.0])
        )

    def test_estimate_memory(self):
        buffer = TimeWindowBuffer(time_window=sc.scalar(100, unit='s'))

        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0]),
            coords={'time': sc.array(dims=['time'], values=[0.0, 1.0, 2.0], unit='s')},
        )
        buffer.append(data)

        memory = buffer.estimate_memory()
        assert memory > 0

    def test_clear(self):
        buffer = TimeWindowBuffer(time_window=sc.scalar(100, unit='s'))

        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0]),
            coords={'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s')},
        )
        buffer.append(data)

        buffer.clear()
        assert buffer.get_buffer() is None

    def test_raises_on_missing_time_dimension(self):
        buffer = TimeWindowBuffer(time_window=sc.scalar(10, unit='s'))

        data = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
        )

        with pytest.raises(ValueError, match="must have 'time' dimension"):
            buffer.append(data)


class TestFixedSizeCircularBuffer:
    def test_append_and_get(self):
        buffer = FixedSizeCircularBuffer(max_size=5, concat_dim='time')

        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2])},
        )
        buffer.append(data)

        result = buffer.get_buffer()
        assert result is not None
        assert sc.identical(result, data)

    def test_circular_eviction(self):
        buffer = FixedSizeCircularBuffer(max_size=5, concat_dim='time')

        # Add 3 elements
        data1 = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2])},
        )
        buffer.append(data1)

        # Add 4 more elements (total 7, should keep last 5)
        data2 = sc.DataArray(
            data=sc.array(dims=['time'], values=[4.0, 5.0, 6.0, 7.0]),
            coords={'time': sc.array(dims=['time'], values=[3, 4, 5, 6])},
        )
        buffer.append(data2)

        result = buffer.get_buffer()
        assert result is not None
        assert len(result) == 5
        # Should have last 5 elements: [3, 4, 5, 6, 7]
        assert sc.identical(
            result.data, sc.array(dims=['time'], values=[3.0, 4.0, 5.0, 6.0, 7.0])
        )

    def test_large_append_truncates(self):
        buffer = FixedSizeCircularBuffer(max_size=3, concat_dim='time')

        # Append 5 elements at once (larger than max_size)
        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0, 4.0, 5.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2, 3, 4])},
        )
        buffer.append(data)

        result = buffer.get_buffer()
        assert result is not None
        assert len(result) == 3
        # Should keep last 3 elements
        assert sc.identical(
            result.data, sc.array(dims=['time'], values=[3.0, 4.0, 5.0])
        )

    def test_get_window(self):
        buffer = FixedSizeCircularBuffer(max_size=10, concat_dim='time')

        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0, 4.0, 5.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2, 3, 4])},
        )
        buffer.append(data)

        window = buffer.get_window(size=3)
        assert window is not None
        assert len(window) == 3
        assert sc.identical(
            window.data, sc.array(dims=['time'], values=[3.0, 4.0, 5.0])
        )

    def test_raises_on_missing_concat_dimension(self):
        buffer = FixedSizeCircularBuffer(max_size=5, concat_dim='time')

        data = sc.DataArray(data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0]))

        with pytest.raises(ValueError, match="must have 'time' dimension"):
            buffer.append(data)

    def test_raises_on_invalid_max_size(self):
        with pytest.raises(ValueError, match="max_size must be positive"):
            FixedSizeCircularBuffer(max_size=0)

        with pytest.raises(ValueError, match="max_size must be positive"):
            FixedSizeCircularBuffer(max_size=-1)


class TestGrowingBuffer:
    def test_append_and_get(self):
        buffer = GrowingBuffer(initial_size=2, max_size=10, concat_dim='time')

        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1])},
        )
        buffer.append(data)

        result = buffer.get_buffer()
        assert result is not None
        assert sc.identical(result, data)

    def test_grows_capacity(self):
        buffer = GrowingBuffer(initial_size=2, max_size=10, concat_dim='time')

        # Add 2 elements (at capacity)
        data1 = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1])},
        )
        buffer.append(data1)

        # Add 1 more element (should trigger capacity growth)
        data2 = sc.DataArray(
            data=sc.array(dims=['time'], values=[3.0]),
            coords={'time': sc.array(dims=['time'], values=[2])},
        )
        buffer.append(data2)

        result = buffer.get_buffer()
        assert result is not None
        assert len(result) == 3

    def test_evicts_when_max_size_reached(self):
        buffer = GrowingBuffer(initial_size=2, max_size=5, concat_dim='time')

        # Add 6 elements (exceeds max_size)
        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2, 3, 4, 5])},
        )
        buffer.append(data)

        result = buffer.get_buffer()
        assert result is not None
        assert len(result) == 5
        # Should keep last 5 elements
        assert sc.identical(
            result.data, sc.array(dims=['time'], values=[2.0, 3.0, 4.0, 5.0, 6.0])
        )

    def test_get_window(self):
        buffer = GrowingBuffer(initial_size=2, max_size=10, concat_dim='time')

        data = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0, 4.0, 5.0]),
            coords={'time': sc.array(dims=['time'], values=[0, 1, 2, 3, 4])},
        )
        buffer.append(data)

        window = buffer.get_window(size=3)
        assert window is not None
        assert len(window) == 3
        assert sc.identical(
            window.data, sc.array(dims=['time'], values=[3.0, 4.0, 5.0])
        )

    def test_raises_on_invalid_sizes(self):
        with pytest.raises(ValueError, match="must be positive"):
            GrowingBuffer(initial_size=0, max_size=10)

        with pytest.raises(ValueError, match="must be positive"):
            GrowingBuffer(initial_size=10, max_size=0)

        with pytest.raises(ValueError, match="cannot exceed max_size"):
            GrowingBuffer(initial_size=20, max_size=10)
