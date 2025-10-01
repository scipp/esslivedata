# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.livedata.handlers.accumulators import LogData
from ess.livedata.handlers.to_nxlog import ToNXlog


def test_to_nxlog_initialization():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)
    assert accumulator is not None


def test_to_nxlog_add_single_value():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    log_data = LogData(time=5000000, value=42.0)
    accumulator.add(timestamp=0, data=log_data)

    # Check the data was added by retrieving it
    result = accumulator.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[42.0], unit='counts'))


def test_to_nxlog_get_single_value():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    log_data = LogData(time=1_000_000_000, value=42.0)
    accumulator.add(timestamp=0, data=log_data)

    result = accumulator.get()

    assert_identical(result.data, sc.array(dims=['time'], values=[42.0], unit='counts'))
    assert_identical(
        result.coords['time'][0], sc.datetime('1970-01-01T00:00:01', unit='ns')
    )


def test_to_nxlog_get_multiple_values():
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs)

    log_data1 = LogData(time=1000000, value=273.15)
    log_data2 = LogData(time=2000000, value=293.15)
    log_data3 = LogData(time=3000000, value=303.15)

    accumulator.add(timestamp=0, data=log_data1)
    accumulator.add(timestamp=0, data=log_data2)
    accumulator.add(timestamp=0, data=log_data3)

    result = accumulator.get()

    assert_identical(
        result.data, sc.array(dims=['time'], values=[273.15, 293.15, 303.15], unit='K')
    )

    # Test timestamps are correct relative to start time
    start_time = sc.epoch(unit='ns')
    expected_times = [start_time.value + t for t in [1000000, 2000000, 3000000]]
    expected_time_coord = sc.array(dims=['time'], values=expected_times, unit='ns')
    assert_identical(result.coords['time'], expected_time_coord)


def test_to_nxlog_clear():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    # Add data and verify it's there by getting it
    log_data = LogData(time=5000000, value=42.0)
    accumulator.add(timestamp=0, data=log_data)
    result = accumulator.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[42.0], unit='counts'))

    # After get(), accumulator should be empty
    # Add another value to check if it's the only one
    log_data2 = LogData(time=6000000, value=43.0)
    accumulator.add(timestamp=0, data=log_data2)

    # Explicitly clear
    accumulator.clear()

    # Verify it's empty by adding a new value and checking it's the only one
    log_data3 = LogData(time=7000000, value=44.0)
    accumulator.add(timestamp=0, data=log_data3)
    result = accumulator.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[44.0], unit='counts'))


def test_to_nxlog_get_does_not_clear_data():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    # Add data and get it
    log_data = LogData(time=5000000, value=42.0)
    accumulator.add(timestamp=0, data=log_data)
    _ = accumulator.get()

    # After get(), adding new values should keep the previous data
    log_data2 = LogData(time=7000000, value=100.0)
    accumulator.add(timestamp=0, data=log_data2)

    result = accumulator.get()
    assert_identical(
        result.data, sc.array(dims=['time'], values=[42.0, 100.0], unit='counts')
    )


def test_to_nxlog_empty_get():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    # Getting data from an empty accumulator should raise RuntimeError
    with pytest.raises(RuntimeError, match="No data has been added yet"):
        accumulator.get()


def test_to_nxlog_missing_attributes():
    # Missing units => unit is None
    attrs = {}
    assert ToNXlog(attrs=attrs).unit is None


def test_to_nxlog_add_values_with_different_timestamps():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    # Add values with increasing timestamps to verify correct ordering
    log_data1 = LogData(time=3000000, value=30.0)
    log_data2 = LogData(time=1000000, value=10.0)
    log_data3 = LogData(time=2000000, value=20.0)

    accumulator.add(timestamp=100, data=log_data1)
    accumulator.add(timestamp=200, data=log_data2)
    accumulator.add(timestamp=300, data=log_data3)

    result = accumulator.get()

    # Verify the values are ordered by log_data.time, not by timestamp
    assert_identical(
        result.data, sc.array(dims=['time'], values=[10.0, 20.0, 30.0], unit='counts')
    )

    # Verify times are also ordered
    start_time = sc.epoch(unit='ns')
    expected_times = [start_time.value + t for t in [1000000, 2000000, 3000000]]
    expected_time_coord = sc.array(dims=['time'], values=expected_times, unit='ns')
    assert_identical(result.coords['time'], expected_time_coord)


def test_capacity_expansion_with_many_adds():
    """Test that capacity expands automatically as needed with many additions."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs)

    # Add initial data with 3 items
    accumulator.add(0, LogData(time=10, value=1.0))
    accumulator.add(0, LogData(time=20, value=1.0))
    accumulator.add(0, LogData(time=30, value=1.0))

    result = accumulator.get()
    assert result.sizes["time"] == 3

    # Add more data that would exceed the initial capacity
    accumulator.add(0, LogData(time=40, value=1.0))
    accumulator.add(0, LogData(time=50, value=1.0))
    accumulator.add(0, LogData(time=60, value=1.0))
    accumulator.add(0, LogData(time=70, value=1.0))

    # Check all data is preserved
    result = accumulator.get()
    assert result.sizes["time"] == 7

    # Continue adding more data to trigger multiple capacity expansions
    for i in range(8, 20):
        accumulator.add(0, LogData(time=i * 10, value=1.0))

    # Verify all data is still correct
    result = accumulator.get()
    assert result.sizes["time"] == 19
    expected_times = [10, 20, 30, 40, 50, 60, 70] + [i * 10 for i in range(8, 20)]
    expected_time_values = [
        sc.datetime('1970-01-01T00:00:00.000000', unit='ns').value + t
        for t in expected_times
    ]
    assert_identical(
        result.coords['time'],
        sc.array(dims=['time'], values=expected_time_values, unit='ns'),
    )


def test_large_capacity_jumps():
    """Test adding data that forces large capacity increases."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs)

    # First add a small item to initialize
    accumulator.add(0, LogData(time=10, value=1.0))

    # Add many items that would greatly exceed any reasonable initial capacity
    many_items = list(range(100, 600))
    for t in many_items:
        accumulator.add(0, LogData(time=t, value=1.0))

    # Check that all items were added correctly
    result = accumulator.get()
    assert result.sizes["time"] == len(many_items) + 1

    # Check first and last values for boundary cases
    start_time = sc.epoch(unit='ns')
    assert_identical(result.coords['time'][0], start_time + sc.scalar(10, unit='ns'))
    assert_identical(result.coords['time'][-1], start_time + sc.scalar(599, unit='ns'))


def test_capacity_against_small_additions():
    """Test capacity behavior with many small additions."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs)

    # Add many items one by one, which should trigger capacity expansion multiple times
    for i in range(50):  # Using a smaller number (50) for test runtime
        value = i * 10
        accumulator.add(0, LogData(time=value, value=1.0))

        # Check that data added so far is preserved correctly
        result = accumulator.get()
        assert result.sizes["time"] == i + 1

    # Final verification
    result = accumulator.get()
    assert result.sizes["time"] == 50
    # Check first and last values
    start_time = sc.epoch(unit='ns')
    assert_identical(result.coords['time'][0], start_time + sc.scalar(0, unit='ns'))
    assert_identical(result.coords['time'][-1], start_time + sc.scalar(490, unit='ns'))


def test_repeated_expand_and_clear_cycles():
    """Test repeated cycles of adding many items and clearing."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs)

    for cycle in range(3):
        # Add enough items to trigger capacity expansion
        times = [(cycle * 1000) + (i * 10) for i in range(20)]
        for t in times:
            accumulator.add(0, LogData(time=t, value=1.0))

        # Verify all items are present
        result = accumulator.get()
        assert result.sizes["time"] == len(times)

        # Sample check for cycle 0, 1, 2
        start_time = sc.epoch(unit='ns')
        expected_first_time = start_time + sc.scalar(cycle * 1000, unit='ns')
        expected_last_time = start_time + sc.scalar((cycle * 1000) + 190, unit='ns')
        assert_identical(result.coords['time'][0], expected_first_time)
        assert_identical(result.coords['time'][-1], expected_last_time)

        # Clear and verify empty
        accumulator.clear()
        empty_result = accumulator.get()
        assert empty_result.sizes["time"] == 0


def test_preservation_of_addition_order():
    """Test that sorting occurs when getting data with multiple adds out of order."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs)

    # Add data with time 30
    accumulator.add(0, LogData(time=30, value=3.0))

    # Add data with earlier time 10
    accumulator.add(0, LogData(time=10, value=1.0))

    # Add data with middle time 20
    accumulator.add(0, LogData(time=20, value=2.0))

    # The data should be sorted by time, not in the order it was added
    result = accumulator.get()
    start_time = sc.epoch(unit='ns')

    # Check values are sorted by time
    assert_identical(
        result.data, sc.array(dims=['time'], values=[1.0, 2.0, 3.0], unit='K')
    )

    # Check times are sorted
    expected_times = [start_time.value + t for t in [10, 20, 30]]
    expected_time_coord = sc.array(dims=['time'], values=expected_times, unit='ns')
    assert_identical(result.coords['time'], expected_time_coord)


def test_to_nxlog_array_data_1d():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs, data_dims=('x',))

    log_data1 = LogData(time=1000000, value=[1.0, 2.0, 3.0])
    log_data2 = LogData(time=2000000, value=[4.0, 5.0, 6.0])

    accumulator.add(timestamp=0, data=log_data1)
    accumulator.add(timestamp=0, data=log_data2)

    result = accumulator.get()

    expected_values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert_identical(
        result.data, sc.array(dims=['time', 'x'], values=expected_values, unit='counts')
    )

    start_time = sc.epoch(unit='ns')
    expected_times = [start_time.value + t for t in [1000000, 2000000]]
    expected_time_coord = sc.array(dims=['time'], values=expected_times, unit='ns')
    assert_identical(result.coords['time'], expected_time_coord)


def test_to_nxlog_array_data_2d():
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs, data_dims=('y', 'x'))

    log_data1 = LogData(time=1000000, value=[[1.0, 2.0], [3.0, 4.0]])
    log_data2 = LogData(time=2000000, value=[[5.0, 6.0], [7.0, 8.0]])

    accumulator.add(timestamp=0, data=log_data1)
    accumulator.add(timestamp=0, data=log_data2)

    result = accumulator.get()

    expected_values = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    assert_identical(
        result.data, sc.array(dims=['time', 'y', 'x'], values=expected_values, unit='K')
    )


def test_to_nxlog_scalar_with_variances():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    log_data1 = LogData(time=1000000, value=10.0, variances=1.0)
    log_data2 = LogData(time=2000000, value=20.0, variances=4.0)

    accumulator.add(timestamp=0, data=log_data1)
    accumulator.add(timestamp=0, data=log_data2)

    result = accumulator.get()

    expected_data = sc.array(
        dims=['time'], values=[10.0, 20.0], variances=[1.0, 4.0], unit='counts'
    )
    assert_identical(result.data, expected_data)


def test_to_nxlog_array_with_variances():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs, data_dims=('x',))

    log_data1 = LogData(time=1000000, value=[1.0, 2.0], variances=[0.1, 0.2])
    log_data2 = LogData(time=2000000, value=[3.0, 4.0], variances=[0.3, 0.4])

    accumulator.add(timestamp=0, data=log_data1)
    accumulator.add(timestamp=0, data=log_data2)

    result = accumulator.get()

    expected_data = sc.array(
        dims=['time', 'x'],
        values=[[1.0, 2.0], [3.0, 4.0]],
        variances=[[0.1, 0.2], [0.3, 0.4]],
        unit='counts',
    )
    assert_identical(result.data, expected_data)


def test_to_nxlog_mixed_variances():
    """Test mixing data with and without variances - should fail gracefully."""
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    # First add data with variances
    log_data1 = LogData(time=1000000, value=10.0, variances=1.0)
    accumulator.add(timestamp=0, data=log_data1)

    # Then add data without variances
    log_data2 = LogData(time=2000000, value=20.0)
    accumulator.add(timestamp=0, data=log_data2)

    result = accumulator.get()

    # The result should have variances, with the second entry having variance 0
    expected_data = sc.array(
        dims=['time'], values=[10.0, 20.0], variances=[1.0, 0.0], unit='counts'
    )
    assert_identical(result.data, expected_data)


def test_to_nxlog_capacity_expansion_with_arrays():
    """Test that capacity expansion works correctly with array data."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs, data_dims=('x',))

    # Add enough array data to trigger capacity expansion
    for i in range(10):
        log_data = LogData(time=(i + 1) * 1000000, value=[float(i), float(i + 1)])
        accumulator.add(timestamp=0, data=log_data)

    result = accumulator.get()
    assert result.sizes["time"] == 10
    assert result.sizes["x"] == 2

    # Check first and last values
    assert_identical(
        result.data['time', 0], sc.array(dims=['x'], values=[0.0, 1.0], unit='K')
    )
    assert_identical(
        result.data['time', -1], sc.array(dims=['x'], values=[9.0, 10.0], unit='K')
    )


def test_to_nxlog_array_sorting():
    """Test that array data is correctly sorted by time."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs, data_dims=('x',))

    # Add data out of time order
    log_data1 = LogData(time=3000000, value=[30.0, 31.0])
    log_data2 = LogData(time=1000000, value=[10.0, 11.0])
    log_data3 = LogData(time=2000000, value=[20.0, 21.0])

    accumulator.add(timestamp=0, data=log_data1)
    accumulator.add(timestamp=0, data=log_data2)
    accumulator.add(timestamp=0, data=log_data3)

    result = accumulator.get()

    # Should be sorted by time
    expected_values = [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
    assert_identical(
        result.data, sc.array(dims=['time', 'x'], values=expected_values, unit='K')
    )

    start_time = sc.epoch(unit='ns')
    expected_times = [start_time.value + t for t in [1000000, 2000000, 3000000]]
    expected_time_coord = sc.array(dims=['time'], values=expected_times, unit='ns')
    assert_identical(result.coords['time'], expected_time_coord)


def test_to_nxlog_different_dtypes():
    """Test that different numeric dtypes work correctly."""
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    # Add integer data
    log_data1 = LogData(time=1000000, value=42)
    log_data2 = LogData(time=2000000, value=43)

    accumulator.add(timestamp=0, data=log_data1)
    accumulator.add(timestamp=0, data=log_data2)

    result = accumulator.get()

    assert_identical(
        result.data, sc.array(dims=['time'], values=[42, 43], unit='counts')
    )


def test_to_nxlog_clear_preserves_structure():
    """Test that clearing preserves the array structure for subsequent additions."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs, data_dims=('x',))

    # Add array data
    log_data1 = LogData(time=1000000, value=[1.0, 2.0], variances=[0.1, 0.2])
    accumulator.add(timestamp=0, data=log_data1)

    result = accumulator.get()
    assert result.sizes["time"] == 1
    assert result.data.variances is not None

    # Clear and add new data
    accumulator.clear()
    log_data2 = LogData(time=2000000, value=[3.0, 4.0], variances=[0.3, 0.4])
    accumulator.add(timestamp=0, data=log_data2)

    result = accumulator.get()
    assert result.sizes["time"] == 1
    expected_data = sc.array(
        dims=['time', 'x'], values=[[3.0, 4.0]], variances=[[0.3, 0.4]], unit='K'
    )
    assert_identical(result.data, expected_data)
