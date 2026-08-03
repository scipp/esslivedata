# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.livedata.core.timestamp import Timestamp
from ess.livedata.preprocessors.accumulators import LogData
from ess.livedata.preprocessors.to_nxlog import ToNXlog, nxlog_for_stream


def test_to_nxlog_initialization():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)
    assert accumulator is not None


def test_to_nxlog_add_single_value():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    log_data = LogData(time=5000000, value=42.0)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data)

    # Check the data was added by retrieving it
    result = accumulator.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[42.0], unit='counts'))


def test_to_nxlog_get_single_value():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    log_data = LogData(time=1_000_000_000, value=42.0)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data)

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

    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data1)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data2)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data3)

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
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data)
    result = accumulator.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[42.0], unit='counts'))

    # After get(), accumulator should be empty
    # Add another value to check if it's the only one
    log_data2 = LogData(time=6000000, value=43.0)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data2)

    # Explicitly clear
    accumulator.clear()

    # Verify it's empty by adding a new value and checking it's the only one
    log_data3 = LogData(time=7000000, value=44.0)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data3)
    result = accumulator.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[44.0], unit='counts'))


def test_to_nxlog_get_does_not_clear_data():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    # Add data and get it
    log_data = LogData(time=5000000, value=42.0)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data)
    _ = accumulator.get()

    # After get(), adding new values should keep the previous data
    log_data2 = LogData(time=7000000, value=100.0)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data2)

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


def test_to_nxlog_add_values_with_increasing_timestamps():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    log_data1 = LogData(time=1000000, value=10.0)
    log_data2 = LogData(time=2000000, value=20.0)
    log_data3 = LogData(time=3000000, value=30.0)

    accumulator.add(timestamp=Timestamp.from_ns(100), data=log_data1)
    accumulator.add(timestamp=Timestamp.from_ns(200), data=log_data2)
    accumulator.add(timestamp=Timestamp.from_ns(300), data=log_data3)

    result = accumulator.get()

    assert_identical(
        result.data, sc.array(dims=['time'], values=[10.0, 20.0, 30.0], unit='counts')
    )

    start_time = sc.epoch(unit='ns')
    expected_times = [start_time.value + t for t in [1000000, 2000000, 3000000]]
    expected_time_coord = sc.array(dims=['time'], values=expected_times, unit='ns')
    assert_identical(result.coords['time'], expected_time_coord)


def test_capacity_expansion_with_many_adds():
    """Test that capacity expands automatically as needed with many additions."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs)

    # Add initial data with 3 items
    accumulator.add(Timestamp.from_ns(0), LogData(time=10, value=1.0))
    accumulator.add(Timestamp.from_ns(0), LogData(time=20, value=1.0))
    accumulator.add(Timestamp.from_ns(0), LogData(time=30, value=1.0))

    result = accumulator.get()
    assert result.sizes["time"] == 3

    # Add more data that would exceed the initial capacity
    accumulator.add(Timestamp.from_ns(0), LogData(time=40, value=1.0))
    accumulator.add(Timestamp.from_ns(0), LogData(time=50, value=1.0))
    accumulator.add(Timestamp.from_ns(0), LogData(time=60, value=1.0))
    accumulator.add(Timestamp.from_ns(0), LogData(time=70, value=1.0))

    # Check all data is preserved
    result = accumulator.get()
    assert result.sizes["time"] == 7

    # Continue adding more data to trigger multiple capacity expansions
    for i in range(8, 20):
        accumulator.add(Timestamp.from_ns(0), LogData(time=i * 10, value=1.0))

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
    accumulator.add(Timestamp.from_ns(0), LogData(time=10, value=1.0))

    # Add many items that would greatly exceed any reasonable initial capacity
    many_items = list(range(100, 600))
    for t in many_items:
        accumulator.add(Timestamp.from_ns(0), LogData(time=t, value=1.0))

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
        accumulator.add(Timestamp.from_ns(0), LogData(time=value, value=1.0))

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
            accumulator.add(Timestamp.from_ns(0), LogData(time=t, value=1.0))

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


def test_out_of_order_timestamps_are_skipped():
    """Out-of-order timestamps are skipped to surface upstream issues."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs)

    accumulator.add(Timestamp.from_ns(0), LogData(time=30, value=3.0))
    accumulator.add(
        Timestamp.from_ns(0), LogData(time=10, value=1.0)
    )  # out of order, skipped
    accumulator.add(
        Timestamp.from_ns(0), LogData(time=20, value=2.0)
    )  # out of order, skipped
    accumulator.add(
        Timestamp.from_ns(0), LogData(time=40, value=4.0)
    )  # in order, accepted

    result = accumulator.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[3.0, 4.0], unit='K'))


def test_to_nxlog_array_data_1d():
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs, data_dims=('x',))

    log_data1 = LogData(time=1000000, value=[1.0, 2.0, 3.0])
    log_data2 = LogData(time=2000000, value=[4.0, 5.0, 6.0])

    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data1)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data2)

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

    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data1)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data2)

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

    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data1)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data2)

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

    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data1)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data2)

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
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data1)

    # Then add data without variances
    log_data2 = LogData(time=2000000, value=20.0)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data2)

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
        accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data)

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


def test_out_of_order_array_data_is_skipped():
    """Out-of-order timestamps with array data are skipped."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs, data_dims=('x',))

    accumulator.add(
        timestamp=Timestamp.from_ns(0),
        data=LogData(time=3000000, value=[30.0, 31.0]),
    )
    accumulator.add(
        timestamp=Timestamp.from_ns(0),
        data=LogData(time=1000000, value=[10.0, 11.0]),
    )
    accumulator.add(
        timestamp=Timestamp.from_ns(0),
        data=LogData(time=2000000, value=[20.0, 21.0]),
    )

    result = accumulator.get()

    # Only the first (in-order) entry should be kept
    expected_values = [[30.0, 31.0]]
    assert_identical(
        result.data, sc.array(dims=['time', 'x'], values=expected_values, unit='K')
    )


def test_to_nxlog_different_dtypes():
    """Test that different numeric dtypes work correctly."""
    attrs = {'units': 'counts'}
    accumulator = ToNXlog(attrs=attrs)

    # Add integer data
    log_data1 = LogData(time=1000000, value=42)
    log_data2 = LogData(time=2000000, value=43)

    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data1)
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data2)

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
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data1)

    result = accumulator.get()
    assert result.sizes["time"] == 1
    assert result.data.variances is not None

    # Clear and add new data
    accumulator.clear()
    log_data2 = LogData(time=2000000, value=[3.0, 4.0], variances=[0.3, 0.4])
    accumulator.add(timestamp=Timestamp.from_ns(0), data=log_data2)

    result = accumulator.get()
    assert result.sizes["time"] == 1
    expected_data = sc.array(
        dims=['time', 'x'], values=[[3.0, 4.0]], variances=[[0.3, 0.4]], unit='K'
    )
    assert_identical(result.data, expected_data)


def test_duplicate_timestamp_same_value_silently_skipped():
    """Re-sent f144 values with identical timestamp and value are silently dropped."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs)

    accumulator.add(0, LogData(time=1000, value=42.0))
    accumulator.add(0, LogData(time=1000, value=42.0))  # duplicate
    accumulator.add(0, LogData(time=1000, value=42.0))  # duplicate
    accumulator.add(0, LogData(time=2000, value=43.0))  # new timestamp

    result = accumulator.get()
    assert_identical(
        result.data, sc.array(dims=['time'], values=[42.0, 43.0], unit='K')
    )


def test_duplicate_timestamp_different_value_skipped():
    """Duplicate timestamp with different value is skipped (not accumulated)."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs)

    accumulator.add(0, LogData(time=1000, value=42.0))
    accumulator.add(0, LogData(time=1000, value=99.0))  # different value, skipped

    result = accumulator.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[42.0], unit='K'))


def test_duplicate_timestamp_array_value_skipped():
    """Duplicate timestamp with array values is skipped."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs, data_dims=('x',))

    accumulator.add(0, LogData(time=1000, value=[1.0, 2.0]))
    accumulator.add(0, LogData(time=1000, value=[1.0, 2.0]))  # same array, skipped
    accumulator.add(0, LogData(time=2000, value=[3.0, 4.0]))

    result = accumulator.get()
    expected = sc.array(dims=['time', 'x'], values=[[1.0, 2.0], [3.0, 4.0]], unit='K')
    assert_identical(result.data, expected)


def test_clear_resets_duplicate_tracking():
    """After clear(), the same timestamp can be added again."""
    attrs = {'units': 'K'}
    accumulator = ToNXlog(attrs=attrs)

    accumulator.add(0, LogData(time=1000, value=42.0))
    accumulator.clear()

    # Same timestamp should be accepted after clear
    accumulator.add(0, LogData(time=1000, value=42.0))
    result = accumulator.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[42.0], unit='K'))


def test_grows_with_target_and_idle_coords():
    accumulator = ToNXlog(attrs={'units': 'mm'}, has_target=True, has_idle=True)
    accumulator.add(
        Timestamp.from_ns(0),
        LogData(time=1000, value=1.0, target=10.0, idle=False),
    )
    accumulator.add(
        Timestamp.from_ns(0),
        LogData(time=2000, value=2.0, target=10.0, idle=True),
    )
    result = accumulator.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[1.0, 2.0], unit='mm'))
    assert_identical(
        result.coords['target'],
        sc.array(dims=['time'], values=[10.0, 10.0], unit='mm'),
    )
    assert_identical(
        result.coords['idle'],
        sc.array(dims=['time'], values=[0, 1], dtype='int32'),
    )


def test_target_coord_grows_past_initial_capacity():
    accumulator = ToNXlog(attrs={'units': 'mm'}, has_target=True)
    for i in range(20):
        accumulator.add(
            Timestamp.from_ns(0),
            LogData(time=(i + 1) * 1000, value=float(i), target=float(i + 1)),
        )
    result = accumulator.get()
    assert result.sizes == {'time': 20}
    assert_identical(
        result.coords['target'],
        sc.array(dims=['time'], values=[float(i + 1) for i in range(20)], unit='mm'),
    )


def test_missing_target_raises_when_has_target():
    accumulator = ToNXlog(attrs={'units': 'mm'}, has_target=True)
    with pytest.raises(ValueError, match="Target expected"):
        accumulator.add(Timestamp.from_ns(0), LogData(time=1000, value=1.0))


def test_missing_idle_raises_when_has_idle():
    accumulator = ToNXlog(attrs={'units': 'mm'}, has_idle=True)
    with pytest.raises(ValueError, match="Idle"):
        accumulator.add(Timestamp.from_ns(0), LogData(time=1000, value=1.0))


class TestRetention:
    """History is bounded by max_size and max_age, whichever binds first."""

    def make(self, *, max_size=1_000_000, max_age_s=30 * 86400, **kwargs):
        return ToNXlog(
            attrs={'units': 'm'},
            max_size=max_size,
            max_age=sc.scalar(max_age_s, unit='s'),
            **kwargs,
        )

    def push(self, acc, times, *, scale=1_000_000_000):
        for t in times:
            acc.add(0, LogData(time=int(t * scale), value=float(t)))

    def times(self, acc):
        return [
            int(t) // 1_000_000_000
            for t in acc.get().coords['time'].values.view('int64')
        ]

    def test_max_size_drops_oldest_samples(self):
        acc = self.make(max_size=5)
        self.push(acc, range(1, 12))
        assert self.times(acc) == [7, 8, 9, 10, 11]

    def test_below_max_size_keeps_everything(self):
        acc = self.make(max_size=100)
        self.push(acc, range(1, 12))
        assert self.times(acc) == list(range(1, 12))

    def test_max_age_drops_expired_samples(self):
        acc = self.make(max_age_s=10)
        self.push(acc, [1, 2, 3, 20, 21])
        # Cutoff is 21-10=11; samples 1 and 2 are expired, 3 is the anchor.
        assert self.times(acc) == [3, 20, 21]

    def test_age_cutoff_retains_anchor_for_previous_mode_lookup(self):
        acc = self.make(max_age_s=10)
        self.push(acc, [0, 100])
        # Everything is older than the cutoff, but dropping the last sample at
        # or before it would make sc.lookup(mode='previous') return NaN.
        assert self.times(acc) == [0, 100]
        lut = sc.lookup(acc.get(), 'time', mode='previous')
        query = sc.epoch(unit='ns') + sc.array(
            dims=['t'], values=[50_000_000_000], unit='ns'
        )
        assert lut[query].values.tolist() == [0.0]

    def test_age_measured_against_newest_sample_not_wall_clock(self):
        acc = self.make(max_age_s=10)
        self.push(acc, [1, 2, 3])
        # A stalled device keeps its history; nothing ages out without new data.
        assert self.times(acc) == [1, 2, 3]

    def test_never_trims_below_one_sample(self):
        acc = self.make(max_size=1, max_age_s=1)
        self.push(acc, [1, 100, 1000])
        assert self.times(acc) == [1000]

    def test_size_and_age_combined_take_the_tighter_bound(self):
        acc = self.make(max_size=3, max_age_s=100)
        self.push(acc, range(1, 10))
        assert self.times(acc) == [7, 8, 9]

    def test_retained_window_survives_many_relocations(self):
        acc = self.make(max_size=4)
        self.push(acc, range(1, 500))
        assert self.times(acc) == [496, 497, 498, 499]
        assert list(acc.get().values) == [496.0, 497.0, 498.0, 499.0]

    def test_values_and_coords_stay_aligned_across_relocation(self):
        acc = ToNXlog(
            attrs={'units': 'm'},
            max_size=4,
            has_target=True,
            has_idle=True,
        )
        for t in range(1, 200):
            acc.add(
                0,
                LogData(
                    time=t * 1_000_000_000,
                    value=float(t),
                    target=float(-t),
                    idle=t % 2,
                ),
            )
        result = acc.get()
        assert list(result.values) == [196.0, 197.0, 198.0, 199.0]
        assert list(result.coords['target'].values) == [-196.0, -197.0, -198.0, -199.0]
        assert list(result.coords['idle'].values) == [0, 1, 0, 1]

    def test_variances_survive_relocation(self):
        acc = ToNXlog(attrs={'units': 'm'}, max_size=3)
        for t in range(1, 100):
            acc.add(
                0,
                LogData(time=t * 1_000_000_000, value=float(t), variances=float(t) * 2),
            )
        result = acc.get()
        assert list(result.values) == [97.0, 98.0, 99.0]
        assert list(result.variances) == [194.0, 196.0, 198.0]

    def test_buffer_capacity_stays_bounded(self):
        acc = self.make(max_size=100)
        self.push(acc, range(1, 5000))
        assert acc.get().sizes['time'] == 100
        assert acc._timeseries.sizes['time'] <= 100 + 100  # max_size + max slack

    def test_get_view_is_not_rewritten_by_later_relocation(self):
        acc = self.make(max_size=4)
        self.push(acc, range(1, 20))
        view = acc.get()
        before = list(view.values)
        self.push(acc, range(20, 500))
        assert list(view.values) == before

    def test_clear_resets_retention_state(self):
        acc = self.make(max_size=4)
        self.push(acc, range(1, 50))
        acc.clear()
        self.push(acc, [100, 101])
        assert self.times(acc) == [100, 101]

    def test_rejects_non_positive_max_size(self):
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            ToNXlog(attrs={}, max_size=0)


class TestRelocationPhase:
    """Equal-rate logs must not relocate in the same batch forever."""

    MAX_SIZE = 64

    def relocation_points(self, phase: int, n: int) -> set[int]:
        """Sample counts at which the buffer is reallocated.

        Points during the initial fill are excluded: capacity doubles from 2
        regardless of phase, so every log necessarily reallocates at the same
        counts then. That is a one-off warm-up, whereas the steady-state period
        repeats for the process lifetime and is what must be spread out.
        """
        acc = ToNXlog(attrs={'units': 'm'}, max_size=self.MAX_SIZE, phase=phase)
        points = set()
        previous = None
        for t in range(1, n):
            acc.add(0, LogData(time=t * 1_000_000_000, value=float(t)))
            current = id(acc._timeseries)
            if previous is not None and current != previous and t > 2 * self.MAX_SIZE:
                points.add(t)
            previous = current
        return points

    def test_distinct_phases_relocate_at_distinct_points(self):
        streams = [self.relocation_points(phase, 2000) for phase in (0, 1, 2, 3)]
        assert all(len(points) > 1 for points in streams)
        # No batch ever relocates every log at once.
        assert set.intersection(*streams) == set()

    def test_phase_is_stable_across_instances(self):
        assert self.relocation_points(5, 500) == self.relocation_points(5, 500)

    def test_stream_name_determines_phase(self):
        from ess.livedata.config.stream import F144Stream

        stream = F144Stream(units='m', topic='t', source='s')
        a = nxlog_for_stream(stream, name='chopper_1')
        b = nxlog_for_stream(stream, name='chopper_2')
        assert a._slack != b._slack
