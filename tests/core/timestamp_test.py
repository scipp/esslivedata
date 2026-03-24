# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import bisect
from datetime import timezone

import pytest

from ess.livedata.core.timestamp import Duration, Timestamp


class TestDurationFactory:
    def test_from_seconds(self):
        d = Duration.from_seconds(2.5)
        assert int(d) == 2_500_000_000

    def test_from_ms(self):
        d = Duration.from_ms(150)
        assert int(d) == 150_000_000

    def test_to_seconds(self):
        d = Duration(1_500_000_000)
        assert d.to_seconds() == 1.5

    def test_to_scipp(self):
        d = Duration(42)
        s = d.to_scipp()
        assert s.unit == 'ns'
        assert s.value == 42


class TestDurationArithmetic:
    def test_add_durations(self):
        result = Duration(3) + Duration(7)
        assert isinstance(result, Duration)
        assert int(result) == 10

    def test_sub_durations(self):
        result = Duration(10) - Duration(3)
        assert isinstance(result, Duration)
        assert int(result) == 7

    def test_mul_int(self):
        result = Duration(5) * 3
        assert isinstance(result, Duration)
        assert int(result) == 15

    def test_rmul_int(self):
        result = 3 * Duration(5)
        assert isinstance(result, Duration)
        assert int(result) == 15

    def test_floordiv_int(self):
        result = Duration(10) // 3
        assert isinstance(result, Duration)
        assert int(result) == 3

    def test_floordiv_duration_returns_int(self):
        result = Duration(10) // Duration(3)
        assert isinstance(result, int)
        assert not isinstance(result, Duration)
        assert result == 3

    def test_truediv_duration_returns_float(self):
        result = Duration(10) / Duration(4)
        assert isinstance(result, float)
        assert result == 2.5

    def test_neg(self):
        result = -Duration(5)
        assert isinstance(result, Duration)
        assert int(result) == -5

    def test_add_timestamp_returns_timestamp(self):
        result = Duration(3) + Timestamp(10)
        assert isinstance(result, Timestamp)
        assert int(result) == 13

    def test_mul_timestamp_raises(self):
        with pytest.raises(TypeError):
            Duration(3) * Timestamp(10)  # type: ignore[operator]

    def test_mul_duration_raises(self):
        with pytest.raises(TypeError):
            Duration(3) * Duration(10)  # type: ignore[operator]

    def test_add_int_raises(self):
        with pytest.raises(TypeError):
            Duration(3) + 5  # type: ignore[operator]

    def test_sub_int_raises(self):
        with pytest.raises(TypeError):
            Duration(3) - 5  # type: ignore[operator]


class TestDurationComparison:
    def test_lt(self):
        assert Duration(1) < Duration(2)
        assert not Duration(2) < Duration(1)

    def test_le(self):
        assert Duration(1) <= Duration(2)
        assert Duration(2) <= Duration(2)

    def test_gt(self):
        assert Duration(2) > Duration(1)

    def test_ge(self):
        assert Duration(2) >= Duration(1)
        assert Duration(2) >= Duration(2)

    def test_eq(self):
        assert Duration(5) == Duration(5)
        assert Duration(5) != Duration(6)

    def test_compare_with_int_raises(self):
        with pytest.raises(TypeError):
            _ = Duration(5) < 10  # type: ignore[operator]

    def test_hash(self):
        assert hash(Duration(5)) == hash(Duration(5))
        assert {Duration(5): 'a'}[Duration(5)] == 'a'

    def test_bool_zero(self):
        assert not Duration(0)

    def test_bool_nonzero(self):
        assert Duration(1)


class TestTimestampFactory:
    def test_now(self):
        before = Timestamp(0)
        ts = Timestamp.now()
        assert ts > before

    def test_from_seconds(self):
        ts = Timestamp.from_seconds(1.5)
        assert int(ts) == 1_500_000_000

    def test_from_ms(self):
        ts = Timestamp.from_ms(1500)
        assert int(ts) == 1_500_000_000

    def test_to_seconds(self):
        ts = Timestamp(1_500_000_000)
        assert ts.to_seconds() == 1.5

    def test_to_datetime_utc(self):
        ts = Timestamp(1_700_000_000_000_000_000)
        dt = ts.to_datetime()
        assert dt.tzinfo is not None
        assert dt.year == 2023

    def test_to_datetime_with_tz(self):
        ts = Timestamp(1_700_000_000_000_000_000)
        dt = ts.to_datetime(tz=timezone.utc)
        assert dt.tzinfo == timezone.utc

    def test_to_scipp(self):
        ts = Timestamp(42)
        s = ts.to_scipp()
        assert s.unit == 'ns'
        assert s.value == 42


class TestTimestampArithmetic:
    def test_sub_timestamps_returns_duration(self):
        result = Timestamp(10) - Timestamp(3)
        assert isinstance(result, Duration)
        assert int(result) == 7

    def test_sub_duration_returns_timestamp(self):
        result = Timestamp(10) - Duration(3)
        assert isinstance(result, Timestamp)
        assert int(result) == 7

    def test_add_duration_returns_timestamp(self):
        result = Timestamp(10) + Duration(3)
        assert isinstance(result, Timestamp)
        assert int(result) == 13

    def test_radd_duration_returns_timestamp(self):
        result = Duration(3) + Timestamp(10)
        assert isinstance(result, Timestamp)
        assert int(result) == 13

    def test_add_timestamp_raises(self):
        with pytest.raises(TypeError):
            Timestamp(10) + Timestamp(3)  # type: ignore[operator]

    def test_add_int_raises(self):
        with pytest.raises(TypeError):
            Timestamp(10) + 3  # type: ignore[operator]

    def test_mul_raises(self):
        with pytest.raises(TypeError):
            Timestamp(10) * 2  # type: ignore[operator]

    def test_sub_int_raises(self):
        with pytest.raises(TypeError):
            Timestamp(10) - 5  # type: ignore[operator]


class TestTimestampComparison:
    def test_lt(self):
        assert Timestamp(1) < Timestamp(2)

    def test_le(self):
        assert Timestamp(1) <= Timestamp(2)
        assert Timestamp(2) <= Timestamp(2)

    def test_gt(self):
        assert Timestamp(2) > Timestamp(1)

    def test_ge(self):
        assert Timestamp(2) >= Timestamp(1)
        assert Timestamp(2) >= Timestamp(2)

    def test_eq(self):
        assert Timestamp(5) == Timestamp(5)
        assert Timestamp(5) != Timestamp(6)

    def test_compare_with_int_raises(self):
        with pytest.raises(TypeError):
            _ = Timestamp(5) < 10  # type: ignore[operator]

    def test_hash(self):
        assert hash(Timestamp(5)) == hash(Timestamp(5))
        assert {Timestamp(5): 'a'}[Timestamp(5)] == 'a'

    def test_bool_always_true(self):
        assert Timestamp(0)
        assert Timestamp(1)


class TestTimestampQuantize:
    def test_quantize_exact(self):
        ts = Timestamp(100)
        assert ts.quantize(Duration(10)) == Timestamp(100)

    def test_quantize_rounds_down(self):
        ts = Timestamp(105)
        assert ts.quantize(Duration(10)) == Timestamp(100)

    def test_quantize_up_exact(self):
        ts = Timestamp(100)
        assert ts.quantize_up(Duration(10)) == Timestamp(100)

    def test_quantize_up_rounds_up(self):
        ts = Timestamp(101)
        assert ts.quantize_up(Duration(10)) == Timestamp(110)


class TestSorting:
    def test_sort_timestamps(self):
        timestamps = [Timestamp(3), Timestamp(1), Timestamp(2)]
        assert sorted(timestamps) == [Timestamp(1), Timestamp(2), Timestamp(3)]

    def test_sort_durations(self):
        durations = [Duration(3), Duration(1), Duration(2)]
        assert sorted(durations) == [Duration(1), Duration(2), Duration(3)]

    def test_bisect_timestamps(self):
        timestamps = [Timestamp(1), Timestamp(3), Timestamp(5)]
        idx = bisect.bisect_right(timestamps, Timestamp(3))
        assert idx == 2

    def test_bisect_insort_timestamps(self):
        timestamps = [Timestamp(1), Timestamp(5)]
        bisect.insort(timestamps, Timestamp(3))
        assert timestamps == [Timestamp(1), Timestamp(3), Timestamp(5)]


class TestRepr:
    def test_timestamp_repr(self):
        assert repr(Timestamp(42)) == "Timestamp(42)"

    def test_duration_repr(self):
        assert repr(Duration(42)) == "Duration(42)"
