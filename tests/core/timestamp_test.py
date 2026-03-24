# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import bisect
from datetime import timezone

import pytest

from ess.livedata.core.timestamp import Duration, Timestamp


class TestDurationFactory:
    def test_from_seconds(self):
        d = Duration.from_seconds(2.5)
        assert d.to_ns() == 2_500_000_000

    def test_from_ms(self):
        d = Duration.from_ms(150)
        assert d.to_ns() == 150_000_000

    def test_to_seconds(self):
        d = Duration.from_ns(1_500_000_000)
        assert d.to_seconds() == 1.5

    def test_to_scipp(self):
        d = Duration.from_ns(42)
        s = d.to_scipp()
        assert s.unit == 'ns'
        assert s.value == 42


class TestDurationArithmetic:
    def test_add_durations(self):
        result = Duration.from_ns(3) + Duration.from_ns(7)
        assert isinstance(result, Duration)
        assert result.to_ns() == 10

    def test_sub_durations(self):
        result = Duration.from_ns(10) - Duration.from_ns(3)
        assert isinstance(result, Duration)
        assert result.to_ns() == 7

    def test_mul_int(self):
        result = Duration.from_ns(5) * 3
        assert isinstance(result, Duration)
        assert result.to_ns() == 15

    def test_rmul_int(self):
        result = 3 * Duration.from_ns(5)
        assert isinstance(result, Duration)
        assert result.to_ns() == 15

    def test_floordiv_int(self):
        result = Duration.from_ns(10) // 3
        assert isinstance(result, Duration)
        assert result.to_ns() == 3

    def test_floordiv_duration_returns_int(self):
        result = Duration.from_ns(10) // Duration.from_ns(3)
        assert isinstance(result, int)
        assert not isinstance(result, Duration)
        assert result == 3

    def test_truediv_duration_returns_float(self):
        result = Duration.from_ns(10) / Duration.from_ns(4)
        assert isinstance(result, float)
        assert result == 2.5

    def test_neg(self):
        result = -Duration.from_ns(5)
        assert isinstance(result, Duration)
        assert result.to_ns() == -5

    def test_add_timestamp_returns_timestamp(self):
        result = Duration.from_ns(3) + Timestamp.from_ns(10)
        assert isinstance(result, Timestamp)
        assert result.to_ns() == 13

    def test_mul_timestamp_raises(self):
        with pytest.raises(TypeError):
            Duration.from_ns(3) * Timestamp.from_ns(10)  # type: ignore[operator]

    def test_mul_duration_raises(self):
        with pytest.raises(TypeError):
            Duration.from_ns(3) * Duration.from_ns(10)  # type: ignore[operator]

    def test_add_int_raises(self):
        with pytest.raises(TypeError):
            Duration.from_ns(3) + 5  # type: ignore[operator]

    def test_sub_int_raises(self):
        with pytest.raises(TypeError):
            Duration.from_ns(3) - 5  # type: ignore[operator]


class TestDurationComparison:
    def test_lt(self):
        assert Duration.from_ns(1) < Duration.from_ns(2)
        assert not Duration.from_ns(2) < Duration.from_ns(1)

    def test_le(self):
        assert Duration.from_ns(1) <= Duration.from_ns(2)
        assert Duration.from_ns(2) <= Duration.from_ns(2)

    def test_gt(self):
        assert Duration.from_ns(2) > Duration.from_ns(1)

    def test_ge(self):
        assert Duration.from_ns(2) >= Duration.from_ns(1)
        assert Duration.from_ns(2) >= Duration.from_ns(2)

    def test_eq(self):
        assert Duration.from_ns(5) == Duration.from_ns(5)
        assert Duration.from_ns(5) != Duration.from_ns(6)

    def test_compare_with_int_raises(self):
        with pytest.raises(TypeError):
            _ = Duration.from_ns(5) < 10  # type: ignore[operator]

    def test_hash(self):
        assert hash(Duration.from_ns(5)) == hash(Duration.from_ns(5))
        assert {Duration.from_ns(5): 'a'}[Duration.from_ns(5)] == 'a'

    def test_bool_zero(self):
        assert not Duration.from_ns(0)

    def test_bool_nonzero(self):
        assert Duration.from_ns(1)


class TestTimestampFactory:
    def test_now(self):
        before = Timestamp.from_ns(0)
        ts = Timestamp.now()
        assert ts > before

    def test_from_seconds(self):
        ts = Timestamp.from_seconds(1.5)
        assert ts.to_ns() == 1_500_000_000

    def test_from_ms(self):
        ts = Timestamp.from_ms(1500)
        assert ts.to_ns() == 1_500_000_000

    def test_to_seconds(self):
        ts = Timestamp.from_ns(1_500_000_000)
        assert ts.to_seconds() == 1.5

    def test_to_datetime_utc(self):
        ts = Timestamp.from_ns(1_700_000_000_000_000_000)
        dt = ts.to_datetime()
        assert dt.tzinfo is not None
        assert dt.year == 2023

    def test_to_datetime_with_tz(self):
        ts = Timestamp.from_ns(1_700_000_000_000_000_000)
        dt = ts.to_datetime(tz=timezone.utc)
        assert dt.tzinfo == timezone.utc

    def test_to_scipp(self):
        ts = Timestamp.from_ns(42)
        s = ts.to_scipp()
        assert s.unit == 'ns'
        assert s.value == 42


class TestTimestampArithmetic:
    def test_sub_timestamps_returns_duration(self):
        result = Timestamp.from_ns(10) - Timestamp.from_ns(3)
        assert isinstance(result, Duration)
        assert result.to_ns() == 7

    def test_sub_duration_returns_timestamp(self):
        result = Timestamp.from_ns(10) - Duration.from_ns(3)
        assert isinstance(result, Timestamp)
        assert result.to_ns() == 7

    def test_add_duration_returns_timestamp(self):
        result = Timestamp.from_ns(10) + Duration.from_ns(3)
        assert isinstance(result, Timestamp)
        assert result.to_ns() == 13

    def test_radd_duration_returns_timestamp(self):
        result = Duration.from_ns(3) + Timestamp.from_ns(10)
        assert isinstance(result, Timestamp)
        assert result.to_ns() == 13

    def test_add_timestamp_raises(self):
        with pytest.raises(TypeError):
            Timestamp.from_ns(10) + Timestamp.from_ns(3)  # type: ignore[operator]

    def test_add_int_raises(self):
        with pytest.raises(TypeError):
            Timestamp.from_ns(10) + 3  # type: ignore[operator]

    def test_mul_raises(self):
        with pytest.raises(TypeError):
            Timestamp.from_ns(10) * 2  # type: ignore[operator]

    def test_sub_int_raises(self):
        with pytest.raises(TypeError):
            Timestamp.from_ns(10) - 5  # type: ignore[operator]


class TestTimestampComparison:
    def test_lt(self):
        assert Timestamp.from_ns(1) < Timestamp.from_ns(2)

    def test_le(self):
        assert Timestamp.from_ns(1) <= Timestamp.from_ns(2)
        assert Timestamp.from_ns(2) <= Timestamp.from_ns(2)

    def test_gt(self):
        assert Timestamp.from_ns(2) > Timestamp.from_ns(1)

    def test_ge(self):
        assert Timestamp.from_ns(2) >= Timestamp.from_ns(1)
        assert Timestamp.from_ns(2) >= Timestamp.from_ns(2)

    def test_eq(self):
        assert Timestamp.from_ns(5) == Timestamp.from_ns(5)
        assert Timestamp.from_ns(5) != Timestamp.from_ns(6)

    def test_compare_with_int_raises(self):
        with pytest.raises(TypeError):
            _ = Timestamp.from_ns(5) < 10  # type: ignore[operator]

    def test_hash(self):
        assert hash(Timestamp.from_ns(5)) == hash(Timestamp.from_ns(5))
        assert {Timestamp.from_ns(5): 'a'}[Timestamp.from_ns(5)] == 'a'

    def test_bool_always_true(self):
        assert Timestamp.from_ns(0)
        assert Timestamp.from_ns(1)


class TestTimestampQuantize:
    def test_quantize_exact(self):
        ts = Timestamp.from_ns(100)
        assert ts.quantize(Duration.from_ns(10)) == Timestamp.from_ns(100)

    def test_quantize_rounds_down(self):
        ts = Timestamp.from_ns(105)
        assert ts.quantize(Duration.from_ns(10)) == Timestamp.from_ns(100)

    def test_quantize_up_exact(self):
        ts = Timestamp.from_ns(100)
        assert ts.quantize_up(Duration.from_ns(10)) == Timestamp.from_ns(100)

    def test_quantize_up_rounds_up(self):
        ts = Timestamp.from_ns(101)
        assert ts.quantize_up(Duration.from_ns(10)) == Timestamp.from_ns(110)


class TestSorting:
    def test_sort_timestamps(self):
        timestamps = [Timestamp.from_ns(3), Timestamp.from_ns(1), Timestamp.from_ns(2)]
        assert sorted(timestamps) == [
            Timestamp.from_ns(1),
            Timestamp.from_ns(2),
            Timestamp.from_ns(3),
        ]

    def test_sort_durations(self):
        durations = [Duration.from_ns(3), Duration.from_ns(1), Duration.from_ns(2)]
        assert sorted(durations) == [
            Duration.from_ns(1),
            Duration.from_ns(2),
            Duration.from_ns(3),
        ]

    def test_bisect_timestamps(self):
        timestamps = [Timestamp.from_ns(1), Timestamp.from_ns(3), Timestamp.from_ns(5)]
        idx = bisect.bisect_right(timestamps, Timestamp.from_ns(3))
        assert idx == 2

    def test_bisect_insort_timestamps(self):
        timestamps = [Timestamp.from_ns(1), Timestamp.from_ns(5)]
        bisect.insort(timestamps, Timestamp.from_ns(3))
        assert timestamps == [
            Timestamp.from_ns(1),
            Timestamp.from_ns(3),
            Timestamp.from_ns(5),
        ]


class TestRepr:
    def test_timestamp_repr(self):
        assert repr(Timestamp.from_ns(42)) == "Timestamp(ns=42)"

    def test_duration_repr(self):
        assert repr(Duration.from_ns(42)) == "Duration(ns=42)"
