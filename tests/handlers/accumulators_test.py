# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from ess.reduce import streaming
from streaming_data_types import logdata_f144

from ess.livedata.core.handler import Accumulator
from ess.livedata.handlers.accumulators import (
    Cumulative,
    LatestValue,
    LatestValueHandler,
    LogData,
    NoCopyAccumulator,
    NoCopyWindowAccumulator,
    NullAccumulator,
)
from ess.livedata.handlers.to_nxevent_data import ToNXevent_data


def test_LogData_from_f144() -> None:
    f144_data = logdata_f144.ExtractedLogData(
        source_name='abc', value=42.0, timestamp_unix_ns=12345
    )

    log_data = LogData.from_f144(f144_data)
    assert log_data.time == 12345
    assert log_data.value == 42.0


@pytest.mark.parametrize(
    'accumulator_cls', [Cumulative, LatestValueHandler, ToNXevent_data]
)
def test_accumulator_raises_if_get_before_add(
    accumulator_cls: type[Accumulator],
) -> None:
    accumulator = accumulator_cls()
    with pytest.raises(ValueError, match="No data has been added"):
        accumulator.get()


@pytest.mark.parametrize('accumulator_cls', [Cumulative])
def test_accumulator_clears_when_data_sizes_changes(
    accumulator_cls: type[Accumulator],
) -> None:
    cumulative = accumulator_cls(config={})
    da = sc.DataArray(
        sc.array(dims=['x'], values=[1.0], unit='counts'),
        coords={'x': sc.arange('x', 2, unit='s')},
    )
    cumulative.add(0, da)
    cumulative.add(1, da)
    da2 = sc.DataArray(
        sc.array(dims=['y'], values=[1.0], unit='counts'),
        coords={'y': sc.arange('y', 2, unit='s')},
    )
    cumulative.add(2, da2)
    cumulative.add(3, da2)
    assert sc.identical(
        cumulative.get().data, sc.array(dims=['y'], values=[2.0], unit='counts')
    )


class TestNullAccumulator:
    def test_add_does_nothing(self) -> None:
        accumulator = NullAccumulator()
        accumulator.add(0, "some data")
        # Should not raise any exceptions

    def test_get_returns_none(self) -> None:
        accumulator = NullAccumulator()
        assert accumulator.get() is None

    def test_clear_does_nothing(self) -> None:
        accumulator = NullAccumulator()
        accumulator.clear()
        # Should not raise any exceptions


class TestLatestValueHandler:
    """Tests for LatestValueHandler (handler-style accumulator with add/get/clear)."""

    def test_get_before_add_raises_error(self) -> None:
        accumulator = LatestValueHandler()
        with pytest.raises(ValueError, match="No data has been added"):
            accumulator.get()

    def test_keeps_latest_value_only(self) -> None:
        accumulator = LatestValueHandler()
        da1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0], unit='m'),
            coords={'x': sc.array(dims=['x'], values=[0.0], unit='s')},
        )
        da2 = sc.DataArray(
            sc.array(dims=['x'], values=[2.0], unit='m'),
            coords={'x': sc.array(dims=['x'], values=[1.0], unit='s')},
        )
        da3 = sc.DataArray(
            sc.array(dims=['y'], values=[3.0, 4.0], unit='m'),
            coords={'y': sc.array(dims=['y'], values=[0.0, 1.0], unit='s')},
        )

        accumulator.add(0, da1)
        result = accumulator.get()
        assert sc.identical(result, da1)

        # Add second value - should replace first
        accumulator.add(1, da2)
        result = accumulator.get()
        assert sc.identical(result, da2)

        # Add third value with different shape - should still replace
        accumulator.add(2, da3)
        result = accumulator.get()
        assert sc.identical(result, da3)

    def test_stores_copy_not_reference(self) -> None:
        accumulator = LatestValueHandler()
        da = sc.DataArray(
            sc.array(dims=['x'], values=[1.0], unit='m'),
            coords={'x': sc.array(dims=['x'], values=[0.0], unit='s')},
        )

        accumulator.add(0, da)
        # Modify original
        da.values[0] = 999.0

        # Should not affect stored value
        result = accumulator.get()
        assert result.values[0] == 1.0

    def test_clear(self) -> None:
        accumulator = LatestValueHandler()
        da = sc.DataArray(
            sc.array(dims=['x'], values=[1.0], unit='m'),
            coords={'x': sc.array(dims=['x'], values=[0.0], unit='s')},
        )

        accumulator.add(0, da)
        accumulator.clear()

        with pytest.raises(ValueError, match="No data has been added"):
            accumulator.get()

    def test_timestamp_parameter_is_ignored(self) -> None:
        accumulator = LatestValueHandler()
        da = sc.DataArray(
            sc.array(dims=['x'], values=[1.0], unit='m'),
            coords={'x': sc.array(dims=['x'], values=[0.0], unit='s')},
        )

        # Timestamp should not affect behavior
        accumulator.add(12345, da)
        result = accumulator.get()
        assert sc.identical(result, da)


class TestLatestValue:
    """Tests for LatestValue (streaming-style accumulator with push/value)."""

    def test_value_before_push_raises_error(self) -> None:
        accumulator: LatestValue[sc.DataArray] = LatestValue()
        with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
            _ = accumulator.value

    def test_is_empty_before_push(self) -> None:
        accumulator: LatestValue[sc.DataArray] = LatestValue()
        assert accumulator.is_empty

    def test_keeps_latest_value_only(self) -> None:
        accumulator: LatestValue[sc.DataArray] = LatestValue()
        da1 = sc.DataArray(sc.array(dims=['x'], values=[1.0], unit='m'))
        da2 = sc.DataArray(sc.array(dims=['x'], values=[2.0], unit='m'))

        accumulator.push(da1)
        assert sc.identical(accumulator.value, da1)
        assert not accumulator.is_empty

        accumulator.push(da2)
        assert sc.identical(accumulator.value, da2)

    def test_clear(self) -> None:
        accumulator: LatestValue[sc.DataArray] = LatestValue()
        da = sc.DataArray(sc.array(dims=['x'], values=[1.0], unit='m'))

        accumulator.push(da)
        accumulator.clear()

        assert accumulator.is_empty
        with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
            _ = accumulator.value


class TestNoCopyAccumulator:
    """Tests for NoCopyAccumulator (streaming accumulator without deepcopy)."""

    def test_value_before_push_raises_error(self) -> None:
        accumulator = NoCopyAccumulator()
        with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
            _ = accumulator.value

    def test_is_empty_before_push(self) -> None:
        accumulator = NoCopyAccumulator()
        assert accumulator.is_empty

    def test_accumulates_values(self) -> None:
        accumulator = NoCopyAccumulator()
        da1 = sc.DataArray(sc.array(dims=['x'], values=[1.0], unit='m'))
        da2 = sc.DataArray(sc.array(dims=['x'], values=[2.0], unit='m'))

        accumulator.push(da1)
        assert sc.identical(accumulator.value, da1)

        accumulator.push(da2)
        expected = sc.DataArray(sc.array(dims=['x'], values=[3.0], unit='m'))
        assert sc.identical(accumulator.value, expected)

    def test_clear(self) -> None:
        accumulator = NoCopyAccumulator()
        da = sc.DataArray(sc.array(dims=['x'], values=[1.0], unit='m'))

        accumulator.push(da)
        accumulator.clear()

        assert accumulator.is_empty


class TestNoCopyWindowAccumulator:
    """Tests for NoCopyWindowAccumulator (clears on finalize)."""

    def test_is_empty_initially(self) -> None:
        acc = NoCopyWindowAccumulator()
        assert acc.is_empty

    def test_push_makes_not_empty(self) -> None:
        acc = NoCopyWindowAccumulator()
        acc.push(sc.array(dims=['x'], values=[1.0, 2.0]))
        assert not acc.is_empty

    def test_value_returns_pushed_data(self) -> None:
        acc = NoCopyWindowAccumulator()
        data = sc.array(dims=['x'], values=[1.0, 2.0, 3.0])
        acc.push(data)
        result = acc.value
        assert sc.identical(result, data)

    def test_on_finalize_clears_accumulator(self) -> None:
        acc = NoCopyWindowAccumulator()
        acc.push(sc.array(dims=['x'], values=[1.0, 2.0]))
        assert not acc.is_empty
        acc.on_finalize()
        assert acc.is_empty

    def test_accumulates_values(self) -> None:
        acc = NoCopyWindowAccumulator()
        data1 = sc.array(dims=['x'], values=[1.0, 2.0])
        data2 = sc.array(dims=['x'], values=[3.0, 4.0])
        acc.push(data1)
        acc.push(data2)
        result = acc.value
        expected = data1 + data2
        assert sc.identical(result, expected)

    def test_value_after_on_finalize_raises(self) -> None:
        acc = NoCopyWindowAccumulator()
        acc.push(sc.array(dims=['x'], values=[1.0]))
        acc.on_finalize()
        with pytest.raises(ValueError, match="empty"):
            _ = acc.value

    def test_differs_from_eternal_accumulator_behavior(self) -> None:
        """NoCopyWindowAccumulator clears after on_finalize.

        Unlike EternalAccumulator which preserves its state.
        """
        window_acc = NoCopyWindowAccumulator()
        eternal_acc = streaming.EternalAccumulator()

        data = sc.array(dims=['x'], values=[1.0, 2.0])
        window_acc.push(data)
        eternal_acc.push(data)

        # Both are non-empty before on_finalize
        assert not window_acc.is_empty
        assert not eternal_acc.is_empty

        # Call on_finalize
        window_acc.on_finalize()
        eternal_acc.on_finalize()

        # NoCopyWindowAccumulator is cleared, EternalAccumulator is not
        assert window_acc.is_empty
        assert not eternal_acc.is_empty


class TestCumulative:
    def test_get_before_add_raises_error(self) -> None:
        accumulator = Cumulative()
        with pytest.raises(ValueError, match="No data has been added"):
            accumulator.get()

    def test_cumulative_no_clear_on_get(self) -> None:
        cumulative = Cumulative(config={}, clear_on_get=False)
        da = sc.DataArray(
            sc.array(dims=['x'], values=[1.0], unit='counts'),
            coords={'x': sc.arange('x', 2, unit='s')},
        )
        cumulative.add(0, da)
        assert sc.identical(
            cumulative.get().data, sc.array(dims=['x'], values=[1.0], unit='counts')
        )
        cumulative.add(1, da)
        assert sc.identical(
            cumulative.get().data, sc.array(dims=['x'], values=[2.0], unit='counts')
        )

    def test_cumulative_clear_on_get(self) -> None:
        cumulative = Cumulative(config={}, clear_on_get=True)
        da = sc.DataArray(
            sc.array(dims=['x'], values=[1.0], unit='counts'),
            coords={'x': sc.arange('x', 2, unit='s')},
        )
        cumulative.add(0, da)
        assert sc.identical(
            cumulative.get().data, sc.array(dims=['x'], values=[1.0], unit='counts')
        )
        cumulative.add(1, da)
        assert sc.identical(
            cumulative.get().data, sc.array(dims=['x'], values=[1.0], unit='counts')
        )

    def test_clears_when_data_sizes_changes(self) -> None:
        cumulative = Cumulative(config={})
        da = sc.DataArray(
            sc.array(dims=['x'], values=[1.0], unit='counts'),
            coords={'x': sc.arange('x', 2, unit='s')},
        )
        cumulative.add(0, da)
        cumulative.add(1, da)
        da2 = sc.DataArray(
            sc.array(dims=['y'], values=[1.0], unit='counts'),
            coords={'y': sc.arange('y', 2, unit='s')},
        )
        cumulative.add(2, da2)
        cumulative.add(3, da2)
        assert sc.identical(
            cumulative.get().data, sc.array(dims=['y'], values=[2.0], unit='counts')
        )

    def test_clears_when_coords_change(self) -> None:
        cumulative = Cumulative(config={})
        da1 = sc.DataArray(
            sc.array(dims=['x'], values=[1.0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[0.0], unit='s')},
        )
        da2 = sc.DataArray(
            sc.array(dims=['x'], values=[2.0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[1.0], unit='s')},
        )
        cumulative.add(0, da1)
        cumulative.add(1, da2)
        # Should have cleared and only contain da2
        assert sc.identical(
            cumulative.get().data, sc.array(dims=['x'], values=[2.0], unit='counts')
        )

    def test_manual_clear(self) -> None:
        cumulative = Cumulative(config={})
        da = sc.DataArray(
            sc.array(dims=['x'], values=[1.0], unit='counts'),
            coords={'x': sc.arange('x', 2, unit='s')},
        )
        cumulative.add(0, da)
        cumulative.clear()
        with pytest.raises(ValueError, match="No data has been added"):
            cumulative.get()

    def test_accumulates_2d_data_without_coordinates(self) -> None:
        """Area detector images are 2D and typically have no coordinates."""
        cumulative = Cumulative(config={})
        da = sc.DataArray(
            sc.array(dims=['y', 'x'], values=[[1, 2], [3, 4]], unit='counts'),
        )
        cumulative.add(0, da)
        cumulative.add(1, da)
        result = cumulative.get()
        expected = sc.array(dims=['y', 'x'], values=[[2, 4], [6, 8]], unit='counts')
        assert sc.identical(result.data, expected)

    def test_2d_data_resets_when_sizes_change(self) -> None:
        cumulative = Cumulative(config={})
        da1 = sc.DataArray(
            sc.array(dims=['y', 'x'], values=[[1, 2], [3, 4]], unit='counts'),
        )
        da2 = sc.DataArray(
            sc.array(dims=['y', 'x'], values=[[5, 6, 7], [8, 9, 10]], unit='counts'),
        )
        cumulative.add(0, da1)
        cumulative.add(1, da1)
        cumulative.add(2, da2)  # Different shape, should reset
        result = cumulative.get()
        assert sc.identical(result.data, da2.data)
