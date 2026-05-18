# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for :class:`ToDeviceLog`."""

from __future__ import annotations

import scipp as sc
from scipp.testing import assert_identical

from ess.livedata.core.timestamp import Timestamp
from ess.livedata.handlers.accumulators import DeviceSample
from ess.livedata.handlers.to_device_log import ToDeviceLog


def _ts(ns: int) -> Timestamp:
    return Timestamp.from_ns(ns)


def test_grows_value_only_dataarray() -> None:
    acc = ToDeviceLog(units='mm')
    acc.add(_ts(0), DeviceSample(time=_ts(1000), value=1.0))
    acc.add(_ts(0), DeviceSample(time=_ts(2000), value=2.0))
    result = acc.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[1.0, 2.0], unit='mm'))
    assert 'target' not in result.coords
    assert 'settled' not in result.coords


def test_grows_with_target_and_settled_coords() -> None:
    acc = ToDeviceLog(units='mm', has_target=True, has_settled=True)
    acc.add(_ts(0), DeviceSample(time=_ts(1000), value=1.0, target=10.0, settled=False))
    acc.add(_ts(0), DeviceSample(time=_ts(2000), value=2.0, target=10.0, settled=True))
    result = acc.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[1.0, 2.0], unit='mm'))
    assert_identical(
        result.coords['target'],
        sc.array(dims=['time'], values=[10.0, 10.0], unit='mm'),
    )
    assert_identical(
        result.coords['settled'],
        sc.array(dims=['time'], values=[False, True]),
    )


def test_dedup_duplicate_timestamps() -> None:
    acc = ToDeviceLog(units='mm', has_target=True, has_settled=True)
    acc.add(_ts(0), DeviceSample(time=_ts(1000), value=1.0, target=5.0, settled=False))
    # Duplicate time — should be silently dropped.
    acc.add(_ts(0), DeviceSample(time=_ts(1000), value=1.5, target=5.0, settled=True))
    acc.add(_ts(0), DeviceSample(time=_ts(2000), value=2.0, target=5.0, settled=True))
    result = acc.get()
    assert result.sizes == {'time': 2}
    assert_identical(result.data, sc.array(dims=['time'], values=[1.0, 2.0], unit='mm'))


def test_out_of_order_dropped() -> None:
    acc = ToDeviceLog(units='mm')
    acc.add(_ts(0), DeviceSample(time=_ts(2000), value=2.0))
    acc.add(_ts(0), DeviceSample(time=_ts(1000), value=1.0))  # earlier — dropped
    acc.add(_ts(0), DeviceSample(time=_ts(3000), value=3.0))
    result = acc.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[2.0, 3.0], unit='mm'))


def test_capacity_grows() -> None:
    acc = ToDeviceLog(units='mm', has_target=True)
    for i in range(20):
        acc.add(
            _ts(0),
            DeviceSample(time=_ts((i + 1) * 1000), value=float(i), target=float(i + 1)),
        )
    result = acc.get()
    assert result.sizes == {'time': 20}
    assert_identical(
        result.coords['target'],
        sc.array(
            dims=['time'],
            values=[float(i + 1) for i in range(20)],
            unit='mm',
        ),
    )


def test_clear_resets_state() -> None:
    acc = ToDeviceLog(units='mm')
    acc.add(_ts(0), DeviceSample(time=_ts(1000), value=1.0))
    acc.clear()
    # Same timestamp accepted after clear
    acc.add(_ts(0), DeviceSample(time=_ts(1000), value=2.0))
    result = acc.get()
    assert_identical(result.data, sc.array(dims=['time'], values=[2.0], unit='mm'))
