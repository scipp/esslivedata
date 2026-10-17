# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.livedata.core.timestamp import Timestamp
from ess.livedata.preprocessors.frame_batch import FrameBatchAccumulator


def frame(value: float) -> sc.DataArray:
    return sc.DataArray(sc.full(dims=['x'], shape=[2], value=value, unit='counts'))


def test_get_returns_every_frame_with_its_timestamp() -> None:
    acc = FrameBatchAccumulator()
    acc.add(Timestamp.from_ns(10), frame(1.0))
    acc.add(Timestamp.from_ns(20), frame(2.0))

    batch = acc.get()

    assert [f.values[0] for f in batch.frames] == [1.0, 2.0]
    expected = sc.epoch(unit='ns') + sc.array(
        dims=['frame'], values=[10, 20], unit='ns', dtype='int64'
    )
    assert sc.identical(batch.times, expected)


def test_get_starts_a_new_batch() -> None:
    acc = FrameBatchAccumulator()
    acc.add(Timestamp.from_ns(10), frame(1.0))
    acc.get()
    acc.add(Timestamp.from_ns(20), frame(2.0))

    assert len(acc.get().frames) == 1


def test_get_without_frames_raises() -> None:
    with pytest.raises(ValueError, match="No data"):
        FrameBatchAccumulator().get()
