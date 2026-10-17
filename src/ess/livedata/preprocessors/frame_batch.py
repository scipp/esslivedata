# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Preprocessor keeping the individual frames of an area detector batch."""

from __future__ import annotations

from dataclasses import dataclass

import scipp as sc

from ..core.preprocessor import Accumulator
from ..core.timestamp import Timestamp


@dataclass(frozen=True)
class FrameBatch:
    """The frames received in one batch, with their detector timestamps.

    Parameters
    ----------
    times:
        Frame timestamps, ``datetime64[ns]`` with dimension ``frame``. Same type as
        the ``time`` coordinate of an NXlog, so frames can be looked up in one.
    frames:
        The frames, in arrival order.
    """

    times: sc.Variable
    frames: list[sc.DataArray]


class FrameBatchAccumulator(Accumulator[sc.DataArray, FrameBatch]):
    """
    Collect every frame of a batch instead of summing them.

    For workflows that must treat frames differently depending on information
    that varies between frames (e.g. an image type log), which a sum over the
    batch would destroy. Frames are held by reference, so this costs no more memory
    than the message batch itself. ``get`` hands out the frames and starts a new
    batch.
    """

    def __init__(self) -> None:
        self._times: list[int] = []
        self._frames: list[sc.DataArray] = []

    def add(self, timestamp: Timestamp, data: sc.DataArray) -> bool:
        self._times.append(timestamp.to_ns())
        self._frames.append(data)
        return True

    def get(self) -> FrameBatch:
        if not self._frames:
            raise ValueError("No data has been added")
        times = sc.epoch(unit='ns') + sc.array(
            dims=['frame'], values=self._times, unit='ns', dtype='int64'
        )
        batch = FrameBatch(times=times, frames=self._frames)
        self.clear()
        return batch

    def clear(self) -> None:
        self._times = []
        self._frames = []
