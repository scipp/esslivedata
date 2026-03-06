# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import scipp as sc
from ess.reduce.nexus import group_event_data

# Apply monkey-patch to make group_event_data idempotent for pre-grouped data.
# See module docstring for rationale.
import ess.livedata.handlers._patch_group_event_data  # noqa: F401

from ..core.handler import Accumulator
from .to_nxevent_data import Events, ToNXevent_data


class GroupByPixel(Accumulator[Events, sc.DataArray]):
    """Accumulator that groups events by detector pixel.

    Wraps a ``ToNXevent_data`` accumulator and applies ``group_event_data``
    in ``get()``, producing events binned by ``detector_number``.

    This allows pixel grouping to happen once in the preprocessor rather than
    independently in every downstream workflow that consumes the same source.

    Parameters
    ----------
    inner:
        The underlying event accumulator.
    detector_number:
        Detector pixel numbers used for grouping.
    """

    def __init__(self, inner: ToNXevent_data, detector_number: sc.Variable) -> None:
        self._inner = inner
        # Always use flat 1D detector_number for grouping. The downstream
        # workflow's assemble_detector_data will fold to the correct
        # multi-dimensional shape using EmptyDetector's detector_number.
        self._detector_number = detector_number.flatten(to='detector_number')

    def add(self, timestamp: int, data: Events) -> None:
        self._inner.add(timestamp, data)

    def get(self) -> sc.DataArray:
        ungrouped = self._inner.get()
        return group_event_data(
            event_data=ungrouped,
            detector_number=self._detector_number,
        )

    def clear(self) -> None:
        self._inner.clear()
