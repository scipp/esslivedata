# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import scipp as sc

from .workflow_factory import Workflow


class TimeseriesStreamProcessor(Workflow):
    """Publishes newly arrived samples of a cumulative timeseries as deltas.

    The upstream preprocessor (``ToNXlog``) hands out its whole cumulative
    buffer on every cycle. This workflow republishes only the samples appended
    since the previous ``finalize``, identified by the timestamp of the last
    published sample rather than by a positional index: the preprocessor may
    drop samples off the front of its buffer to bound retention, which shifts
    every remaining sample's position. Timestamps are strictly increasing
    (enforced by ``ToNXlog.add``), so they are a stable cursor under trimming.
    """

    def __init__(self) -> None:
        self._data: sc.DataArray | None = None
        self._last_returned_time: Any = None

    @staticmethod
    def create_workflow() -> Workflow:
        """Factory method for creating TimeseriesStreamProcessor."""
        return TimeseriesStreamProcessor()

    def accumulate(
        self, data: dict[Hashable, sc.DataArray], *, start_time: int, end_time: int
    ) -> None:
        if len(data) != 1:
            raise ValueError("Timeseries processor expects exactly one data item.")
        # Store the full cumulative data (including history from preprocessor)
        self._data = next(iter(data.values()))

    def finalize(self) -> dict[str, sc.DataArray]:
        if self._data is None:
            raise ValueError("No data has been added")

        times = self._data.coords['time'].values
        if self._last_returned_time is None:
            first_new = 0
        else:
            first_new = int(np.searchsorted(times, self._last_returned_time, 'right'))
        if first_new >= len(times):
            raise ValueError("No new data since last finalize")

        result = self._data['time', first_new:]
        self._last_returned_time = times[-1]

        return {'delta': result}

    def clear(self) -> None:
        self._data = None
        self._last_returned_time = None
