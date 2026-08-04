# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import scipp as sc
import structlog

from .workflow_factory import Workflow

logger = structlog.get_logger(__name__)


class TimeseriesStreamProcessor(Workflow):
    """Publishes newly arrived samples of a cumulative timeseries as deltas.

    The upstream preprocessor (``ToNXlog``) hands out its whole cumulative
    buffer on every cycle. This workflow republishes only the samples appended
    since the previous ``finalize``, identified by the timestamp of the last
    published sample rather than by a positional index: the preprocessor may
    drop samples off the front of its buffer to bound retention, which shifts
    every remaining sample's position. Timestamps are strictly increasing
    (enforced by ``ToNXlog.add``), so they are a stable cursor under trimming.

    Trimming is only safe while it stays behind the cursor. Once more samples
    arrive between two ``finalize`` calls than the preprocessor retains, samples
    are dropped before they are ever published and the delta stream has a silent
    gap; ``finalize`` warns when that becomes possible.
    """

    def __init__(self) -> None:
        self._data: sc.DataArray | None = None
        self._source_name: Hashable | None = None
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
        self._source_name, self._data = next(iter(data.items()))

    def finalize(self) -> dict[str, sc.DataArray]:
        if self._data is None:
            raise ValueError("No data has been added")

        times = self._data.coords['time'].values
        if self._last_returned_time is None:
            first_new = 0
        else:
            if times[0] > self._last_returned_time:
                # The last published sample is gone from the retained window, so
                # anything between it and times[0] was dropped before it could be
                # published. Whether such a sample existed is not observable from
                # the window alone, so this fires whenever loss is possible: in
                # steady state that means one cycle delivered more samples than
                # retention holds, and the retention bound is too tight for the
                # stream's rate.
                logger.warning(
                    "publication_cursor_outside_retained_window",
                    source_name=str(self._source_name),
                    last_published_time=str(self._last_returned_time),
                    oldest_retained_time=str(times[0]),
                )
            first_new = int(np.searchsorted(times, self._last_returned_time, 'right'))
        if first_new >= len(times):
            raise ValueError("No new data since last finalize")

        result = self._data['time', first_new:]
        self._last_returned_time = times[-1]

        return {'delta': result}

    def clear(self) -> None:
        self._data = None
        self._source_name = None
        self._last_returned_time = None
