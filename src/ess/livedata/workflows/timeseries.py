# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Hashable

import scipp as sc

from .workflow_factory import Workflow


class TimeseriesStreamProcessor(Workflow):
    def __init__(self) -> None:
        self._data: sc.DataArray | None = None
        self._last_returned_index = 0

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

        # Return only new data since last finalize to avoid republishing full history
        current_size = self._data.sizes['time']
        if self._last_returned_index >= current_size:
            raise ValueError("No new data since last finalize")

        result = self._data['time', self._last_returned_index :]
        self._last_returned_index = current_size

        return {'delta': result}

    def clear(self) -> None:
        self._data = None
        self._last_returned_index = 0
