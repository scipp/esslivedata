# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import scipp as sc

from ..core.handler import JobBasedPreprocessorFactoryBase
from ..core.message import StreamId, StreamKind
from .accumulators import Accumulator, CollectTOA, Cumulative, MonitorEvents
from .monitor_workflow_specs import MonitorDataParams
from .workflow_factory import Workflow


class MonitorStreamProcessor(Workflow):
    def __init__(self, edges: sc.Variable) -> None:
        self._edges = edges
        self._event_edges = edges.to(unit='ns').values
        self._cumulative: sc.DataArray | None = None
        self._current: sc.DataArray | None = None
        self._current_start_time: int | None = None

    @staticmethod
    def create_workflow(params: MonitorDataParams) -> Workflow:
        """Factory method for creating MonitorStreamProcessor from params."""
        return MonitorStreamProcessor(edges=params.toa_edges.get_edges())

    def accumulate(
        self,
        data: dict[Hashable, sc.DataArray | np.ndarray],
        *,
        start_time: int,
        end_time: int,
    ) -> None:
        if len(data) != 1:
            raise ValueError("MonitorStreamProcessor expects exactly one data item.")

        # Track start time of first data since last finalize
        if self._current_start_time is None:
            self._current_start_time = start_time

        raw = next(iter(data.values()))
        # Note: In theory we should consider rebinning/histogramming only in finalize(),
        # but the current plan is to accumulate before/during preprocessing, i.e.,
        # before we ever get here. That is, there should typically be one finalize()
        # call per accumulate() call.
        if isinstance(raw, np.ndarray):
            # Data from accumulators.CollectTOA.
            # Using NumPy here as for these specific operations with medium-sized data
            # it is a bit faster than Scipp.
            values, _ = np.histogram(raw, bins=self._event_edges)
            hist = sc.DataArray(
                data=sc.array(dims=[self._edges.dim], values=values, unit='counts'),
                coords={self._edges.dim: self._edges},
            )
        else:
            if raw.dim != self._edges.dim:
                raw = raw.rename({raw.dim: self._edges.dim})
            coord = raw.coords.get(self._edges.dim).to(unit=self._edges.unit)
            hist = raw.assign_coords({self._edges.dim: coord}).rebin(
                {self._edges.dim: self._edges}
            )
        if self._current is None:
            self._current = hist
        else:
            self._current += hist

    def finalize(self) -> dict[Hashable, sc.DataArray]:
        if self._current is None:
            raise ValueError("No data has been added")
        if self._current_start_time is None:
            raise RuntimeError(
                "finalize called without any data accumulated via accumulate"
            )

        current = self._current
        if self._cumulative is None:
            self._cumulative = current
        else:
            self._cumulative += current
        self._current = sc.zeros_like(current)

        # Add time coord to current result
        time_coord = sc.scalar(self._current_start_time, unit='ns')
        current = current.assign_coords(time=time_coord)
        self._current_start_time = None

        return {'cumulative': self._cumulative, 'current': current}

    def clear(self) -> None:
        self._cumulative = None
        self._current = None
        self._current_start_time = None


class MonitorHandlerFactory(
    JobBasedPreprocessorFactoryBase[MonitorEvents | sc.DataArray, sc.DataArray]
):
    def make_preprocessor(self, key: StreamId) -> Accumulator | None:
        match key.kind:
            case StreamKind.MONITOR_COUNTS:
                return Cumulative(clear_on_get=True)
            case StreamKind.MONITOR_EVENTS:
                return CollectTOA()
            case _:
                return None
