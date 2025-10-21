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
from .workflow_factory import SpecHandle, Workflow


class MonitorStreamProcessor(Workflow):
    def __init__(self, edges: sc.Variable) -> None:
        self._edges = edges
        self._event_edges = edges.to(unit='ns').values
        self._cumulative: sc.DataArray | None = None
        self._current: sc.DataArray | None = None

    def accumulate(self, data: dict[Hashable, sc.DataArray | np.ndarray]) -> None:
        if len(data) != 1:
            raise ValueError("MonitorStreamProcessor expects exactly one data item.")
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
        current = self._current
        if self._cumulative is None:
            self._cumulative = current
        else:
            self._cumulative += current
        self._current = sc.zeros_like(current)
        return {'cumulative': self._cumulative, 'current': current}

    def clear(self) -> None:
        self._cumulative = None
        self._current = None


def _monitor_data_workflow(params: MonitorDataParams) -> Workflow:
    return MonitorStreamProcessor(edges=params.toa_edges.get_edges())


def attach_monitor_workflow_factory(handle: SpecHandle) -> None:
    """
    Attach monitor workflow factory to the spec handle (heavy dependencies).

    This is the second phase of two-phase registration. Call this from
    instrument factories.py modules.

    Parameters
    ----------
    handle
        The spec handle returned by register_monitor_workflow_specs().
    """
    handle.attach_factory()(_monitor_data_workflow)


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
