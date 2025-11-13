# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Hashable
from typing import NewType

import numpy as np
import scipp as sc

from ess.reduce.nexus.types import Filename, NeXusData, NeXusName
from ess.reduce.time_of_flight import GenericTofWorkflow

from .. import parameter_models
from ..config.workflows import TimeseriesAccumulator
from ..core.handler import JobBasedPreprocessorFactoryBase
from ..core.message import StreamId, StreamKind
from .accumulators import Accumulator, CollectTOA, Cumulative, MonitorEvents
from .monitor_workflow_specs import MonitorDataParams, MonitorTimeseriesParams
from .stream_processor_workflow import StreamProcessorWorkflow
from .workflow_factory import Workflow

# Type aliases for monitor interval timeseries workflow
CustomMonitor = NewType('CustomMonitor', int)
CurrentRun = NewType('CurrentRun', int)
MonitorCountsInInterval = NewType('MonitorCountsInInterval', sc.DataArray)


class MonitorStreamProcessor(Workflow):
    def __init__(self, edges: sc.Variable) -> None:
        self._edges = edges
        self._event_edges = edges.to(unit='ns').values
        self._cumulative: sc.DataArray | None = None
        self._current: sc.DataArray | None = None

    @staticmethod
    def create_workflow(params: MonitorDataParams) -> Workflow:
        """Factory method for creating MonitorStreamProcessor from params."""
        return MonitorStreamProcessor(edges=params.toa_edges.get_edges())

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


def _get_interval(
    data: NeXusData[CustomMonitor, CurrentRun],
    range: parameter_models.TOARange,
) -> MonitorCountsInInterval:
    """Extract monitor counts within a time-of-arrival interval."""
    start, stop = range.range_ns
    if data.bins is not None:
        counts = data.bins['event_time_offset', start:stop].sum()
        counts.coords['time'] = data.coords['event_time_zero'][0]
    else:
        counts = data['time', start:stop].sum()
        counts.coords['time'] = data.coords['frame_time'][0]
    return MonitorCountsInInterval(counts)


def create_monitor_interval_timeseries_factory(instrument):
    """
    Create factory function for monitor interval timeseries workflow.

    This is generic workflow logic that can be used by any instrument.
    Auto-attached by Instrument.load_factories().

    Parameters
    ----------
    instrument
        Instrument instance

    Returns
    -------
    :
        Factory function that can be attached to the spec handle
    """

    def factory(
        source_name: str,
        params: MonitorTimeseriesParams,
    ) -> StreamProcessorWorkflow:
        """Factory function for monitor interval timeseries workflow.

        Parameters
        ----------
        source_name:
            Monitor source name
        params:
            MonitorTimeseriesParams with toa_range configuration

        Returns
        -------
        :
            StreamProcessorWorkflow instance
        """
        wf = GenericTofWorkflow(run_types=[CurrentRun], monitor_types=[CustomMonitor])
        wf[Filename[CurrentRun]] = instrument.nexus_file
        wf[NeXusName[CustomMonitor]] = source_name
        wf[parameter_models.TOARange] = params.toa_range
        wf.insert(_get_interval)
        return StreamProcessorWorkflow(
            base_workflow=wf,
            dynamic_keys={source_name: NeXusData[CustomMonitor, CurrentRun]},
            target_keys={'monitor_counts': MonitorCountsInInterval},
            accumulators={MonitorCountsInInterval: TimeseriesAccumulator},
        )

    return factory
