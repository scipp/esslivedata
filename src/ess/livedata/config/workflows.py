# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""Common workflows that are used by multiple instruments."""

from __future__ import annotations

from typing import Any, NewType

import pydantic
import sciline
import scipp as sc

from ess.livedata import parameter_models
from ess.livedata.config import Instrument
from ess.livedata.handlers.accumulators import LogData
from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow
from ess.livedata.handlers.to_nxlog import ToNXlog
from ess.reduce import streaming
from ess.reduce.nexus.types import Filename, MonitorData, NeXusData, NeXusName
from ess.reduce.time_of_flight import GenericTofWorkflow


class MonitorTimeseriesParams(pydantic.BaseModel):
    """Parameters for the monitor timeseries workflow."""

    toa_range: parameter_models.TOARange = pydantic.Field(
        title="Time of Arrival Range",
        description="Time of arrival range to include.",
        default=parameter_models.TOARange(),
    )


CustomMonitor = NewType('CustomMonitor', int)
CurrentRun = NewType('CurrentRun', int)
MonitorCountsInInterval = NewType('MonitorCountsInInterval', sc.DataArray)


def _get_interval(
    data: MonitorData[CurrentRun, CustomMonitor], range: parameter_models.TOARange
) -> MonitorCountsInInterval:
    start, stop = range.range_ns
    if data.bins is not None:
        counts = data.bins['event_time_offset', start:stop].sum()
        counts.coords['time'] = data.coords['event_time_zero'][0]
    else:
        # Include the full time(of arrival) bin at start and stop. Do we need more
        # precision here?
        # Note the current ECDC convention: time is the time offset w.r.t. the frame,
        # i.e., the pulse, frame_time is the absolute time (since epoch).
        counts = data['time', start:stop].sum()
        counts.coords['time'] = data.coords['frame_time'][0]
    return MonitorCountsInInterval(counts)


class TimeseriesAccumulator(streaming.Accumulator[sc.DataArray]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Use ToNXlog which is used as a preprocessor elsewhere (for raw f144). The
        # interface is similar but not identical, so we wrap instead of inheriting.
        self._to_nxlog: ToNXlog | None = None

    @property
    def is_empty(self) -> bool:
        return self._to_nxlog is None

    def _get_value(self) -> sc.DataArray:
        if self._to_nxlog is None:
            raise ValueError("No data accumulated")
        return self._to_nxlog.get()

    def _do_push(self, value: sc.DataArray) -> None:
        if self._to_nxlog is None:
            self._to_nxlog = ToNXlog(
                attrs={'units': str(value.unit)}, data_dims=tuple(value.dims)
            )
        self._to_nxlog.add(
            0,
            LogData(
                time=value.coords['time'].value,
                value=value.values,
                variances=value.variances,
            ),
        )

    def clear(self) -> None:
        if self._to_nxlog is not None:
            self._to_nxlog.clear()


def _prepare_workflow(instrument: Instrument, monitor_name: str) -> sciline.Pipeline:
    workflow = GenericTofWorkflow(run_types=[CurrentRun], monitor_types=[CustomMonitor])
    workflow[Filename[CurrentRun]] = instrument.nexus_file
    workflow[NeXusName[CustomMonitor]] = monitor_name
    workflow.insert(_get_interval)
    return workflow


def register_monitor_timeseries_workflows(
    instrument: Instrument, source_names: list[str]
) -> None:
    """Register monitor timeseries workflows for the given instrument and source names.

    Parameters
    ----------
    instrument
        The instrument for which to register the workflows.
    source_names
        The source names (monitor names) for which to register the workflows.
    """

    @instrument.register_workflow(
        name='monitor_interval_timeseries',
        version=1,
        title='Monitor Interval Timeseries',
        description='Timeseries of counts in a monitor within a specified '
        'time-of-arrival range.',
        source_names=source_names,
        aux_source_names=[],
    )
    def monitor_timeseries_workflow(
        source_name: str, params: MonitorTimeseriesParams
    ) -> StreamProcessorWorkflow:
        wf = _prepare_workflow(instrument, monitor_name=source_name)
        wf[parameter_models.TOARange] = params.toa_range
        return StreamProcessorWorkflow(
            base_workflow=wf,
            dynamic_keys={source_name: NeXusData[CustomMonitor, CurrentRun]},
            target_keys=(MonitorCountsInInterval,),
            accumulators={MonitorCountsInInterval: TimeseriesAccumulator},
        )
