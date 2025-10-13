# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""Common workflows that are used by multiple instruments."""

from __future__ import annotations

from enum import Enum
from typing import Any, NewType

import pydantic
import scipp as sc

from ess.livedata import parameter_models
from ess.livedata.config import Instrument
from ess.livedata.handlers.accumulators import LogData
from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow
from ess.livedata.handlers.to_nxlog import ToNXlog
from ess.reduce import streaming
from ess.reduce.nexus.types import Filename, MonitorData, NeXusData, NeXusName
from ess.reduce.time_of_flight import GenericTofWorkflow


class IntervalMode(str, Enum):
    """Mode for selecting interval: time-of-arrival or wavelength."""

    TIME_OF_ARRIVAL = 'time_of_arrival'
    WAVELENGTH = 'wavelength'


class MonitorTimeseriesParams(pydantic.BaseModel):
    """Parameters for the monitor timeseries workflow."""

    interval_mode: IntervalMode = pydantic.Field(
        title='Interval Mode',
        description='Select interval by time-of-arrival or wavelength.',
        default=IntervalMode.TIME_OF_ARRIVAL,
    )
    toa_range: parameter_models.TOARange = pydantic.Field(
        title="Time of Arrival Range",
        description=(
            "Time of arrival range to include (used when mode is time_of_arrival)."
        ),
        default=parameter_models.TOARange(),
    )
    wavelength_range: parameter_models.WavelengthRange = pydantic.Field(
        title='Wavelength Range',
        description='Wavelength range to include (used when mode is wavelength).',
        default_factory=parameter_models.WavelengthRange,
    )


CustomMonitor = NewType('CustomMonitor', int)
CurrentRun = NewType('CurrentRun', int)
MonitorCountsInInterval = NewType('MonitorCountsInInterval', sc.DataArray)


def _extract_interval(
    data: sc.DataArray,
    dim: str,
    interval: parameter_models.RangeModel,
) -> sc.DataArray:
    """Extract counts in an interval from monitor data.

    Parameters
    ----------
    data
        Monitor data to extract interval from.
    dim
        Dimension to slice along (e.g., 'event_time_offset', 'time', 'wavelength').
    interval
        Range model specifying the start and stop of the interval.

    Returns
    -------
    :
        Data array with counts in the interval and a 'time' coordinate.
    """
    # Get start/stop from the interval and convert to the unit of the dimension
    start = interval.get_start()
    stop = interval.get_stop()
    if data.bins is not None:
        coord = data.bins.coords[dim]
    else:
        coord = data.coords[dim]
    start = start.to(unit=coord.unit, copy=False)
    stop = stop.to(unit=coord.unit, copy=False)

    # Note the current ECDC convention: time is the time offset w.r.t. the frame,
    # i.e., the pulse, frame_time is the absolute time (since epoch).
    time_coord = 'event_time_zero' if data.bins is not None else 'frame_time'

    if data.bins is not None:
        counts = data.bins[dim, start:stop].sum()
        counts.coords['time'] = data.coords[time_coord][0]
    else:
        # Include the full bin at start and stop. Do we need more precision here?
        counts = data[dim, start:stop].sum()
        counts.coords['time'] = data.coords[time_coord][0]
    return counts


def _get_interval(
    data: MonitorData[CurrentRun, CustomMonitor], range: parameter_models.TOARange
) -> MonitorCountsInInterval:
    dim = 'event_time_offset' if data.bins is not None else 'time'
    counts = _extract_interval(data, dim, range)
    return MonitorCountsInInterval(counts)


def _get_interval_by_wavelength(
    data: MonitorData[CurrentRun, CustomMonitor],
    range: parameter_models.WavelengthRange,
) -> MonitorCountsInInterval:
    """Get monitor counts in a wavelength interval."""
    counts = _extract_interval(data, 'wavelength', range)
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
        'time-of-arrival or wavelength range.',
        source_names=source_names,
        aux_source_names=[],
    )
    def monitor_timeseries_workflow(
        source_name: str, params: MonitorTimeseriesParams
    ) -> StreamProcessorWorkflow:
        wf = GenericTofWorkflow(run_types=[CurrentRun], monitor_types=[CustomMonitor])
        wf[Filename[CurrentRun]] = instrument.nexus_file
        wf[NeXusName[CustomMonitor]] = source_name

        # Insert the appropriate interval function based on mode
        if params.interval_mode == IntervalMode.TIME_OF_ARRIVAL:
            wf[parameter_models.TOARange] = params.toa_range
            wf.insert(_get_interval)
        else:  # WAVELENGTH
            wf[parameter_models.WavelengthRange] = params.wavelength_range
            wf.insert(_get_interval_by_wavelength)

        return StreamProcessorWorkflow(
            base_workflow=wf,
            dynamic_keys={source_name: NeXusData[CustomMonitor, CurrentRun]},
            target_keys=(MonitorCountsInInterval,),
            accumulators={MonitorCountsInInterval: TimeseriesAccumulator},
        )
