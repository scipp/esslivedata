# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Accumulator for synthesised device sample streams.

Consumes :class:`DeviceSample` events and grows a merged
:class:`scipp.DataArray` with ``value`` as data plus optional ``target``
and ``settled`` coords. Mirrors :class:`ToNXlog`'s structural contract:
pre-allocate, double on capacity, dedup duplicate/out-of-order timestamps
at ``add()``.
"""

from __future__ import annotations

import numpy as np
import scipp as sc
import structlog

from ess.livedata.core.handler import Accumulator
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.handlers.accumulators import DeviceSample

logger = structlog.get_logger(__name__)


class ToDeviceLog(Accumulator[DeviceSample, sc.DataArray]):
    """Grow a multi-coord DataArray from a stream of :class:`DeviceSample` events."""

    is_context = True

    def __init__(
        self,
        *,
        units: str | None = None,
        has_target: bool = False,
        has_settled: bool = False,
    ) -> None:
        self._unit = sc.Unit(units) if units is not None else None
        self._has_target = has_target
        self._has_settled = has_settled
        # Hard-coded time unit and start in the ESS NeXus filewriter
        self._time_unit = 'ns'
        self._start = sc.epoch(unit='ns')
        self._timeseries: sc.DataArray | None = None
        self._end = 0
        self._last_time: int | None = None

    @property
    def unit(self) -> sc.Unit | None:
        return self._unit

    def _at_capacity(self) -> bool:
        return self._end >= self._timeseries.sizes['time']

    def _ensure_capacity(self) -> None:
        if self._timeseries is None:
            values = sc.zeros(
                dims=['time'], shape=[2], unit=self._unit, dtype='float64'
            )
            times = sc.zeros(
                dims=['time'], shape=[2], unit=self._time_unit, dtype='int64'
            )
            coords: dict[str, sc.Variable] = {'time': self._start + times}
            if self._has_target:
                coords['target'] = sc.zeros(
                    dims=['time'], shape=[2], unit=self._unit, dtype='float64'
                )
            if self._has_settled:
                # int32 (not bool) because da00 serialization rejects bool.
                # The coord remains 0/1 valued.
                coords['settled'] = sc.zeros(dims=['time'], shape=[2], dtype='int32')
            self._timeseries = sc.DataArray(values, coords=coords)
        elif self._at_capacity():
            self._timeseries = sc.concat(
                [self._timeseries, self._timeseries], dim='time'
            )

    def add(self, timestamp: Timestamp, data: DeviceSample) -> bool:
        _ = timestamp
        sample_time = data.time.to_ns()
        if self._last_time is not None:
            if sample_time < self._last_time:
                logger.warning(
                    "out_of_order_device_sample_skipped",
                    sample_time=sample_time,
                    last_time=self._last_time,
                )
                return False
            if sample_time == self._last_time:
                return False

        self._ensure_capacity()
        self._timeseries.coords['time'].values[self._end] = sample_time
        self._timeseries.data.values[self._end] = data.value
        if self._has_target:
            # target may be None on a device whose VAL hasn't been observed yet
            # (only possible before bootstrap completes — synthesizer should not
            # emit before then). Fall back to NaN so the row is still appendable.
            self._timeseries.coords['target'].values[self._end] = (
                np.nan if data.target is None else data.target
            )
        if self._has_settled:
            self._timeseries.coords['settled'].values[self._end] = int(
                bool(data.settled)
            )
        self._end += 1
        self._last_time = sample_time
        return True

    def get(self) -> sc.DataArray:
        if self._timeseries is None:
            raise RuntimeError("No data has been added yet.")
        return self._timeseries['time', : self._end]

    def clear(self) -> None:
        self._end = 0
        self._last_time = None
