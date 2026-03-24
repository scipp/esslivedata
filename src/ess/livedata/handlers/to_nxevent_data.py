# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import scipp as sc
from streaming_data_types import eventdata_ev44

from ess.livedata.core.handler import Accumulator
from ess.livedata.core.timestamp import Timestamp


def _require_single_pulse(ev44: eventdata_ev44.EventData) -> None:
    index = ev44.reference_time_index
    if len(index) > 1 or index[0] != 0 or len(ev44.reference_time) > 1:
        raise NotImplementedError("Processing multi-pulse messages is not supported.")


@dataclass
class MonitorEvents:
    """
    Dataclass for monitor events.

    Decouples our handlers from upstream schema changes. This also simplifies handler
    testing since tests do not have to construct a full eventdata_ev44.EventData object.

    Note that we keep the raw array of time of arrivals, and the unit. This is to avoid
    unnecessary copies of the data.
    """

    time_of_arrival: Sequence[int]
    unit: str

    @staticmethod
    def from_ev44(ev44: eventdata_ev44.EventData) -> MonitorEvents:
        _require_single_pulse(ev44)
        return MonitorEvents(time_of_arrival=ev44.time_of_flight, unit='ns')


@dataclass
class DetectorEvents(MonitorEvents):
    """
    Dataclass for detector events.

    Decouples our handlers from upstream schema changes. This also simplifies handler
    testing since tests do not have to construct a full eventdata_ev44.EventData object.

    Note that we keep the raw array of time of arrivals, and the unit. This is to avoid
    unnecessary copies of the data.
    """

    pixel_id: Sequence[int]

    def __post_init__(self) -> None:
        if len(self.pixel_id) != len(self.time_of_arrival):
            raise ValueError(
                f"pixel_id and time_of_arrival must have the same length, "
                f"got {len(self.pixel_id)} and {len(self.time_of_arrival)}"
            )

    @staticmethod
    def from_ev44(ev44: eventdata_ev44.EventData) -> DetectorEvents:
        _require_single_pulse(ev44)
        return DetectorEvents(
            pixel_id=ev44.pixel_id, time_of_arrival=ev44.time_of_flight, unit='ns'
        )


Events = TypeVar('Events', DetectorEvents, MonitorEvents)


class _ScippBackedBuffer:
    """A growable buffer backed by a scipp Variable.

    Chunks are copied directly into the Variable's underlying memory via its
    ``.values`` view.  When ``get()`` is called the filled region is returned as
    a zero-copy scipp slice — no additional allocation or memcpy needed.
    """

    def __init__(self, *, dtype: str, unit: str | None):
        self._dtype = dtype
        self._unit = unit
        self._var: sc.Variable | None = None
        self._view: np.ndarray | None = None  # writable view into _var

    def _ensure_capacity(self, n: int) -> None:
        capacity = 0 if self._var is None else self._var.sizes['event']
        if capacity >= max(n, 1):
            return
        new_capacity = max(n, 1, capacity * 2)
        self._var = sc.empty(
            dims=['event'],
            shape=[new_capacity],
            unit=self._unit,
            dtype=self._dtype,
        )
        self._view = self._var.values

    def fill_and_slice(self, chunks: list[np.ndarray]) -> sc.Variable:
        """Copy chunks into the buffer and return a zero-copy scipp slice."""
        total = sum(len(c) for c in chunks)
        self._ensure_capacity(total)
        offset = 0
        for chunk in chunks:
            n = len(chunk)
            self._view[offset : offset + n] = chunk
            offset += n
        return self._var['event', :total]


class _WeightsBuffer:
    """A reusable scipp Variable filled with ones."""

    def __init__(self) -> None:
        self._var: sc.Variable | None = None

    def get(self, n: int) -> sc.Variable:
        if self._var is not None and self._var.sizes['event'] >= n:
            return self._var['event', :n]
        new_capacity = max(n, 0 if self._var is None else self._var.sizes['event'] * 2)
        self._var = sc.ones(
            sizes={'event': new_capacity}, dtype='float64', unit='counts'
        )
        return self._var['event', :n]


class ToNXevent_data(Accumulator[Events, sc.DataArray]):
    def __init__(self):
        self._chunks: list[Events] = []
        self._timestamps: list[int] = []
        self._epoch = sc.epoch(unit='ns')
        self._have_event_id: bool | None = None
        self._toa_buf: _ScippBackedBuffer | None = None
        self._pid_buf: _ScippBackedBuffer | None = None
        self._weights = _WeightsBuffer()
        self._buffers_in_use = False

    def add(self, timestamp: Timestamp, data: Events) -> None:
        if data.unit != 'ns':
            raise ValueError(f"Expected unit 'ns', got '{data.unit}'")
        if self._have_event_id is None:
            self._have_event_id = isinstance(data, DetectorEvents)
            self._toa_buf = _ScippBackedBuffer(dtype='int32', unit='ns')
            if self._have_event_id:
                self._pid_buf = _ScippBackedBuffer(dtype='int32', unit=None)
        elif self._have_event_id != isinstance(data, DetectorEvents):
            # This should never happen, but we check to be safe.
            raise ValueError("Inconsistent event_id")
        self._timestamps.append(int(timestamp))
        self._chunks.append(data)

    def get(self) -> sc.DataArray:
        """Build a binned DataArray from the accumulated chunks.

        The returned DataArray shares its underlying buffers with this
        accumulator.  Callers must call :meth:`release_buffers` before the next
        :meth:`get` call to signal that the result is no longer in use.
        """
        if self._have_event_id is None:
            raise ValueError("No data has been added")
        if self._buffers_in_use:
            raise RuntimeError(
                "Buffers from a previous get() have not been released. "
                "Call release_buffers() after the result has been consumed."
            )
        self._buffers_in_use = True

        if self._toa_buf is None:
            raise RuntimeError("Expected _toa_buf to be initialized")
        toa_var = self._toa_buf.fill_and_slice(
            [d.time_of_arrival for d in self._chunks]
        )
        total = toa_var.sizes['event']

        events = sc.DataArray(
            data=self._weights.get(total),
            coords={'event_time_offset': toa_var},
        )

        if self._have_event_id:
            if self._pid_buf is None:
                raise RuntimeError("Expected _pid_buf to be initialized")
            events.coords['event_id'] = self._pid_buf.fill_and_slice(
                [d.pixel_id for d in self._chunks]
            )

        lens = [len(d.time_of_arrival) for d in self._chunks]
        sizes = sc.array(
            dims=['event_time_zero'], values=lens, unit=None, dtype='int64'
        )
        begin = sc.cumsum(sizes, mode='exclusive')
        binned = sc.DataArray(sc.bins(begin=begin, dim='event', data=events))
        binned.coords['event_time_zero'] = self._epoch + sc.array(
            dims=['event_time_zero'], values=self._timestamps, unit='ns', dtype='int64'
        )
        self.clear()
        return binned

    def release_buffers(self) -> None:
        """Signal that the result from the last :meth:`get` is no longer in use."""
        self._buffers_in_use = False

    def clear(self) -> None:
        self._chunks.clear()
        self._timestamps.clear()
