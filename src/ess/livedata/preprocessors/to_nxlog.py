import zlib
from typing import Any

import numpy as np
import scipp as sc
import structlog

from ess.livedata.config.stream import Device, F144Stream, Stream
from ess.livedata.core.preprocessor import Accumulator
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.preprocessors.accumulators import LogData

logger = structlog.get_logger(__name__)

#: Retained samples per log. Sized so that logs slow enough to matter for
#: long-running experiments keep more than a week of history: a sample every
#: two seconds fills this in 23 days, one per second in 12 days. Fast logs
#: (choppers at the 14 Hz pulse rate) hit it after ~20 hours, which is the
#: intended asymmetry -- they are read for recent behaviour, not for history.
DEFAULT_MAX_SIZE = 1_000_000

#: Retained timespan per log. Binds only for logs too slow to reach
#: :data:`DEFAULT_MAX_SIZE`, bounding the history of a device that ticks once a
#: minute just as the size cap bounds a chopper.
DEFAULT_MAX_AGE = sc.scalar(30, unit='day')


class ToNXlog(Accumulator[LogData, sc.DataArray]):
    """
    Preprocessor for log data.

    Accumulates LogData objects and returns a single DataArray as it would be read from
    an NXlog in a NeXus file.

    History is bounded by ``max_size`` samples and ``max_age`` of timespan,
    whichever binds first; the oldest samples are dropped once either is
    exceeded. Age is measured against the newest buffered sample rather than
    wall clock, so a device that stops reporting keeps its last known value
    instead of ageing out to an empty buffer -- consumers rely on the buffer
    being non-empty, and the last value of a stalled device is exactly what
    NXlog semantics must preserve.

    Timestamps must be monotonically increasing. Messages with duplicate or out-of-order
    timestamps are skipped to prevent unbounded buffer growth from upstream re-sends.

    When ``has_target`` / ``has_idle`` are set, the per-sample ``target`` /
    ``idle`` fields on :class:`LogData` are written as time-dim coords. This
    is used by the device-synthesis path; plain f144 logs leave both flags off.
    """

    is_context = True

    def __init__(
        self,
        *,
        attrs: dict[str, Any],
        data_dims: tuple[str, ...] = (),
        has_target: bool = False,
        has_idle: bool = False,
        max_size: int = DEFAULT_MAX_SIZE,
        max_age: sc.Variable = DEFAULT_MAX_AGE,
        phase: int = 0,
    ) -> None:
        self._attrs = attrs
        # Values with no unit are ok
        maybe_unit = self._attrs.get('units')
        if maybe_unit is None:
            self._unit = None
        else:
            self._unit = sc.Unit(maybe_unit)
        # Hard-coded time unit and start in the ESS NeXus filewriter
        self._time_unit = 'ns'
        self._epoch = sc.epoch(unit='ns')

        # Initialize with None, will be created on first add
        self._timeseries: sc.DataArray | None = None
        self._start = 0
        self._end = 0
        self._last_time: int | None = None
        self._data_dims = data_dims
        self._has_target = has_target
        self._has_idle = has_idle

        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}")
        self._max_size = max_size
        self._max_age = int(max_age.to(unit='ns', dtype='int64').value)
        # Free slots left by a relocation, i.e. how many appends it is amortized
        # over. ``phase`` spreads the relocation point across streams: the period
        # is a fixed number of samples, so equal-rate logs that start together
        # would otherwise relocate in the same batch forever, multiplying the
        # worst-case pause by the number of such logs.
        base = max(1, max_size // 8)
        self._slack = base + phase % base

    @property
    def unit(self) -> sc.Unit | None:
        return self._unit

    def _trim(self, newest_time: int) -> None:
        """Drop samples falling outside the retention window.

        Moves the start offset only, leaving the buffer contents untouched, so
        this is O(1) unless the age cutoff bites and cheap enough to run on
        every append -- which is what makes ``max_age`` an actual bound rather
        than a floor that is only enforced when the buffer happens to fill.

        Always keeps the newest sample at or before the age cutoff.
        ``sc.lookup(..., mode='previous')`` against this buffer (BIFROST's
        rotation lookup) returns NaN for queries before the first retained
        entry, so that sample anchors every query inside the window.
        """
        if self._end == self._start:
            return
        # The incoming sample claims one slot of the size budget. This may empty
        # the buffer when max_size is 1, which is safe because add() writes that
        # sample immediately; the age cutoff below never empties it, since a
        # sample older than the cutoff is by definition an anchor to retain.
        start = max(self._start, self._end - self._max_size + 1)
        if start < self._end:
            times = self._timeseries.coords['time'].values.view('int64')
            cutoff = newest_time - self._max_age
            if times[start] < cutoff:
                expired = int(
                    np.searchsorted(times[start : self._end], cutoff, 'right')
                )
                start += expired - 1  # retain the anchor
        self._start = start

    def _target_capacity(self, needed: int) -> int:
        """Capacity to hold ``needed`` live samples.

        Doubles while the buffer is still filling, then settles at
        ``max_size + slack`` once retention caps the live size, so relocations
        are amortized over ``slack`` appends instead of running per append.
        """
        return max(2, min(2 * needed, self._max_size + self._slack))

    def _relocate(self, capacity: int) -> None:
        """Move the live range into a freshly allocated buffer.

        Reallocates rather than shifting in place because :meth:`get` hands out
        views aliasing this buffer and ``sc.lookup`` holds a reference to them
        rather than copying; an in-place shift rewrites data behind consumers
        that are still reading it. The old buffer stays alive for as long as any
        such view does.
        """
        template = self._timeseries
        live = self._end - self._start
        data = sc.empty(
            dims=template.data.dims,
            shape=(capacity, *template.data.shape[1:]),
            unit=template.data.unit,
            dtype=template.data.dtype,
            with_variances=template.data.variances is not None,
        )
        data['time', :live] = template.data['time', self._start : self._end]
        coords = {}
        for name, coord in template.coords.items():
            new = sc.empty(
                dims=coord.dims, shape=(capacity,), unit=coord.unit, dtype=coord.dtype
            )
            new['time', :live] = coord['time', self._start : self._end]
            coords[name] = new
        self._timeseries = sc.DataArray(data, coords=coords)
        self._start = 0
        self._end = live

    def _ensure_capacity(self, data: LogData) -> None:
        if self._timeseries is None:
            # Initialize with initial capacity of 2
            arr = np.asarray(data.value)
            values = sc.zeros(
                dims=['time', *self._data_dims],
                shape=[2, *arr.shape],
                unit=self._unit,
                dtype=arr.dtype,
                with_variances=data.variances is not None,
            )
            times = sc.zeros(
                dims=['time'], shape=[2], unit=self._time_unit, dtype='int64'
            )
            coords: dict[str, sc.Variable] = {'time': self._epoch + times}
            if self._has_target:
                coords['target'] = sc.zeros(
                    dims=['time'], shape=[2], unit=self._unit, dtype='float64'
                )
            if self._has_idle:
                # int32 (not bool) because the streaming_data_types da00
                # flatbuffer schema has no bool dtype. Values stay 0/1.
                coords['idle'] = sc.zeros(dims=['time'], shape=[2], dtype='int32')
            self._timeseries = sc.DataArray(values, coords=coords)
        elif self._end >= self._timeseries.sizes['time']:
            self._relocate(self._target_capacity(self._end - self._start + 1))

    def add(self, timestamp: Timestamp, data: LogData) -> bool:
        if self._last_time is not None:
            if data.time < self._last_time:
                logger.warning(
                    "out_of_order_timestamp_skipped",
                    source_time=data.time,
                    last_time=self._last_time,
                )
                return False
            if data.time == self._last_time:
                last_value = self._timeseries.data.values[self._end - 1]
                if not np.array_equal(data.value, last_value):
                    logger.warning(
                        "duplicate_timestamp_value_mismatch",
                        source_time=data.time,
                    )
                return False

        if self._timeseries is not None:
            self._trim(data.time)
        self._ensure_capacity(data)
        self._timeseries.coords['time'].values[self._end] = data.time
        self._timeseries.data.values[self._end] = data.value
        if data.variances is not None and self._timeseries.data.variances is not None:
            self._timeseries.data.variances[self._end] = data.variances
        if self._has_target:
            if data.target is None:
                raise ValueError("Target expected but not provided")
            self._timeseries.coords['target'].values[self._end] = data.target
        if self._has_idle:
            if data.idle is None:
                raise ValueError("Idle flag expected but not provided")
            self._timeseries.coords['idle'].values[self._end] = int(data.idle)
        self._end += 1
        self._last_time = data.time
        return True

    def get(self) -> sc.DataArray:
        if self._timeseries is None:
            raise RuntimeError("No data has been added yet.")

        # Monotonic timestamps are enforced by add(), no sorting needed
        return self._timeseries['time', self._start : self._end]

    def clear(self) -> None:
        self._start = 0
        self._end = 0
        self._last_time = None
        # Keep the allocated array to avoid reallocations


def nxlog_for_stream(stream: Stream | None, *, name: str = '') -> ToNXlog | None:
    """ToNXlog preprocessor for a Device or F144Stream entry, or None.

    Maps the two stream types that produce NXlog-shaped time series to a
    correctly-parameterised :class:`ToNXlog`. Returns ``None`` for any other
    stream type or ``None`` input so factories can fall through to their own
    handling.

    ``name`` seeds the buffer's relocation phase and must be the stream name, so
    that logs sharing a publishing rate do not relocate in lock-step. A stable
    hash is used rather than :func:`hash` so the phase survives restarts.
    """
    phase = zlib.crc32(name.encode())
    if isinstance(stream, Device):
        return ToNXlog(
            attrs={'units': stream.units},
            has_target=stream.target is not None,
            has_idle=stream.idle is not None,
            phase=phase,
        )
    if isinstance(stream, F144Stream):
        return ToNXlog(attrs={'units': stream.units}, phase=phase)
    return None
