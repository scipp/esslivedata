# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Reduce detector resolution by remapping event ids before pixel grouping."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import scipp as sc
import structlog

from ..config.detector_downsampling import DetectorDownsampling
from ..core.log_throttle import LogThrottle
from ..core.preprocessor import Accumulator
from ..core.timestamp import Timestamp
from .to_nxevent_data import DetectorEvents

SOURCE_RESOLUTION = 'source_resolution'
"""Coord name carrying the inferred source resolution downstream.

Stamped on every batch :class:`DownsamplePixelIds` emits. Counts remapped from
different source resolutions are not commensurable, so the cumulative
accumulator lists this alongside the detector transform in its ``reset_coords``
and discards its buffer when it changes.
"""

DEFAULT_WINDOW_S = 60.0
"""Window over which event ids count as evidence of the source resolution."""

DEFAULT_WINDOW_BUCKETS = 6
"""Granularity with which the window expires evidence."""

DEFAULT_MIN_EVENTS_TO_SHRINK = 1000
"""Events required in the window before concluding the panel got smaller."""


def _round_up_to_power_of_two(side: int) -> int:
    """Smallest power of two that is at least ``side``."""
    return 1 << (max(side, 1) - 1).bit_length()


class _EvidenceWindow:
    """Largest event id, and how many events, seen in the recent past.

    A ring of buckets, coarse because the window boundary does not need to be
    sharp: what matters is that evidence expires on the order of the window,
    not exactly at it. Buckets expire on write, so a detector that stops
    sending stops the clock rather than losing its resolution to a drought --
    absence of events is not evidence about the panel.
    """

    def __init__(self, *, window: float, buckets: int) -> None:
        self._interval = window / buckets
        self._max: list[int | None] = [None] * buckets
        self._count = [0] * buckets
        self._index: int | None = None

    def add(self, now: float, max_id: int, count: int) -> None:
        index = int(now // self._interval)
        self._expire(index)
        slot = index % len(self._max)
        current = self._max[slot]
        self._max[slot] = max_id if current is None else max(current, max_id)
        self._count[slot] += count

    def _expire(self, index: int) -> None:
        if self._index is None:
            self._index = index
            return
        stale = min(index - self._index, len(self._max))
        for offset in range(1, stale + 1):
            slot = (self._index + offset) % len(self._max)
            self._max[slot] = None
            self._count[slot] = 0
        self._index = max(index, self._index)

    @property
    def max_id(self) -> int | None:
        """Largest id in the window, or None if it holds no events."""
        seen = [value for value in self._max if value is not None]
        return max(seen) if seen else None

    @property
    def count(self) -> int:
        """Events in the window."""
        return sum(self._count)


class DownsamplePixelIds(Accumulator[DetectorEvents, sc.DataArray]):
    """Remap event ids onto a coarser square grid, then delegate.

    Wraps the accumulator that would otherwise receive the raw events and
    rewrites ``pixel_id`` so that every ``block`` x ``block`` square of source
    pixels becomes one target pixel. Downstream, including pixel grouping,
    then only ever sees the coarse grid.

    Applied before grouping this is O(events); applied after, as
    ``bins.concat`` over the full-resolution grid, it is O(source pixels) per
    update regardless of how many events arrived. That distinction is the
    reason this exists: for a 4096x4096 panel the latter costs seconds per
    update whether one event arrived or a million.

    Source resolution
    -----------------
    ``source_resolution`` is the side length of the grid the detector is
    streaming. The detector is reconfigured to different readout resolutions
    during operation and does not announce it on any stream we consume, so it
    is inferred from the largest event id seen recently, rounded up to a power
    of two:

        source = 2 ** ceil(log2(sqrt(max_id + 1)))

    The candidates are the target resolution repeatedly doubled, and the
    estimate is the smallest of them that can hold ``max_id``. Restricting the
    candidates that way is what makes them easy to reach: whatever the panel
    is, an event anywhere past its first quarter of rows pins it, where
    rounding to a multiple of ``resolution`` instead would have needed one in
    the last ``resolution`` rows -- past row 3136 of a 4096 panel downsampled
    to 512. It also means the target grid always tiles the source exactly.
    Neither resolution has to be a power of two itself, only their ratio.

    Evidence expires, because the estimate has to follow the panel downward as
    well as upward. A reconfiguration to a smaller readout produces no id that
    contradicts the old estimate -- it simply stops producing the large ones --
    so an estimate taken over all ids ever seen could only ever grow, and would
    stay pinned at the old resolution for the rest of the run. Taking it over a
    window instead handles both directions with one rule.

    The two directions are not equally well evidenced, though, and the estimate
    is deliberately asymmetric: a large id *proves* the panel is at least that
    big, while the absence of large ids is only suggestive -- a beam spot in
    the low rows looks the same as a smaller panel. So growth is immediate,
    while shrinking additionally requires the window to hold
    ``min_events_to_shrink`` events, which keeps a handful of stray counts
    during a quiet period from dropping the estimate. Neither guard is exact,
    and both are cheap to get wrong: a resolution change resets the cumulative
    accumulators (see ``SOURCE_RESOLUTION``), so an estimate that flips costs a
    restarted image rather than a corrupted one.

    The estimate is bounded by ``max_resolution``, which the instrument
    configuration states because it is a property of the hardware. Ids implying
    more than that are corruption rather than evidence: they are excluded from
    the window before it is consulted, counted and reported, and they map
    outside the target grid so grouping drops them.

    Ids are laid out as ``x * source + y`` with ``x`` the slow axis, which
    :func:`~ess.livedata.config.detector_downsampling.resolve_downsampling`
    verifies against the geometry file. They are taken relative to
    ``downsampling.first_id`` because detectors disagree on where they start
    counting, and guessing wrong does not merely shift the image: for a 1-based
    4096x4096 panel treated as 0-based, ``max_id`` becomes ``4096**2`` rather
    than ``4096**2 - 1`` and the inferred resolution doubles to 8192.

    Parameters
    ----------
    inner:
        Accumulator receiving the remapped events.
    downsampling:
        Resolved settings from ``Instrument.get_downsampling``.
    window:
        Seconds an event id counts as evidence of the source resolution.
    buckets:
        Granularity with which ``window`` expires evidence.
    min_events_to_shrink:
        Events required in the window before the estimate may decrease.
    clock:
        Monotonic seconds, for expiring evidence and throttling reports.
    """

    def __init__(
        self,
        inner: Accumulator[DetectorEvents, sc.DataArray],
        downsampling: DetectorDownsampling,
        *,
        window: float = DEFAULT_WINDOW_S,
        buckets: int = DEFAULT_WINDOW_BUCKETS,
        min_events_to_shrink: int = DEFAULT_MIN_EVENTS_TO_SHRINK,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._resolution = downsampling.resolution
        self._max_resolution = downsampling.max_resolution
        self._corrupt_at = downsampling.max_resolution**2
        self._first_id = downsampling.first_id
        self._source_resolution: int | None = None
        self._window = _EvidenceWindow(window=window, buckets=buckets)
        self._min_events_to_shrink = min_events_to_shrink
        self._clock = clock
        self._out_of_range = 0
        self._below_first_id = 0
        self._out_of_range_throttle = LogThrottle()
        self._below_first_id_throttle = LogThrottle()
        self._logger = structlog.get_logger(__name__)

    @property
    def source_resolution(self) -> int | None:
        """Inferred source grid side length, or None before the first event."""
        return self._source_resolution

    def _estimate(self, max_id: int) -> int:
        """Smallest admissible source resolution that can hold ``max_id``.

        ``isqrt(max_id) + 1`` is the smallest ``s`` with ``s**2 > max_id``,
        exactly -- unlike ``ceil(sqrt(...))``, which is a float operation on
        values that reach 2**24 for a 4096x4096 panel. Rounding the blocks per
        side up to a power of two picks the smallest candidate at or above it.
        """
        smallest = math.isqrt(max_id) + 1
        blocks = _round_up_to_power_of_two(-(-smallest // self._resolution))
        return min(self._resolution * blocks, self._max_resolution)

    def add(self, timestamp: Timestamp, data: DetectorEvents) -> bool:
        raw = np.asarray(data.pixel_id)
        if raw.size == 0:
            return self._inner.add(timestamp, data)
        pixel_id = raw if self._first_id == 0 else raw - self._first_id

        if (lowest := int(pixel_id.min())) < 0:
            self._report_below_first_id(pixel_id, lowest)

        self._observe(pixel_id)

        # An estimate needs at least one admissible id. Until one arrives every
        # id in the batch is out of range, so remapping at the target stride
        # forwards it and grouping drops it, as it would with any other stride.
        source = self._source_resolution or self._resolution
        block = source // self._resolution
        # x is the slow axis: id = x * source + y. Ids at or beyond source**2
        # give x // block >= resolution, so the remapped id falls outside the
        # target grid and grouping drops it, as it already does for ids outside
        # the configured detector_number. Ids below first_id stay negative.
        x, y = np.divmod(pixel_id, source)
        remapped = (x // block) * self._resolution + (y // block)

        return self._inner.add(timestamp, replace(data, pixel_id=remapped))

    def _observe(self, pixel_id: np.ndarray) -> None:
        """Feed the batch's largest admissible id to the evidence window.

        Ids at or beyond ``max_resolution**2``, and ids below ``first_id``,
        cannot have come from this detector, so they are corruption rather
        than evidence and must not reach the estimate. Letting them through
        would defeat the bound they are measured against: one wild id would
        ratchet a reduced readout up to the full panel and hold it there for a
        whole window.
        """
        max_id = int(pixel_id.max())
        if max_id >= self._corrupt_at:
            admissible = pixel_id < self._corrupt_at
            self._report_out_of_range(pixel_id.size - int(admissible.sum()))
            max_id = int(np.max(pixel_id, where=admissible, initial=-1))
        if max_id < 0:
            return
        self._window.add(self._clock(), max_id, count=pixel_id.size)
        self._update_estimate()

    def _update_estimate(self) -> None:
        """Adopt the estimate the window supports, if it differs and is allowed."""
        if (window_max := self._window.max_id) is None:
            return
        estimate = self._estimate(window_max)
        if estimate == self._source_resolution:
            return
        if self._source_resolution is None:
            self._source_resolution = estimate
            self._logger.info(
                'detector_resolution_inferred',
                source_resolution=estimate,
                max_resolution=self._max_resolution,
                target_resolution=self._resolution,
                first_id=self._first_id,
                block=estimate // self._resolution,
            )
            return
        if estimate < self._source_resolution and (
            self._window.count < self._min_events_to_shrink
        ):
            return
        # Deliberately not throttled, unlike the corruption reports below. Those
        # describe a condition that holds for a whole run and would otherwise
        # recur at the cycle rate; this is an event, and one that is expected to
        # be rare. If it stops being rare, that frequency is the finding -- the
        # window or the shrink guard would be mistuned -- so it must not be
        # suppressed.
        self._logger.warning(
            'detector_resolution_changed',
            previous_resolution=self._source_resolution,
            source_resolution=estimate,
            window_max_event_id=self._window.max_id,
            window_events=self._window.count,
            block=estimate // self._resolution,
        )
        self._source_resolution = estimate
        # Events already accumulated this cycle were mapped with the old
        # stride. Discarding them costs at most one update; what was published
        # in earlier cycles is discarded downstream via SOURCE_RESOLUTION.
        self._inner.clear()

    def _report_out_of_range(self, n: int) -> None:
        self._out_of_range += n
        if (suppressed := self._out_of_range_throttle.take(self._clock())) is None:
            return
        self._logger.warning(
            'event_id_above_max_resolution',
            source_resolution=self._source_resolution,
            max_resolution=self._max_resolution,
            dropped=n,
            dropped_total=self._out_of_range,
            suppressed_reports=suppressed,
        )

    def _report_below_first_id(self, pixel_id: np.ndarray, lowest: int) -> None:
        # Ids below where the detector is declared to start counting: the base
        # is wrong, or the data is. Either way they map outside the target grid
        # and grouping drops them.
        n = int(np.count_nonzero(pixel_id < 0))
        self._below_first_id += n
        if (suppressed := self._below_first_id_throttle.take(self._clock())) is None:
            return
        self._logger.warning(
            'event_id_below_first_id',
            first_id=self._first_id,
            min_event_id=lowest + self._first_id,
            dropped=n,
            dropped_total=self._below_first_id,
            suppressed_reports=suppressed,
        )

    def get(self) -> sc.DataArray:
        result = self._inner.get()
        if self._source_resolution is not None:
            result.coords[SOURCE_RESOLUTION] = sc.index(self._source_resolution)
        return result

    def clear(self) -> None:
        self._inner.clear()

    def release_buffers(self) -> None:
        self._inner.release_buffers()
