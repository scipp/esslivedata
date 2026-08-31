# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Reduce detector resolution by remapping event ids before pixel grouping."""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np
import structlog

from ..config.detector_downsampling import DetectorDownsampling
from ..core.log_throttle import LogThrottle
from ..core.preprocessor import Accumulator
from ..core.timestamp import Timestamp
from .to_nxevent_data import DetectorEvents


def _round_up_to_power_of_two(side: int) -> int:
    """Smallest power of two that is at least ``side``."""
    return 1 << (max(side, 1) - 1).bit_length()


class DownsamplePixelIds[T](Accumulator[DetectorEvents, T]):
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
    streaming. It is inferred from the largest event id observed, rounded up to
    a power of two:

        source = 2 ** ceil(log2(sqrt(max_id + 1)))

    Panels come in powers of two -- 4096, 2048, 512 -- so rounding to one is
    both the tightest defensible estimate and much easier to reach than an
    exact one: a 4096 panel is pinned by any event past row 1024, where
    rounding to a multiple of ``resolution`` instead would have needed one past
    row 3136.

    The estimate is a lower bound, so the only possible error is an
    underestimate, and an underestimate is self-announcing: ids at or beyond
    ``source**2`` cannot occur if it is correct. Those ids therefore raise the
    estimate rather than merely being counted, so a run that starts with a
    narrow beam converges on the true panel as soon as anything lands outside
    it.

    Re-estimating is not free, and what it costs is *not* reset automatically.
    This accumulator discards the events it is still holding, which is at most
    one update's worth since ``get()`` empties it every cycle. Everything
    already published is another matter: the workflow's cumulative accumulator
    keeps counts across cycles, and those accumulated before the correction
    were mapped with the old stride. The target grid does not change, so
    nothing breaks or reshapes -- but the cumulative image and any ROI spectrum
    derived from it stay wrong, in proportion to how long the estimate was
    wrong, until someone restarts or resets the workflow. Nothing signals this
    downstream; ``detector_resolution_grown`` is logged at WARNING so an
    operator can decide. Wiring it into the accumulator's reset-on-move
    (``reset_coord`` / ``DetectorGeometry``) would need the resolution to
    travel with the data, since that signal is a static pipeline value today.

    The geometry file is not the authority on the source resolution: it is
    static and describes how the detector was configured when the file was
    written, not what is being streamed now, and a detector reading out a
    subset of its panel would be described wrongly by it. It does, however,
    bound the estimate. Nothing a reconfiguration can do makes the panel
    physically larger than the file describes, so ids implying more than
    ``declared_resolution`` are corruption and are dropped, which is also what
    keeps a single wild id from ratcheting the estimate up for good. Where no
    file was read there is no such bound and a wild id can ratchet; that path
    already warns that it is running blind.

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
    """

    def __init__(
        self,
        inner: Accumulator[DetectorEvents, T],
        downsampling: DetectorDownsampling,
    ) -> None:
        self._inner = inner
        self._resolution = downsampling.resolution
        self._first_id = downsampling.first_id
        self._declared_resolution = downsampling.declared_resolution
        self._source_resolution: int | None = None
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
        values that reach 2**24 for a 4096x4096 panel.
        """
        side = _round_up_to_power_of_two(math.isqrt(max_id) + 1)
        side = max(side, self._resolution)
        if self._declared_resolution is not None:
            side = min(side, self._declared_resolution)
        return side

    def add(self, timestamp: Timestamp, data: DetectorEvents) -> bool:
        raw = np.asarray(data.pixel_id)
        if raw.size == 0:
            return self._inner.add(timestamp, data)
        pixel_id = raw if self._first_id == 0 else raw - self._first_id

        if (lowest := int(pixel_id.min())) < 0:
            self._report_below_first_id(pixel_id, lowest)

        max_id = int(pixel_id.max())
        if self._source_resolution is None:
            self._source_resolution = self._estimate(max_id)
            self._logger.info(
                'detector_resolution_inferred',
                max_event_id=max_id,
                source_resolution=self._source_resolution,
                declared_resolution=self._declared_resolution,
                target_resolution=self._resolution,
                first_id=self._first_id,
                block=self._source_resolution // self._resolution,
            )
        elif max_id >= self._source_resolution**2:
            self._grow(max_id)

        source = self._source_resolution
        if max_id >= source * source:
            # Only reachable once the estimate is at the bound the geometry
            # file sets, so these ids are corruption rather than evidence.
            self._report_out_of_range(pixel_id, source)

        block = source // self._resolution
        # x is the slow axis: id = x * source + y. Ids at or beyond source**2
        # give x // block >= resolution, so the remapped id falls outside the
        # target grid and grouping drops it, as it already does for ids outside
        # the configured detector_number. Ids below first_id stay negative.
        x, y = np.divmod(pixel_id, source)
        remapped = (x // block) * self._resolution + (y // block)

        return self._inner.add(timestamp, replace(data, pixel_id=remapped))

    def _grow(self, max_id: int) -> None:
        """Raise the estimate to cover ``max_id``, if the bound allows it."""
        grown = self._estimate(max_id)
        if grown <= self._source_resolution:
            return
        self._logger.warning(
            'detector_resolution_grown',
            max_event_id=max_id,
            previous_resolution=self._source_resolution,
            source_resolution=grown,
            declared_resolution=self._declared_resolution,
            block=grown // self._resolution,
        )
        self._source_resolution = grown
        # Events already accumulated this cycle were mapped with the old
        # stride. Discarding them costs at most one update.
        self._inner.clear()

    def _report_out_of_range(self, pixel_id: np.ndarray, source: int) -> None:
        n = int(np.count_nonzero(pixel_id >= source * source))
        self._out_of_range += n
        if (suppressed := self._out_of_range_throttle.take(time.monotonic())) is None:
            return
        self._logger.warning(
            'event_id_above_source_resolution',
            source_resolution=source,
            declared_resolution=self._declared_resolution,
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
        if (suppressed := self._below_first_id_throttle.take(time.monotonic())) is None:
            return
        self._logger.warning(
            'event_id_below_first_id',
            first_id=self._first_id,
            min_event_id=lowest + self._first_id,
            dropped=n,
            dropped_total=self._below_first_id,
            suppressed_reports=suppressed,
        )

    def get(self) -> T:
        return self._inner.get()

    def clear(self) -> None:
        self._inner.clear()

    def release_buffers(self) -> None:
        self._inner.release_buffers()
