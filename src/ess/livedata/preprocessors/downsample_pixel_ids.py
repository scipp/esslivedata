# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Reduce detector resolution by remapping event ids before pixel grouping."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import structlog

from ..config.detector_downsampling import DetectorDownsampling
from ..core.preprocessor import Accumulator
from ..core.timestamp import Timestamp
from .to_nxevent_data import DetectorEvents


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
    ``source_resolution`` is inferred from the largest event id observed in the
    first batch, rounded up to a multiple of ``resolution``:

        source = ceil(sqrt(max_id + 1) / resolution) * resolution

    The geometry file is deliberately not the authority here. It is static and
    describes how the detector was configured when the file was written, not
    what is being streamed now; where it declares a different resolution that
    is logged, not obeyed. It is trusted only for what a reconfiguration does
    not change: the id base, and that the grid is square.

    Inference is safe here because ids are laid out as ``x * source + y`` with
    ``x`` the slow axis, so any event at large ``x`` produces a large id: even
    a small illuminated patch away from the first rows pins the resolution, and
    detector noise alone spans the panel. Rounding up to a multiple of
    ``resolution`` -- required anyway for the blocks to tile evenly -- absorbs
    dead pixels in the last rows.

    The estimate is a lower bound, so the only possible error is an
    underestimate, and an underestimate is self-announcing: ids at or beyond
    ``source**2`` cannot occur if it is correct. Those are counted and logged
    rather than raised on, matching what grouping already does with ids outside
    the grid, so a single corrupt id cannot take the service down.

    Ids are taken relative to ``downsampling.first_id``, which the geometry
    file supplies, because detectors disagree on where they start counting and
    guessing wrong does not merely shift the image: for a 1-based 4096x4096
    panel treated as 0-based, ``max_id`` becomes 4096**2 rather than
    4096**2 - 1 and the inferred resolution comes out as 4608.

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
        self._logger = structlog.get_logger(__name__)

    @property
    def source_resolution(self) -> int | None:
        """Inferred source grid side length, or None before the first event."""
        return self._source_resolution

    def _infer_source_resolution(self, max_id: int) -> int:
        blocks = math.ceil(math.sqrt(max_id + 1) / self._resolution)
        source = max(blocks, 1) * self._resolution
        self._logger.info(
            'detector_resolution_inferred',
            max_event_id=int(max_id),
            source_resolution=source,
            declared_resolution=self._declared_resolution,
            target_resolution=self._resolution,
            first_id=self._first_id,
            block=source // self._resolution,
        )
        if (
            self._declared_resolution is not None
            and self._declared_resolution != source
        ):
            # Not an error: the file is static and the detector may be running
            # in a different configuration. Worth saying out loud, because the
            # usual reason is that the geometry file has gone stale.
            self._logger.warning(
                'detector_resolution_differs_from_geometry_file',
                streamed_resolution=source,
                declared_resolution=self._declared_resolution,
            )
        return source

    def add(self, timestamp: Timestamp, data: DetectorEvents) -> bool:
        raw = np.asarray(data.pixel_id)
        if raw.size == 0:
            return self._inner.add(timestamp, data)
        pixel_id = raw if self._first_id == 0 else raw - self._first_id

        if below := int(np.count_nonzero(pixel_id < 0)):
            # Ids below where the detector is declared to start counting: the
            # base is wrong, or the data is. Either way they map outside the
            # target grid and grouping drops them.
            self._below_first_id += below
            self._logger.warning(
                'event_id_below_first_id',
                first_id=self._first_id,
                min_event_id=int(raw.min()),
                dropped=below,
                dropped_total=self._below_first_id,
            )

        max_id = int(pixel_id.max())
        if self._source_resolution is None:
            self._source_resolution = self._infer_source_resolution(max_id)
        source = self._source_resolution
        block = source // self._resolution

        # x is the slow axis: id = x * source + y.
        x, y = np.divmod(pixel_id, source)
        # Ids at or beyond source**2 give x // block >= resolution, so the
        # remapped id falls outside the target grid and grouping drops it, as
        # it already does for ids outside the configured detector_number.
        remapped = (x // block) * self._resolution + (y // block)

        if max_id >= source * source:
            n = int(np.count_nonzero(pixel_id >= source * source))
            self._out_of_range += n
            self._logger.warning(
                'event_id_above_inferred_resolution',
                max_event_id=max_id,
                source_resolution=source,
                dropped=n,
                dropped_total=self._out_of_range,
            )

        return self._inner.add(timestamp, replace(data, pixel_id=remapped))

    def get(self) -> T:
        return self._inner.get()

    def clear(self) -> None:
        self._inner.clear()
