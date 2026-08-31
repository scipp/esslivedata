# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Reduced-resolution detector ingest settings.

Holds the value object the preprocessor layer consumes and the rules that
resolve it against the geometry file. ``Instrument`` keeps only the opt-in
(``configure_detector_downsampling``) and the cached lookup
(``get_downsampling``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipp as sc
import structlog

logger = structlog.get_logger(__name__)


def is_power_of_two(value: int) -> bool:
    """Returns whether ``value`` is a positive power of two."""
    return value > 0 and not value & (value - 1)


@dataclass(frozen=True)
class DetectorDownsampling:
    """Resolved reduced-resolution ingest settings for one detector."""

    #: Side length of the target grid the preprocessor maps event ids onto.
    resolution: int
    #: Lowest event id the detector emits, taken from the geometry file.
    #: Detectors differ: TBL's Timepix3 counts from 0, its He3 banks from 1.
    first_id: int
    #: Side length the geometry file declares, or None if no file was read.
    #: An upper bound rather than the streamed resolution, which is inferred
    #: from the event ids since the file is static and may be stale.
    declared_resolution: int | None
    #: The target grid itself, as a 2D ``detector_number``.
    grid: sc.Variable


def resolve_downsampling(
    name: str, resolution: int, declared: sc.Variable | None
) -> DetectorDownsampling:
    """Resolve ingest settings for one detector against its declared grid.

    The *streamed* resolution is deliberately not taken from ``declared``: the
    geometry file is static and need not describe what the detector is
    currently streaming, so the preprocessor infers it from the observed event
    ids instead (see
    :class:`~ess.livedata.preprocessors.downsample_pixel_ids.DownsamplePixelIds`).

    ``declared`` is consulted for what a reconfiguration cannot change:

    - that the ids are a contiguous row-major enumeration of a square grid,
      which is the layout the remap assumes and the one property whose failure
      would scramble the image rather than announce itself;
    - which id the detector counts from, since detectors disagree and a wrong
      base does not merely shift the image by a pixel, it makes the inferred
      source resolution wrong;
    - the largest grid the detector can be reading out, which bounds the
      preprocessor's estimate.

    Parameters
    ----------
    name:
        Detector name, for diagnostics.
    resolution:
        Side length of the target grid.
    declared:
        The detector's ``detector_number`` as read from the geometry file, or
        None where no file was read.

    Returns
    -------
    :
        The resolved settings.
    """
    grid = sc.arange('detector_number', resolution * resolution, unit=None).fold(
        dim='detector_number', sizes={'dim_0': resolution, 'dim_1': resolution}
    )
    if declared is None:
        # Dev configurations and instruments without a geometry file. The id
        # base cannot be checked, so the common convention is assumed; a
        # detector counting from 1 will show up as dropped ids. Nor is there a
        # bound on the inferred resolution.
        logger.warning(
            'downsampling_without_geometry',
            detector=name,
            resolution=resolution,
            assumed_first_id=0,
        )
        return DetectorDownsampling(resolution, 0, None, grid)
    if declared.ndim != 2 or declared.shape[0] != declared.shape[1]:
        raise ValueError(
            f"Detector {name} is configured for downsampling, which assumes a "
            f"square grid, but its detector_number is {declared.sizes}."
        )
    side = declared.shape[0]
    if not is_power_of_two(side) or side < resolution:
        raise ValueError(
            f"Detector {name} has side {side}, which must be a power of two and "
            f"at least the downsampling resolution {resolution}. Panels come in "
            "powers of two; a detector that does not is not a candidate for "
            "downsampling by id remapping."
        )
    first_id = _check_contiguous_row_major(name, declared)
    return DetectorDownsampling(resolution, first_id, side, grid)


def _check_contiguous_row_major(name: str, declared: sc.Variable) -> int:
    """Return the first id, having checked the layout the remap assumes.

    The remap decomposes an id as ``x * side + y``. That is only the inverse of
    the file's own enumeration if ``detector_number`` runs contiguously in
    row-major order, and a file that enumerated per readout chip instead --
    plausible for a multi-chip panel -- would produce a scrambled image with
    nothing to notice it by. The file is already in memory, so check it.
    """
    flat = declared.values.reshape(-1)
    first_id = int(flat[0])
    expected = np.arange(first_id, first_id + flat.size, dtype=flat.dtype)
    if not np.array_equal(flat, expected):
        raise ValueError(
            f"Detector {name} is configured for downsampling, which maps event "
            f"id to pixel as 'id = x * {declared.shape[0]} + y'. That requires "
            "detector_number to enumerate the grid contiguously in row-major "
            "order, which this one does not."
        )
    return first_id
