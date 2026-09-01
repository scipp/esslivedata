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


def _is_power_of_two(value: int) -> bool:
    """Returns whether ``value`` is a positive power of two."""
    return value > 0 and not value & (value - 1)


def is_reachable_resolution(side: int, resolution: int) -> bool:
    """Returns whether ``side`` is ``resolution`` times a power of two.

    The remap needs the target resolution to tile the source exactly, which on
    its own would allow any integer ratio. The power-of-two restriction comes
    from the inference instead: the preprocessor picks the source resolution
    from the candidates it can reach by repeatedly doubling the target, so a
    source that is not one of them could never be inferred, however cleanly it
    tiles. Neither side has to be a power of two itself -- a 1000x1000 panel
    ingested at 250x250 is fine.
    """
    return (
        resolution > 0
        and side % resolution == 0
        and _is_power_of_two(side // resolution)
    )


@dataclass(frozen=True)
class DetectorDownsampling:
    """Resolved reduced-resolution ingest settings for one detector."""

    #: Side length of the target grid the preprocessor maps event ids onto.
    resolution: int
    #: Largest grid the detector can physically read out, from the instrument
    #: configuration. A hardware fact, unlike the resolution it is currently
    #: streaming, which is operator-reconfigurable and inferred from the ids.
    max_resolution: int
    #: Lowest event id the detector emits, taken from the geometry file.
    #: Detectors differ: TBL's Timepix3 counts from 0, its He3 banks from 1.
    first_id: int
    #: The target grid itself, as a 2D ``detector_number``.
    grid: sc.Variable


def resolve_downsampling(
    name: str, resolution: int, max_resolution: int, declared: sc.Variable | None
) -> DetectorDownsampling:
    """Resolve ingest settings for one detector against its declared grid.

    The *streamed* resolution is deliberately not taken from ``declared``: the
    readout resolution is operator-reconfigurable and changes during a run, so
    the geometry file records one past configuration rather than the current
    one. The preprocessor infers it from the observed event ids instead (see
    :class:`~ess.livedata.preprocessors.downsample_pixel_ids.DownsamplePixelIds`),
    bounded by ``max_resolution``, which is a property of the hardware and so
    belongs in the instrument configuration rather than in a file that
    describes a configuration.

    ``declared`` is consulted only for what a reconfiguration cannot change:

    - that the ids are a contiguous row-major enumeration of a square grid,
      which is the layout the remap assumes and the one property whose failure
      would scramble the image rather than announce itself. The check can only
      be made at the resolution the file happens to record, but a readout that
      enumerates row-major at one resolution does so at all of them;
    - which id the detector counts from, since detectors disagree and a wrong
      base does not merely shift the image by a pixel, it makes the inferred
      source resolution wrong.

    Parameters
    ----------
    name:
        Detector name, for diagnostics.
    resolution:
        Side length of the target grid.
    max_resolution:
        Largest grid the detector can read out.
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
        # detector counting from 1 will show up as dropped ids.
        logger.warning(
            'downsampling_without_geometry',
            detector=name,
            resolution=resolution,
            assumed_first_id=0,
        )
        return DetectorDownsampling(resolution, max_resolution, 0, grid)
    if declared.ndim != 2 or declared.shape[0] != declared.shape[1]:
        raise ValueError(
            f"Detector {name} is configured for downsampling, which assumes a "
            f"square grid, but its detector_number is {declared.sizes}."
        )
    side = declared.shape[0]
    if not is_reachable_resolution(side, resolution):
        raise ValueError(
            f"Detector {name} has side {side}, which must be the downsampling "
            f"resolution {resolution} times a power of two. The target grid has "
            "to tile the source, and the source has to be one of the resolutions "
            "the preprocessor can infer."
        )
    if side > max_resolution:
        raise ValueError(
            f"Detector {name} is declared with side {side} in the geometry file "
            f"but configured with max_resolution={max_resolution}. The "
            "configured maximum is meant to be what the hardware can read out, "
            "so one of the two is wrong."
        )
    first_id = _check_contiguous_row_major(name, declared)
    return DetectorDownsampling(resolution, max_resolution, first_id, grid)


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
