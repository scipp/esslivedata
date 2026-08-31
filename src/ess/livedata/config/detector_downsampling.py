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

import scipp as sc
import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class DetectorDownsampling:
    """Resolved reduced-resolution ingest settings for one detector."""

    #: Side length of the target grid the preprocessor maps event ids onto.
    resolution: int
    #: Lowest event id the detector emits, taken from the geometry file.
    #: Detectors differ: TBL's Timepix3 counts from 0, its He3 banks from 1.
    first_id: int
    #: Side length the geometry file declares, or None if no file was read.
    #: Only a cross-check: the resolution actually streamed is inferred from
    #: the event ids, since the file is static and may be stale.
    declared_resolution: int | None
    #: The target grid itself, as a 2D ``detector_number``.
    grid: sc.Variable


def resolve_downsampling(
    name: str, resolution: int, declared: sc.Variable | None
) -> DetectorDownsampling:
    """Resolve ingest settings for one detector against its declared grid.

    The *source* resolution is deliberately not taken from ``declared``: the
    geometry file is static and need not describe what the detector is
    currently streaming, so the preprocessor infers it from the observed event
    ids instead (see
    :class:`~ess.livedata.preprocessors.downsample_pixel_ids.DownsamplePixelIds`).

    ``declared`` is consulted only for what a reconfiguration does not change:
    that the grid is square, that its side tiles evenly into ``resolution``,
    and which id the detector counts from. That last one is why the file is
    read at all rather than assumed — id bases differ between detectors, and a
    wrong base does not merely shift the image by a pixel, it makes the
    inferred source resolution wrong, silently.

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
        # Dev configurations and instruments without a geometry file. The
        # id base cannot be checked, so the common convention is assumed;
        # a detector counting from 1 will show up as dropped ids.
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
    source = declared.shape[0]
    if source % resolution:
        raise ValueError(
            f"Detector {name} has side {source}, which is not a multiple of the "
            f"downsampling resolution {resolution}, so the blocks do not tile."
        )
    return DetectorDownsampling(resolution, int(declared.min().value), source, grid)
