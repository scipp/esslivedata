# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Selection of a spectral range from monitor and detector-view histograms."""

import scipp as sc

from ..config.spectral_coords import SPECTRAL_RANGE_HIGH_COORD, SPECTRAL_RANGE_LOW_COORD


def select_spectral_range(
    data: sc.DataArray, spectral_range: tuple[sc.Variable, sc.Variable] | None
) -> sc.DataArray:
    """Restrict ``data`` to a range along its last dim, the spectral axis.

    Every output that reports counts summed over the spectral axis goes through
    this function, so they all select the same bins and carry the same interval.
    The result has the selected interval as scalar coords
    :data:`SPECTRAL_RANGE_LOW_COORD` and :data:`SPECTRAL_RANGE_HIGH_COORD`, which
    survive summing over the spectral axis.

    Parameters
    ----------
    data:
        Histogram with bin edges along its last dim.
    spectral_range:
        ``(low, high)`` in the unit of the bin edges. Selects every bin that
        overlaps the range, so the selected interval extends to the enclosing bin
        edges. ``None`` selects all bins.

    Returns
    -------
    :
        The selected bins. If no bin overlaps the range, the result has no bins
        and its interval is ``spectral_range`` itself.
    """
    dim = data.dims[-1]
    if spectral_range is not None:
        data = data[dim, spectral_range[0] : spectral_range[1]]
    if data.sizes[dim] == 0:
        low, high = spectral_range
    else:
        edges = data.coords[dim]
        low, high = edges[dim, 0], edges[dim, -1]
    return data.assign_coords(
        {SPECTRAL_RANGE_LOW_COORD: low, SPECTRAL_RANGE_HIGH_COORD: high}
    )
