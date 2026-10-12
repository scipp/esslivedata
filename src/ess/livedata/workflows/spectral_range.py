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
        ``(low, high)`` in any unit convertible to that of the bin edges. Selects
        every bin that overlaps the range, so the selected interval extends to the
        enclosing bin edges. ``None`` selects all bins.

    Returns
    -------
    :
        The selected bins, with the interval in the unit of the bin edges. If no
        bin overlaps the range, the result has no bins and its interval is
        ``spectral_range`` itself.
    """
    dim = data.dims[-1]
    edges = data.coords[dim]
    if spectral_range is None:
        low, high = edges[dim, 0], edges[dim, -1]
    else:
        low, high = (v.to(unit=edges.unit, dtype=edges.dtype) for v in spectral_range)
        data = data[dim, low:high]
        if data.sizes[dim] > 0:
            selected = data.coords[dim]
            low, high = selected[dim, 0], selected[dim, -1]
    return data.assign_coords(
        {SPECTRAL_RANGE_LOW_COORD: low, SPECTRAL_RANGE_HIGH_COORD: high}
    )
