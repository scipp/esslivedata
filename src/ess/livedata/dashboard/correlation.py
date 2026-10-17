# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Correlation of time series with the values of other time series."""

from __future__ import annotations

from collections.abc import Mapping

import scipp as sc


def correlate[K](
    data: Mapping[K, sc.DataArray], axes: Mapping[str, sc.DataArray]
) -> dict[K, sc.DataArray]:
    """Add to each point of ``data`` the value every axis had at that time.

    Each point is correlated with the axis value in effect at its timestamp, i.e.
    the most recent axis reading at or before it. Points predating the first
    reading of any axis have no such value and are dropped; correlating them with
    a reading taken later would be fabricating the axis history.

    Parameters
    ----------
    data:
        Time series to correlate, each with a ``time`` dim and coord.
    axes:
        Axis time series, keyed by the name of the coord they become in the
        result. Their ``time`` coord must use the same dtype and unit as that of
        ``data``.

    Returns
    -------
    :
        The correlated time series, keyed like ``data``. Entries left without
        points are omitted.
    """
    # sc.values only accepts float dtypes, so integer axes are used as is.
    lookups = {
        name: sc.lookup(
            sc.values(ax) if ax.variances is not None else ax, mode='previous'
        )
        for name, ax in axes.items()
    }
    # Earliest time at which every axis has a reading. Before it, 'previous'
    # lookup yields NaN, which hist()/bin() would drop without a trace.
    start = max(ax.coords['time'].min() for ax in axes.values())

    correlated: dict[K, sc.DataArray] = {}
    for key, da in data.items():
        dependent = da['time', start:].copy(deep=False)
        if dependent.sizes['time'] == 0:
            continue
        for name, lut in lookups.items():
            dependent.coords[name] = lut[dependent.coords['time']]
        correlated[key] = dependent
    return correlated
