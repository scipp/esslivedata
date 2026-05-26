# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Time-based downsampling for timeseries plots.

Long-running timeseries accumulate many points (~100 k for a day at 1 Hz),
which makes per-tick plot rebuilds and the resulting ``pipe.send`` payloads
expensive. ``downsample_timeseries`` reduces a DataArray to two bands of
uniform stride: a recent window at fine resolution plus a coarse floor
that extends the run's older history indefinitely (set the floor period
to 0 to drop older data instead). See issue #940.
"""

from __future__ import annotations

import numpy as np
import scipp as sc


def _bucket_first_indices(times_ns: np.ndarray, period_ns: int) -> np.ndarray:
    """First-occurrence index of each ``period_ns``-wide bucket along ``times_ns``."""
    if period_ns <= 0 or len(times_ns) == 0:
        return np.arange(len(times_ns), dtype=np.int64)
    buckets = (times_ns - times_ns[0]) // period_ns
    keep = np.empty(len(times_ns), dtype=bool)
    keep[0] = True
    keep[1:] = buckets[1:] != buckets[:-1]
    return np.flatnonzero(keep)


def _select_indices(data: sc.DataArray, dim: str, indices: np.ndarray) -> sc.DataArray:
    """Return ``data`` reduced to the given positional indices along ``dim``."""
    new_coords = {}
    for name, coord in data.coords.items():
        if dim in coord.dims:
            new_coords[name] = sc.array(
                dims=coord.dims,
                values=coord.values[indices],
                unit=coord.unit,
                dtype=coord.dtype,
            )
        else:
            new_coords[name] = coord
    variances = None if data.variances is None else data.variances[indices]
    return sc.DataArray(
        data=sc.array(
            dims=data.dims,
            values=data.values[indices],
            variances=variances,
            unit=data.unit,
            dtype=data.dtype,
        ),
        coords=new_coords,
    )


def downsample_timeseries(
    data: sc.DataArray,
    *,
    period_seconds: float,
    recent_seconds: float,
    floor_period_seconds: float,
    concat_dim: str = 'time',
) -> sc.DataArray:
    """Downsample a timeseries DataArray to a fine-recent + coarse-floor layout.

    Bucketing is purely positional: the time coord is split into a recent band
    (last ``recent_seconds``) and a floor band (older). Within each band the
    first sample of every period-wide bucket is kept; the very last sample is
    always force-included so the lag indicator stays accurate.

    A ``floor_period_seconds`` of 0 drops older data entirely.

    Parameters
    ----------
    data:
        Buffered timeseries with a 1-D ``concat_dim`` time coord.
    period_seconds:
        Sample period for the recent band, in seconds. Must be > 0.
    recent_seconds:
        Length of the recent band, in seconds. Must be >= 0.
    floor_period_seconds:
        Sample period for the floor band, in seconds. 0 drops older data.
    concat_dim:
        Name of the time dimension.

    Returns
    -------
    :
        New DataArray with the same dims/coords structure but fewer points.
    """
    if concat_dim not in data.dims:
        return data
    n = data.sizes[concat_dim]
    if n < 2:
        return data
    if concat_dim not in data.coords:
        return data
    # FullHistoryExtractor guarantees a datetime64 time coord; scipp.to does
    # not support astype on datetime64, so reinterpret via numpy.
    times_ns = (
        data.coords[concat_dim]
        .values.astype('datetime64[ns]', copy=False)
        .astype('int64', copy=False)
    )

    latest_ns = times_ns[-1]
    recent_cutoff_ns = latest_ns - int(recent_seconds * 1e9)
    period_ns = max(int(period_seconds * 1e9), 1)
    floor_ns = max(int(floor_period_seconds * 1e9), 0)

    is_recent = times_ns >= recent_cutoff_ns
    parts: list[np.ndarray] = []

    older_indices = np.flatnonzero(~is_recent)
    if floor_ns > 0 and older_indices.size > 0:
        local = _bucket_first_indices(times_ns[older_indices], floor_ns)
        parts.append(older_indices[local])

    recent_indices = np.flatnonzero(is_recent)
    if recent_indices.size > 0:
        local = _bucket_first_indices(times_ns[recent_indices], period_ns)
        parts.append(recent_indices[local])

    if not parts:
        return _select_indices(data, concat_dim, np.array([n - 1], dtype=np.int64))

    keep = np.concatenate(parts)
    if keep[-1] != n - 1:
        keep = np.append(keep, n - 1)
    if keep.size == n:
        return data
    return _select_indices(data, concat_dim, keep)
