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


def downsample_timeseries(
    data: sc.DataArray,
    *,
    period_seconds: float,
    recent_seconds: float,
    floor_period_seconds: float,
    concat_dim: str = 'time',
) -> sc.DataArray:
    """Downsample a timeseries DataArray to a fine-recent + coarse-floor layout.

    The time coord is split into a recent band (last ``recent_seconds``) and a
    floor band (older). Within each band buckets are anchored at the band's
    latest sample and the last sample of each bucket is kept, so the very
    latest sample of the input is always present in the output.

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
    if n < 2 or concat_dim not in data.coords:
        return data

    times = data.coords[concat_dim].to(unit='ns', copy=False)
    latest = times[concat_dim, -1]
    period = sc.scalar(max(int(period_seconds * 1e9), 1), unit='ns')
    floor_period_ns = max(int(floor_period_seconds * 1e9), 0)
    recent_cutoff = latest - sc.scalar(int(recent_seconds * 1e9), unit='ns')
    is_recent = times >= recent_cutoff
    drop_older = floor_period_ns <= 0

    if drop_older:
        buckets = (latest - times) // period
    else:
        # If no older samples, latest_older is unused (gated by is_recent below),
        # but must be a valid datetime scalar for sc.where.
        is_older = ~is_recent
        latest_older = (
            times[is_older][concat_dim, -1] if sc.any(is_older).value else latest
        )
        floor_period = sc.scalar(floor_period_ns, unit='ns')
        band_latest = sc.where(is_recent, latest, latest_older)
        band_period = sc.where(is_recent, period, floor_period)
        buckets = (band_latest - times) // band_period

    # Keep a sample iff it is the last of its (band, bucket) run.
    bucket_changes = buckets[concat_dim, :-1] != buckets[concat_dim, 1:]
    band_changes = is_recent[concat_dim, :-1] != is_recent[concat_dim, 1:]
    trailing_true = sc.array(dims=[concat_dim], values=np.array([True]))
    keep_mask = sc.concat(
        [bucket_changes | band_changes, trailing_true], dim=concat_dim
    )
    if drop_older:
        keep_mask = keep_mask & is_recent

    if keep_mask.values.all():
        return data
    return data[keep_mask]
