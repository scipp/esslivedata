# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Time-based downsampling for timeseries plots.

Long-running timeseries accumulate many points (~100 k for a day at 1 Hz),
which makes per-tick plot rebuilds and the resulting ``pipe.send`` payloads
expensive. ``downsample_timeseries`` reduces a DataArray to two bands of
uniform stride on a fixed time grid: a recent window at fine resolution
plus a coarse floor that extends the run's older history indefinitely
(set the floor period to 0 to drop older data instead).

The bucket grid is anchored to the epoch, so kept samples sit on absolute
time-quanta and do not slide as new samples arrive. The recent-band cutoff
is quantized to the floor period: ``recent_seconds`` is therefore a lower
bound on the actual recent-window length, which can extend by up to one
floor period before a quantum boundary is crossed and a batch of samples
retires from the recent band together. See issue #940.
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

    Bucket boundaries are anchored at the epoch, so kept samples sit on a
    stable absolute grid. Within each band the last sample of each bucket
    is kept (the very latest sample of the input is always present).

    The recent-band cutoff is quantized to the floor grid: actual recent
    length is between ``recent_seconds`` and ``recent_seconds +
    floor_period_seconds``. A ``floor_period_seconds`` of 0 drops older
    data and quantizes the cutoff to the recent period instead.

    Parameters
    ----------
    data:
        Buffered timeseries with a 1-D ``concat_dim`` time coord.
    period_seconds:
        Sample period for the recent band, in seconds. Must be > 0.
    recent_seconds:
        Lower bound on the recent-band length, in seconds. Must be >= 0.
    floor_period_seconds:
        Sample period for the floor band, in seconds. 0 drops older data.
    concat_dim:
        Name of the time dimension.

    Returns
    -------
    :
        New DataArray with the same dims/coords structure but fewer points.
    """
    if concat_dim not in data.dims or concat_dim not in data.coords:
        return data
    if data.sizes[concat_dim] < 2:
        return data

    times = data.coords[concat_dim].to(unit='ns', copy=False)
    latest = times[concat_dim, -1]
    epoch = sc.epoch(unit='ns')
    period = sc.scalar(max(int(period_seconds * 1e9), 1), unit='ns')
    floor_period_ns = max(int(floor_period_seconds * 1e9), 0)
    drop_older = floor_period_ns <= 0
    # 1 ns when drop_older avoids divide-by-zero; older bucket IDs are
    # masked out below so the values are irrelevant.
    floor_period = sc.scalar(floor_period_ns or 1, unit='ns')

    quantum = period if drop_older else floor_period
    raw_cutoff = latest - sc.scalar(int(recent_seconds * 1e9), unit='ns')
    cutoff = ((raw_cutoff - epoch) // quantum) * quantum + epoch
    is_recent = times >= cutoff

    band_period = sc.where(is_recent, period, floor_period)
    buckets = (times - epoch) // band_period

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
