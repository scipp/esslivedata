# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for time-based timeseries downsampling (issue #940)."""

from __future__ import annotations

import numpy as np
import pytest
import scipp as sc

from ess.livedata.dashboard.timeseries_downsample import downsample_timeseries


def _make_timeseries(n: int, period_s: float = 1.0) -> sc.DataArray:
    """1 Hz-by-default timeseries with start_time/end_time coords."""
    times_ns = (np.arange(n, dtype=np.int64) * int(period_s * 1e9)).astype(
        'datetime64[ns]'
    )
    values = np.arange(n, dtype=np.float64)
    return sc.DataArray(
        data=sc.array(dims=['time'], values=values, unit='K'),
        coords={
            'time': sc.array(dims=['time'], values=times_ns),
            'start_time': sc.array(dims=['time'], values=times_ns),
            'end_time': sc.array(dims=['time'], values=times_ns),
        },
    )


class TestDownsampleTimeseries:
    def test_no_downsampling_when_buffer_shorter_than_recent(self):
        data = _make_timeseries(60)  # 1 min at 1 Hz
        result = downsample_timeseries(
            data,
            period_seconds=1.0,
            recent_seconds=3600.0,
            floor_period_seconds=300.0,
        )
        assert result.sizes['time'] == 60
        assert sc.identical(result, data)

    def test_recent_only_when_floor_disabled(self):
        # 2 h of data, recent window 1 h, floor=0 → drop older
        data = _make_timeseries(7200)
        result = downsample_timeseries(
            data,
            period_seconds=1.0,
            recent_seconds=3600.0,
            floor_period_seconds=0.0,
        )
        # ~3600 recent samples at period=1s. Boundary inclusion may add one.
        assert result.sizes['time'] in (3600, 3601)
        # Latest sample preserved
        assert result.coords['time'].values[-1] == data.coords['time'].values[-1]
        # Values are the trailing slice of the original
        assert result.values[-1] == data.values[-1]
        assert result.values[0] >= 3599

    def test_floor_band_for_long_run(self):
        # 1 day @ 1 Hz = 86400 points, recent_seconds=3600, floor=300s.
        # Soft window: recent length is 3600-3900 s; floor: ~275 buckets.
        data = _make_timeseries(86400)
        result = downsample_timeseries(
            data,
            period_seconds=1.0,
            recent_seconds=3600.0,
            floor_period_seconds=300.0,
        )
        n = result.sizes['time']
        assert 3850 <= n <= 4200, n
        # Last sample is always preserved.
        assert result.coords['time'].values[-1] == data.coords['time'].values[-1]

    def test_period_seconds_strides_recent_band(self):
        data = _make_timeseries(100)  # 100 s @ 1 Hz
        result = downsample_timeseries(
            data,
            period_seconds=10.0,
            recent_seconds=100.0,
            floor_period_seconds=0.0,
        )
        # ~10 stride buckets in 100s; bucket 0 is end-anchored, includes latest.
        assert 10 <= result.sizes['time'] <= 11
        assert result.values[-1] == data.values[-1]

    def test_latest_sample_always_kept_under_misaligned_recent_window(self):
        # recent_seconds chosen so the bucket grid does not align to the latest.
        data = _make_timeseries(100)
        result = downsample_timeseries(
            data,
            period_seconds=10.0,
            recent_seconds=95.0,
            floor_period_seconds=0.0,
        )
        assert result.values[-1] == data.values[-1]
        assert result.coords['time'].values[-1] == data.coords['time'].values[-1]

    def test_preserves_start_time_end_time_coords(self):
        data = _make_timeseries(60)
        result = downsample_timeseries(
            data,
            period_seconds=10.0,
            recent_seconds=60.0,
            floor_period_seconds=0.0,
        )
        # 1-D start_time and end_time coords stay aligned with the time dim.
        assert 'start_time' in result.coords
        assert 'end_time' in result.coords
        assert result.coords['start_time'].sizes == {'time': result.sizes['time']}

    def test_preserves_variances(self):
        n = 100
        times_ns = (np.arange(n, dtype=np.int64) * int(1e9)).astype('datetime64[ns]')
        data = sc.DataArray(
            data=sc.array(
                dims=['time'],
                values=np.arange(n, dtype=np.float64),
                variances=np.ones(n, dtype=np.float64),
                unit='K',
            ),
            coords={'time': sc.array(dims=['time'], values=times_ns)},
        )
        result = downsample_timeseries(
            data,
            period_seconds=10.0,
            recent_seconds=100.0,
            floor_period_seconds=0.0,
        )
        assert result.variances is not None
        assert result.variances.shape == result.values.shape

    def test_empty_or_single_sample_returned_unchanged(self):
        for n in (0, 1):
            data = _make_timeseries(n)
            result = downsample_timeseries(
                data,
                period_seconds=1.0,
                recent_seconds=3600.0,
                floor_period_seconds=300.0,
            )
            assert result.sizes['time'] == n

    def test_no_time_dim_passthrough(self):
        data = sc.DataArray(sc.array(dims=['x'], values=np.arange(10.0)))
        result = downsample_timeseries(
            data,
            period_seconds=1.0,
            recent_seconds=3600.0,
            floor_period_seconds=300.0,
        )
        assert sc.identical(result, data)


@pytest.mark.parametrize('n', [10_000, 100_000])
def test_downsampling_caps_output_size(n: int):
    """Sanity check: downsampling actually reduces point count drastically."""
    data = _make_timeseries(n)
    result = downsample_timeseries(
        data,
        period_seconds=1.0,
        recent_seconds=3600.0,
        floor_period_seconds=300.0,
    )
    assert result.sizes['time'] < 5000


def test_kept_samples_stable_between_cutoff_crossings():
    """Epoch-anchored grid: existing kept samples don't move between ticks.

    Picks two consecutive sizes that do not straddle a floor-period quantum
    crossing; the second output must extend the first by exactly the newly
    arrived sample.
    """
    kwargs = {
        "period_seconds": 1.0,
        "recent_seconds": 3600.0,
        "floor_period_seconds": 300.0,
    }
    # Tick from n=7202 to n=7203: latest=7201 -> 7202; raw_cutoff 3601 -> 3602.
    # Quantized to 300 stays at 3600. No quantum crossing.
    a = downsample_timeseries(_make_timeseries(7202), **kwargs)
    b = downsample_timeseries(_make_timeseries(7203), **kwargs)
    a_times = set(a.coords['time'].values.tolist())
    b_times = set(b.coords['time'].values.tolist())
    assert a_times.issubset(b_times)
    assert b_times - a_times == {7202 * 10**9}
