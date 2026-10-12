# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ess.livedata.config.spectral_coords import (
    SPECTRAL_RANGE_HIGH_COORD,
    SPECTRAL_RANGE_LOW_COORD,
)
from ess.livedata.workflows.spectral_range import select_spectral_range


def _histogram() -> sc.DataArray:
    """Counts 0..9 in 10 bins of 1 ms from 0 to 10 ms, behind a pixel dim."""
    return sc.DataArray(
        sc.array(dims=['pixel', 'toa'], values=[list(range(10))], unit='counts'),
        coords={'toa': sc.linspace('toa', 0.0, 10.0, 11, unit='ms')},
    )


def _interval(da: sc.DataArray) -> tuple[sc.Variable, sc.Variable]:
    return da.coords[SPECTRAL_RANGE_LOW_COORD], da.coords[SPECTRAL_RANGE_HIGH_COORD]


def test_without_range_selects_all_bins_and_their_interval() -> None:
    result = select_spectral_range(_histogram(), None)

    assert result.sizes['toa'] == 10
    low, high = _interval(result)
    assert sc.identical(low, sc.scalar(0.0, unit='ms'))
    assert sc.identical(high, sc.scalar(10.0, unit='ms'))


def test_interval_extends_to_edges_of_overlapping_bins() -> None:
    spectral_range = (sc.scalar(2.5, unit='ms'), sc.scalar(5.5, unit='ms'))

    result = select_spectral_range(_histogram(), spectral_range)

    assert result.sum().value == 2 + 3 + 4 + 5
    low, high = _interval(result)
    assert sc.identical(low, sc.scalar(2.0, unit='ms'))
    assert sc.identical(high, sc.scalar(6.0, unit='ms'))


def test_interval_survives_summing_over_spectral_dim() -> None:
    spectral_range = (sc.scalar(2.0, unit='ms'), sc.scalar(4.0, unit='ms'))

    result = select_spectral_range(_histogram(), spectral_range).sum('toa')

    low, high = _interval(result)
    assert sc.identical(low, sc.scalar(2.0, unit='ms'))
    assert sc.identical(high, sc.scalar(4.0, unit='ms'))


def test_range_outside_edges_selects_nothing_and_keeps_requested_interval() -> None:
    spectral_range = (sc.scalar(20.0, unit='ms'), sc.scalar(30.0, unit='ms'))

    result = select_spectral_range(_histogram(), spectral_range)

    assert result.sizes['toa'] == 0
    assert result.sum().value == 0
    low, high = _interval(result)
    assert sc.identical(low, spectral_range[0])
    assert sc.identical(high, spectral_range[1])
