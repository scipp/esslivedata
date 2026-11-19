# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for FREIA logical detector view transforms."""

import numpy as np
import scipp as sc

from ess.livedata.config.instruments.freia.specs import detector_pixel_range
from ess.livedata.config.instruments.freia.views import (
    get_multiblade_view,
    get_spectrum_view,
)


def _raw_counts() -> sc.DataArray:
    """Dense per-pixel counts mimicking grouped raw detector data.

    Sized from the configured pixel range, so the fold tests also check that
    ``ess.freia``'s bank sizes cover exactly the pixels the detector has.
    """
    first, last = detector_pixel_range
    rng = np.random.default_rng(seed=1)
    return sc.DataArray(
        sc.array(
            dims=['detector_number'],
            values=rng.integers(0, 5, size=last - first + 1),
            unit='counts',
        )
    )


def test_multiblade_view_folds_strip_fastest_and_blade_slowest() -> None:
    """Pixel order of ``coda_freia_999999_00030110.hdf``.

    In that file x depends on strip only, and y and z depend on blade and wire only,
    only if consecutive detector numbers step along strip first, then wire, then
    blade.
    """
    da = _raw_counts()
    folded = get_multiblade_view(da, 'multiblade_detector')

    assert folded.dims == ('blade', 'wire', 'strip')
    assert folded.sizes == {'blade': 32, 'wire': 32, 'strip': 64}
    assert folded['blade', 0]['wire', 0].values.tolist() == da.values[:64].tolist()
    assert folded['blade', 0]['wire', 1].values.tolist() == da.values[64:128].tolist()
    assert folded['blade', 1]['wire', 0].values.tolist() == (
        da.values[2048:2112].tolist()
    )


def test_spectrum_view_sums_over_strips() -> None:
    folded = get_multiblade_view(_raw_counts(), 'multiblade_detector')
    spectrum = get_spectrum_view(folded)

    assert spectrum.dims == ('blade', 'wire')
    assert spectrum.sum().value == folded.sum().value
