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


def test_multiblade_view_folds_strip_slowest_and_wire_fastest() -> None:
    da = _raw_counts()
    folded = get_multiblade_view(da, 'multiblade_detector')

    assert folded.sizes == {'strip': 64, 'blade': 32, 'wire': 32}
    # Consecutive detector numbers step along wire first, then blade.
    assert folded['strip', 0]['blade', 0].values.tolist() == da.values[:32].tolist()
    assert folded['strip', 0]['blade', 1].values.tolist() == da.values[32:64].tolist()


def test_spectrum_view_sums_over_strips() -> None:
    folded = get_multiblade_view(_raw_counts(), 'multiblade_detector')
    spectrum = get_spectrum_view(folded)

    assert spectrum.dims == ('blade', 'wire')
    assert spectrum.sum().value == folded.sum().value
