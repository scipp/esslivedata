# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for MAGIC logical detector view transforms."""

import numpy as np
import pytest
import scipp as sc

from ess.livedata.config.instruments.magic.views import (
    DETECTOR_BANK_SIZES,
    get_strip_view,
    get_wire_view,
)


@pytest.fixture(params=list(DETECTOR_BANK_SIZES))
def bank(request) -> str:
    return request.param


def _pixel_count(bank: str) -> int:
    return int(np.prod(list(DETECTOR_BANK_SIZES[bank].values())))


def _raw_counts(bank: str) -> sc.DataArray:
    """Dense per-pixel counts mimicking grouped raw detector data."""
    rng = np.random.default_rng(seed=1)
    return sc.DataArray(
        sc.array(
            dims=['detector_number'],
            values=rng.integers(0, 5, size=_pixel_count(bank)),
            unit='counts',
        )
    )


def test_bank_sizes_match_pixel_counts():
    assert _pixel_count('magic_detector_a') == 245760
    assert _pixel_count('magic_detector_b') == 131072


def test_wire_view_keeps_strip_for_reduction_then_yields_wire(bank: str):
    da = _raw_counts(bank)
    transformed = get_wire_view(da, bank)

    # reduction_dim='strip' must be present and is reduced by the framework.
    assert 'strip' in transformed.dims
    reduced = transformed.sum('strip')
    assert reduced.dims == ('wire', 'other')
    assert reduced.sizes['wire'] == DETECTOR_BANK_SIZES[bank]['wire']
    assert reduced.data.sum().value == da.data.sum().value


def test_strip_view_keeps_other_for_reduction_then_yields_strip(bank: str):
    da = _raw_counts(bank)
    transformed = get_strip_view(da, bank)

    # reduction_dim='other' must be present and is reduced by the framework.
    assert 'other' in transformed.dims
    reduced = transformed.sum('other')
    assert reduced.dims == ('strip',)
    assert reduced.sizes['strip'] == DETECTOR_BANK_SIZES[bank]['strip']
    assert reduced.data.sum().value == da.data.sum().value
