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
def bank(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


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


def test_bank_sizes_match_pixel_counts() -> None:
    # Voxel counts of `detector_number` in coda_magic_999999_00015510.hdf. The
    # folds are meaningless if they do not cover exactly the pixels the files
    # carry, and the coda files are too large to check in.
    assert _pixel_count('magic_detector_a') == 491520
    assert _pixel_count('magic_detector_b') == 131072


def test_wire_view_keeps_strip_for_reduction_then_yields_wire(bank: str) -> None:
    da = _raw_counts(bank)
    transformed = get_wire_view(da, bank)

    # reduction_dim='strip' must be present and is reduced by the framework.
    assert 'strip' in transformed.dims
    reduced = transformed.sum('strip')
    assert reduced.dims == ('wire', 'segment')
    assert reduced.sizes['wire'] == DETECTOR_BANK_SIZES[bank]['wire']
    assert reduced.data.sum().value == da.data.sum().value


def test_strip_view_keeps_wire_segment_for_reduction_then_yields_strip(
    bank: str,
) -> None:
    da = _raw_counts(bank)
    transformed = get_strip_view(da, bank)

    # reduction_dim='wire/segment' must be present and is reduced by the framework.
    assert 'wire/segment' in transformed.dims
    reduced = transformed.sum('wire/segment')
    assert reduced.dims == ('strip',)
    assert reduced.sizes['strip'] == DETECTOR_BANK_SIZES[bank]['strip']
    assert reduced.data.sum().value == da.data.sum().value
