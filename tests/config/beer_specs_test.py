# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the BEER detector view transforms."""

import math

import pytest
import scipp as sc

from ess.livedata.config.instruments.beer.specs import detector_pixel_ranges
from ess.livedata.config.instruments.beer.views import (
    BANK_SIZES,
    get_bank_view,
    get_panel_view,
)


@pytest.fixture
def bank() -> sc.DataArray:
    """One count in every pixel of a bank."""
    npixel = math.prod(BANK_SIZES.values())
    return sc.DataArray(
        sc.ones(dims=['detector_number'], shape=[npixel], unit='counts')
    )


def test_bank_view_yields_single_image_summed_over_panels(bank: sc.DataArray) -> None:
    image = get_bank_view(bank, 'beer_detector_s2').sum(['panel', 'y_bin', 'x_bin'])
    assert image.sizes == {'y': 250, 'x': 250}
    # Each screen pixel collects 12 panels times a 4x4 pixel block.
    assert sc.identical(
        image.data, sc.full(sizes=image.sizes, value=192.0, unit='counts')
    )


def test_panel_view_keeps_panels_separate(bank: sc.DataArray) -> None:
    image = get_panel_view(bank, 'beer_detector_s2').sum(['y_bin', 'x_bin'])
    assert image.sizes == {'panel': 12, 'y': 125, 'x': 125}
    # Each screen pixel collects an 8x8 pixel block of a single panel.
    assert sc.identical(
        image.data, sc.full(sizes=image.sizes, value=64.0, unit='counts')
    )


def test_detector_pixel_ranges_are_contiguous_and_match_bank_shape() -> None:
    npixel = math.prod(BANK_SIZES.values())
    (first_s, last_s), (first_n, last_n) = detector_pixel_ranges.values()
    assert (first_s, last_s) == (1, npixel)
    assert (first_n, last_n) == (npixel + 1, 2 * npixel)
