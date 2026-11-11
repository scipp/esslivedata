# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for TBL logical detector view transforms."""

import warnings

import numpy as np
import pytest
import scipp as sc
import scippnexus as snx
from scipp.testing import assert_identical

from ess.livedata.config.instruments.tbl.views import get_multiblade_view
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename


# The geometry file carries no data, so scippnexus falls back to loading each
# component as a plain DataGroup, with pixel offsets on an unnamed dim.
@pytest.fixture(scope='module')
def multiblade() -> sc.DataArray:
    """Multiblade pixel offsets from the geometry file, folded by the view."""
    with (
        warnings.catch_warnings(),
        snx.File(get_nexus_geometry_filename('tbl')) as f,
    ):
        warnings.filterwarnings('ignore', 'Failed to load', UserWarning)
        dg = f['entry/instrument/multiblade_detector'][()]
    pixels = sc.DataArray(
        dg['detector_number'],
        coords={
            axis: dg[f'{axis}_pixel_offset'].rename_dims(dim_0='detector_number')
            for axis in 'xyz'
        },
    )
    return get_multiblade_view(pixels, 'multiblade_detector')


def _along(coord: sc.Variable, dim: str) -> np.ndarray:
    """Values of ``coord`` along ``dim``, which must be its only varying axis."""
    profile = coord[{d: 0 for d in coord.dims if d != dim}]
    assert_identical(coord, sc.broadcast(profile, sizes=coord.sizes))
    return profile.values


def test_multiblade_view_has_blade_wire_strip_dims(multiblade) -> None:
    assert multiblade.sizes == {'blade': 14, 'wire': 32, 'strip': 64}


def test_multiblade_blades_count_up_from_the_bottom(multiblade) -> None:
    # The blades are inclined, so y also varies along wire within each blade.
    y = multiblade.coords['y'].mean(['wire', 'strip'])
    assert np.all(np.diff(y.values) > 0)


def test_multiblade_wires_run_downstream_along_the_beam(multiblade) -> None:
    assert np.all(np.diff(_along(multiblade.coords['z'], 'wire')) > 0)


def test_multiblade_strips_run_left_to_right_looking_downstream(multiblade) -> None:
    # NeXus x points left when looking downstream, so x decreases left to right.
    assert np.all(np.diff(_along(multiblade.coords['x'], 'strip')) < 0)


def test_multiblade_detector_number_matches_documented_formula(multiblade) -> None:
    blade, wire, strip = np.indices((14, 32, 64))
    expected = 1 + 2048 * (13 - blade) + 64 * (31 - wire) + strip
    np.testing.assert_array_equal(multiblade.data.values, expected)
