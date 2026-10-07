# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for TBL spec registration."""

import pytest
import scippnexus as snx

from ess.livedata.config.instruments.tbl.specs import GEOMETRY_FILE_Z_POSITIONS
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename


# The geometry file carries no data, so scippnexus falls back to loading each
# component as a plain DataGroup. Positions still resolve from depends_on.
@pytest.mark.filterwarnings('ignore:Failed to load:UserWarning')
def test_positions_shown_in_descriptions_match_geometry_file() -> None:
    """The view descriptions warn users with these positions; they must state
    what wavelength mode actually computes with."""
    with snx.File(get_nexus_geometry_filename('tbl')) as f:
        instrument = f['entry/instrument']
        for name, z in GEOMETRY_FILE_Z_POSITIONS.items():
            position = snx.compute_positions(instrument[name][()])['position']
            assert position.fields.z.to(unit='m').mean().value == pytest.approx(
                z, abs=1e-6
            ), name
