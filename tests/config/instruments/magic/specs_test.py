# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for MAGIC spec registration against the geometry file."""

import numpy as np
import pytest
import scippnexus as snx

from ess.livedata.config.instruments.magic import specs
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename


@pytest.mark.parametrize('name', specs.detector_names)
def test_pixel_ranges_match_the_default_geometry_file(name: str) -> None:
    """Events are grouped by the configured range, the projection reads pixel
    positions from the file, so the two must enumerate the same ids."""
    first, last = specs.detector_pixel_ranges[name]
    with snx.File(get_nexus_geometry_filename('magic')) as f:
        declared = f['entry/instrument'][name]['detector_number'][()]
    np.testing.assert_array_equal(declared.values, np.arange(first, last + 1))
