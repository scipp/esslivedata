# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import scipp as sc

from ess.livedata.parameter_models import Scale, WavelengthEdges


def test_log_edges_are_geometrically_spaced_between_bounds() -> None:
    edges = WavelengthEdges(
        start=1.0, stop=100.0, num_bins=2, scale=Scale.LOG
    ).get_edges()

    expected = sc.array(dims=['wavelength'], values=[1.0, 10.0, 100.0], unit='Å')
    assert sc.allclose(edges, expected)
