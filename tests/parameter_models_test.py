# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
from pydantic import ValidationError

from ess.livedata.parameter_models import Scale, WavelengthEdges


def test_log_edges_are_geometrically_spaced_between_bounds() -> None:
    edges = WavelengthEdges(
        start=1.0, stop=100.0, num_bins=2, scale=Scale.LOG
    ).get_edges()

    expected = sc.array(dims=['wavelength'], values=[1.0, 10.0, 100.0], unit='Å')
    assert sc.allclose(edges, expected)


@pytest.mark.parametrize('start', [0.0, -1.0])
def test_log_scale_rejects_non_positive_start(start: float) -> None:
    with pytest.raises(ValidationError, match='start must be positive'):
        WavelengthEdges(start=start, stop=100.0, scale=Scale.LOG)


@pytest.mark.parametrize('start', [0.0, -1.0])
def test_linear_scale_allows_non_positive_start(start: float) -> None:
    WavelengthEdges(start=start, stop=100.0, scale=Scale.LINEAR)
