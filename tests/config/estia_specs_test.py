# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pydantic
import pytest
import scipp as sc

from ess.livedata.config.instruments.estia.specs import (
    BeamDivergenceLimits,
    EstiaLiveDiagnosticsParams,
    IndexLimits,
)


def test_estia_live_diagnostics_default_limits_match_previous_literals():
    params = EstiaLiveDiagnosticsParams()

    assert sc.identical(params.y_index_limits.get_limits()[0], sc.scalar(0))
    assert sc.identical(params.y_index_limits.get_limits()[1], sc.scalar(63))
    assert sc.identical(params.z_index_limits.get_limits()[0], sc.scalar(0))
    assert sc.identical(params.z_index_limits.get_limits()[1], sc.scalar(1535))
    assert sc.identical(
        params.beam_divergence_limits.get_limits()[0],
        sc.scalar(-0.75, unit='deg'),
    )
    assert sc.identical(
        params.beam_divergence_limits.get_limits()[1],
        sc.scalar(0.75, unit='deg'),
    )


def test_estia_live_diagnostics_custom_limits_convert_to_scipp_scalars():
    params = EstiaLiveDiagnosticsParams(
        y_index_limits={'start': 2, 'stop': 5},
        z_index_limits=IndexLimits(start=10, stop=20),
        beam_divergence_limits={'start': -0.1, 'stop': 0.2, 'unit': 'rad'},
    )

    assert sc.identical(params.y_index_limits.get_limits()[0], sc.scalar(2))
    assert sc.identical(params.y_index_limits.get_limits()[1], sc.scalar(5))
    assert sc.identical(params.z_index_limits.get_limits()[0], sc.scalar(10))
    assert sc.identical(params.z_index_limits.get_limits()[1], sc.scalar(20))
    assert sc.identical(
        params.beam_divergence_limits.get_limits()[0],
        sc.scalar(-0.1, unit='rad'),
    )
    assert sc.identical(
        params.beam_divergence_limits.get_limits()[1],
        sc.scalar(0.2, unit='rad'),
    )


def test_estia_index_limits_reject_stop_less_than_start():
    with pytest.raises(pydantic.ValidationError, match='greater than or equal'):
        IndexLimits(start=2, stop=1)


def test_estia_beam_divergence_limits_reject_stop_less_than_start():
    with pytest.raises(pydantic.ValidationError, match='greater than start'):
        BeamDivergenceLimits(start=1.0, stop=0.0)
