# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for ``_pad_linear``, ``_pad_log``, ``_finite_min_max`` edge cases
and ``Plotter.compute`` cleanup on mid-loop exceptions.
"""

from __future__ import annotations

import uuid
from typing import Any

import holoviews as hv
import numpy as np
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.data_roles import PRIMARY
from ess.livedata.dashboard.plot_params import PlotParams1d, PlotScale
from ess.livedata.dashboard.plots import (
    LinePlotter,
    Plotter,
    _finite_min_max,
    _pad_linear,
    _pad_log,
)

hv.extension('bokeh')


def _key(source: str = 'src', output: str = 'out') -> ResultKey:
    return ResultKey(
        workflow_id=WorkflowId(instrument='test', name='test', version=1),
        job_id=JobId(source_name=source, job_number=uuid.uuid4()),
        output_name=output,
    )


class TestPadLinear:
    def test_zero_range_at_zero_returns_non_degenerate(self):
        lo, hi = _pad_linear(0.0, 0.0)
        assert lo < hi

    def test_zero_range_at_five_scales_to_magnitude(self):
        lo, hi = _pad_linear(5.0, 5.0)
        assert lo < 5.0 < hi
        # Symmetric around the value
        assert pytest.approx(5.0 - lo) == hi - 5.0

    def test_zero_range_at_large_value_scales_with_magnitude(self):
        # |lo| * 0.05 = 50 > floor (0.5) so magnitude-based offset wins.
        lo, hi = _pad_linear(1000.0, 1000.0)
        assert pytest.approx(hi - lo) == 100.0

    def test_zero_range_at_negative_value(self):
        lo, hi = _pad_linear(-3.0, -3.0)
        assert lo < -3.0 < hi

    def test_non_degenerate_uses_5_percent_pad(self):
        lo, hi = _pad_linear(0.0, 10.0)
        assert pytest.approx(lo) == -0.5
        assert pytest.approx(hi) == 10.5


class TestPadLog:
    def test_positive_inputs_apply_multiplicative_pad(self):
        lo, hi = _pad_log(2.0, 8.0)
        assert pytest.approx(lo) == 2.0 / 1.1
        assert pytest.approx(hi) == 8.0 * 1.1

    def test_tiny_positive_inputs_stay_positive(self):
        # _pad_log no longer guards; callers must filter. Verify behaviour
        # for a tiny but positive lo: padded result is still positive.
        lo, hi = _pad_log(1e-300, 1.0)
        assert lo > 0.0
        assert hi > 0.0


class TestFiniteMinMax:
    def test_basic_finite_min_max(self):
        assert _finite_min_max(np.array([1.0, 2.0, 3.0])) == (1.0, 3.0)

    def test_drops_non_finite(self):
        result = _finite_min_max(np.array([np.nan, 2.0, np.inf, 3.0]))
        assert result == (2.0, 3.0)

    def test_empty_input_returns_none(self):
        assert _finite_min_max(np.array([])) is None

    def test_all_non_finite_returns_none(self):
        assert _finite_min_max(np.array([np.nan, np.inf, -np.inf])) is None

    def test_log_drops_non_positive(self):
        assert _finite_min_max(np.array([-1.0, -2.0, 3.0, 4.0]), log=True) == (
            3.0,
            4.0,
        )

    def test_log_returns_none_when_no_positive(self):
        assert _finite_min_max(np.array([-1.0, -2.0]), log=True) is None

    def test_log_drops_zero(self):
        assert _finite_min_max(np.array([0.0, 1.0, 5.0]), log=True) == (1.0, 5.0)

    def test_log_returns_none_for_all_zero(self):
        assert _finite_min_max(np.array([0.0, 0.0, 0.0]), log=True) is None

    def test_log_ignores_nan(self):
        assert _finite_min_max(np.array([np.nan, -1.0, 2.0]), log=True) == (2.0, 2.0)


class TestLinePlotterLogYWithZeros:
    def test_constant_zero_series_log_y_skips_y_target(self):
        params = PlotParams1d()
        params.plot_scale.y_scale = PlotScale.log
        plotter = LinePlotter.from_params(params)
        key = _key()
        data = sc.DataArray(
            sc.array(dims=['x'], values=[0.0, 0.0, 0.0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='m')},
        )
        plotter.compute({PRIMARY: {key: data}})

        targets = plotter.get_range_targets(key)
        # x is still computable; y must be absent because no positive values exist.
        if targets is not None:
            assert 'y' not in targets

    def test_negative_x_coord_log_x_skips_x_target(self):
        params = PlotParams1d()
        params.plot_scale.x_scale = PlotScale.log
        plotter = LinePlotter.from_params(params)
        key = _key()
        data = sc.DataArray(
            sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='counts'),
            coords={'x': sc.array(dims=['x'], values=[-3.0, -2.0, -1.0], unit='m')},
        )
        plotter.compute({PRIMARY: {key: data}})

        targets = plotter.get_range_targets(key)
        if targets is not None:
            assert 'x' not in targets


class _RaisingPlotter(Plotter):
    """Test plotter whose ``plot()`` raises after the first successful call."""

    AUTOSCALE_AXES = frozenset({'x'})

    def __init__(self) -> None:
        super().__init__()
        self._call_count = 0

    def plot(
        self, data: sc.DataArray, data_key: ResultKey, *, label: str = '', **kwargs: Any
    ) -> hv.Element:
        self._call_count += 1
        # Always populate targets so we can assert cleanup happens.
        self._range_targets[data_key] = {'x': (0.0, 1.0)}
        if self._call_count == 1:
            return hv.Curve([(0, 0), (1, 1)])
        raise RuntimeError("boom")


class TestComputeExceptionResetsRangeTargets:
    def test_partial_targets_cleared_on_mid_loop_exception(self):
        plotter = _RaisingPlotter()
        key1 = _key('a')
        key2 = _key('b')
        data = {
            PRIMARY: {
                key1: sc.DataArray(sc.array(dims=['x'], values=[1.0, 2.0])),
                key2: sc.DataArray(sc.array(dims=['x'], values=[3.0, 4.0])),
            }
        }

        plotter.compute(data)

        # First call wrote targets, second raised — targets must be cleared.
        assert dict(plotter.iter_range_targets()) == {}
        assert plotter.get_range_targets(key1) is None
        assert plotter.get_range_targets(key2) is None

    def test_successful_compute_keeps_targets(self):
        plotter = _RaisingPlotter()
        # Only one entry -> single successful call, no exception.
        key1 = _key('a')
        data = {
            PRIMARY: {key1: sc.DataArray(sc.array(dims=['x'], values=[1.0, 2.0]))},
        }

        plotter.compute(data)

        assert plotter.get_range_targets(key1) == {'x': (0.0, 1.0)}
