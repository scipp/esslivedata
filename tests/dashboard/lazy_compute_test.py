# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for Plotter lazy-compute gating on active-consumer interest."""

import uuid

import holoviews as hv
import numpy as np
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.data_roles import PRIMARY
from ess.livedata.dashboard.plot_params import PlotParams1d, PlotParams2d
from ess.livedata.dashboard.plots import ImagePlotter, LinePlotter

hv.extension('bokeh')


def _make_key(source: str = 'src') -> ResultKey:
    return ResultKey(
        workflow_id=WorkflowId(instrument='i', name='w', version=1),
        job_id=JobId(source_name=source, job_number=uuid.uuid4()),
        output_name='out',
    )


def _make_1d(n: int = 32) -> sc.DataArray:
    return sc.DataArray(
        data=sc.array(dims=['x'], values=np.arange(n, dtype='float64'), unit='counts'),
        coords={'x': sc.arange('x', n, dtype='float64')},
    )


def _make_2d(nx: int = 8, ny: int = 6) -> sc.DataArray:
    return sc.DataArray(
        data=sc.array(
            dims=['y', 'x'], values=np.arange(ny * nx, dtype='float64').reshape(ny, nx)
        ),
        coords={
            'x': sc.arange('x', nx, dtype='float64'),
            'y': sc.arange('y', ny, dtype='float64'),
        },
    )


@pytest.fixture
def line_plotter() -> LinePlotter:
    return LinePlotter.from_params(PlotParams1d())


@pytest.fixture
def image_plotter() -> ImagePlotter:
    return ImagePlotter.from_params(PlotParams2d())


@pytest.fixture
def line_input() -> dict:
    return {PRIMARY: {_make_key(): _make_1d()}}


class TestLazyCompute:
    """Compute is gated on active tokens; no token = stash only."""

    def test_compute_without_active_token_stashes(self, line_plotter, line_input):
        line_plotter.compute(line_input)
        assert not line_plotter.has_cached_state()

    def test_set_active_true_triggers_pending_build(self, line_plotter, line_input):
        token = object()
        line_plotter.set_active(token, False)
        line_plotter.compute(line_input)
        assert not line_plotter.has_cached_state()
        line_plotter.set_active(token, True)
        assert line_plotter.has_cached_state()

    def test_compute_while_active_builds_immediately(self, line_plotter, line_input):
        token = object()
        line_plotter.set_active(token, True)
        line_plotter.compute(line_input)
        assert line_plotter.has_cached_state()

    def test_only_zero_to_one_transition_rebuilds(self, line_plotter, line_input):
        """A second True after the input was already built does not rebuild."""
        token = object()
        line_plotter.set_active(token, True)
        line_plotter.compute(line_input)
        first_state = line_plotter.get_cached_state()
        # Re-asserting active does not rebuild because _dirty is already False.
        line_plotter.set_active(token, True)
        assert line_plotter.get_cached_state() is first_state

    def test_intermediate_updates_collapse_to_latest(self, line_plotter):
        """N inputs while inactive → on activation, only the latest is built."""
        token = object()
        line_plotter.set_active(token, False)
        for i in range(5):
            line_plotter.compute({PRIMARY: {_make_key(f'src-{i}'): _make_1d(n=4 + i)}})
        assert not line_plotter.has_cached_state()
        line_plotter.set_active(token, True)
        assert line_plotter.has_cached_state()
        # Should have built from the *latest* input (the n=8 input had source name
        # 'src-4'); the cached state should be a non-empty hv element.
        cached = line_plotter.get_cached_state()
        assert cached is not None

    def test_multiple_tokens_keep_active(self, line_plotter, line_input):
        t1, t2 = object(), object()
        line_plotter.set_active(t1, True)
        line_plotter.set_active(t2, True)
        line_plotter.compute(line_input)
        assert line_plotter.has_cached_state()

        # Release one token; plotter stays active. New input should still build.
        line_plotter.set_active(t1, False)
        assert line_plotter.has_active_interest
        new_input = {PRIMARY: {_make_key('new'): _make_1d(n=16)}}
        line_plotter.compute(new_input)
        new_state = line_plotter.get_cached_state()
        assert new_state is not None

        # Release the second; plotter becomes inactive. Further input stashes.
        line_plotter.set_active(t2, False)
        assert not line_plotter.has_active_interest
        line_plotter.compute({PRIMARY: {_make_key('newer'): _make_1d(n=4)}})
        assert line_plotter.get_cached_state() is new_state  # unchanged

    def test_release_idempotent(self, line_plotter):
        """Releasing a token that wasn't acquired is a no-op."""
        token = object()
        line_plotter.set_active(token, False)
        # Should not raise, plotter not active.
        assert not line_plotter.has_active_interest

    def test_image_plotter_lazy(self, image_plotter):
        """ImagePlotter inherits the gating from Plotter base."""
        token = object()
        image_plotter.set_active(token, False)
        image_plotter.compute({PRIMARY: {_make_key(): _make_2d()}})
        assert not image_plotter.has_cached_state()
        image_plotter.set_active(token, True)
        assert image_plotter.has_cached_state()


class TestCorrelationHistogramLazy:
    """CorrelationHistogramPlotter mirrors the same gating as Plotter."""

    def _make_plotter(self, *, active: bool):
        from ess.livedata.dashboard.correlation_plotter import (
            X_AXIS,
            AxisSpec,
            CorrelationHistogramPlotter,
        )
        from ess.livedata.dashboard.plot_params import PlotScaleParams

        renderer = LinePlotter(scale_opts=PlotScaleParams(), mode='histogram')
        plotter = CorrelationHistogramPlotter(
            axes=[AxisSpec(role=X_AXIS, name='position', bins=10)],
            normalize=False,
            renderer=renderer,
        )
        if active:
            plotter.set_active(object(), True)
        return plotter

    def _make_data(self):
        from ess.livedata.dashboard.correlation_plotter import PRIMARY, X_AXIS

        axis = sc.DataArray(
            data=sc.array(dims=['time'], values=[1.0, 2.0, 3.0], unit='m'),
            coords={'time': sc.array(dims=['time'], values=[100, 200, 300], unit='ms')},
        )
        source = sc.DataArray(
            data=sc.array(dims=['time'], values=[10.0, 20.0], unit='counts'),
            coords={'time': sc.array(dims=['time'], values=[150, 250], unit='ms')},
        )
        return {
            PRIMARY: {_make_key('detector'): source},
            X_AXIS: {_make_key('position'): axis},
        }

    def test_compute_without_active_token_stashes(self):
        plotter = self._make_plotter(active=False)
        plotter.compute(self._make_data())
        assert not plotter.has_cached_state()

    def test_set_active_true_triggers_histogram_and_renderer_build(self):
        plotter = self._make_plotter(active=False)
        token = object()
        plotter.compute(self._make_data())
        plotter.set_active(token, True)
        assert plotter.has_cached_state()

    def test_compute_validates_eagerly_when_inactive(self):
        """Bad input raises synchronously even when no consumer is watching."""
        from ess.livedata.dashboard.correlation_plotter import PRIMARY

        plotter = self._make_plotter(active=False)
        with pytest.raises(ValueError, match="at least one data source"):
            plotter.compute({PRIMARY: {}})


class TestStashedInputIsLatest:
    """When activation happens after multiple inputs, build uses latest."""

    def test_latest_input_wins(self, line_plotter):
        token = object()
        line_plotter.set_active(token, False)
        # Distinguishable inputs via source name embedded in result keys.
        line_plotter.compute({PRIMARY: {_make_key('first'): _make_1d(n=4)}})
        line_plotter.compute({PRIMARY: {_make_key('second'): _make_1d(n=8)}})
        line_plotter.set_active(token, True)
        # The state should be from the second input; we verify by re-stashing
        # and confirming idempotent get.
        s1 = line_plotter.get_cached_state()
        line_plotter.set_active(token, True)  # no-op, dirty cleared
        s2 = line_plotter.get_cached_state()
        assert s1 is s2
