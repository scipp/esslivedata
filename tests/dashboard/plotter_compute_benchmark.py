# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Benchmarks for Plotter.compute() with different plotter types."""

import uuid

import holoviews as hv
import numpy as np
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard.plot_params import PlotParams1d, PlotParams2d
from ess.livedata.dashboard.plots import ImagePlotter, LinePlotter, Overlay1DPlotter

hv.extension('bokeh')


def _make_result_key(source_name: str = 'test_source') -> ResultKey:
    workflow_id = WorkflowId(
        instrument='test_instrument',
        name='test_workflow',
        version=1,
    )
    job_id = JobId(source_name=source_name, job_number=uuid.uuid4())
    return ResultKey(workflow_id=workflow_id, job_id=job_id, output_name='test_result')


def _make_1d_curve_data(n: int = 100) -> sc.DataArray:
    return sc.DataArray(
        data=sc.array(dims=['x'], values=np.random.rand(n), unit='counts'),
        coords={'x': sc.array(dims=['x'], values=np.arange(n, dtype='float64'))},
    )


def _make_2d_image_data(nx: int = 64, ny: int = 64) -> sc.DataArray:
    return sc.DataArray(
        data=sc.array(dims=['y', 'x'], values=np.random.rand(ny, nx), unit='counts'),
        coords={
            'x': sc.array(dims=['x'], values=np.arange(nx, dtype='float64')),
            'y': sc.array(dims=['y'], values=np.arange(ny, dtype='float64')),
        },
    )


def _make_2d_overlay_data(n_curves: int = 5, n_points: int = 100) -> sc.DataArray:
    return sc.DataArray(
        data=sc.array(
            dims=['curve', 'x'],
            values=np.random.rand(n_curves, n_points),
            unit='counts',
        ),
        coords={
            'curve': sc.array(
                dims=['curve'], values=np.arange(n_curves, dtype='float64')
            ),
            'x': sc.array(dims=['x'], values=np.arange(n_points, dtype='float64')),
        },
    )


@pytest.fixture
def line_plotter() -> LinePlotter:
    plotter = LinePlotter.from_params(PlotParams1d())
    plotter.set_active(object(), True)
    return plotter


@pytest.fixture
def image_plotter() -> ImagePlotter:
    plotter = ImagePlotter.from_params(PlotParams2d())
    plotter.set_active(object(), True)
    return plotter


@pytest.fixture
def overlay_plotter() -> Overlay1DPlotter:
    plotter = Overlay1DPlotter.from_params(PlotParams1d())
    plotter.set_active(object(), True)
    return plotter


def test_line_plotter_compute(benchmark, line_plotter):
    data_key = _make_result_key()
    da = _make_1d_curve_data()
    benchmark(line_plotter.compute, {'primary': {data_key: da}})


def test_image_plotter_compute(benchmark, image_plotter):
    data_key = _make_result_key()
    da = _make_2d_image_data()
    benchmark(image_plotter.compute, {'primary': {data_key: da}})


def test_overlay_plotter_compute(benchmark, overlay_plotter):
    data_key = _make_result_key()
    da = _make_2d_overlay_data(n_curves=5)
    benchmark(overlay_plotter.compute, {'primary': {data_key: da}})


# Tab-switch latency benchmarks. These measure the cost of a 0→1 active-token
# transition with a dirty pending input, which is the work that happens when
# a user switches to a previously hidden tab containing one or more plotters.


def _stash_input(plotter, data: dict) -> None:
    """Submit input while the plotter has no active token (input gets stashed)."""
    token = object()
    plotter.set_active(token, False)
    plotter.compute(data)
    assert not plotter.has_cached_state()


def _activate_once(plotter) -> None:
    plotter.set_active(object(), True)


def test_tab_switch_latency_line_n100k(benchmark, line_plotter):
    """One LinePlotter, N=100k, dirty: time the activation rebuild."""
    data_key = _make_result_key()
    da = _make_1d_curve_data(n=100_000)

    def setup():
        plotter = LinePlotter.from_params(PlotParams1d())
        _stash_input(plotter, {'primary': {data_key: da}})
        return (plotter,), {}

    benchmark.pedantic(_activate_once, setup=setup, rounds=20)


def test_tab_switch_latency_image_1k(benchmark, image_plotter):
    """One ImagePlotter, 1k x 1k pixels, dirty: time the activation rebuild."""
    data_key = _make_result_key()
    da = _make_2d_image_data(nx=1024, ny=1024)

    def setup():
        plotter = ImagePlotter.from_params(PlotParams2d())
        _stash_input(plotter, {'primary': {data_key: da}})
        return (plotter,), {}

    benchmark.pedantic(_activate_once, setup=setup, rounds=10)


def test_tab_switch_latency_10_line_plotters(benchmark):
    """Worst-case "busy tab" switch: 10 LinePlotters at N=100k, sequential rebuild."""
    data_key = _make_result_key()
    da = _make_1d_curve_data(n=100_000)

    def setup():
        plotters = [LinePlotter.from_params(PlotParams1d()) for _ in range(10)]
        for p in plotters:
            _stash_input(p, {'primary': {data_key: da}})
        return (plotters,), {}

    def activate_all(plotters):
        for p in plotters:
            p.set_active(object(), True)

    benchmark.pedantic(activate_all, setup=setup, rounds=10)
