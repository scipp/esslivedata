# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
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
        namespace='test_namespace',
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
    return LinePlotter.from_params(PlotParams1d())


@pytest.fixture
def image_plotter() -> ImagePlotter:
    return ImagePlotter.from_params(PlotParams2d())


@pytest.fixture
def overlay_plotter() -> Overlay1DPlotter:
    from ess.livedata.dashboard.plot_params import PlotParams3d

    return Overlay1DPlotter.from_params(PlotParams3d())


def test_line_plotter_compute(benchmark, line_plotter):
    data_key = _make_result_key()
    da = _make_1d_curve_data()
    benchmark(line_plotter.compute, {data_key: da})


def test_image_plotter_compute(benchmark, image_plotter):
    data_key = _make_result_key()
    da = _make_2d_image_data()
    benchmark(image_plotter.compute, {data_key: da})


def test_overlay_plotter_compute(benchmark, overlay_plotter):
    data_key = _make_result_key()
    da = _make_2d_overlay_data(n_curves=5)
    benchmark(overlay_plotter.compute, {data_key: da})
