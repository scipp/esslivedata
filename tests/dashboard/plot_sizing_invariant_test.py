# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Render-geometry invariant: a plot's figures must be sized consistently.

The "collapsed detector image" bug (figures left ``stretch_both`` while their
pane asked for ``stretch_width``, so the free axis was unconstrained and the
plot rendered at zero height) passed every functional test: the data and the
figure existed, only the *rendered geometry* was wrong.

A real browser is needed to observe the collapse itself (``inner_height``), but
the upstream cause is observable headlessly: every figure a plotter produces
must adopt the responsive sizing mode the cell's pane will wrap it in. These
tests render each plotter through the real ``compute -> present -> bokeh`` path
and assert that invariant across aspect types and combine modes -- including
the layout-mode sub-figures that the original bug skipped.
"""

from __future__ import annotations

import uuid

import holoviews as hv
import numpy as np
import pytest
import scipp as sc
from bokeh.models import Plot
from holoviews.plotting.bokeh import BokehRenderer

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard import plots
from ess.livedata.dashboard.plot_params import (
    CombineMode,
    LayoutParams,
    PlotAspect,
    PlotAspectType,
    PlotParams1d,
    PlotParams2d,
    StretchMode,
)
from ess.livedata.dashboard.widgets.cell import _get_sizing_mode

hv.extension('bokeh')


def _figures(hv_obj) -> list[Plot]:
    """All Bokeh figures in the rendered HoloViews object (Layout -> one per cell)."""
    state = BokehRenderer.instance().get_plot(hv_obj).state
    return [m for m in state.references() if isinstance(m, Plot)]


def _present_figures(plotter, data: dict) -> list[Plot]:
    plotter.compute({'primary': data})
    presenter = plotter.create_presenter()
    pipe = hv.streams.Pipe(data=plotter.get_cached_state())
    return _figures(presenter.present(pipe))


def _keys(n: int) -> list[ResultKey]:
    wf = WorkflowId(instrument='test', name='wf', version=1)
    return [
        ResultKey(
            workflow_id=wf,
            job_id=JobId(source_name=f's{i}', job_number=uuid.uuid4()),
            output_name='out',
        )
        for i in range(n)
    ]


def _data_1d(n: int) -> dict:
    da = sc.DataArray(
        sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
        coords={'x': sc.array(dims=['x'], values=[10.0, 20.0, 30.0])},
    )
    return dict.fromkeys(_keys(n), da)


def _data_2d(n: int) -> dict:
    da = sc.DataArray(
        sc.array(dims=['y', 'x'], values=np.arange(12.0).reshape(3, 4)),
        coords={
            'x': sc.arange('x', 4, dtype='float64'),
            'y': sc.arange('y', 3, dtype='float64'),
        },
    )
    return dict.fromkeys(_keys(n), da)


_ASPECTS = [
    PlotAspect(aspect_type=PlotAspectType.free),
    PlotAspect(aspect_type=PlotAspectType.square, stretch_mode=StretchMode.width),
    PlotAspect(aspect_type=PlotAspectType.square, stretch_mode=StretchMode.height),
    PlotAspect(aspect_type=PlotAspectType.equal, stretch_mode=StretchMode.width),
    PlotAspect(aspect_type=PlotAspectType.aspect, ratio=2.0),
]


@pytest.mark.parametrize('combine', [CombineMode.overlay, CombineMode.layout])
@pytest.mark.parametrize('n_sources', [1, 2])
@pytest.mark.parametrize('aspect', _ASPECTS, ids=lambda a: a.aspect_type.name)
class TestFigureSizingInvariant:
    """Every figure a plotter renders adopts the aspect's responsive mode.

    Covers both overlay (one shared figure) and layout (one figure per source),
    so a regression that skips the aspect hook for any figure -- as the
    collapsed-image bug did for Layout sub-figures -- fails here.
    """

    def test_line_plotter(self, aspect, n_sources, combine):
        params = PlotParams1d(
            layout=LayoutParams(combine_mode=combine), plot_aspect=aspect
        )
        figures = _present_figures(
            plots.LinePlotter.from_params(params), _data_1d(n_sources)
        )
        expected_n = n_sources if combine == CombineMode.layout else 1
        assert len(figures) == expected_n
        for fig in figures:
            assert fig.sizing_mode == _get_sizing_mode(params)

    def test_image_plotter(self, aspect, n_sources, combine):
        params = PlotParams2d(
            layout=LayoutParams(combine_mode=combine), plot_aspect=aspect
        )
        figures = _present_figures(
            plots.ImagePlotter.from_params(params), _data_2d(n_sources)
        )
        expected_n = n_sources if combine == CombineMode.layout else 1
        assert len(figures) == expected_n
        for fig in figures:
            assert fig.sizing_mode == _get_sizing_mode(params)
