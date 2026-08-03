# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Render-geometry invariant: a plot's figures must be sized consistently.

The "collapsed detector image" bug (figures left ``stretch_both`` while their
pane asked for ``stretch_width``, so the free axis was unconstrained and the
plot rendered at zero height) passed every functional test: the data and the
figure existed, only the *rendered geometry* was wrong.

Observing the collapse itself needs a real browser (see
``render_geometry_test.py``), but its cause is observable headlessly: every
figure a plotter produces must adopt the responsive sizing mode the cell's pane
will wrap it in, i.e. the one :func:`pane_sizing_mode` returns. These tests
render each plotter through the real ``compute -> present -> bokeh`` path and
assert that invariant across aspect types, combine modes and source counts --
including the layout-mode sub-figures that the original bug skipped.
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pytest
import scipp as sc
from bokeh.models import Plot
from holoviews.plotting.bokeh import BokehRenderer

from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.dashboard import plots
from ess.livedata.dashboard.frame_aspect import pane_sizing_mode
from ess.livedata.dashboard.plot_params import (
    CombineMode,
    LayoutParams,
    PlotAspect,
    PlotAspectType,
    PlotParams1d,
    PlotParams2d,
    PlotParams3d,
    StretchMode,
)
from ess.livedata.dashboard.slicer_plotter import SlicerPlotter

hv.extension('bokeh')


def _figures(hv_obj) -> list[Plot]:
    """All Bokeh figures in the rendered HoloViews object (Layout -> one per cell)."""
    state = BokehRenderer.instance().get_plot(hv_obj).state
    return [m for m in state.references() if isinstance(m, Plot)]


def _present_figures(plotter, data: dict) -> list[Plot]:
    plotter.compute({'primary': data})
    state = plotter.get_cached_state()
    # compute() turns any failure into a Text placeholder, which would render a
    # single default-sized figure and make the assertions below pass vacuously.
    assert not isinstance(state, hv.Text), f'plotter did not plot the data: {state}'
    presenter = plotter.create_presenter()
    return _figures(presenter.present(hv.streams.Pipe(data=state)))


def _keys(n: int) -> list[DataKey]:
    wf = WorkflowId(instrument='test', name='wf', version=1)
    return [
        DataKey(workflow_id=wf, source_name=f's{i}', output_name='out')
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


def _data_3d() -> dict:
    da = sc.DataArray(
        sc.array(dims=['z', 'y', 'x'], values=np.arange(24.0).reshape(2, 3, 4)),
        coords={
            'x': sc.arange('x', 4, dtype='float64'),
            'y': sc.arange('y', 3, dtype='float64'),
            'z': sc.arange('z', 2, dtype='float64'),
        },
    )
    return dict.fromkeys(_keys(1), da)


# Every aspect type, and both stretch modes for the constrained ones, since the
# stretch mode decides which axis the pane and the figure must agree on.
_ASPECTS = [
    PlotAspect(aspect_type=PlotAspectType.free),
    PlotAspect(aspect_type=PlotAspectType.square, stretch_mode=StretchMode.width),
    PlotAspect(aspect_type=PlotAspectType.square, stretch_mode=StretchMode.height),
    PlotAspect(aspect_type=PlotAspectType.equal, stretch_mode=StretchMode.width),
    PlotAspect(aspect_type=PlotAspectType.equal, stretch_mode=StretchMode.height),
    PlotAspect(
        aspect_type=PlotAspectType.aspect, ratio=2.0, stretch_mode=StretchMode.width
    ),
    PlotAspect(
        aspect_type=PlotAspectType.data_aspect,
        ratio=0.5,
        stretch_mode=StretchMode.height,
    ),
]


@pytest.mark.parametrize(
    ('plotter_cls', 'params_cls', 'make_data'),
    [
        (plots.LinePlotter, PlotParams1d, _data_1d),
        (plots.ImagePlotter, PlotParams2d, _data_2d),
    ],
    ids=['line', 'image'],
)
@pytest.mark.parametrize('combine', [CombineMode.overlay, CombineMode.layout])
@pytest.mark.parametrize('n_sources', [1, 2])
@pytest.mark.parametrize(
    'aspect', _ASPECTS, ids=lambda a: f'{a.aspect_type.name}-{a.stretch_mode.name}'
)
def test_every_figure_adopts_the_panes_sizing_mode(
    plotter_cls, params_cls, make_data, combine, n_sources, aspect
):
    """A regression skipping the aspect hook for any figure fails here.

    Layout mode yields one figure per source; the collapsed-image bug sized only
    the cell's pane, leaving those sub-figures ``stretch_both``.
    """
    params = params_cls(layout=LayoutParams(combine_mode=combine), plot_aspect=aspect)
    figures = _present_figures(plotter_cls.from_params(params), make_data(n_sources))

    assert len(figures) == (n_sources if combine == CombineMode.layout else 1)
    for fig in figures:
        assert fig.sizing_mode == pane_sizing_mode(aspect)


@pytest.mark.parametrize(
    'aspect', _ASPECTS, ids=lambda a: f'{a.aspect_type.name}-{a.stretch_mode.name}'
)
def test_slicer_figure_adopts_the_panes_sizing_mode(aspect):
    """The slicer styles each rendered slice instead of its DynamicMap.

    It has to: element options on a kdim-carrying DynamicMap wrap it in an
    operation that the grid cell's own ``.opts()`` call then cannot see through.
    That makes the sizing easy to drop on the render path, so pin it here. The
    slicer takes a single source and produces a single figure, hence no
    combine-mode or source-count axis.
    """
    figures = _present_figures(
        SlicerPlotter.from_params(PlotParams3d(plot_aspect=aspect)), _data_3d()
    )

    assert len(figures) == 1
    assert figures[0].sizing_mode == pane_sizing_mode(aspect)
