# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Range-ownership invariant: every figure ends up with bounds around its data.

``Plotter.applies_ranges`` lets a plotter hand HoloViews ``apply_ranges=False``,
which stops it computing bounds from the data on every frame -- a third of a 1D
layer's repaint. That is only sound where something else sets the bounds: the
cell's :class:`CellAutoscaleController`, or the layer being an annotation drawn
in another layer's coordinate space.

The failure mode is silent and does not look like a crash: the figure renders on
Bokeh's default unit range, so a detector image becomes a flat rectangle between
0 and 1. It is invisible to tests that only check that elements and figures
exist. These tests therefore render each plotter through the real
``compute -> hook -> bokeh`` path and assert the resulting axis ranges actually
span the data -- which is what a viewer sees.
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
from ess.livedata.dashboard.cell_autoscale import build_controller_from_layers
from ess.livedata.dashboard.plot_params import (
    CombineMode,
    LayoutParams,
    PlotParams1d,
    PlotParams2d,
    PlotParams3d,
)
from ess.livedata.dashboard.slicer_plotter import SlicerPlotter

hv.extension('bokeh')

X_COORDS = [10.0, 20.0, 30.0]
Y_VALUES = [1.0, 2.0, 3.0]


def _keys(n: int) -> list[DataKey]:
    wf = WorkflowId(instrument='test', name='wf', version=1)
    return [
        DataKey(workflow_id=wf, source_name=f's{i}', output_name='out')
        for i in range(n)
    ]


def _data_1d(n: int) -> dict:
    da = sc.DataArray(
        sc.array(dims=['x'], values=Y_VALUES),
        coords={'x': sc.array(dims=['x'], values=X_COORDS)},
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


def _rendered_figures(plotter, data: dict) -> list[Plot]:
    """Render through the cell's path, including its autoscale hook.

    A non-overlayable layer gets no cell-level hooks -- a Layout's sub-figures
    and a Table have no single figure for them to act on, so ``_compose_plot``
    returns early (``widgets/cell.py``). Mirroring that here is the point: it is
    exactly the case where nothing would write the bounds.
    """
    plotter.compute({'primary': data})
    state = plotter.get_cached_state()
    assert not isinstance(state, hv.Text), f'plotter did not plot the data: {state}'
    obj = plotter.create_presenter().present(hv.streams.Pipe(data=state))
    if plotter.is_overlayable:
        obj = obj.opts(hooks=[build_controller_from_layers([plotter]).make_hook()])
    bokeh_state = BokehRenderer.instance().get_plot(obj).state
    return [m for m in bokeh_state.references() if isinstance(m, Plot)]


def _spans(axis_range, lo: float, hi: float) -> bool:
    start, end = axis_range.start, axis_range.end
    return start is not None and end is not None and start <= lo and end >= hi


@pytest.mark.parametrize('combine', [CombineMode.overlay, CombineMode.layout])
@pytest.mark.parametrize('n_sources', [1, 2])
def test_line_figures_span_their_data(combine, n_sources):
    params = PlotParams1d(layout=LayoutParams(combine_mode=combine))
    figures = _rendered_figures(
        plots.LinePlotter.from_params(params), _data_1d(n_sources)
    )

    assert len(figures) == (n_sources if combine == CombineMode.layout else 1)
    for fig in figures:
        assert _spans(fig.x_range, X_COORDS[0], X_COORDS[-1])
        assert _spans(fig.y_range, Y_VALUES[0], Y_VALUES[-1])


@pytest.mark.parametrize('combine', [CombineMode.overlay, CombineMode.layout])
@pytest.mark.parametrize('n_sources', [1, 2])
def test_image_figures_span_their_data(combine, n_sources):
    """Layout mode is the case that regressed.

    ``RangeHandles`` writes through a single figure's ``x_range`` / ``y_range``,
    which a layout-mode plotter does not have -- it draws one figure per source.
    Opting such a plotter out of HoloViews' range computation leaves nobody to
    set the bounds, and every sub-figure collapses onto the unit range.
    """
    params = PlotParams2d(layout=LayoutParams(combine_mode=combine))
    figures = _rendered_figures(
        plots.ImagePlotter.from_params(params), _data_2d(n_sources)
    )

    assert len(figures) == (n_sources if combine == CombineMode.layout else 1)
    for fig in figures:
        assert _spans(fig.x_range, 0.0, 3.0)
        assert _spans(fig.y_range, 0.0, 2.0)


def test_slicer_figure_spans_its_data():
    """The slicer autoscales only the color axis, so HoloViews keeps x and y."""
    figures = _rendered_figures(SlicerPlotter.from_params(PlotParams3d()), _data_3d())

    assert len(figures) == 1
    assert _spans(figures[0].x_range, 0.0, 3.0)
    assert _spans(figures[0].y_range, 0.0, 2.0)
