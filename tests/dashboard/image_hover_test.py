# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Hover of ImagePlotter: index axes show the pixel index, not the cursor."""

from __future__ import annotations

import holoviews as hv
import numpy as np
import scipp as sc
from bokeh.models import HoverTool
from holoviews.plotting.bokeh import BokehRenderer

from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.dashboard import plots
from ess.livedata.dashboard.plot_params import (
    PlotAspect,
    PlotAspectType,
    PlotParams2d,
)

hv.extension('bokeh')

DATA_KEY = DataKey(
    workflow_id=WorkflowId(instrument='inst', namespace='ns', name='wf', version=1),
    source_name='src',
    output_name='out',
)


def _image(**coords: sc.Variable) -> sc.DataArray:
    return sc.DataArray(
        sc.array(dims=['y', 'x'], values=np.arange(12.0).reshape(3, 4), unit='counts'),
        coords=coords,
    )


def _render(plotter: plots.ImagePlotter, data: sc.DataArray):
    plotter.compute({'primary': {DATA_KEY: data}})
    pipe = hv.streams.Pipe(data=plotter.get_cached_state())
    dmap = plotter.create_presenter().present(pipe)
    return BokehRenderer.instance().get_plot(dmap)


def _hover(plot) -> HoverTool:
    [hover] = [t for t in plot.state.toolbar.tools if isinstance(t, HoverTool)]
    return hover


def test_axes_without_coord_round_cursor_to_pixel_index() -> None:
    plot = _render(plots.ImagePlotter.from_params(PlotParams2d()), _image())
    hover = _hover(plot)
    assert set(hover.formatters) == {'$x', '$y'}
    assert dict(hover.tooltips)['x'] == '$x{x}'
    assert hover.formatters['$x'].args['values_by_dim'] == [[]]


def test_integer_coord_axis_snaps_to_nearest_coord_value() -> None:
    data = _image(x=sc.array(dims=['x'], values=[10, 20, 30, 40], unit=None))
    hover = _hover(_render(plots.ImagePlotter.from_params(PlotParams2d()), data))
    assert hover.formatters['$x'].args['values_by_dim'] == [[10.0, 20.0, 30.0, 40.0]]


def test_float_coord_axis_keeps_cursor_position() -> None:
    data = _image(x=sc.linspace('x', 0.5, 3.5, 4, unit='m'))
    hover = _hover(_render(plots.ImagePlotter.from_params(PlotParams2d()), data))
    assert set(hover.formatters) == {'$y'}
    assert dict(hover.tooltips)['x (m)'] == '$x'


def test_physical_coords_keep_default_hover() -> None:
    data = _image(
        x=sc.linspace('x', 0.5, 3.5, 4, unit='m'),
        y=sc.linspace('y', 0.0, 2.0, 3, unit='m'),
    )
    hover = _hover(_render(plots.ImagePlotter.from_params(PlotParams2d()), data))
    assert not hover.formatters


def test_bin_edges_are_not_index_axes() -> None:
    data = _image(x=sc.arange('x', 5, unit=None))
    hover = _hover(_render(plots.ImagePlotter.from_params(PlotParams2d()), data))
    assert set(hover.formatters) == {'$y'}


def test_fixed_aspect_keeps_hover_and_sizing_hooks() -> None:
    params = PlotParams2d()
    params.plot_aspect = PlotAspect(aspect_type=PlotAspectType.square)
    plot = _render(plots.ImagePlotter.from_params(params), _image())
    assert 'change:inner_width' in plot.state.js_property_callbacks
    assert set(_hover(plot).formatters) == {'$x', '$y'}


def test_non_uniform_integer_coord_keeps_default_quadmesh_hover() -> None:
    data = _image(x=sc.array(dims=['x'], values=[0, 1, 2, 5], unit=None))
    plot = _render(plots.ImagePlotter.from_params(PlotParams2d()), data)
    assert not _hover(plot).formatters
    assert '@image' not in [template for _, template in _hover(plot).tooltips]


def test_overlaid_images_each_get_their_own_hover() -> None:
    plotter = plots.ImagePlotter.from_params(PlotParams2d())
    first = _image(x=sc.arange('x', 0, 4, unit=None))
    second = _image(x=sc.arange('x', 10, 14, unit=None))
    overlay = plotter.plot(first, DATA_KEY) * plotter.plot(second, DATA_KEY)
    fig = BokehRenderer.instance().get_plot(overlay).state
    hovers = [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]
    assert len(hovers) == 2
    assert all(len(h.renderers) == 1 for h in hovers)
    assert {tuple(h.formatters['$x'].args['values_by_dim'][0]) for h in hovers} == {
        (0.0, 1.0, 2.0, 3.0),
        (10.0, 11.0, 12.0, 13.0),
    }
