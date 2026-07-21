# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest
from bokeh.models.tools import BoxEditTool, HoverTool, PanTool, PolyDrawTool
from bokeh.plotting import figure

from ess.livedata.dashboard.hover_suspend import make_hover_suspend_hook


class FakePlot:
    """Stand-in for the HoloViews plot object, which only exposes ``state``."""

    def __init__(self, state):
        self.state = state


def make_figure(*tools):
    fig = figure(tools=[])
    fig.toolbar.tools = list(tools)
    return fig


def callbacks(fig):
    return fig.js_property_callbacks.get('change:inner_width', [])


@pytest.fixture
def hook():
    return make_hover_suspend_hook()


@pytest.mark.parametrize('edit_tool', [BoxEditTool, PolyDrawTool])
def test_wires_hover_and_edit_tools(hook, edit_tool):
    hover = HoverTool()
    edit = edit_tool()
    fig = make_figure(PanTool(), hover, edit)

    hook(FakePlot(fig), None)

    (callback,) = callbacks(fig)
    assert callback.args['hovers'] == [hover]
    assert callback.args['edits'] == [edit]


def test_no_callback_without_edit_tool(hook):
    fig = make_figure(PanTool(), HoverTool())

    hook(FakePlot(fig), None)

    assert callbacks(fig) == []


def test_no_callback_without_hover_tool(hook):
    fig = make_figure(PanTool(), BoxEditTool())

    hook(FakePlot(fig), None)

    assert callbacks(fig) == []


def test_repeated_renders_do_not_stack_callbacks(hook):
    fig = make_figure(HoverTool(), BoxEditTool())

    hook(FakePlot(fig), None)
    hook(FakePlot(fig), None)

    assert len(callbacks(fig)) == 1


def test_wires_a_freshly_swapped_figure(hook):
    """HoloViews replaces the figure on kdim transitions; the new one needs it."""
    hook(FakePlot(make_figure(HoverTool(), BoxEditTool())), None)
    fig = make_figure(HoverTool(), BoxEditTool())

    hook(FakePlot(fig), None)

    assert len(callbacks(fig)) == 1


def test_hook_sees_tools_of_a_rendered_holoviews_overlay(hook):
    """The hook runs late enough to see tools contributed by all overlay layers.

    The hover tool comes from the image layer and the BoxEdit tool from the ROI
    layer, both merged onto the overlay's single figure before hooks run.
    """
    import holoviews as hv
    import numpy as np

    hv.extension('bokeh')

    image_pipe = hv.streams.Pipe(data=np.random.default_rng(0).random((8, 8)))
    image = hv.DynamicMap(hv.Image, streams=[image_pipe]).opts(tools=['hover'])
    roi_pipe = hv.streams.Pipe(data=[])
    rectangles = hv.DynamicMap(hv.Rectangles, streams=[roi_pipe])
    hv.streams.BoxEdit(source=rectangles, num_objects=2)
    overlay = hv.Overlay([image, rectangles]).collate()

    plot = hv.renderer('bokeh').get_plot(overlay.opts(hooks=[hook]))

    assert len(callbacks(plot.state)) == 1
