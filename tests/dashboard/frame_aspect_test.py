# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for frame-aspect sizing opts and their CustomJS hook."""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pytest
from holoviews.streams import Pipe

from ess.livedata.dashboard.frame_aspect import make_frame_aspect_opts
from ess.livedata.dashboard.plot_params import (
    PlotAspect,
    PlotAspectType,
    StretchMode,
)

hv.extension('bokeh')


def _image(data: np.ndarray) -> hv.Image:
    return hv.Image((np.arange(data.shape[1]), np.arange(data.shape[0]), data))


def _n_js_callbacks(model) -> int:
    return sum(len(cbs) for cbs in model.js_property_callbacks.values())


@pytest.fixture
def rendered_dmap():
    """Render a data-aspect DynamicMap and return (figure, pipe) for updates."""
    opts = make_frame_aspect_opts(
        PlotAspect(
            aspect_type=PlotAspectType.data_aspect, stretch_mode=StretchMode.width
        )
    )
    pipe = Pipe(data=np.zeros((4, 8)))
    dmap = hv.DynamicMap(lambda data: _image(data).opts(**opts), streams=[pipe])
    plot = hv.renderer('bokeh').get_plot(dmap)
    return plot.state, pipe


class TestMakeFrameAspectOpts:
    def test_free_aspect_yields_plain_responsive(self) -> None:
        opts = make_frame_aspect_opts(PlotAspect(aspect_type=PlotAspectType.free))
        assert opts == {'responsive': True}

    @pytest.mark.parametrize(
        'aspect_type',
        [
            PlotAspectType.square,
            PlotAspectType.aspect,
            PlotAspectType.equal,
            PlotAspectType.data_aspect,
        ],
    )
    def test_fill_width_fixes_height_so_holoviews_stretches_width(
        self, aspect_type: PlotAspectType
    ) -> None:
        opts = make_frame_aspect_opts(
            PlotAspect(aspect_type=aspect_type, stretch_mode=StretchMode.width)
        )
        fig = hv.render(_image(np.zeros((4, 8))).opts(**opts))
        assert fig.sizing_mode == 'stretch_width'
        assert fig.height == opts['height']
        assert fig.width is None

    def test_fill_height_fixes_width_so_holoviews_stretches_height(self) -> None:
        opts = make_frame_aspect_opts(
            PlotAspect(
                aspect_type=PlotAspectType.square, stretch_mode=StretchMode.height
            )
        )
        fig = hv.render(_image(np.zeros((4, 8))).opts(**opts))
        assert fig.sizing_mode == 'stretch_height'
        assert fig.width == opts['width']
        assert fig.height is None


class TestHookAcrossUpdates:
    def test_callback_set_attached_exactly_once(self, rendered_dmap) -> None:
        fig, pipe = rendered_dmap
        # One callback on each of inner_width/inner_height and range start/end.
        assert _n_js_callbacks(fig) == 2
        assert _n_js_callbacks(fig.x_range) == 2
        assert _n_js_callbacks(fig.y_range) == 2

        for _ in range(3):
            pipe.send(np.ones((4, 8)))

        assert _n_js_callbacks(fig) == 2
        assert _n_js_callbacks(fig.x_range) == 2
        assert _n_js_callbacks(fig.y_range) == 2

    def test_updates_leave_figure_geometry_untouched(self, rendered_dmap) -> None:
        fig, pipe = rendered_dmap
        # Simulate the browser-side CustomJS having adjusted the height.
        fig.height = 513

        for _ in range(3):
            pipe.send(np.ones((4, 8)))

        assert fig.sizing_mode == 'stretch_width'
        assert fig.height == 513
