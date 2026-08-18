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
)

hv.extension('bokeh')

_CONSTRAINED = [
    PlotAspectType.square,
    PlotAspectType.aspect,
    PlotAspectType.equal,
    PlotAspectType.data_aspect,
]


def _image(data: np.ndarray) -> hv.Image:
    return hv.Image((np.arange(data.shape[1]), np.arange(data.shape[0]), data))


def _n_js_callbacks(model) -> int:
    return sum(len(cbs) for cbs in model.js_property_callbacks.values())


@pytest.fixture
def rendered_dmap():
    """Render a data-aspect DynamicMap and return (figure, pipe) for updates."""
    opts = make_frame_aspect_opts(PlotAspect(aspect_type=PlotAspectType.data_aspect))
    pipe = Pipe(data=np.zeros((4, 8)))
    dmap = hv.DynamicMap(lambda data: _image(data).opts(**opts), streams=[pipe])
    plot = hv.renderer('bokeh').get_plot(dmap)
    return plot.state, pipe


class TestMakeFrameAspectOpts:
    def test_free_aspect_yields_plain_responsive(self) -> None:
        opts = make_frame_aspect_opts(PlotAspect(aspect_type=PlotAspectType.free))
        assert opts == {'responsive': True}

    @pytest.mark.parametrize('aspect_type', _CONSTRAINED)
    def test_constrained_aspect_adds_a_hook(self, aspect_type: PlotAspectType) -> None:
        opts = make_frame_aspect_opts(PlotAspect(aspect_type=aspect_type))
        assert opts['responsive'] is True
        assert len(opts['hooks']) == 1

    @pytest.mark.parametrize(
        'aspect_type', [PlotAspectType.free, *_CONSTRAINED], ids=lambda t: t.name
    )
    def test_figure_fills_its_container_on_both_axes(
        self, aspect_type: PlotAspectType
    ) -> None:
        """The frame carries the aspect, so the figure itself never leaves the cell.

        A figure sized along one axis only would overflow its grid cell along
        the other and paint over the neighbouring cell (#931).
        """
        opts = make_frame_aspect_opts(PlotAspect(aspect_type=aspect_type))
        fig = hv.render(_image(np.zeros((4, 8))).opts(**opts))
        assert fig.sizing_mode == 'stretch_both'
        assert fig.width is None
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

    def test_reapplied_opts_reset_the_letterbox_padding(self, rendered_dmap) -> None:
        """Re-applying element opts rewrites ``min_border_*``, wiping the letterbox.

        This fixture re-applies the opts per frame, which the dashboard's own
        update path does not do (data arrives through pipes, leaving figure
        properties alone). It pins the hazard the callback is written against:
        anything that re-applies opts wipes the padding, so the rule has to be
        idempotent and has to re-derive its input from the current layout
        rather than remember what it applied.
        """
        fig, pipe = rendered_dmap
        # Simulate the browser-side CustomJS having letterboxed the frame.
        fig.min_border_right = 120

        pipe.send(np.ones((4, 8)))

        assert fig.sizing_mode == 'stretch_both'
        assert fig.min_border_right != 120
