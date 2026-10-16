# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import asyncio

import holoviews as hv
import numpy as np
from bokeh.document import Document

# Imported for its side effect: the compositor registry the dashboard runs with,
# without which this overlay would update through the map and mask the bug.
import ess.livedata.dashboard.reduction  # noqa: F401
from ess.livedata.dashboard.widgets.cell import (
    CurrentStateHoloViews,
    format_error_html,
    format_freshness_html,
    format_layer_time_html,
    format_stopped_html,
)
from ess.livedata.dashboard.widgets.styles import StatusPill

hv.extension('bokeh')


class TestFreshnessPill:
    def test_none_renders_nothing(self) -> None:
        assert format_freshness_html(None) == ''

    def test_fresh_band_colors_and_label(self) -> None:
        html = format_freshness_html(2.3)
        assert StatusPill.FRESH[0] in html  # background
        assert '2.3s' in html
        assert 'border-radius' in html

    def test_stale_band(self) -> None:
        html = format_freshness_html(12.0)
        assert StatusPill.STALE[0] in html
        assert '12s' in html

    def test_old_band(self) -> None:
        html = format_freshness_html(41.0)
        assert StatusPill.OLD[0] in html
        assert '41s' in html

    def test_minutes_for_large_age(self) -> None:
        assert '3m' in format_freshness_html(200.0)

    def test_no_hover_tooltip_on_pill(self) -> None:
        # A native title tooltip cannot survive the pill's per-frame re-renders
        # as the age ticks; detail lives in the toolbar row instead.
        assert 'title=' not in format_freshness_html(2.0)


class TestStatusPills:
    def test_stopped_pill_is_neutral_gray_with_label(self) -> None:
        html = format_stopped_html()
        assert StatusPill.STOPPED[0] in html
        assert 'stopped' in html

    def test_stopped_pill_has_square_stop_glyph(self) -> None:
        # The dot renders as a square -- the conventional "stop" glyph.
        assert 'border-radius:50%' not in format_stopped_html()

    def test_error_pill_is_red_with_label(self) -> None:
        html = format_error_html()
        assert StatusPill.ERROR[0] in html
        assert 'error' in html


class TestLayerTimeHtml:
    def test_empty_renders_nothing(self) -> None:
        assert format_layer_time_html('') == ''

    def test_wraps_text_in_muted_span(self) -> None:
        html = format_layer_time_html('14:35:01 - 14:35:07 (Lag: 2.3s)')
        assert '14:35:01 - 14:35:07 (Lag: 2.3s)' in html
        assert '<span' in html


class TestCurrentStateHoloViews:
    """A second render of a plot must be built from its current state."""

    def test_second_render_seeds_the_edit_tool_from_current_state(self):
        rois = hv.streams.Pipe(data=[])
        roi_map = hv.DynamicMap(hv.Rectangles, streams=[rois])
        box_edit = hv.streams.BoxEdit(source=roi_map)
        image = hv.streams.Pipe(data=np.zeros((4, 4)))
        plot = (
            hv.Overlay([hv.DynamicMap(hv.Image, streams=[image]), roi_map])
            .collate()
            .opts(clone=True)
        )

        async def render_twice() -> None:
            CurrentStateHoloViews(plot).get_root(Document())
            rois.send([(1.0, 2.0, 5.0, 6.0)])
            CurrentStateHoloViews(plot).get_root(Document())

        # The figures' data-source callbacks schedule tasks on the running loop.
        asyncio.run(render_twice())

        assert list(box_edit.data['x0']) == [1.0]
        assert list(box_edit.data['y1']) == [6.0]
