# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ess.livedata.dashboard.widgets.cell import (
    format_freshness_html,
    format_layer_time_html,
)
from ess.livedata.dashboard.widgets.styles import FreshnessPill


class TestFreshnessPill:
    def test_none_renders_nothing(self) -> None:
        assert format_freshness_html(None) == ''

    def test_fresh_band_colors_and_label(self) -> None:
        html = format_freshness_html(2.3)
        assert FreshnessPill.FRESH[0] in html  # background
        assert '2.3s' in html
        assert 'border-radius' in html

    def test_stale_band(self) -> None:
        html = format_freshness_html(12.0)
        assert FreshnessPill.STALE[0] in html
        assert '12s' in html

    def test_old_band(self) -> None:
        html = format_freshness_html(41.0)
        assert FreshnessPill.OLD[0] in html
        assert '41s' in html

    def test_minutes_for_large_age(self) -> None:
        assert '3m' in format_freshness_html(200.0)

    def test_no_hover_tooltip_on_pill(self) -> None:
        # A native title tooltip cannot survive the pill's per-frame re-renders
        # as the age ticks; detail lives in the toolbar row instead.
        assert 'title=' not in format_freshness_html(2.0)


class TestLayerTimeHtml:
    def test_empty_renders_nothing(self) -> None:
        assert format_layer_time_html('') == ''

    def test_wraps_text_in_muted_span(self) -> None:
        html = format_layer_time_html('14:35:01 - 14:35:07 (Lag: 2.3s)')
        assert '14:35:01 - 14:35:07 (Lag: 2.3s)' in html
        assert '<span' in html
