# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import panel as pn
import pytest
from bokeh.models import ColorBar, InlineStyleSheet, Legend, LinearAxis

from ess.livedata.dashboard.design import LivedataDesign


@pytest.mark.parametrize(
    'pane', [pn.pane.HTML('<b>x</b>'), pn.pane.Markdown('x'), pn.pane.Str('x')]
)
def test_design_keeps_markup_panes_visible(pane: pn.pane.PaneBase) -> None:
    """Markup panes must not depend on Panel's stylesheet-load reveal.

    Panel hides a markup pane until every stylesheet link in it fires ``load``,
    which never happens for panes built after page load, leaving them invisible
    for the rest of the session (see ``dashboard.design``).
    """
    model = pane.get_root()
    LivedataDesign().apply(pane, model)

    css = [
        sheet.css if isinstance(sheet, InlineStyleSheet) else sheet
        for sheet in model.stylesheets
        if isinstance(sheet, InlineStyleSheet | str)
    ]
    assert any('visibility: visible !important' in rule for rule in css)


def test_design_theme_tightens_figure_spacing() -> None:
    """The design's Bokeh theme overrides Material's spacing inside figures."""
    theme = LivedataDesign(theme='default').theme.bokeh_theme
    axis, colorbar, legend = LinearAxis(), ColorBar(), Legend()
    for model in (axis, colorbar, legend):
        theme.apply_to_model(model)

    assert axis.axis_label_standoff < 10
    assert colorbar.padding < 10
    assert legend.spacing < 8
    # Material's own settings are kept.
    assert axis.axis_label_text_font_size == '1.25em'
