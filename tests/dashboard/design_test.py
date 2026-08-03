# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import panel as pn
import pytest
from bokeh.models import InlineStyleSheet

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
