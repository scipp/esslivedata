# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Rendered-geometry smoke test: no plot collapses to zero size in a real browser.

The "collapsed detector image" bug (#1029) left figures with ``inner_height ==
0`` while every functional and headless test stayed green: the data and the
figure models were fine, only the browser-computed layout was broken. Sizing is
settled between Bokeh, HoloViews and Panel in the browser, so the only faithful
guard renders the dashboard in an actual browser and measures the laid-out
figures.

This visits every plot-grid tab of the Kafka-free fake dashboard and asserts
that each figure's data area (the *frame*, not the outer canvas -- a collapsed
plot still draws its axes) is non-degenerate. It is fix-agnostic: it fails
whenever a plot renders collapsed, whatever the cause.
``plot_sizing_invariant_test.py`` is the cheap headless counterpart, pinning the
known cause across the full aspect matrix.

Runs via ``pytest -m browser`` (excluded from the default run; CI runs them
via ``tox -e browser``; skips cleanly where Playwright is absent).
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
from tests.helpers.browser import Dashboard, fake_dashboard, wait_until

# Below this many CSS pixels a figure's data area is considered collapsed. Real
# frames here are hundreds of pixels; the bug produced zero.
_MIN_FRAME_PX = 50

# The last static tab; grid tabs are appended after the static ones.
_LAST_STATIC_TAB = "Manage Plots"

# Geometry of every figure currently laid out on screen. Figures of tabs visited
# earlier stay in the document with their last dimensions, so those without a
# displayed view are excluded -- otherwise a stale figure would be reported
# against whatever tab is active now.
_VISIBLE_FIGURE_GEOMETRY_JS = """() => {
  const out = [];
  for (const doc of (window.Bokeh && Bokeh.documents) || []) {
    for (const m of Array.from(doc._all_models.values())) {
      if (m.type !== 'Figure') continue;
      const view = Bokeh.index.find_one_by_id(m.id);
      if (!view || !view.el || !view.el.offsetParent) continue;
      out.push({iw: m.inner_width, ih: m.inner_height,
                ow: m.outer_width, oh: m.outer_height});
    }
  }
  return out;
}"""


def _visible_figures(dash: Dashboard) -> list[dict]:
    return dash.page.evaluate(_VISIBLE_FIGURE_GEOMETRY_JS)


@pytest.mark.browser
def test_no_plot_renders_collapsed():
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
        tabs = dash.tab_names()
        grid_tabs = tabs[tabs.index(_LAST_STATIC_TAB) + 1 :]
        assert grid_tabs, f"fixture exposes no plot-grid tab, only {tabs}"

        collapsed: dict[str, list] = {}
        for tab in grid_tabs:
            dash.goto_tab(tab)
            wait_until(
                dash,
                lambda: bool(_visible_figures(dash)),
                label=f"figures to render on tab {tab!r}",
            )
            bad = [
                fig
                for fig in _visible_figures(dash)
                if fig["iw"] < _MIN_FRAME_PX or fig["ih"] < _MIN_FRAME_PX
            ]
            if bad:
                collapsed[tab] = bad

        assert not collapsed, (
            f"figures with a data area below {_MIN_FRAME_PX}px: {collapsed}"
        )
