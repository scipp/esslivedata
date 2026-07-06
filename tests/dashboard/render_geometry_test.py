# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Rendered-geometry smoke test: no plot collapses to zero size in a real browser.

The "collapsed detector image" bug left figures with ``inner_height == 0`` while
every functional/headless test stayed green -- the data and figure models were
fine, only the browser-computed layout was broken. The only faithful guard is to
render the dashboard in an actual browser and measure the laid-out figures.

This drives the Kafka-free fake backend (seeded from the committed dummy fixture)
with Playwright, visits every plot-grid tab, and asserts that every Bokeh figure
has a non-degenerate data area. It is fix-agnostic: it fails whenever a plot
renders collapsed, whatever the cause.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

pytest.importorskip("playwright.sync_api")
from drive_dashboard import Dashboard, _fake_dashboard  # noqa: E402

# Below this many CSS pixels a figure's data area is considered collapsed.
_MIN_FRAME_PX = 50

_FIGURE_DIMS_JS = """() => {
  const out = [];
  for (const doc of (window.Bokeh && Bokeh.documents) || []) {
    for (const m of Array.from(doc._all_models.values())) {
      if (m.type === 'Figure') {
        out.push({iw: m.inner_width, ih: m.inner_height,
                  ow: m.outer_width, oh: m.outer_height});
      }
    }
  }
  return out;
}"""


@pytest.fixture(scope="module")
def dashboard():
    """Launch the fake-backend dummy dashboard once and drive it with Playwright."""
    with _fake_dashboard("dummy", 5031) as url, Dashboard.connect(url) as dash:
        yield dash


def _plot_grid_tabs(dash: Dashboard) -> list[str]:
    """Tab titles that hold plot grids (exclude the static control tabs)."""
    static = {"Workflows", "System Status", "Manage Plots"}
    return [t for t in dash.tab_names() if t not in static]


@pytest.mark.browser
def test_no_plot_renders_collapsed(dashboard):
    grid_tabs = _plot_grid_tabs(dashboard)
    assert grid_tabs, "fixture should expose at least one plot-grid tab"

    collapsed: dict[str, list] = {}
    for tab in grid_tabs:
        dashboard.goto_tab(tab)
        figures = dashboard.page.evaluate(_FIGURE_DIMS_JS)
        assert figures, f"tab {tab!r} rendered no figures"
        bad = [f for f in figures if f["iw"] < _MIN_FRAME_PX or f["ih"] < _MIN_FRAME_PX]
        if bad:
            collapsed[tab] = bad

    assert not collapsed, f"figures collapsed below {_MIN_FRAME_PX}px: {collapsed}"
