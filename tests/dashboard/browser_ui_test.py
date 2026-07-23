# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Browser-driven regression tests for dashboard UI flows.

Automates the recurring manual verification items that need no Kafka backend,
driving the fake-backend dashboard (seeded from the committed dummy fixture)
through the stable ``lt-*`` automation hooks:

- session reload restores tabs and live updates;
- a grid created in one session appears in others without stealing focus;
- a cell title survives a no-op Save of the cell-properties modal;
- disabling or removing a grid keeps the remaining tabs resolving and
  updating.

Each test launches its own dashboard on a dedicated port for isolation, since
grid topology changes are process-global. Runs via ``pytest -m browser``
(excluded from the default run; CI runs them via ``tox -e browser``; skips
cleanly where Playwright is absent).
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
from tests.helpers.browser import (
    Dashboard,
    assert_updating,
    fake_dashboard,
    fingerprint,
    wait_until,
)

_CELL_TITLE_INPUT = 'input[placeholder="Leave empty to use the derived title"]'
_GRID_TITLE_INPUT = 'input[placeholder="Enter grid title"]'


def _active_tab(dash: Dashboard) -> str:
    return dash.page.locator(".bk-tab.bk-active").first.inner_text().strip()


def _add_grid(dash: Dashboard, title: str) -> None:
    """Add an empty grid via the Manage Plots form (must be the active tab).

    Panel's TextInput syncs ``value`` on Enter/blur, so press Enter before
    clicking the button. Adding focuses the new grid's tab in this session.
    """
    dash.page.locator(_GRID_TITLE_INPUT).fill(title)
    dash.page.keyboard.press("Enter")
    dash.page.get_by_role("button", name="Add Grid", exact=True).click()
    wait_until(
        dash, lambda: title in dash.tab_names(), label=f"tab {title!r} to appear"
    )


@pytest.mark.browser
def test_session_reload_restores_tabs_and_live_updates():
    with fake_dashboard("dummy", 5033) as url, Dashboard.connect(url) as dash:
        dash.goto_tab("Detectors")
        assert_updating(dash, "session before reload")

        dash.page.reload(wait_until="networkidle")
        dash.page.wait_for_timeout(5000)

        assert "Detectors" in dash.tab_names()
        dash.goto_tab("Detectors")
        fp = fingerprint(dash)
        assert fp["sources"] > 0, "no data sources rendered after reload"
        assert_updating(dash, "session after reload")


@pytest.mark.browser
def test_grid_created_in_one_session_appears_in_other_without_stealing_focus():
    with (
        fake_dashboard("dummy", 5034) as url,
        Dashboard.connect_many(2, url) as (creator, observer),
    ):
        observer_tab = _active_tab(observer)
        creator.goto_tab("Manage Plots")

        _add_grid(creator, "Created Elsewhere")

        # The other session gains the tab via its topology poll...
        wait_until(
            observer,
            lambda: "Created Elsewhere" in observer.tab_names(),
            label="new grid tab in the other session",
        )
        # ...but its active tab must not change: tab focus is local intent,
        # not shared state, so only the creating session focuses the new tab.
        assert _active_tab(observer) == observer_tab
        wait_until(
            creator,
            lambda: _active_tab(creator) == "Created Elsewhere",
            label="creating session to focus its new tab",
        )


@pytest.mark.browser
def test_cell_title_survives_noop_save_of_cell_properties_modal():
    # The per-cell hook (not DOM order) addresses the cell: a rebuilt cell --
    # e.g. after the rename below -- moves to the end of the document.
    pencil = ".lt-cell-r0c0.lt-tool-pencil"
    with fake_dashboard("dummy", 5035) as url, Dashboard.connect(url) as dash:
        page = dash.page
        dash.goto_tab("Detectors")

        # Give the first cell a user-defined title.
        dash.open_modal(pencil)
        page.locator(_CELL_TITLE_INPUT).fill("My Cell")
        page.get_by_role("button", name="Save", exact=True).click()
        wait_until(
            dash,
            page.get_by_text("My Cell", exact=True).first.is_visible,
            label="renamed cell titlebar",
        )

        # Reopen and Save without typing: an untouched Save must not clear
        # the user title (the field pre-fill must round-trip through Save).
        dash.open_modal(pencil)
        assert page.locator(_CELL_TITLE_INPUT).input_value() == "My Cell"
        page.get_by_role("button", name="Save", exact=True).click()
        page.locator("[role=dialog]").first.wait_for(state="hidden", timeout=10000)

        # The titlebar keeps the title, and the persisted state still carries
        # it: reopening the modal pre-fills the unchanged user title.
        assert page.get_by_text("My Cell", exact=True).first.is_visible()
        dash.open_modal(pencil)
        assert page.locator(_CELL_TITLE_INPUT).input_value() == "My Cell"


@pytest.mark.browser
def test_remaining_tabs_keep_updating_after_disabling_and_removing_grids():
    with fake_dashboard("dummy", 5036) as url, Dashboard.connect(url) as dash:
        # Arrange three grids ordered [Bravo, Charlie, Detectors]: two empty
        # grids ahead of the fixture's populated one, so disabling the first
        # and removing the middle both shift the Detectors tab position --
        # the regression class where tab indices fall out of alignment with
        # the grid list once a preceding grid is hidden or gone.
        dash.goto_tab("Manage Plots")
        _add_grid(dash, "Bravo")
        dash.goto_tab("Manage Plots")
        _add_grid(dash, "Charlie")
        dash.goto_tab("Manage Plots")
        for expected in (
            ["Bravo", "Detectors", "Charlie"],
            ["Bravo", "Charlie", "Detectors"],
        ):
            dash.click(".lt-grid-detectors.lt-tool-chevron-down")
            wait_until(
                dash,
                lambda expected=expected: dash.tab_names()[-3:] == expected,
                label=f"grid tab order {expected}",
            )

        # Disable the first grid: its tab vanishes, the rest keep working.
        dash.click(".lt-grid-bravo.lt-tool-eye")
        wait_until(
            dash,
            lambda: "Bravo" not in dash.tab_names(),
            label="disabled grid tab to vanish",
        )
        assert "Charlie" in dash.tab_names()
        dash.goto_tab("Detectors")
        assert_updating(dash, "Detectors tab after disabling first grid")

        # Remove the middle grid of [Bravo (disabled), Charlie, Detectors].
        dash.goto_tab("Manage Plots")
        dash.click(".lt-grid-charlie.lt-tool-x")
        wait_until(
            dash,
            lambda: "Charlie" not in dash.tab_names(),
            label="removed grid tab to vanish",
        )
        dash.goto_tab("Detectors")
        assert_updating(dash, "Detectors tab after removing middle grid")
