# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Browser-driven regression tests for dashboard UI flows.

Automates the recurring manual verification items that need no Kafka backend,
driving the fake-backend dashboard (seeded from the committed dummy fixture)
through the stable ``lt-*`` automation hooks:

- clicking out an empty grid region and completing the plot wizard yields a
  live plot;
- session reload restores tabs and live updates;
- a grid created in one session appears in others without stealing focus;
- a cell title survives a no-op Save of the cell-properties modal;
- a cell rebuilt in an open session renders its titlebar rather than an
  invisible one;
- disabling or removing a grid keeps the remaining tabs resolving and
  updating;
- two sessions racing to save edits on the same grid converge on one title,
  without a server-side exception or a duplicated/lost tab;
- uploading a grid config whose cells claim the same slot is rejected;
- a popped-out plot opens within the viewport, floats above other tabs and
  keeps updating there, and stops costing anything once minimized;
- renaming a popped-out cell leaves its window the size and where the user
  put it, and opening one adds no scrollable whitespace to the page.

Each test launches its own dashboard for isolation, since grid topology changes
are process-global; ports are allocated per launch, so concurrent runs of this
suite do not fight over them. Runs via ``pytest -m browser``
(excluded from the default run; CI runs them via ``tox -e browser``; skips
cleanly where Playwright is absent).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml

pytest.importorskip("playwright.sync_api")
from tests.helpers.browser import (
    REPO_ROOT,
    Dashboard,
    assert_stops_updating,
    assert_updating,
    click_until,
    fake_dashboard,
    fingerprint,
    wait_until,
)

_CELL_TITLE_INPUT = 'input[placeholder="Leave empty to use the derived title"]'
_GRID_TITLE_INPUT = 'input[placeholder="Enter grid title"]'
_FIXTURE = REPO_ROOT / "tests/dashboard/ui_config_fixtures/dummy/plot_configs.yaml"


def _active_tab(dash: Dashboard) -> str:
    return dash.page.locator(".bk-tab.bk-active").first.inner_text().strip()


def _write_grid_config(path: Path, title: str, *, overlapping: bool) -> Path:
    """Write an uploadable grid config derived from the dummy fixture's grid.

    A persisted grid is already in the upload file's format, so the fixture
    doubles as a realistic, workflow-resolvable payload. ``overlapping`` adds a
    copy of the first cell grown to span its whole column -- what a hand-edited
    or concatenated config looks like, two cells claiming the same slots.

    The copy overlaps the original only *partially* and is ordered ahead of the
    cell owning the slots it grows into. That combination is what breaks the
    preview: the region it claims still holds an unplaced slot, which Panel's
    ``GridSpec`` dereferences. A cell that merely repaints fully-occupied slots
    (an exact duplicate, say) is just as invalid but renders without raising,
    so it would not exercise the failure this guards.
    """
    grid = yaml.safe_load(_FIXTURE.read_text())["plot_grids"]["grids"][0]
    grid["title"] = title
    if overlapping:
        clash = copy.deepcopy(grid["cells"][0])
        clash["geometry"]["row_span"] = grid["nrows"]
        grid["cells"].insert(1, clash)
    path.write_text(yaml.safe_dump(grid))
    return path


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


def _wizard_click(dash: Dashboard, label: str) -> None:
    """Click a plot-wizard control once it is rendered.

    Every step rebuilds the dialog body, so a control is briefly absent and can
    detach under the cursor: wait for it to exist, then use the retrying click.
    """
    selector = f'button:text-is("{label}")'
    wait_until(
        dash,
        lambda: dash.page.locator(selector).count() > 0,
        label=f"wizard control {label!r}",
    )
    dash.click(selector)


@pytest.mark.browser
def test_clicking_out_an_empty_region_creates_a_live_plot():
    """The click-to-place flow, from two clicks on a bare grid to live data.

    Every other plot in this suite comes pre-seeded from the fixture, so this
    is the only coverage of the path a user actually takes to create one.
    """
    cell = ".lt-empty-cell-r0c0"
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
        page = dash.page
        dash.goto_tab("Manage Plots")
        _add_grid(dash, "Placed")

        # The first click only arms the region; the relabelled cell is the
        # proof it landed, without which the second click would arm rather
        # than complete the selection.
        dash.click(cell)
        armed = page.get_by_text("Click again for 1x1 plot", exact=True)
        wait_until(dash, lambda: armed.count() > 0, label="the armed cell")
        dash.open_modal(cell)

        for label in ("Monitors", "Beam monitor", "Histogram", "Next", "Lines"):
            _wizard_click(dash, label)
        _wizard_click(dash, "Next")
        _wizard_click(dash, "Add Plot")

        page.locator("[role=dialog]").first.wait_for(state="hidden", timeout=10000)
        wait_until(
            dash,
            lambda: page.locator(".lt-cell-r0c0").count() > 0,
            label="the placed cell's titlebar",
        )
        assert_updating(dash, "plot placed by clicking out a region")


@pytest.mark.browser
def test_session_reload_restores_tabs_and_live_updates():
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
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
        fake_dashboard("dummy") as fake,
        Dashboard.connect_many(2, fake.url) as (creator, observer),
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
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
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
def test_rebuilt_cell_titlebar_panes_are_visible_without_cdn_access():
    """A cell rebuilt in an open session must show its titlebar, not hide it.

    Panel reveals a markup pane's content only once every stylesheet ``<link>``
    in it has fired ``load``, and arms that reveal once, while rendering. A pane
    built after page load -- every cell the poll loop rebuilds -- first renders
    against cdn.holoviz.org URLs, which Panel then swaps for the locally served
    copies; the load events the reveal waits on belong to the discarded links
    (holoviz/panel#8696). The pane's model, text and layout stay correct, so the
    failure is invisible to every assertion except a visibility check.

    Blocking the CDN, as the deployment network does, makes those links fail for
    certain instead of racing the swap, which is what makes this deterministic.
    Which pane loses that race varies, so both titlebar panes are checked.
    """
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
        page = dash.page
        page.route("**cdn.holoviz.org/**", lambda route: route.abort())
        dash.goto_tab("Detectors")

        # Renaming rebuilds the cell, minting fresh titlebar panes.
        dash.open_modal(".lt-cell-r0c0.lt-tool-pencil")
        page.locator(_CELL_TITLE_INPUT).fill("Rebuilt Cell")
        page.get_by_role("button", name="Save", exact=True).click()

        title = page.get_by_text("Rebuilt Cell", exact=True).first
        wait_until(
            dash, lambda: title.count() > 0, label="renamed cell title in the DOM"
        )
        assert title.is_visible(), "rebuilt cell title is in the DOM but invisible"

        # The freshness pill is the same kind of pane and the symptom that was
        # reported; it fills on the first freshness-due poll after the rebuild.
        pill = page.get_by_text(re.compile(r"^\d+(\.\d+)?[sm]$")).first
        wait_until(dash, lambda: pill.count() > 0, label="freshness pill in the DOM")
        assert pill.is_visible(), "freshness pill is in the DOM but invisible"


@pytest.mark.browser
def test_remaining_tabs_keep_updating_after_disabling_and_removing_grids():
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
        # Order the grids [Diagnostics, Bravo, Charlie, Detectors]: two empty
        # grids ahead of the fixture's populated one, so disabling one and
        # removing the next both shift the Detectors tab position -- the
        # regression class where tab indices fall out of alignment with the
        # grid list once a preceding grid is hidden or gone.
        dash.goto_tab("Manage Plots")
        _add_grid(dash, "Bravo")
        dash.goto_tab("Manage Plots")
        _add_grid(dash, "Charlie")
        dash.goto_tab("Manage Plots")
        for expected in (
            ["Diagnostics", "Detectors", "Bravo", "Charlie"],
            ["Diagnostics", "Bravo", "Detectors", "Charlie"],
            ["Diagnostics", "Bravo", "Charlie", "Detectors"],
        ):
            dash.click(".lt-grid-detectors.lt-tool-chevron-down")
            wait_until(
                dash,
                lambda expected=expected: dash.tab_names()[-4:] == expected,
                label=f"grid tab order {expected}",
            )

        # Disable a grid ahead of Detectors: its tab vanishes, the rest keep
        # working.
        dash.click(".lt-grid-bravo.lt-tool-eye")
        wait_until(
            dash,
            lambda: "Bravo" not in dash.tab_names(),
            label="disabled grid tab to vanish",
        )
        assert "Charlie" in dash.tab_names()
        dash.goto_tab("Detectors")
        assert_updating(dash, "Detectors tab after disabling first grid")

        # Remove the grid between the disabled Bravo and Detectors.
        dash.goto_tab("Manage Plots")
        dash.click(".lt-grid-charlie.lt-tool-x")
        wait_until(
            dash,
            lambda: "Charlie" not in dash.tab_names(),
            label="removed grid tab to vanish",
        )
        dash.goto_tab("Detectors")
        assert_updating(dash, "Detectors tab after removing middle grid")


@pytest.mark.browser
def test_multi_layer_cell_gear_picks_the_layer_to_configure():
    # A cell with several layers turns its gear into a layer picker. Both the
    # gear and the entry it routes to are addressed per cell, since DOM order
    # across cells is not stable.
    gear = ".lt-cell-r2c0.lt-tool-settings"
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
        page = dash.page
        dash.goto_tab("Detectors")

        # The picker lists one entry per layer, named after it.
        dash.click(gear)
        entries = [
            page.get_by_text(f"Beam monitor → Histogram → Lines ({source})").first
            for source in ("monitor1", "monitor2")
        ]
        for entry in entries:
            wait_until(dash, entry.is_visible, label="layer menu entry")

        # The dropdown's min-width rule targets the popup's actual DOM class
        # (Bokeh's Dropdown menu is `bk-Menu`, not the lowercase `bk-menu` a
        # stale selector would suggest); a mismatch silently drops the rule
        # and wraps every entry across multiple lines.
        menu_box = page.locator(".bk-Menu").bounding_box()
        assert menu_box is not None
        assert menu_box["width"] >= 200

        # Choosing one opens the config modal for that layer, not the cell's
        # first: the source selector is pre-filled with the chosen layer's
        # source. (The dialog's own inner_text is empty -- Panel renders each
        # widget into its own shadow root -- so assert on the chip itself.)
        dash.open_modal(entries[1])
        chip = page.locator(".choices__list--multiple .choices__item").first
        chip.wait_for(state="visible", timeout=10000)
        assert chip.inner_text().startswith("monitor2")


@pytest.mark.browser
def test_concurrent_grid_property_edits_resolve_to_one_title_without_crash():
    """Two sessions racing to save edits on the same grid's properties.

    Grid edit is inline (a Manage Plots form switching into an edit state),
    not a modal, despite the wording in
    ``.claude/rules/dashboard-widgets.md``.

    There is no true last-writer-wins conflict resolution here: each
    session's Save Changes targets the grid it started editing, captured by
    id when the edit form opened. Whichever save reaches the orchestrator
    while that grid still exists commits; a save that arrives once the grid
    is already gone (replaced by the other session) is dropped -- with an
    error notification, not an unhandled exception (the bug this test
    guards: it used to be an unhandled ``KeyError``). In a tight race,
    which of the two saves actually lands first is not something this test
    controls or asserts on -- only the invariants that must hold regardless:
    exactly one of the two submitted titles survives, the losing session is
    told its edit did not apply, no server-side exception is logged, and the
    tab set stays intact and live.
    """
    pencil = ".lt-grid-detectors.lt-tool-pencil"
    first_title = "First Session Title"
    second_title = "Second Session Title"
    with (
        fake_dashboard("dummy") as fake,
        Dashboard.connect_many(2, fake.url) as (first, second),
    ):
        for dash in (first, second):
            dash.goto_tab("Manage Plots")

        # Both sessions open the same grid's edit form and type a different
        # title before either saves.
        first.click(pencil)
        first.page.locator(_GRID_TITLE_INPUT).fill(first_title)
        first.page.keyboard.press("Enter")

        second.click(pencil)
        second.page.locator(_GRID_TITLE_INPUT).fill(second_title)
        second.page.keyboard.press("Enter")

        log_offset = fake.log.stat().st_size

        # Fire both saves back-to-back, no wait in between, to race each
        # click against the other session's own topology poll.
        first.page.get_by_role("button", name="Save Changes", exact=True).click()
        second.page.get_by_role("button", name="Save Changes", exact=True).click()

        wait_until(
            first,
            lambda: (
                first_title in first.tab_names() or second_title in first.tab_names()
            ),
            label="one of the two submitted titles to win the race",
        )
        winning_title, losing_title = (
            (first_title, second_title)
            if first_title in first.tab_names()
            else (second_title, first_title)
        )
        winner, loser = (
            (first, second) if winning_title == first_title else (second, first)
        )

        for dash in (first, second):
            wait_until(
                dash,
                lambda dash=dash: winning_title in dash.tab_names(),
                label="both sessions to converge on the same grid title",
            )
            tabs = dash.tab_names()
            assert tabs.count(winning_title) == 1, tabs
            assert losing_title not in tabs
            assert "Detectors" not in tabs

        # The losing session is told its edit did not apply -- a dropped
        # save is a deliberate, user-visible outcome, not silent data loss.
        wait_until(
            loser,
            lambda: loser.page.locator('.notyf__message').count() > 0,
            label="the losing session's error notification",
        )
        assert (
            'grid was removed'
            in loser.page.locator('.notyf__message').first.inner_text()
        )

        new_log = fake.log.read_text()[log_offset:]
        assert "Traceback" not in new_log, (
            f"server logged an exception while saving concurrent grid edits:\n{new_log}"
        )

        winner.goto_tab(winning_title)
        assert_updating(winner, "surviving grid after concurrent edit race")


@pytest.mark.browser
def test_uploading_a_grid_config_with_overlapping_cells_is_rejected(tmp_path):
    """Import enforces the no-overlap invariant the click-to-place editor has.

    A config whose cells claim the same slot renders in the live view (cells
    overwrite, last wins) but the resulting grid is uneditable: building the
    edit preview feeds the overlapping region into Panel's ``GridSpec``, which
    dereferences the still-empty slot and raises, so edit mode aborts before
    the Save/Copy buttons appear -- leaving no way to repair the grid from the
    UI. Import is therefore the last point at which the user can still act on
    it, and is where it is refused. The second half re-uploads the same grid
    tiled, pinning the refusal to the overlap rather than to uploading at all.
    """
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
        page = dash.page
        dash.goto_tab("Manage Plots")
        page.get_by_role("button", name="Upload", exact=True).click()
        file_input = page.locator('input[type="file"]')
        log_offset = fake.log.stat().st_size

        file_input.set_input_files(
            _write_grid_config(
                tmp_path / "overlapping.yaml", "Overlapping Import", overlapping=True
            )
        )

        wait_until(
            dash,
            lambda: page.locator(".notyf__message").count() > 0,
            label="the rejection notification",
        )
        assert "overlaps" in page.locator(".notyf__message").first.inner_text()
        # Refused outright, not merely un-previewed: the form keeps its
        # defaults, so a following "Add Grid" cannot import the layout anyway.
        assert page.locator(_GRID_TITLE_INPUT).input_value() == "New Grid"
        assert "Overlapping Import" not in dash.tab_names()
        new_log = fake.log.read_text()[log_offset:]
        assert "Traceback" not in new_log, (
            f"server logged an exception rejecting the upload:\n{new_log}"
        )

        # Same grid, cells tiled: it imports and the new tab goes live.
        file_input.set_input_files(
            _write_grid_config(
                tmp_path / "tiled.yaml", "Tiled Import", overlapping=False
            )
        )
        wait_until(
            dash,
            lambda: page.locator(_GRID_TITLE_INPUT).input_value() == "Tiled Import",
            label="the accepted upload to populate the form",
        )
        page.get_by_role("button", name="Add Grid", exact=True).click()
        wait_until(
            dash,
            lambda: "Tiled Import" in dash.tab_names(),
            label="tab 'Tiled Import' to appear",
        )
        dash.goto_tab("Tiled Import")
        assert_updating(dash, "imported grid")


# Tallest canvas inside the pop-out window. The plot renders into per-widget
# shadow roots, which descendant CSS selectors do not cross, so walk them.
_POPOUT_PLOT_HEIGHT = """
() => {
  const deep = (root, out) => {
    root.querySelectorAll('*').forEach(e => {
      if (e.tagName === 'CANVAS') out.push(e);
      if (e.shadowRoot) deep(e.shadowRoot, out);
    });
    return out;
  };
  const panel = document.querySelector('.jsPanel');
  if (!panel) return 0;
  const heights = deep(panel, []).map(c => c.getBoundingClientRect().height);
  return heights.length ? Math.round(Math.max(...heights)) : 0;
}
"""


def _window_height(dash: Dashboard) -> float:
    box = dash.page.locator(".jsPanel").first.bounding_box()
    assert box is not None
    return box["height"]


@pytest.mark.browser
def test_popped_out_plot_resizes_with_its_window():
    """The plot must track the window, maximize included.

    jsPanel resizes its own content element, but the wrappers Panel puts below
    it carry no height, and Panel re-lays out only on a drag-resize. Without
    both gaps closed the plot keeps whatever size it had when it opened, and
    the window grows around it into whitespace.

    Uses the free-aspect cell: an aspect-locked plot derives its height from
    the window's *width*, so it legitimately overflows a wider window.
    """
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
        del fake
        page = dash.page
        page.set_viewport_size({"width": 1280, "height": 900})
        dash.goto_tab("Detectors")

        click_until(
            dash,
            ".lt-cell-r2c0.lt-tool-arrows-maximize",
            lambda: page.locator(".lt-popout-r2c0").count() == 1,
            label="the pop-out window to open",
        )
        wait_until(
            dash,
            lambda: page.evaluate(_POPOUT_PLOT_HEIGHT) > 0,
            label="the popped-out plot to render",
        )
        opened = page.evaluate(_POPOUT_PLOT_HEIGHT)
        assert opened > _window_height(dash) * 0.8, (
            f"plot {opened} does not fill the window it opened in"
        )

        page.locator(".jsPanel-btn-maximize").first.click()
        wait_until(
            dash,
            lambda: page.evaluate(_POPOUT_PLOT_HEIGHT) > opened,
            label="the plot to grow with the maximized window",
        )
        assert page.evaluate(_POPOUT_PLOT_HEIGHT) > _window_height(dash) * 0.8


@pytest.mark.browser
@pytest.mark.parametrize("viewport_height", [700, 1000])
def test_popped_out_window_fits_the_viewport_it_opens_on(viewport_height):
    """The window must never open with its title bar off the top of the screen.

    That title bar carries the only controls for closing, minimizing and
    maximizing, so a window opening above the fold cannot be dismissed at all.
    A fixed pixel height centred vertically does exactly that on any screen
    shorter than the window, which is most laptops.
    """
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
        del fake
        page = dash.page
        page.set_viewport_size({"width": 1280, "height": viewport_height})
        dash.goto_tab("Detectors")

        click_until(
            dash,
            ".lt-cell-r0c0.lt-tool-arrows-maximize",
            lambda: page.locator(".lt-popout-r0c0").count() == 1,
            label="the pop-out window to open",
        )

        box = page.locator(".jsPanel").first.bounding_box()
        assert box is not None
        assert box["y"] >= 0, f"window opened above the viewport: {box}"
        assert box["y"] + box["height"] <= viewport_height, (
            f"window opened taller than the viewport: {box}"
        )
        # The controls specifically, not just the window's box.
        close_button = page.locator(".jsPanel-btn-close").first.bounding_box()
        assert close_button is not None
        assert close_button["y"] >= 0


# Deepest *scrollable* vertical overflow outside the pop-out window itself,
# which scrolls its own content by design. Only elements that actually scroll
# count: a clipped one hides its overflow and shows no scrollbar. Light DOM
# only -- the template's scrolling main region and the boxes rooting the
# windows in the component tree both live there.
_PAGE_OVERFLOW = """
() => {
  let worst = document.documentElement.scrollHeight
      - document.documentElement.clientHeight;
  document.querySelectorAll('*').forEach(e => {
    if (e.closest('.jsPanel')) return;
    const overflow = getComputedStyle(e).overflowY;
    if (overflow !== 'auto' && overflow !== 'scroll') return;
    worst = Math.max(worst, e.scrollHeight - e.clientHeight);
  });
  return worst;
}
"""


@pytest.mark.browser
def test_opening_a_popout_does_not_make_the_page_scroll():
    """A window must cost the page under it no layout at all.

    jsPanel moves the window's content out to ``document.body``, but the box
    Panel wrapped it in stays behind at the window's full size, empty.
    Overflowing the zero-height container that roots the windows, it lands
    below the fold and hands the page a scrollbar onto a window-sized band of
    whitespace.
    """
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
        del fake
        page = dash.page
        dash.goto_tab("Detectors")
        wait_until(
            dash,
            lambda: page.locator(".lt-cell-r0c0.lt-tool-arrows-maximize").count() == 1,
            label="the plot grid to render",
        )
        before = page.evaluate(_PAGE_OVERFLOW)

        click_until(
            dash,
            ".lt-cell-r0c0.lt-tool-arrows-maximize",
            lambda: page.locator(".lt-popout-r0c0").count() == 1,
            label="the pop-out window to open",
        )
        wait_until(
            dash,
            lambda: page.evaluate(_POPOUT_PLOT_HEIGHT) > 0,
            label="the popped-out plot to render",
        )

        assert page.evaluate(_PAGE_OVERFLOW) == before


def _drag(dash: Dashboard, selector: str, *, dx: int, dy: int) -> None:
    """Drag an element's centre by (dx, dy), as a mouse gesture."""
    page = dash.page
    grip = page.locator(selector).first.bounding_box()
    assert grip is not None
    x, y = grip["x"] + grip["width"] / 2, grip["y"] + grip["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + dx, y + dy, steps=10)
    page.mouse.up()


@pytest.mark.browser
def test_renaming_a_popped_out_cell_leaves_its_window_where_it_was():
    """A cell rebuild must replace the window's contents, not the window.

    jsPanel owns the window's size and position and reports neither back to
    Panel, so a rebuild that replaces the window can only bring it back at the
    default size in the next cascade slot -- from the user's side, a rename
    throws the window across the screen and shrinks it. Renaming is both the
    cheapest rebuild to drive from the UI and the one most likely to happen
    with a window open alongside.
    """
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
        del fake
        page = dash.page
        dash.goto_tab("Detectors")

        click_until(
            dash,
            ".lt-cell-r0c0.lt-tool-arrows-maximize",
            lambda: page.locator(".lt-popout-r0c0").count() == 1,
            label="the pop-out window to open",
        )
        wait_until(
            dash,
            lambda: page.evaluate(_POPOUT_PLOT_HEIGHT) > 0,
            label="the popped-out plot to render",
        )
        opened = page.locator(".jsPanel").first.bounding_box()
        # Place and size it by hand, as a user would: clear of the cell
        # titlebar the rename is driven from, and far enough from the default
        # that a replacement window could not land here by chance.
        # Shrunk before it is moved: the resize handle sits on the window's
        # bottom-right corner, which a window moved down and right first has
        # already carried off the bottom of the viewport.
        _drag(dash, ".jsPanel-resizeit-se", dx=-160, dy=-220)
        _drag(dash, ".jsPanel-title", dx=140, dy=180)
        placed = page.locator(".jsPanel").first.bounding_box()
        assert placed["x"] != opened["x"], "the window did not move"
        assert placed["height"] != opened["height"], "the window did not resize"

        dash.open_modal(".lt-cell-r0c0.lt-tool-pencil")
        page.locator(_CELL_TITLE_INPUT).fill("Renamed")
        page.get_by_role("button", name="Save", exact=True).click()
        wait_until(
            dash,
            lambda: (
                page.locator(".jsPanel-title").first.inner_text().strip() == "Renamed"
            ),
            label="the window header to follow the rename",
        )

        assert page.locator(".jsPanel").first.bounding_box() == pytest.approx(
            placed, abs=1
        )
        assert page.evaluate(_POPOUT_PLOT_HEIGHT) > 0
        assert_updating(dash, "pop-out whose cell was renamed")


@pytest.mark.browser
def test_popped_out_plot_stays_live_across_tabs_and_sleeps_when_minimized():
    """A pop-out is a live second view, and costs nothing while it shows none.

    Switching tabs is the interesting case: ``dynamic=True`` tears down the
    hidden grid's Bokeh models, so the window is then the *only* plot in the
    document -- and the poll loop must still be feeding it. Minimizing it must
    stop that feed, or popping out many cells and minimizing the windows would
    pin every one of those cells live for nothing on screen.
    """
    popout = ".lt-cell-r0c0.lt-tool-arrows-maximize"
    with fake_dashboard("dummy") as fake, Dashboard.connect(fake.url) as dash:
        del fake
        page = dash.page
        dash.goto_tab("Detectors")

        click_until(
            dash,
            popout,
            lambda: page.locator(".lt-popout-r0c0").count() == 1,
            label="the pop-out window to open",
        )
        # The cell is a detail *view*: the grid cell keeps its own plot and
        # its whole titlebar, including the button that opened the window.
        assert page.locator(popout).count() == 1

        dash.goto_tab("Workflows")
        assert page.locator(".jsPanel").first.is_visible()
        assert_updating(dash, "popped-out plot while another tab is shown")

        # Minimizing must put the cell back to sleep, and restoring must wake
        # it. Proving both end to end matters because the guard rests on
        # jsPanel round-tripping the user's click as a ``status`` change.
        page.locator(".jsPanel-btn-minimize").first.click()
        assert_stops_updating(dash, "minimized pop-out on a hidden tab")

        # A minimized window is parked off-screen and replaced by a strip of
        # small buttons; normalize from there rather than from the panel.
        page.locator(".jsPanel-btn-sm.jsPanel-btn-normalize").click()
        assert_updating(dash, "restored pop-out on a hidden tab")

        page.locator(".jsPanel-btn-close").first.click()
        wait_until(
            dash,
            lambda: page.locator(".lt-popout-r0c0").count() == 0,
            label="the pop-out window to close",
        )
