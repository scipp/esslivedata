# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""A hidden tab is reaped after a grace, and comes back when shown.

A tab hidden for longer than the grace stops sending heartbeats, so the reaper
drops its session while its websocket is still open. Left on the page, such a
tab would accept input that changes shared state but never update again
(#1348). It must instead park on the paused page while hidden and come back in
a fresh, live session, on the tab it was showing, once shown. A tab shown again
before the reap keeps its session.

Hiding the tab is emulated by overriding ``document.visibilityState`` from an
init script, so the override survives the navigations under test; the switch
lives in ``sessionStorage`` for the same reason.

Runs via ``pytest -m browser`` (excluded from the default run).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

pytest.importorskip("playwright.sync_api")

from ess.livedata.dashboard.session_paused import SESSION_PAUSED_PATH
from ess.livedata.dashboard.widgets.plot_grid_tabs import TAB_QUERY_PARAM
from tests.helpers.browser import (
    Dashboard,
    assert_updating,
    fake_dashboard,
    open_browser,
)

# Well above page load plus the first heartbeat, so a session is reaped for
# being hidden and not before it ever heartbeats.
SESSION_STALE_TIMEOUT_SECONDS = 15
HIDDEN_TAB_GRACE_SECONDS = 2
# Grace plus stale timeout, with margin: the registry reaps from the background
# update thread, between its other work.
REAP_TIMEOUT_MS = 2 * (HIDDEN_TAB_GRACE_SECONDS + SESSION_STALE_TIMEOUT_SECONDS) * 1000
# How long a hidden tab is watched for staying parked.
PARKED_MS = 3000

_VISIBILITY_KEY = 'lt-test-visibility'
_VISIBILITY_OVERRIDE_JS = f"""
Object.defineProperty(Document.prototype, 'visibilityState', {{
  configurable: true,
  get() {{
    try {{ return sessionStorage.getItem('{_VISIBILITY_KEY}') || 'visible'; }}
    catch (e) {{ return 'visible'; }}
  }},
}});
"""


def _active_tab(dash: Dashboard) -> str:
    return dash.page.locator(".bk-tab.bk-active").first.inner_text().strip()


def _set_visibility(dash: Dashboard, state: str) -> None:
    dash.page.evaluate(
        "([key, state]) => {"
        "  sessionStorage.setItem(key, state);"
        "  document.dispatchEvent(new Event('visibilitychange'));"
        "}",
        [_VISIBILITY_KEY, state],
    )


@contextmanager
def _dashboard_on_detectors() -> Iterator[Dashboard]:
    with (
        fake_dashboard(
            "dummy",
            session_stale_timeout_seconds=SESSION_STALE_TIMEOUT_SECONDS,
            hidden_tab_grace_seconds=HIDDEN_TAB_GRACE_SECONDS,
        ) as fake,
        open_browser() as browser,
    ):
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        context.add_init_script(_VISIBILITY_OVERRIDE_JS)
        dash = Dashboard(context.new_page())
        dash.page.goto(fake.url, wait_until="networkidle")
        dash.goto_tab("Detectors")
        # The active tab rides in the URL, so a resumed session reopens it.
        dash.page.wait_for_url(f"**/?{TAB_QUERY_PARAM}=Detectors")
        yield dash


@pytest.mark.browser
def test_tab_shown_before_the_reap_keeps_its_session() -> None:
    with _dashboard_on_detectors() as dash:
        # Survives only if the page is never left: a reaped tab that is visible
        # goes through the paused page straight back to the same URL.
        dash.page.evaluate("() => { window.__lt_never_left = true; }")
        _set_visibility(dash, 'hidden')
        # Past the grace, so heartbeats have stopped. The last one went out up
        # to one heartbeat interval (5 s) before hiding, so this stays well
        # short of the reap.
        dash.page.wait_for_timeout((HIDDEN_TAB_GRACE_SECONDS + 3) * 1000)
        _set_visibility(dash, 'visible')
        # Long past the point where the session would have been reaped had the
        # heartbeat not resumed.
        dash.page.wait_for_timeout((SESSION_STALE_TIMEOUT_SECONDS + 5) * 1000)
        assert dash.page.evaluate("() => window.__lt_never_left === true"), (
            "a tab shown before the reap lost its session"
        )
        assert_updating(dash, "session shown again before the reap")


@pytest.mark.browser
def test_hidden_tab_parks_after_grace_and_resumes_on_its_tab_when_shown() -> None:
    with _dashboard_on_detectors() as dash:
        assert_updating(dash, "session before the reap")
        dashboard_url = dash.page.url

        _set_visibility(dash, 'hidden')
        dash.page.wait_for_url(f"**/{SESSION_PAUSED_PATH}", timeout=REAP_TIMEOUT_MS)

        dash.page.wait_for_timeout(PARKED_MS)
        assert dash.page.url.endswith(SESSION_PAUSED_PATH), (
            "a hidden tab left the paused page, starting a session nobody looks at"
        )

        _set_visibility(dash, 'visible')
        dash.page.wait_for_url(dashboard_url)
        dash.settle()
        assert _active_tab(dash) == "Detectors"
        assert_updating(dash, "session resumed from the paused page")
