# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""A reaped session whose browser is still connected must leave the dashboard.

The reaper decides on the browser heartbeat alone, and browsers throttle the
heartbeat timer of hidden tabs (Chrome: about once a minute after five minutes),
so it reaps tabs that are merely in the background. Left on the page, such a tab
accepts input that changes shared state but never updates again (#1348). The
tab must instead park on the paused page while hidden and come back in a fresh,
live session once shown.

Hiding the tab is emulated by overriding ``document.visibilityState`` from an
init script, so the override survives the navigations under test; the switch
lives in ``sessionStorage`` for the same reason. Stopping the page's timers
stands in for the throttled heartbeat.

Runs via ``pytest -m browser`` (excluded from the default run).
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

from ess.livedata.dashboard.session_paused import SESSION_PAUSED_PATH
from tests.helpers.browser import (
    Dashboard,
    assert_updating,
    fake_dashboard,
    open_browser,
)

# Well above page load plus the first heartbeat, so the session is reaped for
# the stopped timers and not before it ever heartbeats.
SESSION_STALE_TIMEOUT_SECONDS = 15
# The registry reaps from the background update thread, between its other work.
REAP_TIMEOUT_MS = 3 * SESSION_STALE_TIMEOUT_SECONDS * 1000
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
_STOP_TIMERS_JS = """() => {
  const top = setTimeout(() => {}, 0);
  for (let i = 0; i <= top; i++) { clearInterval(i); clearTimeout(i); }
}"""


def _set_visibility(dash: Dashboard, state: str) -> None:
    dash.page.evaluate(
        "([key, state]) => {"
        "  sessionStorage.setItem(key, state);"
        "  document.dispatchEvent(new Event('visibilitychange'));"
        "}",
        [_VISIBILITY_KEY, state],
    )


@pytest.mark.browser
def test_reaped_hidden_tab_parks_and_resumes_when_shown() -> None:
    with (
        fake_dashboard(
            "dummy", session_stale_timeout_seconds=SESSION_STALE_TIMEOUT_SECONDS
        ) as fake,
        open_browser() as browser,
    ):
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        context.add_init_script(_VISIBILITY_OVERRIDE_JS)
        dash = Dashboard(context.new_page())
        dash.page.goto(fake.url, wait_until="networkidle")
        dash.goto_tab("Detectors")
        assert_updating(dash, "session before the reap")
        dashboard_url = dash.page.url

        _set_visibility(dash, 'hidden')
        dash.page.evaluate(_STOP_TIMERS_JS)
        dash.page.wait_for_url(f"**/{SESSION_PAUSED_PATH}", timeout=REAP_TIMEOUT_MS)

        dash.page.wait_for_timeout(PARKED_MS)
        assert dash.page.url.endswith(SESSION_PAUSED_PATH), (
            "a hidden tab left the paused page, starting a session nobody looks at"
        )

        _set_visibility(dash, 'visible')
        dash.page.wait_for_url(dashboard_url)
        dash.goto_tab("Detectors")
        assert_updating(dash, "session resumed from the paused page")
