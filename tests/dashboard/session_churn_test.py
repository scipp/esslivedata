# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Session churn: repeated connect/disconnect must not leak dashboard sessions.

Every session holds per-session state -- a Bokeh document, a periodic callback,
a notification-queue registration, per-layer viewer tokens. Two distinct
teardown paths release it:

* **Clean close** -- the websocket closes, Bokeh destroys the session, and
  ``on_session_destroyed`` unregisters it; teardown runs inline on the
  session's own IOLoop.
* **Reaper** -- the browser vanishes *without* closing the socket (network
  partition, suspended machine). The server still sees an open connection, so
  only the missing browser heartbeat gives it away: ``SessionRegistry`` reaps
  the session from a background thread and defers the document-touching half of
  teardown, which may never run (#955, #1095).

Both had unit coverage but neither had ever run end to end through a real
browser, which is what #1116 asks for: churn sessions and check that the count
returns to baseline and the server stays usable.

**What is observable.** This app enables neither the Bokeh admin panel nor a
metrics endpoint, so the session count is read from the UI: the *System Status*
tab renders a "Dashboard Sessions" summary straight from ``SessionRegistry``.
The observing session therefore stays on that tab -- ``dynamic=True`` means
only the active tab's models exist, and the widget skips refreshes while
hidden. What the count cannot show is whether the *deeper* teardown ran: a
leaked viewer token keeps a hidden layer computing, which no amount of UI
inspection reveals (that gap stays manual, #1097). What the closing check does
buy is that the reaper's deferred, off-IOLoop teardown left shared state and
the background update thread intact -- the failure mode that stalls every
session at once.

**Why one test.** Both paths run against one server, with the clean closes
draining *inside* the reaper's window, so the stale timeout is waited out once
rather than twice.

Runs via ``pytest -m browser`` (excluded from the default run; CI runs them via
``tox -e browser``; skips cleanly where Playwright is absent).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Browser, BrowserContext

from tests.helpers.browser import (
    Dashboard,
    assert_updating,
    fake_dashboard,
    fingerprint,
)

CHURN_CYCLES = 3
SESSIONS_PER_CYCLE = 2
ABANDONED_SESSIONS = 3

# A churn session only has to reach the server and take up per-layer state, not
# to look right, so it settles far shorter than a session we assert on.
CHURN_SETTLE_MS = 1500
POLL_INTERVAL_MS = 250

# The summary refreshes at most every 2 s, so a session takes a moment to show up.
REGISTRATION_TIMEOUT_MS = 15_000

# Closing the websocket does not unregister the session immediately: Bokeh
# discards it once it has been unused for 15 s, checked every 17 s (upstream
# defaults, which this app does not pin). The ceiling matters as much as the
# floor -- it has to stay below the point where the reaper would clean up a
# cleanly closed session too, and so hide a broken close path.
CLEAN_RECOVERY_TIMEOUT_MS = 40_000

# Measured from the moment the browsers go offline: past the window above, so a
# session still counted here cannot have been cleanly closed, yet short of the
# reaper's own deadline (SessionRegistry's 60 s stale timeout, set in
# dashboard_services.py). This is what keeps the reaper path distinguishable
# from the clean-close path instead of the two silently collapsing into one.
NO_CLEAN_CLOSE_CHECKPOINT_MS = 40_000
REAPER_RECOVERY_TIMEOUT_MS = 45_000

# The "Dashboard Sessions" summary as rendered by SessionStatusWidget:
# "<b>3</b> sessions (including you)", "1 active session (just you)" or
# "No active sessions". Walks shadow roots because dashboard widgets render into
# them, and matches textContent so no node needs layout.
SESSION_COUNT_JS = r"""() => {
  const counts = [];
  const walk = (root) => root.querySelectorAll('*').forEach((el) => {
    const text = (el.textContent || '').trim();
    if (/^No active sessions$/.test(text)) counts.push(0);
    const match = text.match(/^(\d+) (?:active )?sessions?\b/);
    if (match) counts.push(Number(match[1]));
    if (el.shadowRoot) walk(el.shadowRoot);
  });
  walk(document);
  return counts.length ? counts[0] : null;
}"""


def session_count(observer: Dashboard) -> int:
    """Sessions the server reports, read from the observer's System Status tab."""
    count = observer.page.evaluate(SESSION_COUNT_JS)
    assert count is not None, (
        "No 'Dashboard Sessions' summary in the DOM -- is the observer still on "
        "the System Status tab?"
    )
    return count


def wait_for_sessions(
    observer: Dashboard,
    condition: Callable[[int], bool],
    *,
    timeout_ms: int,
    label: str,
) -> None:
    """Poll the reported session count until ``condition`` holds.

    Unlike ``wait_until``, the failure carries the count last seen, which is the
    whole diagnostic when sessions leak.
    """
    waited = 0
    while not condition(count := session_count(observer)):
        if waited >= timeout_ms:
            raise AssertionError(
                f"{label}: session count stuck at {count} after {timeout_ms // 1000} s"
            )
        observer.page.wait_for_timeout(POLL_INTERVAL_MS)
        waited += POLL_INTERVAL_MS


def open_churn_session(browser: Browser, url: str) -> BrowserContext:
    """Open a throwaway session that visits a plot tab.

    Visiting a plot tab is what makes the session worth tearing down: it takes
    out per-layer viewer tokens that teardown has to release again. Both steps
    raise if the session never came up, so a session that returns from here is
    one the server registered.
    """
    context = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = context.new_page()
    page.goto(url, wait_until="networkidle")
    page.get_by_text("Detectors", exact=True).first.click()
    page.wait_for_timeout(CHURN_SETTLE_MS)
    return context


def assert_server_still_serves(browser: Browser, url: str) -> None:
    """A session opened after the churn must render populated plots and update."""
    context = browser.new_context(viewport={"width": 1600, "height": 1000})
    try:
        fresh = Dashboard(context.new_page())
        fresh.page.goto(url, wait_until="networkidle")
        fresh.goto_tab("Detectors")
        fp = fingerprint(fresh)
        assert fp["sources"] > 0, "session opened after churn rendered no data sources"
        assert fp["length"] > 0, "session opened after churn rendered empty sources"
        assert_updating(fresh, "session opened after churn")
    finally:
        context.close()


@pytest.mark.browser
def test_session_churn_returns_to_baseline_and_server_stays_usable() -> None:
    with fake_dashboard("dummy", 5070) as url, Dashboard.connect(url) as observer:
        observer.goto_tab("System Status")
        browser = observer.page.context.browser
        assert browser is not None

        for cycle in range(CHURN_CYCLES):
            contexts = [
                open_churn_session(browser, url) for _ in range(SESSIONS_PER_CYCLE)
            ]
            wait_for_sessions(
                observer,
                lambda n: n >= 1 + SESSIONS_PER_CYCLE,
                timeout_ms=REGISTRATION_TIMEOUT_MS,
                label=f"cycle {cycle}: churn sessions never showed up",
            )
            for context in contexts:
                context.close()

        # Cutting the browsers off the network abandons these sessions without a
        # websocket close, leaving the server with an open connection and a
        # stalled heartbeat -- the only thing the reaper has to go on. Their
        # deadline runs from here, and the clean closes above drain inside it.
        abandoned = [
            open_churn_session(browser, url) for _ in range(ABANDONED_SESSIONS)
        ]
        for context in abandoned:
            context.set_offline(True)
        offline_at = time.monotonic()
        survivors = 1 + ABANDONED_SESSIONS

        closed = CHURN_CYCLES * SESSIONS_PER_CYCLE
        wait_for_sessions(
            observer,
            lambda n: n == survivors,
            timeout_ms=CLEAN_RECOVERY_TIMEOUT_MS,
            label=f"{closed} sessions closed cleanly, expected {survivors} left",
        )

        elapsed_ms = (time.monotonic() - offline_at) * 1000
        observer.page.wait_for_timeout(
            max(0.0, NO_CLEAN_CLOSE_CHECKPOINT_MS - elapsed_ms)
        )
        assert (count := session_count(observer)) == survivors, (
            f"session count is {count} {NO_CLEAN_CLOSE_CHECKPOINT_MS // 1000} s "
            f"after the browsers went offline, expected {survivors}: fewer means "
            "the server saw a clean close and the reaper is not being exercised, "
            "more means the cleanly closed sessions were never unregistered"
        )

        wait_for_sessions(
            observer,
            lambda n: n == 1,
            timeout_ms=REAPER_RECOVERY_TIMEOUT_MS,
            label=f"{ABANDONED_SESSIONS} abandoned sessions, expected 1 left",
        )
        for context in abandoned:
            context.close()

        assert_server_still_serves(browser, url)
