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
browser, which is what #1116 asks for: churn sessions, and check that the
server neither leaks them nor stops working.

**What is observed.** ``SessionRegistry`` names every session it registers,
unregisters and reaps, so this test follows session ids through the server log
instead of counting sessions: every id registered while browsers churn must
come back as *unregistered*, every id registered by a browser that then went
offline must come back as *reaped*, and neither set may turn up under the
other's teardown line. Naming the path is what keeps the two apart. A count can
only tell them apart by *when* it drops, which pins the test to Bokeh's
unused-session defaults (15 s lifetime, swept every 17 s) that this app does
not set. Counting is also less literal than it looks: the harness's readiness
probe renders the app over plain HTTP, which registers a browser-less session
of its own.

What the log cannot show is whether the *deeper* teardown ran: a leaked viewer
token keeps a hidden layer computing, and nothing logs that. Closing that gap
needs the app in-process (#1147); until then it stays manual (#1097). What the
closing check does buy is that the reaper's deferred, off-IOLoop teardown left
shared state and the background update thread intact -- the failure mode that
stalls every session at once.

**Why one test.** Both paths run against one server, with the clean closes
draining *inside* the reaper's window, so the stale timeout is waited out once
rather than twice.

Runs via ``pytest -m browser`` (excluded from the default run; CI runs them via
``tox -e browser``; skips cleanly where Playwright is absent).
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Browser, BrowserContext

from tests.helpers.browser import (
    Dashboard,
    assert_updating,
    fake_dashboard,
    fingerprint,
    open_browser,
)

CHURN_CYCLES = 3
SESSIONS_PER_CYCLE = 2
ABANDONED_SESSIONS = 3

# A churn session only has to reach the server and take up per-layer state, not
# to look right, so it settles far shorter than a session we assert on.
CHURN_SETTLE_MS = 1500
POLL_INTERVAL_SECONDS = 0.5

# Closing the websocket does not unregister the session immediately: Bokeh
# discards it once it has been unused for 15 s, checked every 17 s. No upper
# bound is needed to keep this honest -- a session the reaper got to first
# shows up under the wrong teardown line, which is asserted separately.
CLEAN_CLOSE_TIMEOUT_SECONDS = 60

# SessionRegistry's stale timeout is 60 s (dashboard_services.py), and the
# background update thread reaps between its other work.
REAP_TIMEOUT_SECONDS = 90

# Session ids as SessionRegistry logs them. The console renderer wraps each
# message in ANSI escapes, which terminate the id capture on their own.
_REGISTERED = re.compile(r"Registered new session: ([\w-]+)")
_UNREGISTERED = re.compile(r"Unregistered session: ([\w-]+)")
_REAPED = re.compile(r"Cleaned up stale session: ([\w-]+)")


def _ids(pattern: re.Pattern[str], text: str) -> set[str]:
    return set(pattern.findall(text))


def _log_since(log: Path, offset: int) -> str:
    """Server log written after ``offset``, so earlier sessions cannot leak in."""
    return log.read_text()[offset:]


def _wait_for_ids(
    log: Path,
    offset: int,
    pattern: re.Pattern[str],
    expected: set[str],
    *,
    timeout_seconds: float,
    label: str,
) -> None:
    """Wait until ``pattern`` has reported every expected session id."""
    deadline = time.monotonic() + timeout_seconds
    while missing := expected - _ids(pattern, _log_since(log, offset)):
        if time.monotonic() > deadline:
            raise AssertionError(
                f"{label}: {len(missing)} of {len(expected)} sessions unaccounted "
                f"for after {timeout_seconds:.0f} s: {sorted(missing)}"
            )
        time.sleep(POLL_INTERVAL_SECONDS)


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
    with fake_dashboard("dummy") as fake, open_browser() as browser:
        churned_at = fake.log.stat().st_size
        for _ in range(CHURN_CYCLES):
            contexts = [
                open_churn_session(browser, fake.url) for _ in range(SESSIONS_PER_CYCLE)
            ]
            for context in contexts:
                context.close()
        churn_ids = _ids(_REGISTERED, _log_since(fake.log, churned_at))
        expected = CHURN_CYCLES * SESSIONS_PER_CYCLE
        assert len(churn_ids) == expected, (
            f"{expected} browsers were churned but the server registered "
            f"{len(churn_ids)} sessions"
        )

        # Cutting the browsers off the network abandons these sessions without a
        # websocket close, leaving the server with an open connection and a
        # stalled heartbeat -- the only thing the reaper has to go on. Their
        # stale timeout runs from here, and the clean closes above drain inside
        # it.
        abandoned_at = fake.log.stat().st_size
        for _ in range(ABANDONED_SESSIONS):
            # The context lives until the browser closes: closing it here would
            # let Bokeh destroy the very session whose reaping is asserted.
            open_churn_session(browser, fake.url).set_offline(True)
        abandoned_ids = _ids(_REGISTERED, _log_since(fake.log, abandoned_at))
        assert len(abandoned_ids) == ABANDONED_SESSIONS, (
            f"{ABANDONED_SESSIONS} browsers were abandoned but the server "
            f"registered {len(abandoned_ids)} sessions"
        )

        _wait_for_ids(
            fake.log,
            churned_at,
            _UNREGISTERED,
            churn_ids,
            timeout_seconds=CLEAN_CLOSE_TIMEOUT_SECONDS,
            label="sessions closed cleanly but never unregistered",
        )
        _wait_for_ids(
            fake.log,
            abandoned_at,
            _REAPED,
            abandoned_ids,
            timeout_seconds=REAP_TIMEOUT_SECONDS,
            label="sessions abandoned but never reaped",
        )

        churn_log = _log_since(fake.log, churned_at)
        assert not churn_ids & _ids(_REAPED, churn_log), (
            "cleanly closed sessions were reaped: the close path did not run "
            "and the reaper silently covered for it"
        )
        assert not abandoned_ids & _ids(_UNREGISTERED, churn_log), (
            "abandoned sessions were unregistered: the server saw a clean close, "
            "so the reaper was never exercised"
        )
        assert "Error cleaning up updater" not in churn_log, (
            f"teardown raised while releasing a session:\n{churn_log}"
        )
        assert "Error in periodic update step" not in churn_log, (
            f"a session kept updating after teardown:\n{churn_log}"
        )

        assert_server_still_serves(browser, fake.url)
