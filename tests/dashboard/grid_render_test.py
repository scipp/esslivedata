# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Plot-grid rendering assumptions that a Panel/HoloViews/Bokeh upgrade can break.

Both tests run on the LOKI fixture: 9 bank images, a timeseries and a bar chart in
one grid. That grid is big enough to show the failures below; the smaller dummy
grid does not reliably show them.

- **Every session renders after a reveal.** With Bokeh 3.10, session callbacks of
  different sessions run concurrently on worker threads (#1413). With three
  sessions, a session that reveals the grid while the others are live can be
  left with a blank grid and ``n is not iterable`` page errors.
- **One PATCH-DOC per session pass.** :func:`batched_update` holds all document
  events of a pass and relies on Panel to send them to the browser as one
  message. Each message is a separate render pass in BokehJS, so splitting a pass
  into one message per event multiplies the browser's work. Panel 1.10 on Bokeh
  3.10 does that: ~57 messages/s instead of ~1/s on this grid.

Runs via ``pytest -m browser``.
"""

from __future__ import annotations

import re
import time

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import WebSocket

from tests.helpers.browser import (
    Dashboard,
    fake_dashboard,
    open_browser,
    wait_until,
)

GRID = "Banks"

# Canvases across all shadow roots: one or more per rendered figure. A grid whose
# figures failed to build shows its frame but no canvas.
CANVAS_COUNT_JS = """() => {
  let n = 0;
  const walk = (root) => {
    n += root.querySelectorAll('canvas').length;
    root.querySelectorAll('*').forEach(e => e.shadowRoot && walk(e.shadowRoot));
  };
  walk(document);
  return n;
}"""

PATCH_DOC = re.compile(r'"msgtype":\s*"PATCH-DOC"')

# The fake backend publishes at 1 Hz, and each publish wakes one pass per session;
# housekeeping passes add a few more. A healthy session gets about one message per
# second. The ceiling leaves room for passes that happen to fall close together,
# and is still far below the one-message-per-event rate (tens per second).
MAX_PATCH_DOCS_PER_S = 3
WINDOW_S = 8


def _canvases(dash: Dashboard) -> int:
    return dash.page.evaluate(CANVAS_COUNT_JS)


def _page_errors(dash: Dashboard) -> list[str]:
    return [line for line in dash.console if line.startswith("pageerror:")]


@pytest.mark.browser
def test_grid_renders_in_every_session_after_a_reveal() -> None:
    with (
        fake_dashboard("loki") as fake,
        Dashboard.connect_many(3, fake.url) as sessions,
    ):
        for dash in sessions:
            dash.goto_tab(GRID)
        wait_until(
            sessions[0],
            lambda: (
                len({_canvases(d) for d in sessions}) == 1
                and _canvases(sessions[0]) > 0
            ),
            label="the grid to render in all sessions",
        )
        expected = _canvases(sessions[0])

        sessions[1].goto_tab("Workflows")
        sessions[1].goto_tab(GRID)

        for index, dash in enumerate(sessions):
            wait_until(
                dash,
                lambda dash=dash: _canvases(dash) == expected,
                label=f"session {index} to show {expected} canvases "
                f"(has {_canvases(dash)})",
            )
        errors = {i: _page_errors(d) for i, d in enumerate(sessions)}
        assert not any(errors.values()), errors


@pytest.mark.browser
def test_session_pass_reaches_the_browser_as_one_patch_doc() -> None:
    received: list[str] = []

    def record(payload: str | bytes) -> None:
        # Message headers are text frames; binary frames carry array buffers.
        if isinstance(payload, str):
            received.append(payload)

    def tap(ws: WebSocket) -> None:
        ws.on("framereceived", record)

    with fake_dashboard("loki") as fake, open_browser() as browser:
        page = browser.new_context(viewport={"width": 1600, "height": 1000}).new_page()
        # Registered before the page loads: Playwright reports only websockets
        # opened after the listener.
        page.on("websocket", tap)
        dash = Dashboard(page)
        page.goto(fake.url, wait_until="networkidle")
        dash.settle()
        dash.goto_tab(GRID)
        wait_until(dash, lambda: _canvases(dash) > 0, label="the grid to render")

        received.clear()
        start = time.monotonic()
        page.wait_for_timeout(WINDOW_S * 1000)
        elapsed = time.monotonic() - start
        patch_docs = sum(1 for frame in received if PATCH_DOC.search(frame))

    assert patch_docs > 0, "the session received no updates"
    assert patch_docs <= MAX_PATCH_DOCS_PER_S * elapsed, (
        f"{patch_docs} PATCH-DOC messages in {elapsed:.1f} s: the events of a "
        f"session pass are no longer sent together"
    )
