# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Shared helpers for browser-driven (Playwright) dashboard tests.

Importing this module requires Playwright; test modules must call
``pytest.importorskip("playwright.sync_api")`` before importing it.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import Browser, sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Re-exported so tests reach the driving kit through this module, which owns the
# scripts/ sys.path seam above, rather than each repeating it.
import drive_dashboard  # noqa: E402,F401
from drive_dashboard import (  # noqa: E402
    Dashboard,
    _fake_dashboard,
    _launch_browser,
)

fake_dashboard = _fake_dashboard

# How long to watch for a data update before declaring a session stalled. The
# fake backend emits at 1 Hz, so this is a generous margin.
UPDATE_WINDOW_MS = 6000

# Cross-session state propagates via per-session version polling (~1 s tick);
# generous margin for a change made in one session to appear in another.
PROPAGATION_TIMEOUT_MS = 20_000

# Fingerprint of all rendered ColumnDataSource data in the page: source and
# column-length counts plus a value checksum (sampling nested/typed arrays, so
# image payloads contribute). Any data update changes it.
DATA_FINGERPRINT_JS = """() => {
  let sources = 0, length = 0, checksum = 0;
  const sample = (column) => {
    for (let i = 0; i < Math.min(column.length, 64); i++) {
      const v = column[i];
      if (typeof v === 'number' && Number.isFinite(v)) checksum += v;
      // Strings also have a numeric length, but 'x'[0] === 'x', so recursing
      // into one never terminates. A string column contributes via `length`.
      else if (v && typeof v !== 'string' && typeof v.length === 'number')
        sample(v);
    }
  };
  for (const doc of (window.Bokeh && Bokeh.documents) || []) {
    for (const m of Array.from(doc._all_models.values())) {
      if (m.type === 'ColumnDataSource') {
        sources++;
        for (const column of Object.values(m.data)) {
          length += column.length;
          sample(column);
        }
      }
    }
  }
  return {sources, length, checksum};
}"""


@contextmanager
def open_browser() -> Iterator[Browser]:
    """A running browser with no dashboard session open on it.

    ``Dashboard.connect`` opens a session as it hands the browser over; tests
    that assert on which sessions the server holds need to control that
    themselves.
    """
    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        try:
            yield browser
        finally:
            browser.close()


def fingerprint(dash: Dashboard) -> dict:
    return dash.page.evaluate(DATA_FINGERPRINT_JS)


def assert_updating(dash: Dashboard, label: str) -> None:
    """Assert the session's rendered data changes within the update window.

    Polls rather than sleeping out the window: the fake backend emits at 1 Hz,
    so a healthy session passes in about a second instead of always paying the
    full window.
    """
    before = fingerprint(dash)
    wait_until(
        dash,
        lambda: fingerprint(dash) != before,
        label=f"a data update ({label} stopped receiving data updates: {before})",
        timeout_ms=UPDATE_WINDOW_MS,
    )


def wait_until(
    dash: Dashboard,
    condition: Callable[[], bool],
    *,
    label: str,
    timeout_ms: int = PROPAGATION_TIMEOUT_MS,
    interval_ms: int = 250,
) -> None:
    """Poll ``condition`` until true, failing after ``timeout_ms``."""
    waited = 0
    while not condition():
        if waited >= timeout_ms:
            raise AssertionError(f"Timed out after {timeout_ms} ms waiting for {label}")
        dash.page.wait_for_timeout(interval_ms)
        waited += interval_ms
