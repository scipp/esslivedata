# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Multi-session smoke test: every session sees plots and keeps receiving data.

Dashboard state (orchestrators, DataService, plot layers) is process-global
while widgets and Bokeh documents are per-session, so the classic regression
class is asymmetry between sessions: a plot renders in one session but not
another (#789), a late joiner sees stale or missing data, or activity in one
session stalls delivery to another. Until now this was only covered by the
recurring manual test-plan item "check with 2+ browser sessions" (e.g. the
manual checklists in #1009 and #1043).

This drives the Kafka-free fake backend (seeded from the committed dummy
fixture, updating at 1 Hz) with two Playwright sessions and walks the manual
checklist once: a late-joining session must see the same tabs and populated
plots, both sessions must keep receiving updates, and tab switching in one
session must disturb neither. It is deliberately behavioral -- it observes
rendered ColumnDataSource contents, not internals -- so it stays valid across
delivery-mechanism refactors (#1045, #1046).

Runs locally via ``pytest -m browser`` (excluded from the default run and CI;
skips cleanly where Playwright is absent).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

pytest.importorskip("playwright.sync_api")
from drive_dashboard import Dashboard, _fake_dashboard  # noqa: E402

# How long to watch for a data update before declaring a session stalled. The
# fake backend emits at 1 Hz, so this is a generous margin.
_UPDATE_WINDOW_MS = 6000

# Fingerprint of all rendered ColumnDataSource data in the page: source and
# column-length counts plus a value checksum (sampling nested/typed arrays, so
# image payloads contribute). Any data update changes it.
_DATA_FINGERPRINT_JS = """() => {
  let sources = 0, length = 0, checksum = 0;
  const sample = (column) => {
    for (let i = 0; i < Math.min(column.length, 64); i++) {
      const v = column[i];
      if (typeof v === 'number' && Number.isFinite(v)) checksum += v;
      else if (v && typeof v.length === 'number') sample(v);
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


def _fingerprint(dash: Dashboard) -> dict:
    return dash.page.evaluate(_DATA_FINGERPRINT_JS)


def _assert_updating(dash: Dashboard, label: str) -> None:
    before = _fingerprint(dash)
    dash.page.wait_for_timeout(_UPDATE_WINDOW_MS)
    after = _fingerprint(dash)
    assert after != before, f"{label} stopped receiving data updates: {before}"


@pytest.mark.browser
def test_two_sessions_see_plots_and_keep_receiving_updates():
    with (
        _fake_dashboard("dummy", 5032) as url,
        Dashboard.connect_many(2, url) as (
            first,
            second,
        ),
    ):
        # The late joiner (second) must see the same UI as the first session.
        assert second.tab_names() == first.tab_names()
        assert "Detectors" in second.tab_names()

        first.goto_tab("Detectors")
        second.goto_tab("Detectors")

        # Both sessions render populated plots, including the session that
        # joined after the workflows were already running.
        for label, dash in [("first session", first), ("second session", second)]:
            fp = _fingerprint(dash)
            assert fp["sources"] > 0, f"{label} rendered no data sources"
            assert fp["length"] > 0, f"{label} rendered only empty data sources"

        # Both sessions keep receiving live updates.
        _assert_updating(first, "first session")
        _assert_updating(second, "second session")

        # Tab switching in one session must not stall delivery to either: the
        # switching session must resume on return, the other must be unaffected.
        second.goto_tab("Workflows")
        second.goto_tab("Detectors")
        _assert_updating(second, "second session after tab switch")
        _assert_updating(first, "first session while other session switched tabs")
