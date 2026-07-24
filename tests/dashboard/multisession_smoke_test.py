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

Runs via ``pytest -m browser`` (excluded from the default run; CI runs them
via ``tox -e browser``; skips cleanly where Playwright is absent).
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
from tests.helpers.browser import (
    Dashboard,
    assert_updating,
    fake_dashboard,
    fingerprint,
)


@pytest.mark.browser
def test_two_sessions_see_plots_and_keep_receiving_updates():
    with (
        fake_dashboard("dummy", 5032) as url,
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
            fp = fingerprint(dash)
            assert fp["sources"] > 0, f"{label} rendered no data sources"
            assert fp["length"] > 0, f"{label} rendered only empty data sources"

        # Both sessions keep receiving live updates.
        assert_updating(first, "first session")
        assert_updating(second, "second session")

        # Tab switching in one session must not stall delivery to either: the
        # switching session must resume on return, the other must be unaffected.
        second.goto_tab("Workflows")
        second.goto_tab("Detectors")
        assert_updating(second, "second session after tab switch")
        assert_updating(first, "first session while other session switched tabs")
