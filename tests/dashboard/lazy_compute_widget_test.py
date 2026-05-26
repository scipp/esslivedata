# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Tests for the PlotGridTabs static helpers that drive lazy-compute interest.

These exercise only the helpers themselves, decoupled from Panel widget
construction, so the suite remains runnable in environments where the wider
``plot_grid_tabs_test.py`` is broken by upstream Panel deprecations.
"""

from ess.livedata.dashboard.session_layer import SessionLayer
from ess.livedata.dashboard.widgets.plot_grid_tabs import PlotGridTabs


class _FakePlotter:
    def __init__(self) -> None:
        self.calls: list[tuple[object, bool]] = []

    def set_active(self, token: object, active: bool) -> None:
        self.calls.append((token, active))


def _make_session_layer() -> SessionLayer:
    return SessionLayer(layer_id='layer-1', last_seen_version=0)  # type: ignore[arg-type]


class TestSyncPlotterInterest:
    def test_first_call_records_active_plotter(self):
        plotter = _FakePlotter()
        sl = _make_session_layer()
        PlotGridTabs._sync_plotter_interest(sl, plotter, True)
        assert sl.active_plotter is plotter
        assert plotter.calls == [(sl, True)]

    def test_inactive_call_still_records_plotter(self):
        plotter = _FakePlotter()
        sl = _make_session_layer()
        PlotGridTabs._sync_plotter_interest(sl, plotter, False)
        assert sl.active_plotter is plotter
        assert plotter.calls == [(sl, False)]

    def test_plotter_replacement_releases_old_acquires_new(self):
        old = _FakePlotter()
        new = _FakePlotter()
        sl = _make_session_layer()
        PlotGridTabs._sync_plotter_interest(sl, old, True)
        PlotGridTabs._sync_plotter_interest(sl, new, True)
        assert sl.active_plotter is new
        assert old.calls == [(sl, True), (sl, False)]
        assert new.calls == [(sl, True)]

    def test_none_plotter_does_not_call_set_active(self):
        sl = _make_session_layer()
        PlotGridTabs._sync_plotter_interest(sl, None, True)
        assert sl.active_plotter is None  # unchanged

    def test_none_plotter_after_real_plotter_releases(self):
        plotter = _FakePlotter()
        sl = _make_session_layer()
        PlotGridTabs._sync_plotter_interest(sl, plotter, True)
        PlotGridTabs._sync_plotter_interest(sl, None, True)
        assert sl.active_plotter is None
        assert plotter.calls == [(sl, True), (sl, False)]

    def test_plotter_without_set_active_is_tolerated(self):
        sl = _make_session_layer()
        sentinel = object()  # no set_active method
        PlotGridTabs._sync_plotter_interest(sl, sentinel, True)
        assert sl.active_plotter is sentinel


class TestReleasePlotterInterest:
    def test_release_clears_active_plotter(self):
        plotter = _FakePlotter()
        sl = _make_session_layer()
        PlotGridTabs._sync_plotter_interest(sl, plotter, True)
        PlotGridTabs._release_plotter_interest(sl)
        assert sl.active_plotter is None
        assert plotter.calls == [(sl, True), (sl, False)]

    def test_release_when_no_plotter_is_noop(self):
        sl = _make_session_layer()
        PlotGridTabs._release_plotter_interest(sl)
        assert sl.active_plotter is None
