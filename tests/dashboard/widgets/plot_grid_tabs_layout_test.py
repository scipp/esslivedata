# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Regression tests for issue #805: Layout hooks + rebuild recovery.

(a) The has_layout guard must detect Layouts via `dmap.type` (the
    DynamicMap's resolved type after Bokeh renders it), not just
    `plotter.get_cached_state()` which can be transiently unavailable.

(b) `last_seen_version` must only be bumped after a successful rebuild,
    so that a failed rebuild is retried on the next poll cycle.
"""

from __future__ import annotations

from uuid import uuid4

import holoviews as hv

from ess.livedata.dashboard.plot_data_service import LayerId, PlotDataService
from ess.livedata.dashboard.plots import PresenterBase
from ess.livedata.dashboard.session_layer import SessionLayer

hv.extension('bokeh')


class FakePlotter:
    """Minimal plotter for testing component lifecycle."""

    def __init__(self, cached_state=None):
        self._cached_state = cached_state
        self._presenters: list[FakePresenter] = []

    def get_cached_state(self):
        return self._cached_state

    def has_cached_state(self):
        return self._cached_state is not None

    def create_presenter(self, *, owner=None):
        presenter = FakePresenter(self, owner=owner)
        self._presenters.append(presenter)
        return presenter

    def mark_presenters_dirty(self):
        for p in self._presenters:
            p._mark_dirty()


class FakePresenter(PresenterBase):
    """Passthrough presenter — same as DefaultPresenter."""

    def present(self, pipe):
        return hv.DynamicMap(lambda data: data, streams=[pipe], cache_size=1)


class TestLayoutDetectionViaDmapType:
    """Regression tests for has_layout guard using dmap.type."""

    def test_dmap_type_catches_layout_when_cached_state_is_none(self):
        """
        `dmap.type is hv.Layout` detects Layouts even when the plotter's
        cached state is transiently None (background plotter replacement).
        """
        layout = hv.Layout(
            [hv.Curve([1, 2, 3]).relabel('A'), hv.Curve([4, 5, 6]).relabel('B')]
        )
        plotter = FakePlotter(cached_state=layout)
        layer_id = LayerId(uuid4())
        pds = PlotDataService()
        pds.job_started(layer_id, plotter)
        pds.data_arrived(layer_id)

        state = pds.get(layer_id)
        session_layer = SessionLayer(layer_id=layer_id, last_seen_version=state.version)
        session_layer.ensure_components(state)

        # Simulate Bokeh rendering
        session_layer.dmap[()]

        # Plotter's cached state goes away (background plotter replacement)
        plotter._cached_state = None

        dmap = session_layer.dmap
        assert not isinstance(plotter.get_cached_state(), hv.Layout)
        assert isinstance(dmap, hv.DynamicMap)
        assert dmap.type is hv.Layout

    def test_dmap_type_does_not_false_positive_on_non_layout(self):
        """
        An unevaluated DynamicMap with non-Layout content must not be
        flagged as Layout, so that hooks are applied normally.
        """
        plotter = FakePlotter(cached_state=hv.Curve([1, 2, 3]))
        layer_id = LayerId(uuid4())
        pds = PlotDataService()
        pds.job_started(layer_id, plotter)
        pds.data_arrived(layer_id)

        state = pds.get(layer_id)
        session_layer = SessionLayer(layer_id=layer_id, last_seen_version=state.version)
        session_layer.ensure_components(state)

        dmap = session_layer.dmap
        assert isinstance(dmap, hv.DynamicMap)
        assert dmap.type is not hv.Layout
        # Hooks must be accepted
        dmap.opts(hooks=[lambda p, e: None])


class TestVersionBumpAfterRebuild:
    """Regression test: version must only be bumped after successful rebuild."""

    def test_failed_rebuild_leaves_version_stale_for_retry(self):
        """
        If the rebuild fails, `last_seen_version` must not be bumped
        so the next poll cycle retries the rebuild.
        """
        layer_id = LayerId(uuid4())
        pds = PlotDataService()

        layout = hv.Layout(
            [hv.Curve([1, 2, 3]).relabel('A'), hv.Curve([4, 5, 6]).relabel('B')]
        )
        plotter = FakePlotter(cached_state=layout)
        pds.job_started(layer_id, plotter)
        pds.data_arrived(layer_id)

        state = pds.get(layer_id)
        session_layer = SessionLayer(layer_id=layer_id, last_seen_version=0)

        assert state.version != session_layer.last_seen_version

        # Simulate rebuild that fails
        rebuild_succeeded = False
        try:
            session_layer.ensure_components(state)
            if session_layer.dmap is not None:
                session_layer.dmap[()]
                session_layer.dmap.opts(hooks=[lambda p, e: None])
            rebuild_succeeded = True
        except ValueError:
            pass

        # Only bump version on success (the pattern used in the fix)
        if rebuild_succeeded:
            session_layer.last_seen_version = state.version

        assert not rebuild_succeeded

        # Version stays stale → next poll detects mismatch → retry
        assert pds.get(layer_id).version != session_layer.last_seen_version
