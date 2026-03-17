# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Tests for issue #805: Layout-producing DynamicMaps and hook application.

Two bugs are tested:
(a) The `has_layout` guard in `_get_session_composed_plot` checks
    `plotter.get_cached_state()` at call time. If the plotter's cached
    state is transiently None (e.g., plotter being replaced), the guard
    misses the Layout and hooks are applied to a DynamicMap that has
    already been typed as Layout — causing a ValueError.

(b) `_poll_for_plot_updates` bumps `last_seen_version` before the cell
    rebuild. If the rebuild raises, the version is already marked as
    seen, preventing future retry.
"""

from __future__ import annotations

from uuid import uuid4

import holoviews as hv
import pytest

from ess.livedata.dashboard.plot_data_service import LayerId, PlotDataService
from ess.livedata.dashboard.plots import PresenterBase
from ess.livedata.dashboard.session_layer import SessionLayer

hv.extension('bokeh')


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Issue (a): has_layout guard is racy
# ---------------------------------------------------------------------------


class TestHasLayoutGuardRace:
    """
    The has_layout check uses `plotter.get_cached_state()` at call time.
    If the cached state is transiently unavailable while the DynamicMap has
    already resolved to Layout, `.opts(hooks=...)` will fail.
    """

    def test_opts_hooks_fails_on_evaluated_layout_dmap(self):
        """Applying hooks to a DynamicMap that has been evaluated as Layout raises."""
        layout = hv.Layout(
            [hv.Curve([1, 2, 3]).relabel('A'), hv.Curve([4, 5, 6]).relabel('B')]
        )
        pipe = hv.streams.Pipe(data=layout)
        dmap = hv.DynamicMap(lambda data: data, streams=[pipe], cache_size=1)

        # Force evaluation (simulates Bokeh rendering the plot)
        dmap[()]
        assert dmap.type is hv.Layout

        with pytest.raises(ValueError, match="Unexpected option 'hooks' for Layout"):
            dmap.opts(hooks=[lambda plot, element: None])

    def test_opts_hooks_ok_on_unevaluated_layout_dmap(self):
        """Hooks succeed on a fresh (unevaluated) DynamicMap, even with Layout data."""
        layout = hv.Layout(
            [hv.Curve([1, 2, 3]).relabel('A'), hv.Curve([4, 5, 6]).relabel('B')]
        )
        pipe = hv.streams.Pipe(data=layout)
        dmap = hv.DynamicMap(lambda data: data, streams=[pipe], cache_size=1)

        # No evaluation yet — type is unknown, so opts is accepted
        assert dmap.type is None
        dmap.opts(hooks=[lambda plot, element: None])  # should not raise

    def test_has_layout_guard_misses_layout_when_cached_state_is_none(self):
        """
        Reproduce the race: components exist with an evaluated Layout DynamicMap,
        but the plotter's cached state is transiently None (plotter being replaced).

        The `isinstance(plotter.get_cached_state(), hv.Layout)` guard returns False,
        so hooks are applied to the evaluated DynamicMap — which fails.
        """
        layout = hv.Layout(
            [hv.Curve([1, 2, 3]).relabel('A'), hv.Curve([4, 5, 6]).relabel('B')]
        )

        # Step 1: plotter has Layout, components created, DynamicMap evaluated
        plotter = FakePlotter(cached_state=layout)
        layer_id = LayerId(uuid4())
        pds = PlotDataService()
        pds.job_started(layer_id, plotter)
        pds.data_arrived(layer_id)

        state = pds.get(layer_id)
        session_layer = SessionLayer(layer_id=layer_id, last_seen_version=state.version)
        session_layer.ensure_components(state)
        assert session_layer.dmap is not None

        # Simulate Bokeh having rendered this DynamicMap
        session_layer.dmap[()]
        assert session_layer.dmap.type is hv.Layout

        # Step 2: plotter's cached state becomes None (e.g., new plotter arriving
        # via job_started, or background thread resetting state)
        plotter._cached_state = None

        # Step 3: the has_layout guard evaluates to False
        has_layout = isinstance(plotter.get_cached_state(), hv.Layout)
        assert not has_layout, "Guard fails to detect Layout when cached state is None"

        # Step 4: hooks applied to the evaluated Layout DynamicMap → ERROR
        with pytest.raises(ValueError, match="Unexpected option 'hooks' for Layout"):
            session_layer.dmap.opts(hooks=[lambda plot, element: None])

    def test_has_layout_guard_works_when_cached_state_is_layout(self):
        """
        When the guard has access to the Layout cached state, it correctly
        skips hook application.
        """
        layout = hv.Layout(
            [hv.Curve([1, 2, 3]).relabel('A'), hv.Curve([4, 5, 6]).relabel('B')]
        )
        plotter = FakePlotter(cached_state=layout)

        has_layout = isinstance(plotter.get_cached_state(), hv.Layout)
        assert has_layout, "Guard should detect Layout when cached state is available"


# ---------------------------------------------------------------------------
# Issue (b): last_seen_version bumped before rebuild → no recovery
# ---------------------------------------------------------------------------


class TestVersionBumpBeforeRebuild:
    """
    `_poll_for_plot_updates` sets `session_layer.last_seen_version` at
    line 1055 before the rebuild at line 1068. If the rebuild fails,
    the version is already marked as "seen" and the cell is stuck.
    """

    def test_version_bumped_before_rebuild_prevents_retry(self):
        """
        Simulate the poll+rebuild sequence to show that a failed rebuild
        leaves the version as "seen", preventing future retries.

        This reproduces issue (b): the cell remains as a placeholder
        because no subsequent poll cycle will detect a version mismatch.
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

        # Session layer created on first poll (version=0 initially)
        session_layer = SessionLayer(layer_id=layer_id, last_seen_version=0)

        # --- Simulate _poll_for_plot_updates ---
        # The poll loop bumps version BEFORE rebuild:
        assert state.version != session_layer.last_seen_version
        session_layer.last_seen_version = state.version  # line 1055

        # Now simulate the rebuild failing (e.g., ValueError from hooks on Layout)
        # In the actual code, this would be caught by SessionUpdater's exception handler
        rebuild_failed = False
        try:
            session_layer.ensure_components(state)
            # Simulate Bokeh evaluation + hook application failure
            if session_layer.dmap is not None:
                session_layer.dmap[()]
                session_layer.dmap.opts(hooks=[lambda p, e: None])
        except ValueError:
            rebuild_failed = True

        assert rebuild_failed, "Rebuild should have failed"

        # --- Next poll cycle ---
        # The version hasn't changed (no new state transitions), so the poll
        # loop doesn't detect a mismatch and won't retry the rebuild.
        current_state = pds.get(layer_id)
        version_mismatch = current_state.version != session_layer.last_seen_version

        assert not version_mismatch, (
            "Version should match (already bumped), so poll won't retry the rebuild. "
            "The cell is stuck as a placeholder forever."
        )

    def test_version_bump_after_rebuild_would_allow_retry(self):
        """
        Demonstrate that if the version were bumped AFTER a successful rebuild
        (the fix), a failed rebuild would leave the version stale, allowing
        the next poll cycle to retry.
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

        # --- Simulate FIXED poll: version bumped only on success ---
        assert state.version != session_layer.last_seen_version

        rebuild_succeeded = False
        try:
            session_layer.ensure_components(state)
            if session_layer.dmap is not None:
                session_layer.dmap[()]
                session_layer.dmap.opts(hooks=[lambda p, e: None])
            rebuild_succeeded = True
        except ValueError:
            pass

        # Only bump version if rebuild succeeded
        if rebuild_succeeded:
            session_layer.last_seen_version = state.version

        assert not rebuild_succeeded, "Rebuild should have failed"

        # --- Next poll cycle ---
        # Version was NOT bumped because rebuild failed → mismatch → retry possible
        current_state = pds.get(layer_id)
        version_mismatch = current_state.version != session_layer.last_seen_version

        assert version_mismatch, (
            "Version should NOT match (wasn't bumped after failure), "
            "so the next poll will retry the rebuild."
        )
