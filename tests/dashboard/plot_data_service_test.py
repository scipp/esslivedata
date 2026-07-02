# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for PlotDataService and LayerStateMachine."""

from uuid import uuid4

from ess.livedata.dashboard.plot_data_service import (
    LayerId,
    LayerState,
    LayerStateMachine,
    PlotDataService,
)


class FakePlotter:
    """Minimal fake plotter for testing state machine transitions."""

    def __init__(self):
        self._cached_state = None
        self.compute_calls: list[tuple[dict, dict]] = []

    def compute(self, data, **kwargs):
        self._cached_state = data
        self.compute_calls.append((data, kwargs))

    def get_cached_state(self):
        return self._cached_state

    def has_cached_state(self):
        return self._cached_state is not None

    def mark_presenters_dirty(self):
        pass


class TestLayerStateMachineVersionInvariant:
    """Test version invariant: plotter change always increments version."""

    def test_job_started_increments_version_from_waiting_for_job(self):
        """Version increments when job_started called from WAITING_FOR_JOB."""
        state = LayerStateMachine()
        assert state.version == 0
        assert state.state == LayerState.WAITING_FOR_JOB

        plotter = FakePlotter()
        state.job_started(plotter)

        assert state.version == 1
        assert state.plotter is plotter

    def test_job_started_increments_version_from_stopped(self):
        """Version increments when job_started called from STOPPED."""
        state = LayerStateMachine()
        plotter_a = FakePlotter()
        plotter_a.compute({'data': 1})

        state.job_started(plotter_a)
        state.data_arrived()
        state.job_stopped()

        version_after_stop = state.version
        assert state.state == LayerState.STOPPED

        # Simulate workflow restart with new plotter
        plotter_b = FakePlotter()
        state.job_started(plotter_b)

        assert state.version == version_after_stop + 1
        assert state.plotter is plotter_b

    def test_job_started_increments_version_from_ready(self):
        """Version increments when job_started called from READY (workflow restart)."""
        state = LayerStateMachine()
        plotter_a = FakePlotter()
        plotter_a.compute({'data': 1})

        state.job_started(plotter_a)
        state.data_arrived()

        version_after_ready = state.version
        assert state.state == LayerState.READY

        # Simulate workflow restart with new plotter while still running
        plotter_b = FakePlotter()
        state.job_started(plotter_b)

        assert state.version == version_after_ready + 1
        assert state.plotter is plotter_b

    def test_job_started_increments_version_from_waiting_for_data(self):
        """Version increments when plotter replaced while waiting for data."""
        state = LayerStateMachine()
        plotter_a = FakePlotter()
        state.job_started(plotter_a)

        version_after_first_start = state.version
        assert state.state == LayerState.WAITING_FOR_DATA

        # Workflow restarted before data arrived
        plotter_b = FakePlotter()
        state.job_started(plotter_b)

        assert state.version == version_after_first_start + 1
        assert state.plotter is plotter_b

    def test_job_started_increments_version_from_error(self):
        """Version increments when job_started called from ERROR state."""
        state = LayerStateMachine()
        state.error_occurred("test error")

        version_after_error = state.version
        assert state.state == LayerState.ERROR

        plotter = FakePlotter()
        state.job_started(plotter)

        assert state.version == version_after_error + 1
        assert state.plotter is plotter

    def test_job_started_with_same_plotter_still_increments_version(self):
        """Version increments even when called with the same plotter instance."""
        state = LayerStateMachine()
        plotter = FakePlotter()

        state.job_started(plotter)
        version_after_first = state.version

        state.job_stopped()
        state.job_started(plotter)  # Same plotter instance

        assert state.version == version_after_first + 2  # +1 for stop, +1 for start


class TestLayerStateMachineOtherTransitions:
    """Tests for other state transitions that also affect version."""

    def test_data_arrived_increments_version(self):
        """Version increments when data arrives."""
        state = LayerStateMachine()
        plotter = FakePlotter()
        state.job_started(plotter)
        version_before = state.version

        state.data_arrived()

        assert state.version == version_before + 1
        assert state.state == LayerState.READY

    def test_data_arrived_no_op_when_already_ready(self):
        """Subsequent data arrivals don't increment version."""
        state = LayerStateMachine()
        plotter = FakePlotter()
        state.job_started(plotter)
        state.data_arrived()
        version_at_ready = state.version

        # Subsequent data arrivals are no-ops
        state.data_arrived()
        state.data_arrived()

        assert state.version == version_at_ready

    def test_job_stopped_increments_version(self):
        """Version increments when job is stopped."""
        state = LayerStateMachine()
        plotter = FakePlotter()
        state.job_started(plotter)
        state.data_arrived()
        version_before = state.version

        state.job_stopped()

        assert state.version == version_before + 1
        assert state.state == LayerState.STOPPED

    def test_error_occurred_increments_version(self):
        """Version increments when error occurs."""
        state = LayerStateMachine()
        version_before = state.version

        state.error_occurred("test error")

        assert state.version == version_before + 1
        assert state.state == LayerState.ERROR


class TestPlotDataService:
    """Tests for PlotDataService."""

    def test_job_started_creates_layer_if_not_exists(self):
        """job_started creates layer entry if it doesn't exist."""
        service = PlotDataService()
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()

        assert service.get(layer_id) is None

        service.job_started(layer_id, plotter)

        state = service.get(layer_id)
        assert state is not None
        assert state.plotter is plotter
        assert state.version == 1

    def test_job_started_on_existing_layer_increments_version(self):
        """job_started on existing layer increments version."""
        service = PlotDataService()
        layer_id = LayerId(uuid4())
        plotter_a = FakePlotter()
        plotter_a.compute({'data': 1})

        service.job_started(layer_id, plotter_a)
        service.data_arrived(layer_id)
        service.job_stopped(layer_id)

        version_before = service.get(layer_id).version

        plotter_b = FakePlotter()
        service.job_started(layer_id, plotter_b)

        assert service.get(layer_id).version == version_before + 1
        assert service.get(layer_id).plotter is plotter_b


class _Token:
    """Weakref-compatible stand-in for a viewer (real callers are SessionLayer)."""


class TestLayerComputeGate:
    """Tests for the lazy compute gate on LayerStateMachine."""

    def _ready_state(self) -> tuple[LayerStateMachine, FakePlotter]:
        state = LayerStateMachine()
        plotter = FakePlotter()
        state.job_started(plotter)
        return state, plotter

    def test_stash_without_active_token_does_not_flush(self):
        state, plotter = self._ready_state()
        assert state.stash_pending({'k': 1}) is None
        assert plotter.compute_calls == []

    def test_set_active_then_stash_flushes_immediately(self):
        state, plotter = self._ready_state()
        token = _Token()
        state.set_active(token, True)
        task = state.stash_pending({'k': 1})
        assert task is not None
        task.run()
        assert plotter.compute_calls == [({'k': 1}, {'title_resolver': None})]

    def test_set_active_reports_only_zero_to_one_transition(self):
        state, _plotter = self._ready_state()
        token = _Token()
        assert state.set_active(token, True) is True
        # Re-asserting True is not a transition.
        assert state.set_active(token, True) is False
        state.set_active(token, False)
        assert state.set_active(token, True) is True

    def test_has_viewers_tracks_token_count(self):
        state, _plotter = self._ready_state()
        t1, t2 = _Token(), _Token()
        assert not state.has_viewers
        state.set_active(t1, True)
        state.set_active(t2, True)
        assert state.has_viewers
        state.set_active(t1, False)
        assert state.has_viewers
        state.set_active(t2, False)
        assert not state.has_viewers

    def test_multiple_tokens_keep_active_until_last_released(self):
        state, plotter = self._ready_state()
        t1, t2 = _Token(), _Token()
        assert state.set_active(t1, True) is True
        # Second token while already active is not a transition.
        assert state.set_active(t2, True) is False
        task = state.stash_pending({'k': 1})
        assert task is not None
        task.run()
        # Release one; still active — stash flushes immediately.
        state.set_active(t1, False)
        task = state.stash_pending({'k': 2})
        assert task is not None
        task.run()
        # Release the second; gate closes, next stash returns None.
        state.set_active(t2, False)
        assert state.stash_pending({'k': 3}) is None
        # Only the first two computes ran; the third is stashed.
        assert [d for d, _ in plotter.compute_calls] == [{'k': 1}, {'k': 2}]

    def test_release_of_unknown_token_is_noop(self):
        state, _plotter = self._ready_state()
        # Releasing a token never seen does not raise or change anything.
        unknown = _Token()
        assert state.set_active(unknown, False) is False
        assert not state.has_viewers

    def test_stash_returns_no_task_before_job_started(self):
        state = LayerStateMachine()
        token = _Token()
        state.set_active(token, True)
        # No plotter yet → no flush.
        assert state.stash_pending({'k': 1}) is None

    def test_token_auto_released_on_garbage_collection(self):
        """Safety net: if the caller is gc'd without releasing, the gate closes."""
        import gc

        state, _plotter = self._ready_state()
        token = _Token()
        state.set_active(token, True)
        # Active: stash flushes immediately.
        assert state.stash_pending({'k': 1}) is not None
        # Drop the only reference and force gc; finalizer must release the key.
        del token
        gc.collect()
        # Gate is now closed (no active tokens) → next stash returns None.
        assert not state.has_viewers
        assert state.stash_pending({'k': 2}) is None
