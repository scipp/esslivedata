# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for PlotDataService and LayerSnapshot."""

from uuid import uuid4

from ess.livedata.dashboard.plot_data_service import (
    LayerId,
    LayerSnapshot,
    LayerState,
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


class TestLayerSnapshotVersionInvariant:
    """Test version invariant: plotter change always increments version."""

    def test_job_started_increments_version_from_initial_state(self):
        """Version increments when job_started called on a fresh snapshot."""
        state = LayerSnapshot()
        assert state.version == 0
        assert state.state == LayerState.WAITING_FOR_DATA

        plotter = FakePlotter()
        state = state.job_started(plotter)

        assert state.version == 1
        assert state.plotter is plotter

    def test_job_started_increments_version_from_stopped(self):
        """Version increments when job_started called from STOPPED."""
        plotter_a = FakePlotter()
        plotter_a.compute({'data': 1})

        state = LayerSnapshot().job_started(plotter_a)
        state = state.data_arrived().job_stopped()

        version_after_stop = state.version
        assert state.state == LayerState.STOPPED

        # Simulate workflow restart with new plotter
        plotter_b = FakePlotter()
        state = state.job_started(plotter_b)

        assert state.version == version_after_stop + 1
        assert state.plotter is plotter_b

    def test_job_started_increments_version_from_ready(self):
        """Version increments when job_started called from READY (workflow restart)."""
        plotter_a = FakePlotter()
        plotter_a.compute({'data': 1})

        state = LayerSnapshot().job_started(plotter_a)
        state = state.data_arrived()

        version_after_ready = state.version
        assert state.state == LayerState.READY

        # Simulate workflow restart with new plotter while still running
        plotter_b = FakePlotter()
        state = state.job_started(plotter_b)

        assert state.version == version_after_ready + 1
        assert state.plotter is plotter_b

    def test_job_started_increments_version_from_waiting_for_data(self):
        """Version increments when plotter replaced while waiting for data."""
        state = LayerSnapshot().job_started(FakePlotter())

        version_after_first_start = state.version
        assert state.state == LayerState.WAITING_FOR_DATA

        # Workflow restarted before data arrived
        plotter_b = FakePlotter()
        state = state.job_started(plotter_b)

        assert state.version == version_after_first_start + 1
        assert state.plotter is plotter_b

    def test_job_started_increments_version_from_error(self):
        """Version increments when job_started called from ERROR state."""
        state = LayerSnapshot().error_occurred("test error")

        version_after_error = state.version
        assert state.state == LayerState.ERROR

        plotter = FakePlotter()
        state = state.job_started(plotter)

        assert state.version == version_after_error + 1
        assert state.plotter is plotter

    def test_job_started_with_same_plotter_still_increments_version(self):
        """Version increments even when called with the same plotter instance."""
        plotter = FakePlotter()

        state = LayerSnapshot().job_started(plotter)
        version_after_first = state.version

        state = state.job_stopped()
        state = state.job_started(plotter)  # Same plotter instance

        assert state.version == version_after_first + 2  # +1 for stop, +1 for start


class TestLayerSnapshotOtherTransitions:
    """Tests for other state transitions that also affect version."""

    def test_data_arrived_increments_version(self):
        """Version increments when data arrives."""
        state = LayerSnapshot().job_started(FakePlotter())
        version_before = state.version

        state = state.data_arrived()

        assert state.version == version_before + 1
        assert state.state == LayerState.READY

    def test_data_arrived_no_op_when_already_ready(self):
        """Subsequent data arrivals are rejected, leaving the snapshot in place."""
        state = LayerSnapshot().job_started(FakePlotter())
        state = state.data_arrived()

        # Subsequent data arrivals are no-ops
        assert state.data_arrived() is None
        assert state.data_arrived() is None

    def test_job_stopped_increments_version(self):
        """Version increments when job is stopped."""
        state = LayerSnapshot().job_started(FakePlotter())
        state = state.data_arrived()
        version_before = state.version

        state = state.job_stopped()

        assert state.version == version_before + 1
        assert state.state == LayerState.STOPPED

    def test_error_occurred_increments_version(self):
        """Version increments when error occurs."""
        state = LayerSnapshot()
        version_before = state.version

        state = state.error_occurred("test error")

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


class TestPlotDataServiceVersion:
    """Aggregate version counter gating each session's poll pass.

    Only transitions that actually took effect may advance it: the counter is
    read once per tick to decide whether a session pays a full hold+freeze
    pass, so a counter that moved without any layer changing costs every
    session a pointless pass.
    """

    def test_job_started_advances_version(self):
        service = PlotDataService()
        before = service.version

        service.job_started(LayerId(uuid4()), FakePlotter())

        assert service.version == before + 1

    def test_first_data_arrived_advances_version(self):
        service = PlotDataService()
        layer_id = LayerId(uuid4())
        service.job_started(layer_id, FakePlotter())
        before = service.version

        service.data_arrived(layer_id)

        assert service.version == before + 1

    def test_further_data_arrived_does_not_advance_version(self):
        """This is what keeps the poll gate quiet under steady data flow: once
        a layer is READY, every subsequent data message is a no-op transition
        and must not arm every session's poll pass."""
        service = PlotDataService()
        layer_id = LayerId(uuid4())
        service.job_started(layer_id, FakePlotter())
        service.data_arrived(layer_id)
        version_at_ready = service.version

        service.data_arrived(layer_id)
        service.data_arrived(layer_id)

        assert service.version == version_at_ready

    def test_job_stopped_advances_version(self):
        service = PlotDataService()
        layer_id = LayerId(uuid4())
        service.job_started(layer_id, FakePlotter())
        service.data_arrived(layer_id)
        before = service.version

        service.job_stopped(layer_id)

        assert service.version == before + 1

    def test_error_occurred_advances_version(self):
        service = PlotDataService()
        layer_id = LayerId(uuid4())
        service.job_started(layer_id, FakePlotter())
        before = service.version

        service.error_occurred(layer_id, 'boom')

        assert service.version == before + 1

    def test_rejected_transition_does_not_advance_version(self):
        service = PlotDataService()
        layer_id = LayerId(uuid4())
        service.job_started(layer_id, FakePlotter())
        service.job_stopped(layer_id)
        version_at_stopped = service.version

        service.job_stopped(layer_id)  # rejected: not valid from STOPPED

        assert service.version == version_at_stopped

    def test_transitions_on_unknown_layer_do_not_advance_version(self):
        service = PlotDataService()
        before = service.version

        service.data_arrived(LayerId(uuid4()))
        service.job_stopped(LayerId(uuid4()))

        assert service.version == before


class _Token:
    """Weakref-compatible stand-in for a viewer (real callers are SessionLayer)."""


class TestLayerViewerGate:
    """Tests for the viewer gate on PlotDataService.

    The gate tracks viewer interest tokens, keyed per layer. It is held apart
    from the lifecycle snapshots because interest is per (session, layer)
    while lifecycle state is per layer. On the 0→1 transition (first viewer)
    the orchestrator rebuilds the layer from DataService; on 1→0 (last viewer
    released), the layer is skipped at frame flushes.
    """

    def _started_layer(self) -> tuple[PlotDataService, LayerId]:
        service = PlotDataService()
        layer_id = LayerId(uuid4())
        service.job_started(layer_id, FakePlotter())
        return service, layer_id

    def test_set_active_reports_zero_to_one_transition(self):
        """set_active returns True on the 0→1 transition only."""
        service, layer_id = self._started_layer()
        token = _Token()
        assert service.set_active(layer_id, token, True) is True
        # Re-activating the same token is not a transition.
        assert service.set_active(layer_id, token, True) is False
        service.set_active(layer_id, token, False)
        assert service.set_active(layer_id, token, True) is True

    def test_has_viewers_tracks_active_tokens(self):
        """has_viewers reflects whether any viewer holds a token."""
        service, layer_id = self._started_layer()
        t1, t2 = _Token(), _Token()
        assert not service.has_viewers(layer_id)
        service.set_active(layer_id, t1, True)
        service.set_active(layer_id, t2, True)
        assert service.has_viewers(layer_id)
        service.set_active(layer_id, t1, False)
        assert service.has_viewers(layer_id)
        service.set_active(layer_id, t2, False)
        assert not service.has_viewers(layer_id)

    def test_multiple_tokens_keep_layer_active(self):
        """Layer stays active while any viewer holds a token."""
        service, layer_id = self._started_layer()
        t1, t2 = _Token(), _Token()
        assert service.set_active(layer_id, t1, True) is True
        # Second token while already active is not a transition.
        assert service.set_active(layer_id, t2, True) is False
        assert service.has_viewers(layer_id)
        # Release one; still active (t2 holds it).
        service.set_active(layer_id, t1, False)
        assert service.has_viewers(layer_id)
        # Release the second; gate closes.
        service.set_active(layer_id, t2, False)
        assert not service.has_viewers(layer_id)

    def test_gate_is_per_layer(self):
        """A viewer on one layer must not open another layer's gate."""
        service, layer_id = self._started_layer()
        other_id = LayerId(uuid4())
        service.job_started(other_id, FakePlotter())
        token = _Token()

        service.set_active(layer_id, token, True)

        assert service.has_viewers(layer_id)
        assert not service.has_viewers(other_id)

    def test_release_of_unknown_token_is_noop(self):
        """Releasing a token never acquired is safe."""
        service, layer_id = self._started_layer()
        unknown = _Token()
        assert service.set_active(layer_id, unknown, False) is False
        assert not service.has_viewers(layer_id)

    def test_has_viewers_on_unknown_layer_is_false(self):
        service = PlotDataService()
        assert not service.has_viewers(LayerId(uuid4()))

    def test_remove_drops_viewer_gate(self):
        """Removing a layer must not leave its gate open for a recreated id."""
        service, layer_id = self._started_layer()
        token = _Token()
        service.set_active(layer_id, token, True)

        service.remove(layer_id)

        assert not service.has_viewers(layer_id)

    def test_token_auto_released_on_garbage_collection(self):
        """Finalizer closes gate if caller is gc'd without explicit release."""
        import gc

        service, layer_id = self._started_layer()
        token = _Token()
        service.set_active(layer_id, token, True)
        assert service.has_viewers(layer_id)
        # Drop the only reference and force gc; finalizer must release the key.
        del token
        gc.collect()
        # Gate is now closed.
        assert not service.has_viewers(layer_id)


class TestSnapshotIdentity:
    """Snapshot identity is the orchestrator's staleness check.

    ``PlotOrchestrator`` captures a snapshot before a pull and, after compute,
    re-reads it and compares by identity to decide whether its result is still
    current. That only works if an effective transition always yields a new
    object and a rejected one never does.
    """

    def test_effective_transition_replaces_snapshot(self):
        service = PlotDataService()
        layer_id = LayerId(uuid4())
        service.job_started(layer_id, FakePlotter())
        before = service.get(layer_id)

        service.data_arrived(layer_id)

        assert service.get(layer_id) is not before

    def test_rejected_transition_keeps_snapshot(self):
        service = PlotDataService()
        layer_id = LayerId(uuid4())
        service.job_started(layer_id, FakePlotter())
        service.data_arrived(layer_id)
        before = service.get(layer_id)

        service.data_arrived(layer_id)  # no-op: already READY

        assert service.get(layer_id) is before

    def test_plotter_swap_replaces_snapshot(self):
        """The pairing #1060 was about: a swap must invalidate a held capture."""
        service = PlotDataService()
        layer_id = LayerId(uuid4())
        service.job_started(layer_id, FakePlotter())
        captured = service.get(layer_id)

        service.job_started(layer_id, FakePlotter())

        current = service.get(layer_id)
        assert current is not captured
        assert captured.plotter is not current.plotter
        # The capture still describes the state it was taken in.
        assert captured.version == 1
        assert current.version == 2
