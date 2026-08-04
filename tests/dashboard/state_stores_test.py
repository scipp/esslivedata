# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for PlotDataService and LayerSnapshot."""

import weakref

from ess.livedata.dashboard.plot_data_service import (
    LayerId,
    LayerSnapshot,
    LayerState,
    PlotDataService,
)
from ess.livedata.dashboard.plots import PresenterBase


class FakePlotter:
    """Fake plotter that caches state and tracks presenters for testing."""

    def __init__(self, state=None, *, scale: str = 'linear'):
        self._cached_state = state
        self.scale = scale
        self._presenters: weakref.WeakSet = weakref.WeakSet()

    def compute(self, data):
        self._cached_state = data
        self.mark_presenters_dirty()
        return data

    def get_cached_state(self):
        return self._cached_state

    def has_cached_state(self):
        return self._cached_state is not None

    def create_presenter(self):
        presenter = FakePresenter(self, scale=self.scale)
        self._presenters.add(presenter)
        return presenter

    def mark_presenters_dirty(self):
        """Mark all registered presenters as having pending updates."""
        for presenter in self._presenters:
            presenter._mark_dirty()


class FakePresenter(PresenterBase):
    """Fake presenter that implements the Presenter protocol."""

    def __init__(self, plotter, *, scale: str = 'linear'):
        super().__init__(plotter)
        self.scale = scale

    def present(self, pipe):
        import holoviews as hv

        def render(data):
            return hv.Curve([])

        return hv.DynamicMap(render, streams=[pipe], cache_size=1)


class TestPlotDataService:
    """Basic PlotDataService tests for layer management."""

    def test_get_unknown_layer_returns_none(self):
        service = PlotDataService()
        assert service.get(LayerId('unknown')) is None

    def test_job_started_and_get(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        plotter = FakePlotter(state={'plot': 'data'})
        service.job_started(layer_id, plotter)
        state = service.get(layer_id)

        assert state is not None
        assert state.plotter is plotter
        assert state.plotter.get_cached_state() == {'plot': 'data'}
        assert state.error_message is None
        assert state.state == LayerState.WAITING_FOR_DATA

    def test_job_started_replaces_previous_plotter(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        plotter1 = FakePlotter(state='v1')
        service.job_started(layer_id, plotter1)
        assert service.get(layer_id).plotter is plotter1

        plotter2 = FakePlotter(state='v2')
        service.job_started(layer_id, plotter2)
        state = service.get(layer_id)

        assert state.plotter is plotter2
        # Starting a new job resets error state
        assert state.error_message is None
        assert state.state == LayerState.WAITING_FOR_DATA

    def test_error_occurred_marks_presenters_dirty(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        plotter = FakePlotter(state='data')
        service.job_started(layer_id, plotter)

        # Create a presenter so we can track dirty flag
        presenter = plotter.create_presenter()
        presenter._dirty = False  # Reset after creation

        service.error_occurred(layer_id, 'Something went wrong')

        assert service.get(layer_id).error_message == 'Something went wrong'
        assert presenter.has_pending_update()

    def test_job_stopped_marks_presenters_dirty(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        plotter = FakePlotter(state='data')
        service.job_started(layer_id, plotter)

        # Create a presenter so we can track dirty flag
        presenter = plotter.create_presenter()
        presenter._dirty = False  # Reset after creation

        service.job_stopped(layer_id)

        assert service.get(layer_id).state == LayerState.STOPPED
        assert presenter.has_pending_update()

    def test_error_occurred_on_nonexistent_layer_creates_it(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        # Error on nonexistent layer creates it in ERROR state
        service.error_occurred(layer_id, 'Error occurred')

        state = service.get(layer_id)
        assert state.error_message == 'Error occurred'
        assert state.plotter is None
        assert state.state == LayerState.ERROR

    def test_job_stopped_on_nonexistent_layer_is_noop(self):
        """Calling job_stopped on a layer that doesn't exist is a no-op.

        With the state machine model, you can't stop a job that hasn't started.
        """
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        # Stop before job started - should be a no-op
        service.job_stopped(layer_id)

        assert service.get(layer_id) is None

    def test_remove_layer(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')
        plotter = FakePlotter(state='v1')
        service.job_started(layer_id, plotter)

        service.remove(layer_id)

        assert service.get(layer_id) is None

    def test_remove_unknown_layer_is_noop(self):
        service = PlotDataService()
        service.remove(LayerId('unknown'))

    def test_clear(self):
        service = PlotDataService()
        plotter = FakePlotter(state='v1')
        service.job_started(LayerId('layer-1'), plotter)

        service.clear()

        assert service.get(LayerId('layer-1')) is None


class TestLayerSnapshot:
    """Tests for LayerSnapshot's pure state transitions."""

    def test_initial_state_is_waiting_for_data(self):
        snapshot = LayerSnapshot()
        assert snapshot.state == LayerState.WAITING_FOR_DATA
        assert snapshot.version == 0
        assert snapshot.plotter is None
        assert snapshot.error_message is None

    def test_transitions_do_not_mutate_the_receiver(self):
        """The whole point of the type: readers holding one are never surprised."""
        original = LayerSnapshot().job_started(FakePlotter(state='data'))

        updated = original.data_arrived()

        assert updated is not original
        assert updated.state == LayerState.READY
        assert original.state == LayerState.WAITING_FOR_DATA
        assert original.version == 1

    def test_job_started_transitions_to_waiting_for_data(self):
        plotter = FakePlotter(state=None)

        snapshot = LayerSnapshot().job_started(plotter)

        assert snapshot.state == LayerState.WAITING_FOR_DATA
        assert snapshot.version == 1
        assert snapshot.plotter is plotter

    def test_data_arrived_transitions_to_ready(self):
        plotter = FakePlotter(state='data')
        snapshot = LayerSnapshot().job_started(plotter)

        snapshot = snapshot.data_arrived()

        assert snapshot.state == LayerState.READY
        assert snapshot.version == 2
        assert snapshot.plotter is plotter

    def test_job_stopped_from_waiting_for_data(self):
        snapshot = LayerSnapshot().job_started(FakePlotter(state=None))

        snapshot = snapshot.job_stopped()

        assert snapshot.state == LayerState.STOPPED
        assert snapshot.version == 2

    def test_job_stopped_from_ready(self):
        snapshot = LayerSnapshot().job_started(FakePlotter(state='data'))
        snapshot = snapshot.data_arrived()

        snapshot = snapshot.job_stopped()

        assert snapshot.state == LayerState.STOPPED
        assert snapshot.version == 3

    def test_error_occurred_from_any_state(self):
        snapshot = LayerSnapshot().error_occurred("test error")

        assert snapshot.state == LayerState.ERROR
        assert snapshot.error_message == "test error"
        assert snapshot.version == 1

    def test_job_started_from_stopped_restarts(self):
        snapshot = LayerSnapshot().job_started(FakePlotter(state='data'))
        snapshot = snapshot.data_arrived().job_stopped()
        assert snapshot.state == LayerState.STOPPED

        plotter2 = FakePlotter(state=None)
        snapshot = snapshot.job_started(plotter2)

        assert snapshot.state == LayerState.WAITING_FOR_DATA
        assert snapshot.plotter is plotter2
        assert snapshot.version == 4

    def test_job_started_from_error_recovers(self):
        snapshot = LayerSnapshot().error_occurred("some error")
        assert snapshot.state == LayerState.ERROR

        plotter = FakePlotter(state=None)
        snapshot = snapshot.job_started(plotter)

        assert snapshot.state == LayerState.WAITING_FOR_DATA
        assert snapshot.error_message is None
        assert snapshot.version == 2

    def test_job_started_from_ready_restarts(self):
        """Test restart while in READY state (workflow restart with data)."""
        snapshot = LayerSnapshot().job_started(FakePlotter(state='data'))
        snapshot = snapshot.data_arrived()
        assert snapshot.state == LayerState.READY

        plotter2 = FakePlotter(state=None)
        snapshot = snapshot.job_started(plotter2)

        assert snapshot.state == LayerState.WAITING_FOR_DATA
        assert snapshot.plotter is plotter2
        assert snapshot.version == 3

    def test_invalid_data_arrived_from_ready_is_ignored(self):
        snapshot = LayerSnapshot().job_started(FakePlotter(state='data'))
        snapshot = snapshot.data_arrived()

        assert snapshot.data_arrived() is None  # Invalid: already in READY

    def test_invalid_job_stopped_when_already_stopped_is_ignored(self):
        snapshot = LayerSnapshot().job_stopped()

        assert snapshot.job_stopped() is None  # Invalid: already stopped

    def test_data_arrived_while_stopped_is_silent_noop(self):
        snapshot = LayerSnapshot().job_stopped()

        # Retained data rendered while stopped
        assert snapshot.data_arrived() is None

    def test_has_displayable_plot_ready_with_data(self):
        snapshot = LayerSnapshot().job_started(FakePlotter(state='data'))
        snapshot = snapshot.data_arrived()

        assert snapshot.has_displayable_plot() is True

    def test_has_displayable_plot_stopped_with_data(self):
        snapshot = LayerSnapshot().job_started(FakePlotter(state='data'))
        snapshot = snapshot.data_arrived().job_stopped()

        assert snapshot.has_displayable_plot() is True

    def test_has_displayable_plot_waiting_for_data_is_false(self):
        snapshot = LayerSnapshot().job_started(FakePlotter(state=None))

        assert snapshot.has_displayable_plot() is False

    def test_has_displayable_plot_error_is_false(self):
        snapshot = LayerSnapshot().error_occurred("error")

        assert snapshot.has_displayable_plot() is False


class TestPlotDataServiceStateMachine:
    """Tests for PlotDataService state machine transitions."""

    def test_job_started_creates_layer(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')
        plotter = FakePlotter(state='data')

        service.job_started(layer_id, plotter)

        state = service.get(layer_id)
        assert state is not None
        assert state.state == LayerState.WAITING_FOR_DATA
        assert state.plotter is plotter

    def test_data_arrived_transitions_to_ready(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')
        plotter = FakePlotter(state='data')
        service.job_started(layer_id, plotter)

        service.data_arrived(layer_id)

        state = service.get(layer_id)
        assert state.state == LayerState.READY
        assert state.version == 2

    def test_job_stopped_transitions_to_stopped(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')
        plotter = FakePlotter(state='data')
        service.job_started(layer_id, plotter)
        service.data_arrived(layer_id)

        service.job_stopped(layer_id)

        state = service.get(layer_id)
        assert state.state == LayerState.STOPPED

    def test_error_occurred_transitions_to_error(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        service.error_occurred(layer_id, "test error")

        state = service.get(layer_id)
        assert state.state == LayerState.ERROR
        assert state.error_message == "test error"

    def test_version_increments_on_transitions(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')
        plotter = FakePlotter(state='data')

        service.job_started(layer_id, plotter)
        assert service.get(layer_id).version == 1

        service.data_arrived(layer_id)
        assert service.get(layer_id).version == 2

        service.job_stopped(layer_id)
        assert service.get(layer_id).version == 3

        # Restart
        service.job_started(layer_id, plotter)
        assert service.get(layer_id).version == 4
