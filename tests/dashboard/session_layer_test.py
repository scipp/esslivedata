# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for SessionLayer."""

import weakref
from uuid import uuid4

import holoviews as hv
import pytest

from ess.livedata.dashboard.plot_data_service import LayerId, PlotDataService
from ess.livedata.dashboard.plots import PresenterBase
from ess.livedata.dashboard.session_layer import SessionLayer


class FakePlot:
    """Fake plot state for testing."""

    pass


class FakePlotter:
    """Fake plotter for testing."""

    def __init__(self, *, scale: str = 'linear'):
        self.scale = scale
        self._cached_state = None
        self._presenters: weakref.WeakSet = weakref.WeakSet()

    def compute(self, data):
        result = FakePlot()
        self._cached_state = result
        self.mark_presenters_dirty()

    def get_cached_state(self):
        return self._cached_state

    def has_cached_state(self):
        return self._cached_state is not None

    def create_presenter(self, *, owner=None):
        presenter = FakePresenter(self, scale=self.scale, owner=owner)
        self._presenters.add(presenter)
        return presenter

    def mark_presenters_dirty(self):
        """Mark all registered presenters as having pending updates."""
        for presenter in self._presenters:
            presenter._mark_dirty()


class FakePresenter(PresenterBase):
    """Fake presenter for testing."""

    def __init__(self, plotter, *, scale: str = 'linear', owner=None):
        super().__init__(plotter, owner=owner)
        self.scale = scale

    def present(self, pipe: hv.streams.Pipe) -> hv.DynamicMap:
        def render(data):
            return hv.Curve([])

        return hv.DynamicMap(render, streams=[pipe], cache_size=1)


@pytest.fixture
def plot_data_service():
    """Create a PlotDataService instance."""
    return PlotDataService()


class TestSessionLayerCreate:
    """Tests for SessionLayer.create() class method."""

    def test_create_returns_session_layer_when_data_available(self, plot_data_service):
        """Test that create() returns a SessionLayer when data is available."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})  # Populate cached state
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)  # Transition to READY

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer.create(layer_id, state)

        assert session_layer is not None
        assert session_layer.layer_id == layer_id
        assert isinstance(session_layer.dmap, hv.DynamicMap)
        assert session_layer.presenter is not None
        assert session_layer.pipe is not None

    def test_create_returns_none_when_no_displayable_plot(self, plot_data_service):
        """Test that create() returns None when layer has no displayable plot."""
        layer_id = LayerId(uuid4())
        # Create an entry in WAITING_FOR_DATA state - no cached data yet
        plotter = FakePlotter()
        plot_data_service.job_started(layer_id, plotter)

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer.create(layer_id, state)

        assert session_layer is None

    def test_create_captures_current_version(self, plot_data_service):
        """Test that create() captures the current version from state."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer.create(layer_id, state)

        assert session_layer.last_seen_version == state.version


class TestSessionLayerUpdatePipe:
    """Tests for SessionLayer.update_pipe() method."""

    def test_update_pipe_returns_false_when_no_pending_update(self, plot_data_service):
        """Test that update_pipe() returns False when no update pending."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer.create(layer_id, state)

        # No new data since creation, so no pending update
        # Note: create() consumes the initial dirty flag via presenter.present()
        # which reads cached state, so presenter is not dirty
        result = session_layer.update_pipe()

        assert result is False

    def test_update_pipe_returns_true_and_sends_when_pending_update(
        self, plot_data_service
    ):
        """Test that update_pipe() sends data and returns True when update pending."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer.create(layer_id, state)

        # Compute new data - marks presenter dirty
        plotter.compute({'data': 2})

        result = session_layer.update_pipe()

        assert result is True

    def test_update_pipe_clears_dirty_flag(self, plot_data_service):
        """Test that update_pipe() clears the dirty flag after sending."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer.create(layer_id, state)

        # Compute new data - marks presenter dirty
        plotter.compute({'data': 2})

        # First call sends update
        assert session_layer.update_pipe() is True
        # Second call has nothing to send
        assert session_layer.update_pipe() is False


class TestSessionLayerIsValidFor:
    """Tests for SessionLayer.is_valid_for() method."""

    def test_is_valid_for_returns_true_for_same_plotter(self, plot_data_service):
        """Test that is_valid_for() returns True for the original plotter."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer.create(layer_id, state)

        assert session_layer.is_valid_for(plotter) is True

    def test_is_valid_for_returns_false_for_different_plotter(self, plot_data_service):
        """Test that is_valid_for() returns False when plotter is replaced."""
        layer_id = LayerId(uuid4())
        plotter_a = FakePlotter()
        plotter_a.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter_a)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer.create(layer_id, state)

        # Create a different plotter
        plotter_b = FakePlotter()
        plotter_b.compute({'data': 2})

        assert session_layer.is_valid_for(plotter_b) is False

    def test_is_valid_for_returns_false_for_none(self, plot_data_service):
        """Test that is_valid_for() returns False when plotter is None."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer.create(layer_id, state)

        assert session_layer.is_valid_for(None) is False


class TestSessionLayerWithStoppedState:
    """Tests for SessionLayer with STOPPED state (workflow ended)."""

    def test_create_works_with_stopped_state(self, plot_data_service):
        """Test that create() works when layer is in STOPPED state with data."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)
        plot_data_service.job_stopped(layer_id)  # Transition to STOPPED

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer.create(layer_id, state)

        # Should still be able to create session layer from stopped state with data
        assert session_layer is not None
        assert isinstance(session_layer.dmap, hv.DynamicMap)
