# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for SessionPlotManager."""

import weakref
from uuid import uuid4

import holoviews as hv
import pytest

from ess.livedata.dashboard.plot_data_service import LayerId, PlotDataService
from ess.livedata.dashboard.plots import PresenterBase
from ess.livedata.dashboard.session_plot_manager import SessionPlotManager


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
        self._mark_presenters_dirty()
        return result

    def get_cached_state(self):
        return self._cached_state

    def has_cached_state(self):
        return self._cached_state is not None

    def create_presenter(self):
        presenter = FakePresenter(self, scale=self.scale)
        self._presenters.add(presenter)
        return presenter

    def _mark_presenters_dirty(self):
        """Mark all registered presenters as having pending updates."""
        for presenter in self._presenters:
            presenter._mark_dirty()


class FakePresenter(PresenterBase):
    """Fake presenter for testing."""

    def __init__(self, plotter, *, scale: str = 'linear'):
        super().__init__(plotter)
        self.scale = scale

    def present(self, pipe: hv.streams.Pipe) -> hv.DynamicMap:
        def render(data):
            return hv.Curve([])

        return hv.DynamicMap(render, streams=[pipe], cache_size=1)


@pytest.fixture
def plot_data_service():
    """Create a PlotDataService instance."""
    return PlotDataService()


@pytest.fixture
def session_manager(plot_data_service):
    """Create a SessionPlotManager instance."""
    return SessionPlotManager(plot_data_service)


class TestSessionPlotManager:
    """Tests for SessionPlotManager."""

    def test_get_or_create_layer_creates_dmap_when_data_available(
        self, plot_data_service, session_manager
    ):
        """Test that get_or_create_layer creates a DynamicMap when data is available."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})  # Populate cached state
        plot_data_service.set_plotter(layer_id, plotter)

        result = session_manager.get_or_create_layer(layer_id)

        assert isinstance(result, hv.DynamicMap)
        assert session_manager.get_dmap(layer_id) is not None

    def test_get_or_create_layer_returns_cached_dmap_on_second_call(
        self, plot_data_service, session_manager
    ):
        """Test that get_or_create_layer returns cached DynamicMap if already set up."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})  # Populate cached state
        plot_data_service.set_plotter(layer_id, plotter)

        dmap1 = session_manager.get_or_create_layer(layer_id)
        dmap2 = session_manager.get_or_create_layer(layer_id)

        assert dmap1 is dmap2

    def test_get_or_create_layer_returns_placeholder_if_no_data(
        self, plot_data_service, session_manager
    ):
        """Test that get_or_create_layer returns placeholder if no data yet."""
        layer_id = LayerId(uuid4())
        # Create an entry with plotter but no data
        plotter = FakePlotter()
        plot_data_service.set_plotter(layer_id, plotter)

        result = session_manager.get_or_create_layer(layer_id)

        # Should return a placeholder (hv.Text), not None
        assert isinstance(result, hv.Text)
        # Should NOT be cached as a real DynamicMap
        assert session_manager.get_dmap(layer_id) is None

    def test_get_or_create_layer_returns_placeholder_if_no_state(
        self, plot_data_service, session_manager
    ):
        """Test that get_or_create_layer returns placeholder if no state exists."""
        layer_id = LayerId(uuid4())
        # No state in PlotDataService

        result = session_manager.get_or_create_layer(layer_id)

        assert isinstance(result, hv.Text)

    def test_get_or_create_layer_returns_error_placeholder_on_error(
        self, plot_data_service, session_manager
    ):
        """Test that get_or_create_layer returns error placeholder."""
        layer_id = LayerId(uuid4())
        plot_data_service.set_error(layer_id, "Something went wrong")

        result = session_manager.get_or_create_layer(layer_id)

        assert isinstance(result, hv.Text)
        # The text should contain the error

    def test_invalidate_layer_clears_cached_components(
        self, plot_data_service, session_manager
    ):
        """Test that invalidate_layer removes cached components."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})  # Populate cached state
        plot_data_service.set_plotter(layer_id, plotter)

        # Set up the layer
        session_manager.get_or_create_layer(layer_id)
        assert session_manager.get_dmap(layer_id) is not None

        # Invalidate
        session_manager.invalidate_layer(layer_id)

        assert session_manager.get_dmap(layer_id) is None

    def test_update_pipes_cleans_up_orphaned_layers(
        self, plot_data_service, session_manager
    ):
        """Test that update_pipes cleans up layers removed from PlotDataService.

        This is the mechanism for handling config changes: when
        update_layer_config() creates a new layer_id, the old one is removed
        from PlotDataService. The session detects this and cleans up the
        orphaned cache entry.
        """
        layer_id = LayerId(uuid4())
        plotter = FakePlotter(scale='linear')
        plotter.compute({'data': 1})  # Populate cached state
        plot_data_service.set_plotter(layer_id, plotter)

        # Set up layer in session (creates real DynamicMap)
        session_manager.get_or_create_layer(layer_id)
        assert session_manager.get_dmap(layer_id) is not None

        # Simulate PlotOrchestrator.update_layer_config():
        # Old layer_id is removed from PlotDataService
        plot_data_service.remove(layer_id)

        # Call update_pipes - should detect orphan and clean up
        session_manager.update_pipes()

        # Session cache should be cleared
        assert session_manager.get_dmap(layer_id) is None

    def test_new_layer_id_gets_fresh_components(
        self, plot_data_service, session_manager
    ):
        """Test that a new layer_id gets fresh components.

        This verifies the full config change flow:
        1. Old layer_id exists with cached components
        2. Config change creates new layer_id, removes old one
        3. Session cleans up orphaned old layer_id
        4. New layer_id gets fresh components with new config
        """
        old_layer_id = LayerId(uuid4())
        plotter_linear = FakePlotter(scale='linear')
        plotter_linear.compute({'data': 1})  # Populate cached state
        plot_data_service.set_plotter(old_layer_id, plotter_linear)

        # Set up with linear scale
        dmap_linear = session_manager.get_or_create_layer(old_layer_id)
        presenter_linear = session_manager._presenters[old_layer_id]
        assert presenter_linear.scale == 'linear'

        # Simulate config change: old removed, new created with different config
        plot_data_service.remove(old_layer_id)
        new_layer_id = LayerId(uuid4())
        plotter_log = FakePlotter(scale='log')
        plotter_log.compute({'data': 2})  # Populate cached state
        plot_data_service.set_plotter(new_layer_id, plotter_log)

        # update_pipes cleans up orphaned old layer
        session_manager.update_pipes()
        assert session_manager.get_dmap(old_layer_id) is None

        # New layer_id gets fresh components
        dmap_log = session_manager.get_or_create_layer(new_layer_id)
        presenter_log = session_manager._presenters[new_layer_id]

        assert dmap_log is not dmap_linear
        assert presenter_log.scale == 'log'

    def test_update_pipes_forwards_data_updates(
        self, plot_data_service, session_manager
    ):
        """Test that update_pipes forwards data updates to session pipes.

        When plotter.compute() is called, it marks all presenters dirty.
        update_pipes() should detect this and forward the update to the pipe.
        """
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})  # Populate cached state and mark dirty
        plot_data_service.set_plotter(layer_id, plotter)

        # Set up layer - presenter is created and dirty flag is reset
        session_manager.get_or_create_layer(layer_id)

        # Update data by computing new state - marks presenter dirty
        plotter.compute({'data': 2})

        # update_pipes should detect dirty flag and forward the update
        transitioned = session_manager.update_pipes()
        # No transitions expected - layer was already set up with data
        assert transitioned == set()

    def test_update_pipes_returns_transitioned_layers(
        self, plot_data_service, session_manager
    ):
        """Test that update_pipes returns layers that transitioned to ready state.

        When a layer returns a placeholder (waiting for data) and data later
        becomes available, update_pipes should return that layer_id as transitioned.
        """
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        # No cached state yet - plotter exists but no data
        plot_data_service.set_plotter(layer_id, plotter)

        # First call returns placeholder (pending)
        result = session_manager.get_or_create_layer(layer_id)
        assert isinstance(result, hv.Text)  # Placeholder

        # Now data arrives
        plotter.compute({'data': 1})

        # update_pipes should detect the transition
        transitioned = session_manager.update_pipes()
        assert layer_id in transitioned

        # Simulate what PlotGridTabs does: call get_or_create_layer to rebuild
        # This creates the real DynamicMap and clears the pending state
        session_manager.get_or_create_layer(layer_id)

        # Now the second call should NOT return it (no longer pending)
        transitioned_again = session_manager.update_pipes()
        assert layer_id not in transitioned_again

    def test_update_pipes_clears_pending_on_transition(
        self, plot_data_service, session_manager
    ):
        """Test that pending layers are cleared when transitioned.

        After a transition is detected, the layer should be removed from
        pending so it's not reported again.
        """
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plot_data_service.set_plotter(layer_id, plotter)

        # Get placeholder (adds to pending)
        session_manager.get_or_create_layer(layer_id)
        assert layer_id in session_manager._pending_layers

        # Data arrives
        plotter.compute({'data': 1})

        # First update_pipes detects transition
        transitioned = session_manager.update_pipes()
        assert layer_id in transitioned

        # Layer should still be in pending until it's actually set up
        # (get_or_create_layer creates the real DynamicMap)
        assert layer_id in session_manager._pending_layers

        # After get_or_create_layer, pending is cleared
        session_manager.get_or_create_layer(layer_id)
        assert layer_id not in session_manager._pending_layers
