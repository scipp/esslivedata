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


@pytest.fixture
def session_manager(plot_data_service):
    """Create a SessionPlotManager instance."""
    return SessionPlotManager(plot_data_service)


class TestSessionPlotManager:
    """Tests for SessionPlotManager."""

    def test_get_dmap_creates_dmap_when_data_available(
        self, plot_data_service, session_manager
    ):
        """Test that get_dmap creates a DynamicMap when data is available."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})  # Populate cached state
        plot_data_service.set_plotter(layer_id, plotter)

        dmap = session_manager.get_dmap(layer_id)

        assert dmap is not None
        assert isinstance(dmap, hv.DynamicMap)
        assert session_manager.has_layer(layer_id)

    def test_get_dmap_returns_cached_dmap_on_second_call(
        self, plot_data_service, session_manager
    ):
        """Test that get_dmap returns cached DynamicMap if already set up."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})  # Populate cached state
        plot_data_service.set_plotter(layer_id, plotter)

        dmap1 = session_manager.get_dmap(layer_id)
        dmap2 = session_manager.get_dmap(layer_id)

        assert dmap1 is dmap2

    def test_get_dmap_returns_none_if_no_data(self, plot_data_service, session_manager):
        """Test that get_dmap returns None if layer has no data."""
        layer_id = LayerId(uuid4())
        # Create an entry with no plotter - represents waiting for data
        plotter = FakePlotter()
        plot_data_service.set_plotter(layer_id, plotter)

        dmap = session_manager.get_dmap(layer_id)

        assert dmap is None

    def test_invalidate_layer_clears_cached_components(
        self, plot_data_service, session_manager
    ):
        """Test that invalidate_layer removes cached components.

        In the real flow, invalidate_layer() is called when a layer is orphaned
        (removed from PlotDataService). This test simulates that scenario.
        """
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})  # Populate cached state
        plot_data_service.set_plotter(layer_id, plotter)

        # Set up the layer via get_dmap
        session_manager.get_dmap(layer_id)
        assert session_manager.has_layer(layer_id)

        # Simulate layer being removed from PlotDataService (orphaned)
        plot_data_service.remove(layer_id)

        # Invalidate (as would happen in update_pipes orphan cleanup)
        session_manager.invalidate_layer(layer_id)

        assert not session_manager.has_layer(layer_id)
        # get_dmap() attempts auto-setup, but fails since layer not in PlotDataService
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

        # Set up layer in session via get_dmap
        dmap_original = session_manager.get_dmap(layer_id)
        assert dmap_original is not None
        assert session_manager.has_layer(layer_id)

        # Simulate PlotOrchestrator.update_layer_config():
        # Old layer_id is removed from PlotDataService
        plot_data_service.remove(layer_id)

        # Call update_pipes - should detect orphan and clean up
        session_manager.update_pipes()

        # Session cache should be cleared
        assert not session_manager.has_layer(layer_id)

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

        # Set up with linear scale via get_dmap
        dmap_linear = session_manager.get_dmap(old_layer_id)
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
        assert not session_manager.has_layer(old_layer_id)

        # New layer_id gets fresh components via get_dmap
        dmap_log = session_manager.get_dmap(new_layer_id)
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

        # Set up layer via get_dmap - presenter is created and dirty flag is reset
        session_manager.get_dmap(layer_id)

        # Update data by computing new state - marks presenter dirty
        plotter.compute({'data': 2})

        # update_pipes should detect dirty flag and forward the update
        updated = session_manager.update_pipes()

        assert layer_id in updated

    def test_update_pipes_detects_plotter_replacement(
        self, plot_data_service, session_manager
    ):
        """Test that update_pipes detects when a plotter is replaced.

        When a workflow restarts, a new plotter is created for the same layer_id.
        The session should detect this and invalidate its cached components so
        they get recreated with the new plotter.
        """
        layer_id = LayerId(uuid4())
        plotter_a = FakePlotter()
        plotter_a.compute({'data': 1})
        plot_data_service.set_plotter(layer_id, plotter_a)

        # Set up layer via get_dmap
        session_manager.get_dmap(layer_id)
        presenter_a = session_manager._presenters[layer_id]
        assert presenter_a.is_owned_by(plotter_a)

        # Simulate workflow restart: new plotter for same layer_id
        plotter_b = FakePlotter()
        plotter_b.compute({'data': 2})
        plot_data_service.set_plotter(layer_id, plotter_b)

        # update_pipes should detect plotter change and invalidate
        session_manager.update_pipes()

        # Session cache should be cleared
        assert not session_manager.has_layer(layer_id)

        # Re-setup via get_dmap creates fresh components with new plotter
        session_manager.get_dmap(layer_id)
        presenter_b = session_manager._presenters[layer_id]
        assert presenter_b.is_owned_by(plotter_b)
        assert presenter_b is not presenter_a
