# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for SessionPlotManager."""

from uuid import uuid4

import holoviews as hv
import pytest

from ess.livedata.dashboard.plot_data_service import LayerId, PlotDataService
from ess.livedata.dashboard.session_plot_manager import SessionPlotManager


class FakePlotter:
    """Fake plotter for testing."""

    def __init__(self, *, scale: str = 'linear'):
        self.scale = scale

    def create_presenter(self):
        return FakePresenter(scale=self.scale)


class FakePresenter:
    """Fake presenter for testing."""

    def __init__(self, scale: str):
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

    def test_setup_layer_creates_dmap(self, plot_data_service, session_manager):
        """Test that setup_layer creates a DynamicMap."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plot_data_service.update(layer_id, state={'data': 1}, plotter=plotter)

        dmap = session_manager.setup_layer(layer_id)

        assert dmap is not None
        assert isinstance(dmap, hv.DynamicMap)
        assert session_manager.has_layer(layer_id)

    def test_setup_layer_returns_cached_dmap_on_second_call(
        self, plot_data_service, session_manager
    ):
        """Test that setup_layer returns cached DynamicMap if already set up."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plot_data_service.update(layer_id, state={'data': 1}, plotter=plotter)

        dmap1 = session_manager.setup_layer(layer_id)
        dmap2 = session_manager.setup_layer(layer_id)

        assert dmap1 is dmap2

    def test_setup_layer_returns_none_if_no_data(
        self, plot_data_service, session_manager
    ):
        """Test that setup_layer returns None if layer has no data."""
        layer_id = LayerId(uuid4())
        plot_data_service.create_entry(layer_id)

        dmap = session_manager.setup_layer(layer_id)

        assert dmap is None

    def test_invalidate_layer_clears_cached_components(
        self, plot_data_service, session_manager
    ):
        """Test that invalidate_layer removes cached components."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plot_data_service.update(layer_id, state={'data': 1}, plotter=plotter)

        # Set up the layer
        session_manager.setup_layer(layer_id)
        assert session_manager.has_layer(layer_id)

        # Invalidate
        session_manager.invalidate_layer(layer_id)

        assert not session_manager.has_layer(layer_id)
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
        plot_data_service.update(layer_id, state={'data': 1}, plotter=plotter)

        # Set up layer in session
        dmap_original = session_manager.setup_layer(layer_id)
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
        plot_data_service.update(
            old_layer_id, state={'data': 1}, plotter=plotter_linear
        )

        # Set up with linear scale
        dmap_linear = session_manager.setup_layer(old_layer_id)
        presenter_linear = session_manager._presenters[old_layer_id]
        assert presenter_linear.scale == 'linear'

        # Simulate config change: old removed, new created with different config
        plot_data_service.remove(old_layer_id)
        new_layer_id = LayerId(uuid4())
        plotter_log = FakePlotter(scale='log')
        plot_data_service.update(new_layer_id, state={'data': 2}, plotter=plotter_log)

        # update_pipes cleans up orphaned old layer
        session_manager.update_pipes()
        assert not session_manager.has_layer(old_layer_id)

        # New layer_id gets fresh components
        dmap_log = session_manager.setup_layer(new_layer_id)
        presenter_log = session_manager._presenters[new_layer_id]

        assert dmap_log is not dmap_linear
        assert presenter_log.scale == 'log'

    def test_update_pipes_forwards_data_updates(
        self, plot_data_service, session_manager
    ):
        """Test that update_pipes forwards data updates to session pipes."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plot_data_service.update(layer_id, state={'data': 1}, plotter=plotter)

        # Set up layer
        session_manager.setup_layer(layer_id)

        # Update data
        plot_data_service.update(layer_id, state={'data': 2})

        # update_pipes should forward the update
        updated = session_manager.update_pipes()

        assert layer_id in updated
