# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for SessionLayer and SessionComponents."""

import weakref
from uuid import uuid4

import holoviews as hv
import pytest

from ess.livedata.dashboard.plot_data_service import LayerId, PlotDataService
from ess.livedata.dashboard.plots import PresenterBase
from ess.livedata.dashboard.session_layer import SessionComponents, SessionLayer


class FakePlot:
    """Fake plot state for testing."""

    pass


class FakePlotter:
    """Fake plotter for testing."""

    def __init__(self, *, scale: str = 'linear'):
        self.scale = scale
        self._cached_state = None
        self._presenters: weakref.WeakSet = weakref.WeakSet()
        self.set_active_calls: list[tuple[object, bool]] = []

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

    def set_active(self, token: object, active: bool) -> None:
        self.set_active_calls.append((token, active))


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


class TestSessionComponentsCreate:
    """Tests for SessionComponents.create() class method."""

    def test_create_returns_components_when_data_available(self, plot_data_service):
        """Test that create() returns components when data is available."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        components = SessionComponents.create(state)

        assert components is not None
        assert isinstance(components.dmap, hv.DynamicMap)
        assert components.presenter is not None
        assert components.pipe is not None

    def test_create_returns_none_when_no_displayable_plot(self, plot_data_service):
        """Test that create() returns None when layer has no displayable plot."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plot_data_service.job_started(layer_id, plotter)

        state = plot_data_service.get(layer_id)
        components = SessionComponents.create(state)

        assert components is None


class TestSessionComponentsUpdatePipe:
    """Tests for SessionComponents.update_pipe() method."""

    def test_update_pipe_returns_false_when_no_pending_update(self, plot_data_service):
        """Test that update_pipe() returns False when no update pending."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        components = SessionComponents.create(state)

        result = components.update_pipe()

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
        components = SessionComponents.create(state)

        plotter.compute({'data': 2})

        result = components.update_pipe()

        assert result is True

    def test_update_pipe_clears_dirty_flag(self, plot_data_service):
        """Test that update_pipe() clears the dirty flag after sending."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        components = SessionComponents.create(state)

        plotter.compute({'data': 2})

        assert components.update_pipe() is True
        assert components.update_pipe() is False


class TestSessionComponentsIsValidFor:
    """Tests for SessionComponents.is_valid_for() method."""

    def test_is_valid_for_returns_true_for_same_plotter(self, plot_data_service):
        """Test that is_valid_for() returns True for the original plotter."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        components = SessionComponents.create(state)

        assert components.is_valid_for(plotter) is True

    def test_is_valid_for_returns_false_for_different_plotter(self, plot_data_service):
        """Test that is_valid_for() returns False when plotter is replaced."""
        layer_id = LayerId(uuid4())
        plotter_a = FakePlotter()
        plotter_a.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter_a)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        components = SessionComponents.create(state)

        plotter_b = FakePlotter()
        plotter_b.compute({'data': 2})

        assert components.is_valid_for(plotter_b) is False

    def test_is_valid_for_returns_false_for_none(self, plot_data_service):
        """Test that is_valid_for() returns False when plotter is None."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        components = SessionComponents.create(state)

        assert components.is_valid_for(None) is False


class TestSessionLayer:
    """Tests for SessionLayer."""

    def test_dmap_returns_none_without_components(self):
        """Test that dmap is None when no components exist."""
        layer_id = LayerId(uuid4())
        session_layer = SessionLayer(layer_id=layer_id, last_seen_version=1)

        assert session_layer.dmap is None

    def test_dmap_returns_dmap_with_components(self, plot_data_service):
        """Test that dmap returns the component's dmap when available."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        components = SessionComponents.create(state)
        session_layer = SessionLayer(
            layer_id=layer_id, last_seen_version=state.version, components=components
        )

        assert session_layer.dmap is components.dmap

    def test_update_pipe_returns_false_without_components(self):
        """Test that update_pipe() returns False when no components exist."""
        layer_id = LayerId(uuid4())
        session_layer = SessionLayer(layer_id=layer_id, last_seen_version=1)

        assert session_layer.update_pipe() is False


class TestSessionLayerEnsureComponents:
    """Tests for SessionLayer.ensure_components() method."""

    def test_ensure_components_creates_when_data_available(self, plot_data_service):
        """Test that ensure_components() creates components when data is available."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer(layer_id=layer_id, last_seen_version=state.version)

        assert session_layer.components is None

        result = session_layer.ensure_components(state)

        assert result is True
        assert session_layer.components is not None
        assert session_layer.dmap is not None

    def test_ensure_components_returns_false_when_no_data(self, plot_data_service):
        """Test that ensure_components() returns False when no displayable data."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plot_data_service.job_started(layer_id, plotter)

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer(layer_id=layer_id, last_seen_version=state.version)

        result = session_layer.ensure_components(state)

        assert result is False
        assert session_layer.components is None

    def test_ensure_components_keeps_valid_components(self, plot_data_service):
        """Test that ensure_components() keeps existing valid components."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        components = SessionComponents.create(state)
        session_layer = SessionLayer(
            layer_id=layer_id, last_seen_version=state.version, components=components
        )

        original_components = session_layer.components

        result = session_layer.ensure_components(state)

        assert result is True
        assert session_layer.components is original_components

    def test_ensure_components_invalidates_on_plotter_change(self, plot_data_service):
        """Test that ensure_components() invalidates components when plotter changes."""
        layer_id = LayerId(uuid4())
        plotter_a = FakePlotter()
        plotter_a.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter_a)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        components = SessionComponents.create(state)
        session_layer = SessionLayer(
            layer_id=layer_id, last_seen_version=state.version, components=components
        )

        # Simulate plotter replacement
        plotter_b = FakePlotter()
        plotter_b.compute({'data': 2})
        plot_data_service.job_started(layer_id, plotter_b)
        plot_data_service.data_arrived(layer_id)

        new_state = plot_data_service.get(layer_id)
        result = session_layer.ensure_components(new_state)

        assert result is True
        assert session_layer.components is not components  # New components created


class TestSessionLayerWithStoppedState:
    """Tests for SessionLayer with STOPPED state (workflow ended)."""

    def test_ensure_components_works_with_stopped_state(self, plot_data_service):
        """Test that ensure_components() works when layer is in STOPPED state."""
        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'data': 1})
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)
        plot_data_service.job_stopped(layer_id)

        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer(layer_id=layer_id, last_seen_version=state.version)

        result = session_layer.ensure_components(state)

        assert result is True
        assert session_layer.components is not None
        assert isinstance(session_layer.dmap, hv.DynamicMap)


class TestSessionLayerSyncActivePlotter:
    """Tests for SessionLayer.sync_active_plotter and release_active_plotter."""

    def test_first_call_records_active_plotter(self):
        plotter = FakePlotter()
        sl = SessionLayer(layer_id=LayerId(uuid4()), last_seen_version=0)
        sl.sync_active_plotter(plotter, True)
        assert sl.active_plotter is plotter
        assert plotter.set_active_calls == [(sl, True)]

    def test_inactive_call_still_records_plotter(self):
        plotter = FakePlotter()
        sl = SessionLayer(layer_id=LayerId(uuid4()), last_seen_version=0)
        sl.sync_active_plotter(plotter, False)
        assert sl.active_plotter is plotter
        assert plotter.set_active_calls == [(sl, False)]

    def test_plotter_replacement_releases_old_acquires_new(self):
        old = FakePlotter()
        new = FakePlotter()
        sl = SessionLayer(layer_id=LayerId(uuid4()), last_seen_version=0)
        sl.sync_active_plotter(old, True)
        sl.sync_active_plotter(new, True)
        assert sl.active_plotter is new
        assert old.set_active_calls == [(sl, True), (sl, False)]
        assert new.set_active_calls == [(sl, True)]

    def test_none_plotter_does_not_call_set_active(self):
        sl = SessionLayer(layer_id=LayerId(uuid4()), last_seen_version=0)
        sl.sync_active_plotter(None, True)
        assert sl.active_plotter is None

    def test_none_plotter_after_real_plotter_releases(self):
        plotter = FakePlotter()
        sl = SessionLayer(layer_id=LayerId(uuid4()), last_seen_version=0)
        sl.sync_active_plotter(plotter, True)
        sl.sync_active_plotter(None, True)
        assert sl.active_plotter is None
        assert plotter.set_active_calls == [(sl, True), (sl, False)]

    def test_release_clears_active_plotter(self):
        plotter = FakePlotter()
        sl = SessionLayer(layer_id=LayerId(uuid4()), last_seen_version=0)
        sl.sync_active_plotter(plotter, True)
        sl.release_active_plotter()
        assert sl.active_plotter is None
        assert plotter.set_active_calls == [(sl, True), (sl, False)]

    def test_release_when_no_plotter_is_noop(self):
        sl = SessionLayer(layer_id=LayerId(uuid4()), last_seen_version=0)
        sl.release_active_plotter()
        assert sl.active_plotter is None
