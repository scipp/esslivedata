# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for PlotDataService."""

import weakref

from ess.livedata.dashboard.plot_data_service import (
    LayerId,
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
    def test_set_plotter_and_get(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        plotter = FakePlotter(state={'plot': 'data'})
        service.set_plotter(layer_id, plotter)
        state = service.get(layer_id)

        assert state is not None
        assert state.plotter is plotter
        assert state.plotter.get_cached_state() == {'plot': 'data'}
        assert state.error is None
        assert state.stopped is False

    def test_get_unknown_layer_returns_none(self):
        service = PlotDataService()
        assert service.get(LayerId('unknown')) is None

    def test_set_plotter_replaces_previous_state(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        plotter1 = FakePlotter(state='v1')
        service.set_plotter(layer_id, plotter1)
        assert service.get(layer_id).plotter is plotter1

        plotter2 = FakePlotter(state='v2')
        service.set_plotter(layer_id, plotter2)
        state = service.get(layer_id)

        assert state.plotter is plotter2
        # Setting a new plotter resets error/stopped state
        assert state.error is None
        assert state.stopped is False

    def test_set_error_marks_presenters_dirty(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        plotter = FakePlotter(state='data')
        service.set_plotter(layer_id, plotter)

        # Create a presenter so we can track dirty flag
        presenter = plotter.create_presenter()
        presenter._dirty = False  # Reset after creation

        service.set_error(layer_id, 'Something went wrong')

        assert service.get(layer_id).error == 'Something went wrong'
        assert presenter.has_pending_update()

    def test_set_stopped_marks_presenters_dirty(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        plotter = FakePlotter(state='data')
        service.set_plotter(layer_id, plotter)

        # Create a presenter so we can track dirty flag
        presenter = plotter.create_presenter()
        presenter._dirty = False  # Reset after creation

        service.set_stopped(layer_id)

        assert service.get(layer_id).stopped is True
        assert presenter.has_pending_update()

    def test_set_error_on_entry_without_plotter(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        # Set error before setting plotter
        service.set_error(layer_id, 'Error occurred')

        state = service.get(layer_id)
        assert state.error == 'Error occurred'
        assert state.plotter is None

    def test_set_stopped_on_entry_without_plotter(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        # Set stopped before setting plotter
        service.set_stopped(layer_id)

        state = service.get(layer_id)
        assert state.stopped is True
        assert state.plotter is None

    def test_remove_layer(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')
        plotter = FakePlotter(state='v1')
        service.set_plotter(layer_id, plotter)

        service.remove(layer_id)

        assert service.get(layer_id) is None

    def test_remove_unknown_layer_is_noop(self):
        service = PlotDataService()
        service.remove(LayerId('unknown'))

    def test_clear(self):
        service = PlotDataService()
        plotter = FakePlotter(state='v1')
        service.set_plotter(LayerId('layer-1'), plotter)

        service.clear()

        assert service.get(LayerId('layer-1')) is None
