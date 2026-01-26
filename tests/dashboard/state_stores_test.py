# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for PlotDataService."""

from ess.livedata.dashboard.plot_data_service import (
    LayerId,
    PlotDataService,
)


class FakePlotter:
    """Fake plotter that caches state for testing."""

    def __init__(self, state=None):
        self._cached_state = state

    def compute(self, data):
        self._cached_state = data
        return data

    def get_cached_state(self):
        return self._cached_state

    def has_cached_state(self):
        return self._cached_state is not None


class TestPlotDataService:
    def test_update_and_get(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        plotter = FakePlotter(state={'plot': 'data'})
        service.update(layer_id, plotter=plotter)
        state = service.get(layer_id)

        assert state is not None
        assert state.plotter.get_cached_state() == {'plot': 'data'}
        assert state.version == 1

    def test_get_unknown_layer_returns_none(self):
        service = PlotDataService()
        assert service.get(LayerId('unknown')) is None

    def test_update_increments_version(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        plotter = FakePlotter(state='v1')
        service.update(layer_id, plotter=plotter)
        assert service.get_version(layer_id) == 1

        plotter._cached_state = 'v2'
        service.update(layer_id, plotter=plotter)
        assert service.get_version(layer_id) == 2

    def test_get_version_unknown_layer_returns_zero(self):
        service = PlotDataService()
        assert service.get_version(LayerId('unknown')) == 0

    def test_get_updates_since(self):
        service = PlotDataService()
        layer1 = LayerId('layer-1')
        layer2 = LayerId('layer-2')

        plotter1 = FakePlotter(state='v1')
        service.update(layer1, plotter=plotter1)
        checkpoint = {layer1: service.get_version(layer1)}
        plotter2 = FakePlotter(state='v2')
        service.update(layer2, plotter=plotter2)
        plotter1._cached_state = 'v1-updated'
        service.update(layer1, plotter=plotter1)

        updates = service.get_updates_since(checkpoint)

        assert set(updates.keys()) == {layer1, layer2}
        assert updates[layer1].plotter.get_cached_state() == 'v1-updated'
        assert updates[layer2].plotter.get_cached_state() == 'v2'

    def test_get_updates_since_empty_versions(self):
        service = PlotDataService()
        layer1 = LayerId('layer-1')

        plotter = FakePlotter(state='v1')
        service.update(layer1, plotter=plotter)

        updates = service.get_updates_since({})

        assert layer1 in updates

    def test_remove_layer(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')
        plotter = FakePlotter(state='v1')
        service.update(layer_id, plotter=plotter)

        service.remove(layer_id)

        assert service.get(layer_id) is None

    def test_remove_unknown_layer_is_noop(self):
        service = PlotDataService()
        service.remove(LayerId('unknown'))

    def test_clear(self):
        service = PlotDataService()
        plotter = FakePlotter(state='v1')
        service.update(LayerId('layer-1'), plotter=plotter)

        service.clear()

        assert service.get(LayerId('layer-1')) is None
