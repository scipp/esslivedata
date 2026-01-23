# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for PlotDataService."""

from ess.livedata.dashboard.state_stores import (
    LayerId,
    PlotDataService,
)


class TestPlotDataService:
    def test_update_and_get(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        service.update(layer_id, {'plot': 'data'})
        state = service.get(layer_id)

        assert state is not None
        assert state.state == {'plot': 'data'}
        assert state.version == 1

    def test_get_unknown_layer_returns_none(self):
        service = PlotDataService()
        assert service.get(LayerId('unknown')) is None

    def test_update_increments_version(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')

        service.update(layer_id, 'v1')
        assert service.get_version(layer_id) == 1

        service.update(layer_id, 'v2')
        assert service.get_version(layer_id) == 2

    def test_get_version_unknown_layer_returns_zero(self):
        service = PlotDataService()
        assert service.get_version(LayerId('unknown')) == 0

    def test_get_updates_since(self):
        service = PlotDataService()
        layer1 = LayerId('layer-1')
        layer2 = LayerId('layer-2')

        service.update(layer1, 'v1')
        checkpoint = {layer1: service.get_version(layer1)}
        service.update(layer2, 'v2')
        service.update(layer1, 'v1-updated')

        updates = service.get_updates_since(checkpoint)

        assert set(updates.keys()) == {layer1, layer2}
        assert updates[layer1].state == 'v1-updated'
        assert updates[layer2].state == 'v2'

    def test_get_updates_since_empty_versions(self):
        service = PlotDataService()
        layer1 = LayerId('layer-1')

        service.update(layer1, 'v1')

        updates = service.get_updates_since({})

        assert layer1 in updates

    def test_remove_layer(self):
        service = PlotDataService()
        layer_id = LayerId('layer-1')
        service.update(layer_id, 'v1')

        service.remove(layer_id)

        assert service.get(layer_id) is None

    def test_remove_unknown_layer_is_noop(self):
        service = PlotDataService()
        service.remove(LayerId('unknown'))

    def test_clear(self):
        service = PlotDataService()
        service.update(LayerId('layer-1'), 'v1')

        service.clear()

        assert service.get(LayerId('layer-1')) is None
