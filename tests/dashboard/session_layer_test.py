# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for SessionLayer and SessionComponents."""

from uuid import uuid4

import holoviews as hv
import pytest

from ess.livedata.dashboard.frame import Frame
from ess.livedata.dashboard.plot_data_service import LayerId, PlotDataService
from ess.livedata.dashboard.session_layer import SessionComponents, SessionLayer
from tests.helpers.plot_fakes import FakePlotter

hv.extension('bokeh')


def _curves(*values: float) -> hv.Overlay:
    """A frame that can be written directly, with one curve per value."""
    return hv.Overlay([hv.Curve([v, v + 1], label=f'c{v}') for v in values])


def _layout(n: int) -> hv.Layout:
    """A frame drawn through HoloViews, of layout shape ``n`` curves."""
    return hv.Layout([hv.Curve([i, i + 1]) for i in range(n)])


@pytest.fixture
def plot_data_service() -> PlotDataService:
    return PlotDataService()


@pytest.fixture
def layer_id() -> LayerId:
    return LayerId(uuid4())


def _ready(
    plot_data_service: PlotDataService, layer_id: LayerId, plotter: FakePlotter
) -> None:
    plot_data_service.job_started(layer_id, plotter)
    plot_data_service.data_arrived(layer_id)


class TestSessionComponentsCreate:
    def test_create_starts_from_the_plotters_current_frame(
        self, plot_data_service, layer_id
    ):
        plotter = FakePlotter(cached_state=_curves(1))
        _ready(plot_data_service, layer_id, plotter)

        components = SessionComponents.create(plot_data_service.get(layer_id))

        assert components.frame is plotter.get_frame()
        assert isinstance(components.dmap, hv.DynamicMap)

    def test_create_returns_none_when_no_displayable_plot(
        self, plot_data_service, layer_id
    ):
        plot_data_service.job_started(layer_id, FakePlotter())

        assert SessionComponents.create(plot_data_service.get(layer_id)) is None


class TestSessionComponentsPushUpdate:
    @pytest.fixture
    def plotter(self) -> FakePlotter:
        return FakePlotter(cached_state=_curves(1))

    @pytest.fixture
    def components(self, plotter, plot_data_service, layer_id) -> SessionComponents:
        _ready(plot_data_service, layer_id, plotter)
        return SessionComponents.create(plot_data_service.get(layer_id))

    @pytest.fixture
    def received(self, components) -> list[Frame]:
        frames: list[Frame] = []
        components.subscribe(frames.append)
        return frames

    def test_returns_false_when_no_pending_update(self, components):
        assert components.push_update() is False

    def test_pushes_frame_of_same_spec_to_subscribers(
        self, plotter, components, received
    ):
        plotter.compute(_curves(1))

        assert components.push_update() is True
        assert received == [plotter.get_frame()]
        assert components.frame is plotter.get_frame()

    def test_clears_the_pending_update(self, plotter, components):
        plotter.compute(_curves(1))

        assert components.push_update() is True
        assert components.push_update() is False

    def test_frame_of_other_spec_is_not_pushed(self, plotter, components, received):
        original = components.frame
        plotter.compute(_curves(1, 2))

        assert components.push_update() is False
        assert received == []
        assert components.frame is original

    def test_failing_subscriber_does_not_stop_the_others(
        self, plotter, components, received
    ):
        def fail(frame: Frame) -> None:
            raise RuntimeError('broken figure')

        components.unsubscribe(received.append)
        components.subscribe(fail)
        components.subscribe(received.append)
        plotter.compute(_curves(1))

        assert components.push_update() is True
        assert received == [plotter.get_frame()]

    def test_unsubscribed_callback_gets_no_frames(self, plotter, components, received):
        components.unsubscribe(received.append)
        plotter.compute(_curves(1))

        components.push_update()

        assert received == []

    def test_dynamicmap_shows_frames_pushed_after_it_was_created(
        self, plotter, components
    ):
        dmap = components.dmap
        frame = _curves(1)
        plotter.compute(frame)

        components.push_update()

        assert dmap.streams[0].data is frame

    def test_dynamicmap_starts_from_the_last_pushed_frame(self, plotter, components):
        frame = _curves(1)
        plotter.compute(frame)
        components.push_update()

        assert components.dmap.streams[0].data is frame


class TestSessionComponentsIsValidFor:
    @pytest.fixture
    def plotter(self) -> FakePlotter:
        return FakePlotter(cached_state=_curves(1))

    @pytest.fixture
    def components(self, plotter, plot_data_service, layer_id) -> SessionComponents:
        _ready(plot_data_service, layer_id, plotter)
        return SessionComponents.create(plot_data_service.get(layer_id))

    def test_valid_for_the_same_plotter(self, plotter, components):
        assert components.is_valid_for(plotter) is True

    def test_invalid_for_a_replaced_plotter(self, components):
        assert components.is_valid_for(FakePlotter(cached_state=_curves(1))) is False

    def test_invalid_for_none(self, components):
        assert components.is_valid_for(None) is False

    def test_spec_change_invalidates_components(self, plotter, components):
        plotter.compute(_curves(1, 2))

        assert components.is_valid_for(plotter) is False


class TestSessionComponentsLayoutShape:
    """A DynamicMap of a layout accepts only frames of the shape it was built with."""

    @pytest.fixture
    def plotter(self) -> FakePlotter:
        return FakePlotter(cached_state=_layout(1))

    @pytest.fixture
    def components(self, plotter, plot_data_service, layer_id) -> SessionComponents:
        _ready(plot_data_service, layer_id, plotter)
        return SessionComponents.create(plot_data_service.get(layer_id))

    def test_frame_of_same_shape_is_sent(self, plotter, components):
        pipe = components.dmap.streams[0]
        frame = _layout(1)
        plotter.compute(frame)

        assert components.push_update() is True
        assert pipe.data is frame

    def test_frame_of_different_shape_is_not_sent(self, plotter, components):
        pipe = components.dmap.streams[0]
        original = pipe.data
        plotter.compute(_layout(2))

        assert components.push_update() is False
        assert pipe.data is original


class TestSessionLayer:
    def test_push_update_returns_false_without_components(self, layer_id):
        assert SessionLayer(layer_id=layer_id).push_update() is False


class TestSessionLayerEnsureComponents:
    def test_creates_components_when_data_available(self, plot_data_service, layer_id):
        _ready(plot_data_service, layer_id, FakePlotter(cached_state=_curves(1)))
        session_layer = SessionLayer(layer_id=layer_id)

        assert session_layer.ensure_components(plot_data_service.get(layer_id))
        assert session_layer.components is not None

    def test_returns_false_when_no_data(self, plot_data_service, layer_id):
        plot_data_service.job_started(layer_id, FakePlotter())
        session_layer = SessionLayer(layer_id=layer_id)

        assert not session_layer.ensure_components(plot_data_service.get(layer_id))
        assert session_layer.components is None

    def test_keeps_valid_components(self, plot_data_service, layer_id):
        _ready(plot_data_service, layer_id, FakePlotter(cached_state=_curves(1)))
        state = plot_data_service.get(layer_id)
        session_layer = SessionLayer(layer_id=layer_id)
        session_layer.ensure_components(state)
        original = session_layer.components

        assert session_layer.ensure_components(state)
        assert session_layer.components is original

    def test_replaces_components_on_plotter_change(self, plot_data_service, layer_id):
        _ready(plot_data_service, layer_id, FakePlotter(cached_state=_curves(1)))
        session_layer = SessionLayer(layer_id=layer_id)
        session_layer.ensure_components(plot_data_service.get(layer_id))
        original = session_layer.components

        _ready(plot_data_service, layer_id, FakePlotter(cached_state=_curves(1)))

        assert session_layer.ensure_components(plot_data_service.get(layer_id))
        assert session_layer.components is not original

    def test_creates_components_for_stopped_layer(self, plot_data_service, layer_id):
        _ready(plot_data_service, layer_id, FakePlotter(cached_state=_curves(1)))
        plot_data_service.job_stopped(layer_id)
        session_layer = SessionLayer(layer_id=layer_id)

        assert session_layer.ensure_components(plot_data_service.get(layer_id))
        assert session_layer.components is not None
