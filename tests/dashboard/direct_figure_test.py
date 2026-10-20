# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for DirectFigure: a figure updated by writing its data sources."""

import gc
import weakref
from typing import Any

import holoviews as hv
import pytest
from bokeh.document import Document

from ess.livedata.dashboard.direct_figure import DirectFigure
from ess.livedata.dashboard.session_layer import SessionComponents
from tests.helpers.plot_fakes import FakePlotter

hv.extension('bokeh')


def _curves(*ys: float) -> hv.Overlay:
    """A frame with one two-point curve per value, at that height."""
    return hv.Overlay([hv.Curve([y, y], label=f'c{i}') for i, y in enumerate(ys)])


class _Layer:
    """A plotter and the session layer showing its frames."""

    def __init__(self, *ys: float) -> None:
        self.plotter = FakePlotter(cached_state=_curves(*ys))
        self.components = SessionComponents(
            presenter=self.plotter.create_presenter(), frame=self.plotter.get_frame()
        )

    def push(self, *ys: float) -> None:
        self.plotter.compute(_curves(*ys))
        assert self.components.push_update()


def _figure(*layers: _Layer, update_hooks=()) -> DirectFigure:
    frames = [layer.components.frame.element for layer in layers]
    return DirectFigure(
        hv.Overlay(frames),
        [layer.components for layer in layers],
        doc=Document(),
        margin=0,
        update_hooks=update_hooks,
    )


def _shown(figure: DirectFigure) -> list[list[float]]:
    """The y values of each curve, in drawing order."""
    return [list(r.data_source.data['y']) for r in figure.figure.renderers]


def test_pushed_frames_are_written_into_the_layers_own_curves():
    first, second = _Layer(1.0, 2.0), _Layer(3.0)
    figure = _figure(first, second)

    second.push(30.0)
    first.push(10.0, 20.0)

    assert _shown(figure) == [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]


def test_update_hooks_run_after_each_write():
    layer = _Layer(1.0)
    seen_by_hook: list[list[float]] = []

    def hook(plot: Any, element: Any) -> None:
        (renderer,) = plot.state.renderers
        seen_by_hook.append(list(renderer.data_source.data['y']))

    _figure(layer, update_hooks=[hook])

    layer.push(5.0)

    assert seen_by_hook == [[5.0, 5.0]]


def test_disposed_figure_receives_no_frames():
    layer = _Layer(1.0)
    figure = _figure(layer)

    figure.dispose()
    layer.push(5.0)

    assert _shown(figure) == [[1.0, 1.0]]


def test_disposed_figure_is_released():
    """Building registers the plot with its document, which would keep it alive."""
    layer = _Layer(1.0)
    figure = _figure(layer)
    plot: Any = weakref.ref(figure.plot)

    figure.dispose()
    del figure
    gc.collect()

    assert plot() is None


def test_figure_needs_one_subplot_per_element():
    layer = _Layer(1.0, 2.0)

    with pytest.raises(RuntimeError, match='subplots'):
        DirectFigure(_curves(1.0), [layer.components], doc=Document(), margin=0)
