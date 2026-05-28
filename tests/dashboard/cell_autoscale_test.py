# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for :class:`CellAutoscaleController`.

Uses lightweight stubs in the spirit of ``range_hook_test.py``: real
``Plotter`` instances aren't needed -- only the public surface
(``AUTOSCALE_AXES``, ``get_range_targets``, ``iter_range_targets``) is
exercised. Bokeh ``CustomAction`` instances are stubbed for the toggle/Fit
state so tests don't require a live Bokeh document.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.dashboard import cell_autoscale
from ess.livedata.dashboard.cell_autoscale import CellAutoscaleController
from ess.livedata.dashboard.range_hook import Axis


def _key(source: str = 'src', output: str = 'out') -> ResultKey:
    return ResultKey(
        workflow_id=WorkflowId(instrument='test', name='test', version=1),
        job_id=JobId(source_name=source, job_number=uuid.uuid4()),
        output_name=output,
    )


class _FakePlotter:
    """Stand-in for a real :class:`Plotter`.

    Implements the minimum surface used by :class:`CellAutoscaleController`:
    ``AUTOSCALE_AXES`` class attribute and ``iter_range_targets()``.
    """

    def __init__(
        self,
        axes: frozenset[Axis],
        targets_by_key: dict[ResultKey, dict[Axis, tuple[float, float]]] | None = None,
    ) -> None:
        self.AUTOSCALE_AXES = axes
        self._targets = targets_by_key or {}

    def iter_range_targets(self):
        return iter(self._targets.items())


class _StubAction:
    """Stub for ``bokeh.models.CustomAction`` -- mutable ``active`` flag."""

    def __init__(self, *, active: bool, description: str, icon: Any | None) -> None:
        self.active = active
        self.description = description
        self.icon = icon
        self._callbacks: list[Any] = []

    def on_change(self, attr: str, callback: Any) -> None:
        assert attr == 'active'
        self._callbacks.append(callback)

    def fire_active(self, new: bool) -> None:
        """Simulate Bokeh firing the ``active`` change callback."""
        old, self.active = self.active, new
        for cb in self._callbacks:
            cb('active', old, new)


class _StubRange:
    def __init__(self) -> None:
        self.start: float | None = None
        self.end: float | None = None
        self.document = None


class _StubColorMapper:
    def __init__(self) -> None:
        self.low: float | None = None
        self.high: float | None = None
        self.document = None


class _StubToolbar:
    def __init__(self) -> None:
        self.tools: list[Any] = []


class _StubFigState:
    def __init__(self, toolbar: _StubToolbar) -> None:
        self.toolbar = toolbar
        self.document = None


class _StubPlot:
    """Stub for HoloViews' ``plot`` argument to a hook."""

    def __init__(
        self,
        *,
        x_range: _StubRange | None = None,
        y_range: _StubRange | None = None,
        color_mapper: _StubColorMapper | None = None,
    ) -> None:
        self.handles: dict[str, Any] = {}
        if x_range is not None:
            self.handles['x_range'] = x_range
        if y_range is not None:
            self.handles['y_range'] = y_range
        if color_mapper is not None:
            self.handles['color_mapper'] = color_mapper
        self.state = _StubFigState(_StubToolbar())


@pytest.fixture(autouse=True)
def patch_custom_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace Bokeh's ``CustomAction`` factory with a stub for tests."""

    def factory(*, active: bool, description: str, icon: Any | None) -> Any:
        return _StubAction(active=active, description=description, icon=icon)

    monkeypatch.setattr(cell_autoscale, '_make_custom_action', factory)


def _make_plot_all_handles() -> tuple[
    _StubPlot, _StubRange, _StubRange, _StubColorMapper
]:
    x = _StubRange()
    y = _StubRange()
    c = _StubColorMapper()
    return _StubPlot(x_range=x, y_range=y, color_mapper=c), x, y, c


class TestGetTarget:
    def test_unions_targets_across_layers(self) -> None:
        k1, k2 = _key('s1'), _key('s2')
        p1 = _FakePlotter(
            frozenset({'x', 'y'}),
            {k1: {'x': (0.0, 10.0), 'y': (1.0, 5.0)}},
        )
        p2 = _FakePlotter(
            frozenset({'x', 'y'}),
            {k2: {'x': (5.0, 20.0), 'y': (-1.0, 3.0)}},
        )
        controller = CellAutoscaleController([p1, p2])

        assert controller.get_target('x') == (0.0, 20.0)
        assert controller.get_target('y') == (-1.0, 5.0)

    def test_skips_layers_without_axis(self) -> None:
        k = _key()
        p1 = _FakePlotter(frozenset({'c'}), {k: {'c': (1.0, 2.0)}})
        p2 = _FakePlotter(frozenset({'x'}), {_key('s2'): {'x': (3.0, 4.0)}})
        controller = CellAutoscaleController([p1, p2])

        assert controller.get_target('x') == (3.0, 4.0)
        assert controller.get_target('c') == (1.0, 2.0)
        assert controller.get_target('y') is None

    def test_returns_none_when_no_targets_computed(self) -> None:
        controller = CellAutoscaleController([_FakePlotter(frozenset({'x', 'y'}), {})])
        assert controller.get_target('x') is None
        assert controller.get_target('y') is None

    def test_unions_multiple_keys_within_single_plotter(self) -> None:
        k1, k2 = _key('a'), _key('b')
        plotter = _FakePlotter(
            frozenset({'x'}),
            {k1: {'x': (0.0, 1.0)}, k2: {'x': (2.0, 3.0)}},
        )
        controller = CellAutoscaleController([plotter])
        assert controller.get_target('x') == (0.0, 3.0)


class TestHookWrites:
    def test_writes_all_axes_when_toggles_on(self) -> None:
        k = _key()
        plotter = _FakePlotter(
            frozenset({'x', 'y', 'c'}),
            {k: {'x': (0.0, 1.0), 'y': (2.0, 3.0), 'c': (4.0, 5.0)}},
        )
        controller = CellAutoscaleController([plotter])
        plot, x, y, c = _make_plot_all_handles()

        controller.make_hook()(plot, None)

        assert (x.start, x.end) == (0.0, 1.0)
        assert (y.start, y.end) == (2.0, 3.0)
        assert (c.low, c.high) == (4.0, 5.0)

    def test_skips_axis_when_toggle_off(self) -> None:
        k = _key()
        plotter = _FakePlotter(
            frozenset({'x', 'y'}),
            {k: {'x': (0.0, 1.0), 'y': (2.0, 3.0)}},
        )
        controller = CellAutoscaleController([plotter])
        plot, x, y, _c = _make_plot_all_handles()

        # First render installs tools; capture them then flip X off.
        hook = controller.make_hook()
        hook(plot, None)
        # Toggles default to True so first render wrote both axes.
        assert (x.start, x.end) == (0.0, 1.0)
        assert (y.start, y.end) == (2.0, 3.0)

        # Now turn X off, advance targets, render again.
        controller._toggles['x'].active = False
        plotter._targets = {k: {'x': (10.0, 11.0), 'y': (20.0, 21.0)}}
        hook(plot, None)

        # X-axis is frozen at previous values; Y followed the new target.
        assert (x.start, x.end) == (0.0, 1.0)
        assert (y.start, y.end) == (20.0, 21.0)

    def test_no_writes_when_target_is_none(self) -> None:
        plotter = _FakePlotter(frozenset({'x', 'y'}), {})  # no compute() yet
        controller = CellAutoscaleController([plotter])
        plot, x, y, _c = _make_plot_all_handles()

        controller.make_hook()(plot, None)

        assert (x.start, x.end) == (None, None)
        assert (y.start, y.end) == (None, None)


class TestFitButton:
    def test_fit_writes_all_axes_regardless_of_toggle_state(self) -> None:
        k = _key()
        plotter = _FakePlotter(
            frozenset({'x', 'y', 'c'}),
            {k: {'x': (0.0, 1.0), 'y': (2.0, 3.0), 'c': (4.0, 5.0)}},
        )
        controller = CellAutoscaleController([plotter])
        plot, x, y, c = _make_plot_all_handles()
        hook = controller.make_hook()
        hook(plot, None)

        # Turn all toggles off and clear what the first render wrote.
        for axis in controller.axes:
            controller._toggles[axis].active = False
        x.start = x.end = None
        y.start = y.end = None
        c.low = c.high = None
        # Advance targets so we can see what Fit wrote.
        plotter._targets = {
            k: {'x': (10.0, 11.0), 'y': (12.0, 13.0), 'c': (14.0, 15.0)}
        }

        # Simulate user pressing Fit.
        controller._fit_tool.fire_active(True)

        assert (x.start, x.end) == (10.0, 11.0)
        assert (y.start, y.end) == (12.0, 13.0)
        assert (c.low, c.high) == (14.0, 15.0)
        assert controller._fit_tool.active is False

    def test_fit_with_no_targets_is_safe(self) -> None:
        plotter = _FakePlotter(frozenset({'x'}), {})
        controller = CellAutoscaleController([plotter])
        plot, _x, _y, _c = _make_plot_all_handles()
        controller.make_hook()(plot, None)

        # Should not raise.
        controller._fit_tool.fire_active(True)
        assert controller._fit_tool.active is False


class TestEmptyController:
    def test_hook_is_noop_when_no_axes(self) -> None:
        plotter = _FakePlotter(frozenset(), {})
        controller = CellAutoscaleController([plotter])
        plot, x, y, c = _make_plot_all_handles()

        controller.make_hook()(plot, None)

        # No tools installed, no writes performed.
        assert plot.state.toolbar.tools == []
        assert (x.start, x.end) == (None, None)
        assert (y.start, y.end) == (None, None)
        assert (c.low, c.high) == (None, None)

    def test_build_controller_returns_none_when_no_axes(self) -> None:
        plotter = _FakePlotter(frozenset(), {})
        assert cell_autoscale.build_controller_from_layers([plotter]) is None

    def test_build_controller_returns_none_when_no_plotters(self) -> None:
        assert cell_autoscale.build_controller_from_layers([]) is None


class TestIdempotentInstallation:
    def test_hook_installs_tools_only_once(self) -> None:
        k = _key()
        plotter = _FakePlotter(
            frozenset({'x', 'y'}),
            {k: {'x': (0.0, 1.0), 'y': (2.0, 3.0)}},
        )
        controller = CellAutoscaleController([plotter])
        plot, _x, _y, _c = _make_plot_all_handles()
        hook = controller.make_hook()

        hook(plot, None)
        tools_after_first = list(plot.state.toolbar.tools)
        hook(plot, None)
        hook(plot, None)
        tools_after_third = list(plot.state.toolbar.tools)

        # X, Y toggles + Fit = 3 tools, no duplicates across renders.
        assert len(tools_after_first) == 3
        assert tools_after_third == tools_after_first


class TestHandlesCaptured:
    def test_first_render_captures_handles(self) -> None:
        k = _key()
        plotter = _FakePlotter(
            frozenset({'x'}),
            {k: {'x': (0.0, 1.0)}},
        )
        controller = CellAutoscaleController([plotter])
        plot, x, y, c = _make_plot_all_handles()

        controller.make_hook()(plot, None)

        assert controller.handles.x_range is x
        assert controller.handles.y_range is y
        assert controller.handles.color_mapper is c
