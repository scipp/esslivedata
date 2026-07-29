# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for :class:`CellAutoscaleController`.

Uses lightweight stubs in the spirit of ``range_hook_test.py``: real
``Plotter`` instances aren't needed -- only the public surface
(``AUTOSCALE_AXES``, ``iter_range_targets``) is exercised. Bokeh
``CustomAction`` instances are stubbed for the toggle/Fit state so tests
don't require a live Bokeh document.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any

import pytest

from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.dashboard import cell_autoscale
from ess.livedata.dashboard.cell_autoscale import CellAutoscaleController
from ess.livedata.dashboard.range_hook import Axis


def _key(source: str = 'src', output: str = 'out') -> DataKey:
    return DataKey(
        workflow_id=WorkflowId(instrument='test', name='test', version=1),
        source_name=source,
        output_name=output,
    )


class _FakePlotter:
    """Stand-in for a real :class:`Plotter`.

    Implements the minimum surface used by :class:`CellAutoscaleController`:
    the ``autoscale_axes`` property and ``iter_range_targets()``.
    """

    def __init__(
        self,
        axes: frozenset[Axis],
        targets_by_key: dict[DataKey, dict[Axis, tuple[float, float]]] | None = None,
    ) -> None:
        self.autoscale_axes = axes
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

    def remove_on_change(self, attr: str, callback: Any) -> None:
        assert attr == 'active'
        self._callbacks.remove(callback)

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


class _StubSubPlot:
    """Stub for a HoloViews sub-plot (e.g., the image inside an Overlay)."""

    def __init__(self, *, color_mapper: _StubColorMapper | None = None) -> None:
        self.handles: dict[str, Any] = {}
        if color_mapper is not None:
            self.handles['color_mapper'] = color_mapper
        self.clim: tuple[float, float] | None = None


class _StubPlot:
    """Stub for HoloViews' ``plot`` argument to a hook."""

    def __init__(
        self,
        *,
        x_range: _StubRange | None = None,
        y_range: _StubRange | None = None,
        color_mapper: _StubColorMapper | None = None,
        subplots: dict[str, _StubSubPlot] | None = None,
    ) -> None:
        self.handles: dict[str, Any] = {}
        if x_range is not None:
            self.handles['x_range'] = x_range
        if y_range is not None:
            self.handles['y_range'] = y_range
        if color_mapper is not None:
            self.handles['color_mapper'] = color_mapper
        self.subplots = subplots
        self.state = _StubFigState(_StubToolbar())
        self.clim: tuple[float, float] | None = None


@pytest.fixture(autouse=True)
def patch_custom_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace Bokeh ``CustomAction`` factories with stubs for tests."""

    def toggle_factory(
        *,
        active: bool,
        description: str,
        on_icon: Any | None,
        off_icon: Any | None,
    ) -> Any:
        icon = on_icon if active else off_icon
        return _StubAction(active=active, description=description, icon=icon)

    def fit_factory(*, description: str, icon: Any | None) -> Any:
        return _StubAction(active=False, description=description, icon=icon)

    monkeypatch.setattr(cell_autoscale, '_make_toggle_action', toggle_factory)
    monkeypatch.setattr(cell_autoscale, '_make_fit_action', fit_factory)


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

    def test_c_axis_re_writes_last_target_when_toggled_off(self) -> None:
        """HoloViews unconditionally re-writes color_mapper.low/high every
        render from data extent. When the c-toggle is off we must re-apply
        the previously written target each render to keep the colorbar
        frozen at the user's chosen state.
        """
        k = _key()
        plotter = _FakePlotter(frozenset({'c'}), {k: {'c': (0.0, 10.0)}})
        controller = CellAutoscaleController([plotter])
        plot, _x, _y, c = _make_plot_all_handles()

        hook = controller.make_hook()
        hook(plot, None)
        assert (c.low, c.high) == (0.0, 10.0)

        # User turns off the c-toggle, then HoloViews' next render overwrites
        # the color_mapper from new data extent (simulated here).
        controller._toggles['c'].active = False
        c.low, c.high = 99.0, 999.0
        plotter._targets = {k: {'c': (100.0, 200.0)}}
        hook(plot, None)

        # Hook must re-write the last-known target, overriding HV's update.
        assert (c.low, c.high) == (0.0, 10.0)


class TestClimFreeze:
    def test_clim_written_when_c_axis_target_known(self) -> None:
        k = _key()
        plotter = _FakePlotter(frozenset({'c'}), {k: {'c': (4.0, 5.0)}})
        controller = CellAutoscaleController([plotter])
        plot, _x, _y, _c = _make_plot_all_handles()

        controller.make_hook()(plot, None)

        # Single image plot: top-level plot carries both handles and clim.
        assert plot.clim == (4.0, 5.0)

    def test_clim_freezes_when_toggle_off(self) -> None:
        k = _key()
        plotter = _FakePlotter(frozenset({'c'}), {k: {'c': (4.0, 5.0)}})
        controller = CellAutoscaleController([plotter])
        plot, _x, _y, _c = _make_plot_all_handles()

        hook = controller.make_hook()
        hook(plot, None)
        assert plot.clim == (4.0, 5.0)

        # Toggle off, advance targets: clim must stay at the frozen value.
        controller._toggles['c'].active = False
        plotter._targets = {k: {'c': (100.0, 200.0)}}
        plot.clim = None  # simulate HV resetting it
        hook(plot, None)

        assert plot.clim == (4.0, 5.0)

    def test_clim_written_when_toggle_on(self) -> None:
        """Toggle on -> clim follows current target every render."""
        k = _key()
        plotter = _FakePlotter(frozenset({'c'}), {k: {'c': (4.0, 5.0)}})
        controller = CellAutoscaleController([plotter])
        plot, _x, _y, _c = _make_plot_all_handles()
        hook = controller.make_hook()

        hook(plot, None)
        assert plot.clim == (4.0, 5.0)

        plotter._targets = {k: {'c': (100.0, 200.0)}}
        hook(plot, None)
        assert plot.clim == (100.0, 200.0)


class TestOverlaySubPlot:
    def test_color_mapper_found_in_subplot(self) -> None:
        """Overlay top-level plot has no color_mapper -- it's on the image
        sub-plot. The hook must find it via plot.subplots."""
        k = _key()
        plotter = _FakePlotter(frozenset({'c'}), {k: {'c': (4.0, 5.0)}})
        controller = CellAutoscaleController([plotter])
        mapper = _StubColorMapper()
        sub = _StubSubPlot(color_mapper=mapper)
        plot = _StubPlot(subplots={'Image': sub})

        controller.make_hook()(plot, None)

        assert (mapper.low, mapper.high) == (4.0, 5.0)
        # clim is set on the sub-plot, not the top-level overlay.
        assert sub.clim == (4.0, 5.0)


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

        # Simulate user pressing Fit: sets a pending flag honoured at the
        # next render. Robust to figure swaps between click and render.
        controller._fit_tool.fire_active(True)
        assert controller._fit_tool.active is False
        hook(plot, None)

        assert (x.start, x.end) == (10.0, 11.0)
        assert (y.start, y.end) == (12.0, 13.0)
        assert (c.low, c.high) == (14.0, 15.0)

    def test_fit_pending_cleared_after_render(self) -> None:
        """A Fit click drives one render; later renders honour toggles again."""
        k = _key()
        plotter = _FakePlotter(frozenset({'x'}), {k: {'x': (0.0, 1.0)}})
        controller = CellAutoscaleController([plotter])
        plot, x, _y, _c = _make_plot_all_handles()
        hook = controller.make_hook()
        hook(plot, None)
        controller._toggles['x'].active = False

        plotter._targets = {k: {'x': (10.0, 11.0)}}
        controller._fit_tool.fire_active(True)
        hook(plot, None)
        assert (x.start, x.end) == (10.0, 11.0)

        plotter._targets = {k: {'x': (20.0, 21.0)}}
        hook(plot, None)
        # Toggle still off, no Fit pending -> previous values stick.
        assert (x.start, x.end) == (10.0, 11.0)

    def test_fit_with_no_targets_is_safe(self) -> None:
        plotter = _FakePlotter(frozenset({'x'}), {})
        controller = CellAutoscaleController([plotter])
        plot, _x, _y, _c = _make_plot_all_handles()
        hook = controller.make_hook()
        hook(plot, None)

        # Should not raise on click or on the following render.
        controller._fit_tool.fire_active(True)
        hook(plot, None)
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

    def test_install_retries_when_toolbar_absent(self) -> None:
        """Missing toolbar on first call must not lock the controller out.

        Real Bokeh plots always have a toolbar; this guards against a
        regression where _tools_installed is set prematurely and a
        subsequent render with a live toolbar misses tool installation.
        """
        k = _key()
        plotter = _FakePlotter(frozenset({'x'}), {k: {'x': (0.0, 1.0)}})
        controller = CellAutoscaleController([plotter])

        plot_no_toolbar = _StubPlot(x_range=_StubRange())
        plot_no_toolbar.state = None  # type: ignore[assignment]
        hook = controller.make_hook()
        hook(plot_no_toolbar, None)
        assert controller._tools_installed is False

        plot, _x, _y, _c = _make_plot_all_handles()
        hook(plot, None)
        assert controller._tools_installed is True
        assert len(plot.state.toolbar.tools) == 2  # x toggle + Fit


class TestHandleRefreshPerRender:
    def test_writes_to_current_plot_handle_each_render(self) -> None:
        """Figure swap (kdim/Layout) replaces the Bokeh handle. The hook
        must write to whichever handle the current plot exposes, not a
        cached reference to a now-detached model."""
        k = _key()
        plotter = _FakePlotter(frozenset({'x'}), {k: {'x': (0.0, 1.0)}})
        controller = CellAutoscaleController([plotter])
        plot, first_x, _y, _c = _make_plot_all_handles()
        hook = controller.make_hook()

        hook(plot, None)
        assert (first_x.start, first_x.end) == (0.0, 1.0)

        # Simulate figure swap: replace x_range with a fresh handle.
        new_x = _StubRange()
        plot.handles['x_range'] = new_x
        plotter._targets = {k: {'x': (10.0, 11.0)}}
        hook(plot, None)

        assert (new_x.start, new_x.end) == (10.0, 11.0)
        # Old handle untouched after swap.
        assert (first_x.start, first_x.end) == (0.0, 1.0)


class TestMultiSession:
    def test_two_controllers_drive_separate_plots(self) -> None:
        """Two sessions of the same cell: each controller owns its toggles
        and writes only to its own session's Bokeh handles."""
        k = _key()
        plotters = [
            _FakePlotter(
                frozenset({'x', 'y'}),
                {k: {'x': (0.0, 1.0), 'y': (2.0, 3.0)}},
            )
        ]
        ctrl_a = CellAutoscaleController(plotters)
        ctrl_b = CellAutoscaleController(plotters)
        plot_a, xa, ya, _ = _make_plot_all_handles()
        plot_b, xb, yb, _ = _make_plot_all_handles()

        ctrl_a.make_hook()(plot_a, None)
        ctrl_b.make_hook()(plot_b, None)

        # Session A turns X off; session B turns Y off. Advance targets.
        ctrl_a._toggles['x'].active = False
        ctrl_b._toggles['y'].active = False
        plotters[0]._targets = {k: {'x': (10.0, 11.0), 'y': (20.0, 21.0)}}

        ctrl_a.make_hook()(plot_a, None)
        ctrl_b.make_hook()(plot_b, None)

        # A: X frozen at first values, Y follows the new target.
        assert (xa.start, xa.end) == (0.0, 1.0)
        assert (ya.start, ya.end) == (20.0, 21.0)
        # B: X follows the new target, Y frozen at first values.
        assert (xb.start, xb.end) == (10.0, 11.0)
        assert (yb.start, yb.end) == (2.0, 3.0)
        # Tool sets are independent.
        assert ctrl_a._toggles['x'] is not ctrl_b._toggles['x']


class TestDispose:
    def test_dispose_removes_fit_on_change(self) -> None:
        plotter = _FakePlotter(frozenset({'x'}), {})
        controller = CellAutoscaleController([plotter])
        plot, _x, _y, _c = _make_plot_all_handles()
        controller.make_hook()(plot, None)

        fit_tool = controller._fit_tool
        assert fit_tool is not None
        assert fit_tool._callbacks, "Fit on_change must be installed"

        controller.dispose()
        assert fit_tool._callbacks == []
        assert controller._fit_tool is None
        assert controller._toggles == {}
        assert controller._tools_installed is False

    def test_controller_collectable_after_dispose(self) -> None:
        """The on_change cycle (controller -> tool -> bound method ->
        controller) keeps a controller alive across cell rebuilds. After
        ``dispose()`` it must be reachable by the cyclic GC immediately."""
        plotter = _FakePlotter(frozenset({'x'}), {})
        controller = CellAutoscaleController([plotter])
        plot, _x, _y, _c = _make_plot_all_handles()
        controller.make_hook()(plot, None)

        controller.dispose()
        ref = weakref.ref(controller)
        del controller
        gc.collect()

        assert ref() is None


class TestHookIdempotency:
    def test_hook_writes_latest_target_each_render(self) -> None:
        k = _key()
        plotter = _FakePlotter(frozenset({'x'}), {k: {'x': (0.0, 1.0)}})
        controller = CellAutoscaleController([plotter])
        plot, x, _y, _c = _make_plot_all_handles()
        hook = controller.make_hook()

        for i in range(5):
            plotter._targets = {k: {'x': (float(i), float(i + 1))}}
            hook(plot, None)
            assert (x.start, x.end) == (float(i), float(i + 1))
