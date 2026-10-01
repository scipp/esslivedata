# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for :class:`CellAutoscaleController`.

Uses lightweight stubs in the spirit of ``range_hook_test.py``: real
``Plotter`` instances aren't needed -- only the public surface
(``AUTOSCALE_AXES``, ``iter_range_targets``) is exercised. The toolbar tools
are real Bokeh ``CustomAction`` models: setting ``active`` on one fires the
controller's handler as a click in the browser would, without a document.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.dashboard.cell_autoscale import (
    CellAutoscaleController,
    build_controller_from_layers,
)
from ess.livedata.dashboard.plots import YProfile
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
    the ``autoscale_axes`` property, ``iter_range_targets()`` and
    ``get_y_profile()``.
    """

    def __init__(
        self,
        axes: frozenset[Axis],
        targets_by_key: dict[DataKey, dict[Axis, tuple[float, float]]] | None = None,
        profiles_by_key: dict[DataKey, YProfile] | None = None,
    ) -> None:
        self.autoscale_axes = axes
        self._targets = targets_by_key or {}
        self._profiles = profiles_by_key or {}

    def iter_range_targets(self):
        return iter(self._targets.items())

    def get_y_profile(self, data_key: DataKey) -> YProfile | None:
        return self._profiles.get(data_key)


class _RaisingPlotter(_FakePlotter):
    """A plotter whose targets fail to read once ``fail`` is set."""

    fail = False

    def iter_range_targets(self):
        if self.fail:
            raise RuntimeError("targets unavailable")
        return super().iter_range_targets()


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
    """Stub for the Bokeh figure: its toolbar, plus the ranges Fit may reset."""

    def __init__(self, toolbar: _StubToolbar) -> None:
        from bokeh.models import Range1d

        self.toolbar = toolbar
        self.x_range = Range1d()
        self.y_range = Range1d()
        self.document = None
        self.event_callbacks: dict[str, list[Any]] = {}

    def on_event(self, event: str, *callbacks: Any) -> None:
        self.event_callbacks.setdefault(event, []).extend(callbacks)


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


_TOGGLE = {
    'x': 'X-axis autoscale on data change',
    'y': 'Y-axis autoscale on data change',
    'c': 'Color autoscale',
}
_FIT = 'Fit ranges to current data'


def _tool(plot: _StubPlot, description: str) -> Any:
    """The tool with the given tooltip on the plot's figure."""
    return next(t for t in plot.state.toolbar.tools if t.description == description)


def _click_toggle(plot: _StubPlot, axis: Axis, active: bool = False) -> None:
    """Simulate the user switching the plot's ``axis`` toggle to ``active``."""
    _tool(plot, _TOGGLE[axis]).active = active


def _click_fit(plot: _StubPlot) -> None:
    """Simulate the user clicking the plot's Fit button."""
    _tool(plot, _FIT).active = True


def _end_pan_or_zoom(plot: _StubPlot) -> None:
    """Simulate the ``RangesUpdate`` Bokeh sends at the end of a pan or zoom."""
    event = SimpleNamespace(model=plot.state)
    for callback in plot.state.event_callbacks.get('rangesupdate', []):
        callback(event)


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
        _click_toggle(plot, 'x')
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
        _click_toggle(plot, 'c')
        c.low, c.high = 99.0, 999.0
        plotter._targets = {k: {'c': (100.0, 200.0)}}
        hook(plot, None)

        # Hook must re-write the last-known target, overriding HV's update.
        assert (c.low, c.high) == (0.0, 10.0)


class TestAutoscaleOnChange:
    """An active x/y toggle writes only when the data extent changed."""

    @pytest.fixture
    def rendered(self) -> tuple[_FakePlotter, Any, _StubPlot, _StubRange]:
        plotter = _FakePlotter(frozenset({'x', 'y'}), {_key(): {'x': (0.0, 10.0)}})
        controller = CellAutoscaleController([plotter])
        hook = controller.make_hook()
        plot, x, _y, _c = _make_plot_all_handles()
        hook(plot, None)
        return plotter, hook, plot, x

    def test_zoom_survives_render_with_unchanged_extent(self, rendered) -> None:
        _plotter, hook, plot, x = rendered
        x.start, x.end = 2.0, 3.0  # user zooms in

        hook(plot, None)

        assert (x.start, x.end) == (2.0, 3.0)

    def test_changed_extent_resets_zoom(self, rendered) -> None:
        plotter, hook, plot, x = rendered
        x.start, x.end = 2.0, 3.0
        plotter._targets = {_key(): {'x': (0.0, 20.0)}}

        hook(plot, None)

        assert (x.start, x.end) == (0.0, 20.0)

    def test_source_joining_refits(self, rendered) -> None:
        plotter, hook, plot, x = rendered
        x.start, x.end = 2.0, 3.0
        plotter._targets = {
            _key('a'): {'x': (0.0, 10.0)},
            _key('b'): {'x': (5.0, 30.0)},
        }

        hook(plot, None)

        assert (x.start, x.end) == (0.0, 30.0)

    def test_switching_toggle_on_refits_unchanged_extent(self, rendered) -> None:
        _plotter, hook, plot, x = rendered
        _click_toggle(plot, 'x', active=False)
        x.start, x.end = 2.0, 3.0
        hook(plot, None)
        assert (x.start, x.end) == (2.0, 3.0)

        _click_toggle(plot, 'x', active=True)

        assert (x.start, x.end) == (0.0, 10.0)

    def test_fit_refits_unchanged_extent_with_toggle_on(self, rendered) -> None:
        _plotter, _hook, plot, x = rendered
        x.start, x.end = 2.0, 3.0

        _click_fit(plot)

        assert (x.start, x.end) == (0.0, 10.0)

    def test_new_figure_is_fitted(self, rendered) -> None:
        _plotter, hook, _plot, _x = rendered
        popout = _StubPlot(x_range=_StubRange())

        hook(popout, None)

        x = popout.handles['x_range']
        assert (x.start, x.end) == (0.0, 10.0)

    def test_new_figure_is_fitted_with_toggle_off(self, rendered) -> None:
        """Nothing else sets the range, so it would stay at Bokeh's (0, 1)."""
        _plotter, hook, plot, _x = rendered
        _click_toggle(plot, 'x', active=False)
        popout = _StubPlot(x_range=_StubRange())

        hook(popout, None)

        x = popout.handles['x_range']
        assert (x.start, x.end) == (0.0, 10.0)

    def test_color_written_every_render(self) -> None:
        """HoloViews re-derives the color mapper from data on every render."""
        plotter = _FakePlotter(frozenset({'c'}), {_key(): {'c': (0.0, 10.0)}})
        controller = CellAutoscaleController([plotter])
        hook = controller.make_hook()
        plot, _x, _y, c = _make_plot_all_handles()
        hook(plot, None)
        c.low, c.high = 99.0, 999.0

        hook(plot, None)

        assert (c.low, c.high) == (0.0, 10.0)


def _peak_profile() -> YProfile:
    """Flat values at x = 0..4 next to a peak at x = 8, unpadded."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0])
    y = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 100.0])
    return YProfile(x_lo=x, x_hi=x, y_lo=y, y_hi=y, pad=0.0, log=False)


class TestYFitsVisibleX:
    """The y-range of 1-D data is fitted to the values within the x-range."""

    @pytest.fixture
    def rendered(self) -> tuple[Any, _StubPlot, _StubRange, _StubRange]:
        profile = _peak_profile()
        plotter = _FakePlotter(
            frozenset({'x', 'y'}),
            {_key(): {'x': (0.0, 8.0), 'y': profile.target()}},
            {_key(): profile},
        )
        hook = CellAutoscaleController([plotter]).make_hook()
        plot, x, y, _c = _make_plot_all_handles()
        hook(plot, None)
        return hook, plot, x, y

    def test_first_render_fits_all_values(self, rendered) -> None:
        _hook, _plot, _x, y = rendered

        assert (y.start, y.end) == (1.0, 100.0)

    def test_render_fits_values_within_zoomed_x(self, rendered) -> None:
        hook, plot, x, y = rendered
        x.start, x.end = 0.0, 4.0  # user zooms in, away from the peak

        hook(plot, None)

        assert (x.start, x.end) == (0.0, 4.0)
        assert (y.start, y.end) == (1.0, 3.0)

    def test_end_of_pan_refits_without_a_render(self, rendered) -> None:
        _hook, plot, x, y = rendered
        x.start, x.end = 2.5, 4.5

        _end_pan_or_zoom(plot)

        assert (y.start, y.end) == (2.0, 3.0)

    def test_end_of_pan_keeps_y_with_toggle_off(self, rendered) -> None:
        _hook, plot, x, y = rendered
        _click_toggle(plot, 'y', active=False)
        x.start, x.end = 0.0, 4.0

        _end_pan_or_zoom(plot)

        assert (y.start, y.end) == (1.0, 100.0)

    def test_end_of_pan_refits_y_moved_with_unchanged_visible_values(
        self, rendered
    ) -> None:
        """A small pan moves y along with x without changing the visible values,
        so the target equals the last one written."""
        _hook, plot, x, y = rendered
        x.start, x.end = 0.1, 8.1
        y.start, y.end = 11.0, 110.0

        _end_pan_or_zoom(plot)

        assert (y.start, y.end) == (1.0, 100.0)

    def test_x_refit_refits_y_zoomed_before_the_gesture_event(self) -> None:
        """A growing extent refits x while y still shows a zoom whose
        RangesUpdate has not arrived; y must follow in the same render even
        though its target equals the last one written."""
        profile = _peak_profile()
        plotter = _FakePlotter(
            frozenset({'x', 'y'}),
            {_key(): {'x': (0.0, 8.0), 'y': profile.target()}},
            {_key(): profile},
        )
        hook = CellAutoscaleController([plotter]).make_hook()
        plot, x, y, _c = _make_plot_all_handles()
        hook(plot, None)
        x.start, x.end = 0.0, 4.0
        y.start, y.end = 50.0, 60.0
        plotter._targets[_key()]['x'] = (0.0, 9.0)

        hook(plot, None)

        assert (x.start, x.end) == (0.0, 9.0)
        assert (y.start, y.end) == (1.0, 100.0)

    def test_window_without_values_keeps_y(self, rendered) -> None:
        _hook, plot, x, y = rendered
        x.start, x.end = 5.0, 7.0

        _end_pan_or_zoom(plot)

        assert (y.start, y.end) == (1.0, 100.0)

    def test_fit_fits_all_values(self, rendered) -> None:
        _hook, plot, x, y = rendered
        x.start, x.end = 0.0, 4.0
        _end_pan_or_zoom(plot)

        _click_fit(plot)

        assert (x.start, x.end) == (0.0, 8.0)
        assert (y.start, y.end) == (1.0, 100.0)

    def test_figure_does_not_keep_disposed_controller_alive(self) -> None:
        """Bokeh cannot unsubscribe from ``RangesUpdate``; the subscription
        must not keep the controller and its plotters alive."""
        controller = CellAutoscaleController([_FakePlotter(frozenset({'x', 'y'}))])
        plot, *_ = _make_plot_all_handles()
        controller.make_hook()(plot, None)

        controller.dispose()
        ref = weakref.ref(controller)
        del controller
        gc.collect()

        assert ref() is None
        _end_pan_or_zoom(plot)  # a dead controller's subscription is a no-op


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
        _click_toggle(plot, 'c')
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
            _click_toggle(plot, axis)
        x.start = x.end = None
        y.start = y.end = None
        c.low = c.high = None
        # Advance targets so we can see what Fit wrote.
        plotter._targets = {
            k: {'x': (10.0, 11.0), 'y': (12.0, 13.0), 'c': (14.0, 15.0)}
        }

        _click_fit(plot)

        assert _tool(plot, _FIT).active is False

        assert (x.start, x.end) == (10.0, 11.0)
        assert (y.start, y.end) == (12.0, 13.0)
        assert (c.low, c.high) == (14.0, 15.0)

    def test_fit_is_one_shot(self) -> None:
        """A Fit writes once; later renders honour the toggles again."""
        k = _key()
        plotter = _FakePlotter(frozenset({'x'}), {k: {'x': (0.0, 1.0)}})
        controller = CellAutoscaleController([plotter])
        plot, x, _y, _c = _make_plot_all_handles()
        hook = controller.make_hook()
        hook(plot, None)
        _click_toggle(plot, 'x')

        plotter._targets = {k: {'x': (10.0, 11.0)}}
        _click_fit(plot)
        assert (x.start, x.end) == (10.0, 11.0)

        plotter._targets = {k: {'x': (20.0, 21.0)}}
        hook(plot, None)
        # Toggle still off -> the fitted values stick.
        assert (x.start, x.end) == (10.0, 11.0)

    def test_fit_with_no_targets_is_safe(self) -> None:
        plotter = _FakePlotter(frozenset({'x'}), {})
        controller = CellAutoscaleController([plotter])
        plot, _x, _y, _c = _make_plot_all_handles()
        hook = controller.make_hook()
        hook(plot, None)

        # Should not raise on click or on the following render.
        _click_fit(plot)
        hook(plot, None)
        assert _tool(plot, _FIT).active is False


class TestFitReplacesReset:
    def test_bokeh_reset_tool_removed(self) -> None:
        from bokeh.models import PanTool, ResetTool

        controller = CellAutoscaleController([_FakePlotter(frozenset({'x', 'y'}))])
        plot, *_ = _make_plot_all_handles()
        pan = PanTool()
        plot.state.toolbar.tools = [pan, ResetTool()]

        controller.make_hook()(plot, None)

        tools = plot.state.toolbar.tools
        assert pan in tools
        assert not any(isinstance(tool, ResetTool) for tool in tools)
        assert _tool(plot, _FIT).icon == 'reset'

    def test_fit_writes_without_waiting_for_a_render(self) -> None:
        """A stopped layer renders no more frames; Fit must still act."""
        plotter = _FakePlotter(frozenset({'x'}), {_key(): {'x': (0.0, 10.0)}})
        controller = CellAutoscaleController([plotter])
        plot, x, _y, _c = _make_plot_all_handles()
        controller.make_hook()(plot, None)
        x.start, x.end = 2.0, 3.0

        _click_fit(plot)

        assert (x.start, x.end) == (0.0, 10.0)

    @pytest.mark.parametrize(
        ('axes', 'reset_axes'),
        [
            (frozenset({'x', 'y', 'c'}), ()),
            (frozenset({'c'}), ('x', 'y')),
        ],
    )
    def test_fit_resets_axes_the_controller_does_not_own(
        self, axes: frozenset[Axis], reset_axes: tuple[Axis, ...]
    ) -> None:
        controller = CellAutoscaleController([_FakePlotter(axes)])
        plot, *_ = _make_plot_all_handles()

        controller.make_hook()(plot, None)

        ranges = [getattr(plot.state, f'{axis}_range') for axis in reset_axes]
        assert _tool(plot, _FIT).callback.args['ranges'] == ranges

    def test_fit_rearmed_when_fitting_raises(self) -> None:
        """The browser only sets ``active`` to True; a tool left active would
        ignore every later click."""
        plotter = _RaisingPlotter(frozenset({'x'}), {_key(): {'x': (0.0, 1.0)}})
        controller = CellAutoscaleController([plotter])
        plot, *_ = _make_plot_all_handles()
        controller.make_hook()(plot, None)
        plotter.fail = True

        with pytest.raises(RuntimeError):
            _click_fit(plot)

        assert _tool(plot, _FIT).active is False

    def test_figure_collectable_while_controller_lives(self) -> None:
        """A closed pop-out's figure must not outlive it until the cell is
        rebuilt."""
        controller = CellAutoscaleController([_FakePlotter(frozenset({'c'}))])
        hook = controller.make_hook()
        plot, *_ = _make_plot_all_handles()
        hook(plot, None)
        ref = weakref.ref(plot.state)

        del plot
        gc.collect()

        assert ref() is None


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
        assert build_controller_from_layers([plotter]) is None

    def test_build_controller_returns_none_when_no_plotters(self) -> None:
        assert build_controller_from_layers([]) is None


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
        regression where the controller records the installation prematurely
        and a subsequent render with a live toolbar misses it.
        """
        k = _key()
        plotter = _FakePlotter(frozenset({'x'}), {k: {'x': (0.0, 1.0)}})
        controller = CellAutoscaleController([plotter])

        plot_no_toolbar = _StubPlot(x_range=_StubRange())
        plot_no_toolbar.state = None  # type: ignore[assignment]
        hook = controller.make_hook()
        hook(plot_no_toolbar, None)

        plot, _x, _y, _c = _make_plot_all_handles()
        hook(plot, None)
        assert len(plot.state.toolbar.tools) == 2  # x toggle + Fit

    def test_second_figure_also_gets_tools(self) -> None:
        """Every figure the cell's hook renders into must carry the tools.

        The hook lives on the session's DynamicMap, which HoloViews can render
        into more than one Bokeh figure (a pop-out, a rebuilt cell whose
        previous pane is still in the document, or a kdim/Layout figure swap).
        Installing only into the figure that happens to render first leaves
        the figure the user sees with no toggles.
        """
        k = _key()
        plotter = _FakePlotter(frozenset({'x', 'y'}), {k: {'x': (0.0, 1.0)}})
        controller = CellAutoscaleController([plotter])
        hook = controller.make_hook()

        first, *_ = _make_plot_all_handles()
        second, *_ = _make_plot_all_handles()
        hook(first, None)
        hook(second, None)

        first_tools = first.state.toolbar.tools
        second_tools = second.state.toolbar.tools
        assert [t.description for t in second_tools] == [
            t.description for t in first_tools
        ]
        assert len(second_tools) == 3

    def test_figures_get_tool_models_of_their_own(self) -> None:
        """A tool model in two toolbars cannot be toggled.

        BokehJS runs a tool's ``CustomJS`` once per figure holding it, so a
        shared toggle flips once per figure on every click and, with two
        figures, lands back where it started.
        """
        plotter = _FakePlotter(frozenset({'x', 'y'}), {})
        controller = CellAutoscaleController([plotter])
        hook = controller.make_hook()

        first, *_ = _make_plot_all_handles()
        second, *_ = _make_plot_all_handles()
        hook(first, None)
        hook(second, None)

        shared = {id(t) for t in first.state.toolbar.tools} & {
            id(t) for t in second.state.toolbar.tools
        }
        assert shared == set()


class TestToggleStateAcrossFigures:
    """A cell rendered into two figures (grid cell and pop-out) has one state."""

    @pytest.fixture
    def figures(self) -> tuple[_FakePlotter, Any, _StubPlot, _StubPlot]:
        plotter = _FakePlotter(frozenset({'x', 'y'}), {_key(): {'x': (0.0, 1.0)}})
        controller = CellAutoscaleController([plotter])
        hook = controller.make_hook()
        first = _StubPlot(x_range=_StubRange())
        second = _StubPlot(x_range=_StubRange())
        hook(first, None)
        hook(second, None)
        return plotter, hook, first, second

    def test_toggle_in_one_figure_shows_in_the_other(self, figures) -> None:
        _plotter, _hook, first, second = figures
        other = _tool(second, _TOGGLE['x'])
        on_icon = other.icon

        _click_toggle(first, 'x')

        assert other.active is False
        assert other.icon != on_icon
        assert _tool(second, _TOGGLE['y']).active is True

    def test_toggle_in_one_figure_freezes_both(self, figures) -> None:
        plotter, hook, first, second = figures

        _click_toggle(second, 'x')
        plotter._targets = {_key(): {'x': (10.0, 11.0)}}
        hook(first, None)
        hook(second, None)

        for plot in (first, second):
            x = plot.handles['x_range']
            assert (x.start, x.end) == (0.0, 1.0)

    def test_figure_rendered_later_starts_in_the_current_state(self) -> None:
        plotter = _FakePlotter(frozenset({'x'}), {})
        controller = CellAutoscaleController([plotter])
        hook = controller.make_hook()
        first, *_ = _make_plot_all_handles()
        hook(first, None)
        _click_toggle(first, 'x')

        later, *_ = _make_plot_all_handles()
        hook(later, None)

        assert _tool(later, _TOGGLE['x']).active is False

    def test_fit_in_one_figure_fits_both(self, figures) -> None:
        plotter, _hook, first, second = figures
        _click_toggle(first, 'x')
        plotter._targets = {_key(): {'x': (10.0, 11.0)}}

        _click_fit(second)

        for plot in (first, second):
            x = plot.handles['x_range']
            assert (x.start, x.end) == (10.0, 11.0)


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
        _click_toggle(plot_a, 'x')
        _click_toggle(plot_b, 'y')
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
        assert _tool(plot_a, _TOGGLE['x']) is not _tool(plot_b, _TOGGLE['x'])


class TestDispose:
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
