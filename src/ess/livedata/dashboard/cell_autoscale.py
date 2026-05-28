# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Per-cell, per-session autoscale controller for plot toggles + Fit.

A :class:`CellAutoscaleController` owns the Bokeh ``CustomAction`` tools that
appear on a plot cell's toolbar (one per autoscalable axis, plus one for Fit),
captures handles to the figure's ``Range1d`` / color mapper, and on each
HoloViews render writes per-axis ranges based on the toggle state.

See ``docs/developer/plans/plot-axis-autoscale-implementation.md`` §2 Phase 3.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .plots import Plotter
from .range_hook import Axis, RangeHandles

_TOGGLE_DESCRIPTIONS: dict[Axis, str] = {
    'x': 'X-axis autoscale',
    'y': 'Y-axis autoscale',
    'c': 'Color autoscale',
}


def _union(
    a: tuple[float, float] | None, b: tuple[float, float] | None
) -> tuple[float, float] | None:
    """Union of two ``(lo, hi)`` ranges; ``None`` entries are dropped."""
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]))


def _make_toggle_action(
    *,
    active: bool,
    description: str,
    on_icon: str | None,
    off_icon: str | None,
) -> Any:
    """Create a stateful toggle ``CustomAction`` toolbar tool.

    Bokeh's ``active_callback="auto"`` is documented to toggle ``active`` on
    click, but its JS implementation only fires when ``callback`` is also
    non-null (``_execute`` short-circuits when ``callback == null``). We
    attach an explicit ``CustomJS`` that flips ``active`` and swaps the icon
    so the button gives clear visual feedback alongside Bokeh's active-tool
    highlight.
    """
    from bokeh.models import CustomAction, CustomJS

    initial_icon = on_icon if active else off_icon
    tool = CustomAction(active=active, description=description, icon=initial_icon)
    tool.callback = CustomJS(
        args={'tool': tool, 'on_icon': on_icon, 'off_icon': off_icon},
        code=(
            'tool.active = !tool.active;tool.icon = tool.active ? on_icon : off_icon;'
        ),
    )
    return tool


def _make_fit_action(*, description: str, icon: str | None) -> Any:
    """Create a one-shot Fit ``CustomAction``.

    Clicking sets ``active = true`` server-side via on_change, which triggers
    the controller's Fit handler; the handler resets ``active`` to ``false``
    so the button returns to its neutral visual state.
    """
    from bokeh.models import CustomAction, CustomJS

    tool = CustomAction(active=False, description=description, icon=icon)
    tool.callback = CustomJS(args={'tool': tool}, code='tool.active = true')
    return tool


class CellAutoscaleController:
    """Per-cell, per-session controller for axis autoscale toggles + Fit.

    Owns one Bokeh ``CustomAction`` per autoscalable axis (toggle) plus one
    for Fit. Owns a :class:`RangeHandles`. Exposes a single HoloViews-compatible
    hook that installs the tools on the figure toolbar, captures handles, and
    on each render writes per-axis ranges based on each toggle's ``.active``.

    Toggles default to ``True`` so the very first render with real data snaps
    the range away from the pipe's dummy bounds (strategy §4.5).

    Parameters
    ----------
    layer_plotters:
        Plotters for the cell's layers (deduplicated). Targets are unioned
        across all plotters' computed ``_range_targets`` entries.
    """

    def __init__(self, layer_plotters: list[Plotter]) -> None:
        self._plotters: list[Plotter] = list(layer_plotters)
        self._axes: frozenset[Axis] = frozenset().union(
            *(plotter.AUTOSCALE_AXES for plotter in self._plotters)
        )
        self._handles = RangeHandles()
        # Lazy-created on first render so each session's tools live in the
        # session's own Bokeh document (see dashboard-widgets rules).
        self._toggles: dict[Axis, Any] = {}
        self._fit_tool: Any | None = None
        self._tools_installed = False
        # Last target written per axis. Read back on subsequent off-state
        # renders so the c-axis freeze and the Fit button have a stable
        # value to apply.
        self._last_targets: dict[Axis, tuple[float, float]] = {}

    @property
    def axes(self) -> frozenset[Axis]:
        """Axes for which this controller exposes a toggle."""
        return self._axes

    @property
    def handles(self) -> RangeHandles:
        """The cell's captured Bokeh handles (populated on first render)."""
        return self._handles

    def get_target(self, axis: Axis) -> tuple[float, float] | None:
        """Union of per-plotter ``(lo, hi)`` targets for ``axis``.

        Skips plotters that do not expose ``axis`` and plotters with no
        computed targets yet. Returns ``None`` when no plotter contributes.
        """
        result: tuple[float, float] | None = None
        for plotter in self._plotters:
            if axis not in plotter.AUTOSCALE_AXES:
                continue
            for _key, targets in plotter.iter_range_targets():
                target = targets.get(axis)
                if target is None:
                    continue
                result = _union(result, target)
        return result

    def make_hook(self) -> Callable[[Any, Any], None]:
        """HoloViews hook that drives this cell's autoscale state.

        On every render the hook:

        1. Installs the ``CustomAction`` tools on the figure toolbar (once
           per session, idempotent).
        2. Captures ``x_range`` / ``y_range`` / ``color_mapper`` handles.
        3. For each axis whose toggle is active, writes ``(lo, hi)`` to the
           corresponding Bokeh handle.

        When :attr:`axes` is empty the hook is a no-op.
        """
        if not self._axes:
            return _noop_hook

        def hook(plot: Any, element: Any) -> None:
            del element
            self._install_tools(plot)
            self._handles.capture(plot)
            self._apply_targets()
            # HV's ColorbarPlot reads ``self.clim`` on every render; when both
            # entries are finite it uses them instead of deriving low/high from
            # data extent (see holoviews/plotting/bokeh/element.py
            # ``_get_colormapper``). Setting clim on the colormapped sub-plot
            # makes the next render's ``_get_colormapper`` pass through our
            # value untouched -- the exact mechanism the prototype relies on
            # (when the toggle is off it omits ``clim`` from element opts so
            # HV's plot param sticks at the last finite value).
            cm_plot = self._handles.color_mapper_plot
            if (
                cm_plot is not None
                and hasattr(cm_plot, 'clim')
                and 'c' in self._axes
                and (clim := self._last_targets.get('c')) is not None
            ):
                cm_plot.clim = clim

        return hook

    def _install_tools(self, plot: Any) -> None:
        """Install ``CustomAction`` tools on the figure toolbar once.

        Idempotent: subsequent calls are no-ops, mirroring the pattern in
        ``flatten_plotter._make_hover_hook``.
        """
        if self._tools_installed:
            return
        try:
            from .widgets.icons import get_icon_data_uri

            on_icon = get_icon_data_uri('lock-open')
            off_icon = get_icon_data_uri('lock')
            fit_icon = get_icon_data_uri('arrows-maximize')
        except Exception:
            # Tests with stub plots don't require real icons.
            on_icon = None
            off_icon = None
            fit_icon = None

        for axis in sorted(self._axes):
            self._toggles[axis] = _make_toggle_action(
                active=True,
                description=_TOGGLE_DESCRIPTIONS[axis],
                on_icon=on_icon,
                off_icon=off_icon,
            )
        self._fit_tool = _make_fit_action(
            description='Fit ranges to current data',
            icon=fit_icon,
        )
        self._fit_tool.on_change('active', self._on_fit_active_change)

        toolbar = getattr(getattr(plot, 'state', None), 'toolbar', None)
        if toolbar is not None:
            # Tools are added via assignment to keep Bokeh's property setter
            # notified; in-place append would not trigger change events.
            toolbar.tools = [
                *toolbar.tools,
                *self._toggles.values(),
                self._fit_tool,
            ]
        self._tools_installed = True

    def _apply_targets(self) -> None:
        """Write current targets to handles based on each axis's toggle state.

        For x/y the hook writes only when the toggle is active; HoloViews
        honours ``framewise=False`` for ``Range1d`` so the previous range
        (and any manual pan/zoom) is preserved when we skip. For ``c`` the
        hook also re-applies the last target when the toggle is off -- a
        belt-and-suspenders safety net alongside the ``plot.clim`` write
        in ``make_hook`` (which is what actually freezes the colorbar by
        making HV's next ``_get_colormapper`` use our value).
        """
        for axis in self._axes:
            toggle = self._toggles.get(axis)
            if toggle is None:
                continue
            if toggle.active:
                target = self.get_target(axis)
                if target is None:
                    continue
                self._last_targets[axis] = target
                self._write_axis(axis, target)
            elif axis == 'c' and (frozen := self._last_targets.get(axis)) is not None:
                self._write_axis(axis, frozen)

    def _write_axis(self, axis: Axis, target: tuple[float, float]) -> None:
        """Write a single axis target onto its captured handle."""
        lo, hi = target
        if axis == 'c':
            mapper = self._handles.color_mapper
            if mapper is None:
                return
            mapper.low = lo
            mapper.high = hi
        else:
            handle = self._handles.x_range if axis == 'x' else self._handles.y_range
            if handle is None:
                return
            handle.start = lo
            handle.end = hi

    def _on_fit_active_change(self, attr: str, old: bool, new: bool) -> None:
        """Bokeh server-side handler for the Fit tool's ``active`` property.

        When the user clicks Fit, ``active`` flips to ``True``; we write
        current targets to all handles regardless of toggle state, then
        reset ``active`` to ``False`` so the button returns to its
        neutral visual state.
        """
        del attr, old
        if not new:
            return
        for axis in self._axes:
            target = self.get_target(axis)
            if target is None:
                continue
            self._last_targets[axis] = target
            self._write_axis(axis, target)
        if self._fit_tool is not None:
            self._fit_tool.active = False


def _noop_hook(plot: Any, element: Any) -> None:
    """No-op hook used when a cell has no autoscalable axes."""
    del plot, element


def build_controller_from_layers(
    layer_plotters: list[Plotter],
) -> CellAutoscaleController | None:
    """Build a controller for a cell, or ``None`` when no axes are autoscalable.

    Helper for the wiring in ``widgets/plot_grid_tabs.py`` — keeps that
    callsite a one-liner and returns ``None`` so the caller can simply skip
    appending a hook when no plotter exposes any axes.
    """
    if not layer_plotters:
        return None
    controller = CellAutoscaleController(layer_plotters)
    if not controller.axes:
        return None
    return controller
