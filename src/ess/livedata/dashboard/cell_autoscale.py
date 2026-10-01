# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Per-cell, per-session autoscale controller for plot toggles + Fit.

A :class:`CellAutoscaleController` owns the Bokeh ``CustomAction`` tools that
appear on a plot cell's toolbar (one per autoscalable axis, plus one for Fit)
and on each HoloViews render writes per-axis ranges based on the toggle state.

An active ``x``/``y`` toggle means "autoscale on change": the range is written
when the data extent differs from what was last written to that range, and is
otherwise left alone. A fixed extent (a spectrum's x-axis, an image's pixel
grid) is therefore fitted once and then keeps the user's pan/zoom, while a
growing one (a timeseries, a correlation histogram) keeps being followed. The
same rule fits a new figure, and refits when a layer or source joins or leaves
and the union of extents changes. A range nothing has been written to yet is
fitted whatever its toggle says: HoloViews leaves the ranges the controller
owns alone (``Plotter.applies_ranges``), so it would otherwise stay at Bokeh's
default of (0, 1). The color axis has no pan/zoom to preserve and is written
on every render while its toggle is active.

The y-range of a 1-D plot is fitted to the values within the figure's visible
x-range rather than to all data (see :class:`~.plots.YProfile`), so a zoom onto
a small feature is not flattened by a peak outside the view. The fit therefore
depends on each figure's x-range, and is redone when the user pans or zooms,
which changes no data: Bokeh's ``RangesUpdate`` event, sent once at the end of
each pan or zoom, applies the y toggle to that figure. With the y toggle active
any pan or zoom therefore ends with y fitted to the visible values, including
one that only changed the y-range.

HoloViews' ``autorange='y'`` option does this fit in the browser, but does not
fit here (HoloViews 1.23):

- It is not set up for an element plotted with ``apply_ranges=False``, which
  1-D layers use to skip HoloViews' own range computation
  (``Plotter.applies_ranges``).
- It pads the y-range linearly, so on a log axis the lower bound can go
  negative.
- It ignores the y toggle, and it would compete with the y-range this
  controller writes on each data update: the browser could draw the full
  range first and the fitted one only after its data callback ran.

The controller removes Bokeh's own reset tool, which returns to the view the
figure was created with -- stale on a live plot -- and gives Fit the reset
tool's icon. Fit writes the current data extent to the axes the controller
owns and, in the browser, resets any x/y range HoloViews owns (e.g. a slicer's
image axes). Fit and switching a toggle on act at once rather than at the next
render, which never comes for a stopped layer.

One cell can be rendered into several figures at once -- its grid cell and a
pop-out window (``widgets/plot_popout.py``). They share one controller, which
holds a single toggle state per axis: turning autoscale off in the pop-out
turns it off in the cell too. Each figure nevertheless gets its *own* tool
models, which the controller keeps in step. A tool model must not sit in two
toolbars: BokehJS creates one tool view per figure, every view runs the tool's
``CustomJS`` on a click, and a toggle flipped once per figure ends up where it
started. Anything the controller tracks per render is keyed by figure, or by
the figure's range handle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary, WeakMethod, WeakSet

import structlog

from .batched_update import batched_update
from .plots import Plotter
from .range_hook import Axis, RangeHandles

logger = structlog.get_logger(__name__)

_TOGGLE_DESCRIPTIONS: dict[Axis, str] = {
    'x': 'X-axis autoscale on data change',
    'y': 'Y-axis autoscale on data change',
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


def _toggle_icons(axis: Axis) -> dict[bool, str]:
    """A toggle's icon when autoscale is on (``True``) and off (``False``)."""
    from .widgets.icons import get_icon_data_uri

    return {
        True: get_icon_data_uri(f'autoscale-{axis}-on'),
        False: get_icon_data_uri(f'autoscale-{axis}'),
    }


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


def _make_fit_action(*, description: str, reset_ranges: list[Any]) -> Any:
    """Create a one-shot Fit ``CustomAction``.

    Clicking sets ``active = true`` server-side via on_change, which triggers
    the controller's Fit handler; the handler resets ``active`` to ``false``
    so the button returns to its neutral visual state. The click also resets
    ``reset_ranges`` in the browser, which is what Bokeh's reset tool does for
    them.

    The tool references the ranges rather than their figure: the controller
    keys each figure's tools by the figure in a weak dictionary, and a tool
    referencing its own key would keep the figure alive.
    """
    from bokeh.models import CustomAction, CustomJS

    tool = CustomAction(active=False, description=description, icon='reset')
    tool.callback = CustomJS(
        args={'tool': tool, 'ranges': reset_ranges},
        code='for (const range of ranges) range.reset();tool.active = true;',
    )
    return tool


# A target written to a range, with the x-range it was fitted to (see
# ``CellAutoscaleController._written``).
_Written = tuple[tuple[float, float] | None, tuple[float, float]]


@dataclass(frozen=True)
class _FigureTools:
    """The autoscale tools on one figure's toolbar."""

    toggles: dict[Axis, Any]
    fit: Any


class CellAutoscaleController:
    """Per-cell, per-session controller for axis autoscale toggles + Fit.

    Installs one Bokeh ``CustomAction`` per autoscalable axis (toggle) plus
    one for Fit on every figure the cell renders into, via a single
    HoloViews-compatible hook. On each render the hook writes that figure's
    per-axis ranges based on the cell's toggle state, which a click on any
    figure's toggle sets for all of them.

    Toggles default to ``True``, so the ranges follow changes in the data
    extent from the first render on.

    Parameters
    ----------
    layer_plotters:
        Plotters for the cell's layers. Targets are unioned
        across all plotters' computed ``_range_targets`` entries.
    """

    def __init__(self, layer_plotters: list[Plotter]) -> None:
        self._plotters: list[Plotter] = list(layer_plotters)
        self._axes: frozenset[Axis] = frozenset().union(
            *(plotter.autoscale_axes for plotter in self._plotters)
        )
        # The cell's toggle state, shown by every figure's toggle of the axis.
        self._active: dict[Axis, bool] = dict.fromkeys(self._axes, True)
        # Toggle icon per axis and state.
        self._toggle_icons: dict[Axis, dict[bool, str]] = {
            axis: _toggle_icons(axis) for axis in self._axes
        }
        # One change handler per axis, shared by that axis's toggles on every
        # figure, so dispose() can detach them by identity.
        self._toggle_handlers: dict[Axis, Callable[[str, bool, bool], None]] = {
            axis: self._make_toggle_handler(axis) for axis in self._axes
        }
        # Figures this cell renders into, as seen by the hook, with the tools
        # installed on each. Created on first render so each session's tools
        # live in the session's own Bokeh document (see dashboard-widgets
        # rules).
        self._figure_tools: WeakKeyDictionary[Any, _FigureTools] = WeakKeyDictionary()
        # x/y axes the controller writes, and those it leaves to HoloViews,
        # which Fit resets in the browser instead.
        self._range_axes: tuple[Axis, ...] = tuple(
            axis for axis in ('x', 'y') if axis in self._axes
        )
        self._reset_axes: tuple[Axis, ...] = tuple(
            axis for axis in ('x', 'y') if axis not in self._axes
        )
        # Last color range written. Re-applied while the toggle is off, so the
        # colorbar stays frozen at it.
        self._clim: tuple[float, float] | None = None
        # Last target written to each x/y range handle, per axis, with the
        # x-range a y target was fitted to (None for x). An active toggle writes
        # only when either differs, so a user's pan/zoom survives renders that
        # do not change the data extent. The y-range is written whenever the
        # x-range moved since, even to a target equal to the last one: the user
        # may have zoomed y with x in between (the gesture's RangesUpdate can
        # arrive after the next render), and y must be refitted together with
        # x. Keyed by the handle rather than the figure: a figure swap or a
        # second figure brings a fresh handle, which nothing has been written
        # to yet.
        self._written: dict[Axis, WeakKeyDictionary[Any, _Written]] = {
            axis: WeakKeyDictionary() for axis in self._range_axes
        }
        # HoloViews plots that rendered into this cell's figures, so Fit and a
        # toggle switched on can write through their current handles at once.
        self._plots: WeakSet = WeakSet()
        # Subscribed to each figure's RangesUpdate. Holds the controller weakly:
        # Bokeh has no way to unsubscribe, and a figure that outlives the cell
        # would otherwise keep the controller and its plotters' data alive.
        on_ranges_update = WeakMethod(self._on_ranges_update)

        def ranges_update_callback(event: Any) -> None:
            if (method := on_ranges_update()) is not None:
                method(event)

        self._ranges_update_callback = ranges_update_callback

    @property
    def axes(self) -> frozenset[Axis]:
        """Axes for which this controller exposes a toggle."""
        return self._axes

    def get_target(
        self, axis: Axis, *, x_window: tuple[float, float] | None = None
    ) -> tuple[float, float] | None:
        """Union of per-plotter ``(lo, hi)`` targets for ``axis``.

        Skips plotters that do not expose ``axis`` and plotters with no
        computed targets yet. Returns ``None`` when no plotter contributes.

        With ``x_window``, the ``y`` target of data with a
        :class:`~.plots.YProfile` is fitted to the values within the window.
        """
        result: tuple[float, float] | None = None
        for plotter in self._plotters:
            if axis not in plotter.autoscale_axes:
                continue
            for key, targets in plotter.iter_range_targets():
                target = targets.get(axis)
                if (
                    axis == 'y'
                    and x_window is not None
                    and (profile := plotter.get_y_profile(key)) is not None
                ):
                    target = profile.target(x_window)
                if target is None:
                    continue
                result = _union(result, target)
        return result

    def make_hook(self) -> Callable[[Any, Any], None]:
        """HoloViews hook that drives this cell's autoscale state.

        On every render the hook:

        1. Installs the ``CustomAction`` tools on the figure's toolbar (once
           per figure, idempotent).
        2. Writes the current targets to the figure's handles, as described
           in :meth:`_apply_targets`. Handles are read from ``plot.handles``
           per render -- HoloViews swaps the figure on kdim/Layout
           transitions, so a cached handle would soon point at a detached
           model.

        When :attr:`axes` is empty the hook is a no-op.
        """
        if not self._axes:
            return _noop_hook

        def hook(plot: Any, element: Any) -> None:
            del element
            self._plots.add(plot)
            self._install_tools(plot)
            self._apply_targets(plot, fit=False)

        return hook

    def dispose(self) -> None:
        """Detach Bokeh callbacks and drop tool references.

        Breaks the controller → tool → on_change-callback → controller
        reference cycle so long sessions don't accumulate detached
        controllers when cells are rebuilt or removed.
        """
        for tools in self._figure_tools.values():
            tools.fit.remove_on_change('active', self._on_fit_active_change)
            for axis, toggle in tools.toggles.items():
                toggle.remove_on_change('active', self._toggle_handlers[axis])
        self._figure_tools.clear()
        self._plots.clear()

    def _install_tools(self, plot: Any) -> None:
        """Ensure this cell's ``CustomAction`` tools are on the figure's toolbar.

        Per figure rather than a controller-wide latch: a cell's hook is
        attached to the session's ``DynamicMap``, which HoloViews can render
        into more than one Bokeh figure (a pop-out window showing the cell a
        second time, a rebuilt cell whose previous pane is still in the
        document, a kdim/Layout figure swap). A one-shot latch let whichever
        figure rendered first consume the installation and left the figure the
        user sees with no toggles at all.

        Each figure gets tools of its own (see the module docstring for why
        they cannot be shared), created in the cell's current toggle state, so
        that state survives a figure swap -- and a pop-out's toolbar drives,
        and displays, the same state as its grid cell's.

        Bokeh's reset tool is removed: Fit takes its place (see the module
        docstring).
        """
        from bokeh.models import ResetTool

        figure = getattr(plot, 'state', None)
        toolbar = getattr(figure, 'toolbar', None)
        if toolbar is None:
            logger.warning(
                "No Bokeh toolbar found for cell autoscale controller; "
                "toggles will be unavailable until the next render."
            )
            return
        tools = self._figure_tools.get(figure)
        if tools is None:
            tools = self._figure_tools[figure] = self._create_tools(figure)
            if 'y' in self._range_axes:
                figure.on_event('rangesupdate', self._ranges_update_callback)
        elif any(tool is tools.fit for tool in toolbar.tools):
            return
        # Tools are set via assignment to keep Bokeh's property setter
        # notified; in-place edits would not trigger change events.
        toolbar.tools = [
            *(tool for tool in toolbar.tools if not isinstance(tool, ResetTool)),
            *tools.toggles.values(),
            tools.fit,
        ]

    def _create_tools(self, figure: Any) -> _FigureTools:
        """Create one figure's per-axis toggles and Fit action."""
        toggles = {}
        for axis in sorted(self._axes):
            icons = self._toggle_icons[axis]
            toggle = _make_toggle_action(
                active=self._active[axis],
                description=_TOGGLE_DESCRIPTIONS[axis],
                on_icon=icons[True],
                off_icon=icons[False],
            )
            toggle.on_change('active', self._toggle_handlers[axis])
            toggles[axis] = toggle
        fit = _make_fit_action(
            description='Fit ranges to current data',
            reset_ranges=[
                getattr(figure, f'{axis}_range') for axis in self._reset_axes
            ],
        )
        fit.on_change('active', self._on_fit_active_change)
        return _FigureTools(toggles=toggles, fit=fit)

    def _make_toggle_handler(self, axis: Axis) -> Callable[[str, bool, bool], None]:
        """Bokeh server-side handler for the ``active`` property of a toggle.

        A click flips one figure's toggle; the handler records the new state
        for the cell and shows it on the axis's toggle in every other figure.
        Those writes fire this handler again, which the state check turns into
        a no-op. Switching an ``x``/``y`` toggle on forgets what was written to
        the axis's ranges and fits them at once, even if the data extent has
        not changed since the user panned or zoomed.
        """
        icons = self._toggle_icons[axis]

        def handler(attr: str, old: bool, new: bool) -> None:
            del attr, old
            if self._active[axis] == new:
                return
            self._active[axis] = new
            with batched_update():
                for tools in list(self._figure_tools.values()):
                    toggle = tools.toggles[axis]
                    # The clicked toggle already shows the state; its client
                    # sets the icon itself, and a server write would only echo
                    # it back.
                    if toggle.active != new:
                        toggle.active = new
                        toggle.icon = icons[new]
                if new:
                    if axis in self._written:
                        self._written[axis].clear()
                    self._apply_to_all_plots(fit=False)

        return handler

    def _apply_targets(self, plot: Any, *, fit: bool) -> None:
        """Write the current targets to ``plot``'s handles.

        An x/y range is written for a Fit, or when the target differs from
        the last one written to that range handle and either the toggle is
        active or nothing has been written to the handle yet (see the module
        docstring). A skipped range keeps its previous value, including any
        manual pan/zoom.

        The color range follows the target while the toggle is active (or for
        a Fit) and stays at the last one written while it is off. It is
        written on every render: HoloViews re-derives the color mapper from
        the data each time, and ``_apply_clim`` is what makes it keep ours.
        """
        # x first: the y target depends on the x-range.
        for axis in self._range_axes:
            self._apply_range(plot, axis, fit=fit)
        if 'c' not in self._axes:
            return
        if (fit or self._active['c']) and (target := self.get_target('c')) is not None:
            self._clim = target
        if self._clim is not None:
            RangeHandles.write(plot, 'c', *self._clim)
            self._apply_clim(plot, self._clim)

    def _apply_range(self, plot: Any, axis: Axis, *, fit: bool) -> None:
        """Write the current target to ``plot``'s x/y range if due.

        See :meth:`_apply_targets` for when it is due.
        """
        handle = RangeHandles.axis_range(plot, axis)
        if handle is None:
            return
        written = self._written[axis]
        if not (fit or self._active[axis] or handle not in written):
            return
        x_window = self._x_window(plot) if axis == 'y' else None
        target = self.get_target(axis, x_window=x_window)
        if target is None or (not fit and written.get(handle) == (x_window, target)):
            return
        # Write the exact (padded) data extent whenever it moved -- there
        # is no hysteresis here. For live data whose min/max drifts every
        # tick (typically the value axis of a 1-D plot) this means one
        # small range patch per update and some range "breathing", and any
        # pan/zoom on that axis is undone on the next update. That is
        # intentional: an active autoscale toggle is meant to track the
        # data, and pan/zoom is kept by turning the toggle off. If the
        # visual jitter proves problematic in practice, introduce a
        # grow/shrink threshold here so the range only moves once the
        # extent leaves a deadband.
        RangeHandles.write(plot, axis, *target)
        written[handle] = (x_window, target)

    def _x_window(self, plot: Any) -> tuple[float, float] | None:
        """The x-range ``plot`` shows, or ``None`` before the controller set it.

        Until then the range holds Bokeh's default, not a view of the data.
        """
        handle = RangeHandles.axis_range(plot, 'x')
        if handle is None or handle not in self._written.get('x', ()):
            return None
        return RangeHandles.read(plot, 'x')

    @staticmethod
    def _apply_clim(plot: Any, clim: tuple[float, float]) -> None:
        """Pin ``cm_plot.clim`` so HV's next render keeps our color range.

        HV's ``ColorbarPlot`` reads ``self.clim`` on every render; when both
        entries are finite it uses them instead of deriving low/high from
        data extent (see holoviews/plotting/bokeh/element.py
        ``_get_colormapper``). Setting ``clim`` on the colormapped sub-plot
        makes the next render's ``_get_colormapper`` pass through our value
        untouched, which is what freezes the colorbar while the toggle is off.
        Writing the color mapper itself as well covers the render in
        progress.

        Note: this reaches past HoloViews' public surface and is likely to
        break on HV bumps. If a documented hook becomes available, switch to
        it and file an upstream issue tracking the need.
        """
        cm_plot = RangeHandles.color_mapper_plot(plot)
        if cm_plot is not None and hasattr(cm_plot, 'clim'):
            cm_plot.clim = clim

    def _apply_to_all_plots(self, *, fit: bool) -> None:
        """Apply the current targets to every figure this cell renders into.

        Batched into one message, as a render's writes are. Sent one by one,
        the browser can paint with x fitted and y not yet, and a data-aspect
        figure (see ``frame_aspect.py``) resizes its frame to that transient
        x/y ratio for a frame.
        """
        with batched_update():
            for plot in list(self._plots):
                self._apply_targets(plot, fit=fit)

    def _on_ranges_update(self, event: Any) -> None:
        """Refit the y-range of the figure the user panned or zoomed, if active.

        ``event`` is Bokeh's ``RangesUpdate``, for the figure ``event.model``.

        The refit is written as for a Fit, even when it equals the last target
        written: pan and wheel zoom move y along with x, so the range need no
        longer show what was last written to it.
        """
        if not self._active['y']:
            return
        with batched_update():
            for plot in list(self._plots):
                if plot.state is event.model:
                    self._apply_range(plot, 'y', fit=True)

    def _on_fit_active_change(self, attr: str, old: bool, new: bool) -> None:
        """Bokeh server-side handler for the Fit tool's ``active`` property.

        When the user clicks Fit, ``active`` flips to ``True``; every figure
        this cell renders into is fitted at once, regardless of toggle state,
        through its plot's current handles. One click on either figure's Fit
        therefore fits the grid cell and the pop-out alike.

        The tools are re-armed even if fitting raises: the browser only ever
        sets ``active`` to ``True``, so a tool left active would ignore every
        later click.
        """
        del attr, old
        if not new:
            return
        with batched_update():
            try:
                self._apply_to_all_plots(fit=True)
            finally:
                for tools in list(self._figure_tools.values()):
                    tools.fit.active = False


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
