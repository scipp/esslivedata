# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""HoloViews hooks that drive per-axis range and color-mapper limits.

The hook factory here is the foundation of the per-axis autoscale model
(see ``docs/developer/plans/plot-axis-autoscale-implementation.md``). A
hook is created once at plot compose time and re-invoked on every render;
it reads the latest target range through a closure and writes it onto
Bokeh's ``Range1d`` / ``LinearColorMapper`` handles directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

Axis = Literal['x', 'y', 'c']
RangeTargets = dict[Axis, tuple[float, float]]


class RangeHandles:
    """Bokeh handles for a cell's x/y ranges and color mapper.

    Populated on the first hook invocation per session. The cell-level
    autoscale controller keeps a reference so a Fit action can mutate the
    handles without a Pipe round-trip.

    For Overlay plots (e.g., image + ROI rectangles), ``x_range``/``y_range``
    live on the top-level ``OverlayPlot`` while ``color_mapper`` lives on the
    image's sub-plot. We capture the sub-plot too so the controller can set
    its ``clim`` parameter -- the prototype's freeze mechanism (a finite
    ``self.clim`` causes HoloViews to skip data-range derivation on the next
    render).
    """

    def __init__(self) -> None:
        self.x_range: Any | None = None
        self.y_range: Any | None = None
        self.color_mapper: Any | None = None
        self.color_mapper_plot: Any | None = None

    def capture(self, plot: Any) -> None:
        """Capture handles from ``plot.handles``; idempotent across renders.

        Searches sub-plots when the top-level plot doesn't carry a
        ``color_mapper`` (Overlay case).
        """
        handles = plot.handles
        if self.x_range is None:
            self.x_range = handles.get('x_range')
        if self.y_range is None:
            self.y_range = handles.get('y_range')
        if self.color_mapper is None:
            cm = handles.get('color_mapper')
            if cm is not None:
                self.color_mapper = cm
                self.color_mapper_plot = plot
            else:
                for sub in (getattr(plot, 'subplots', None) or {}).values():
                    sub_cm = sub.handles.get('color_mapper')
                    if sub_cm is not None:
                        self.color_mapper = sub_cm
                        self.color_mapper_plot = sub
                        break


def make_range_hook(
    axis: Axis,
    get_target: Callable[[], tuple[float, float] | None],
    get_enabled: Callable[[], bool],
    handles: RangeHandles,
) -> Callable[[Any, Any], None]:
    """Create a HoloViews hook that writes a per-axis range every render.

    The hook captures handles into ``handles`` on first call, then -- if
    ``get_enabled()`` is true and ``get_target()`` returns a range -- writes
    ``(lo, hi)`` onto the corresponding Bokeh handle. For ``axis='x'`` /
    ``'y'`` it mutates ``Range1d.start`` / ``.end``; for ``axis='c'`` it
    mutates ``LinearColorMapper.low`` / ``.high``.

    Parameters
    ----------
    axis:
        Which axis this hook drives.
    get_target:
        Returns the current ``(lo, hi)`` target, or ``None`` when no target
        is available yet.
    get_enabled:
        Returns whether the per-axis autoscale toggle is currently on.
    handles:
        Shared container for the cell's Bokeh handles; populated here on
        first invocation.

    Returns
    -------
    :
        A hook function compatible with ``hv.Element.opts(hooks=[...])``.
    """

    def hook(plot: Any, element: Any) -> None:
        del element
        handles.capture(plot)
        if not get_enabled():
            return
        target = get_target()
        if target is None:
            return
        lo, hi = target
        if axis == 'c':
            mapper = handles.color_mapper
            if mapper is None:
                return
            mapper.low = lo
            mapper.high = hi
        else:
            range_handle = handles.x_range if axis == 'x' else handles.y_range
            if range_handle is None:
                return
            range_handle.start = lo
            range_handle.end = hi

    return hook
