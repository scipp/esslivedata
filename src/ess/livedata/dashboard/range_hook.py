# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""HoloViews hooks that drive per-axis range and color-mapper limits.

A hook is created once at plot compose time and re-invoked on every render;
it reads the latest target range through a closure and writes it onto
Bokeh's ``Range1d`` / ``LinearColorMapper`` handles directly.

Handles are read from ``plot.handles`` on every render rather than cached:
HoloViews swaps the Bokeh figure on kdim changes and Layout transitions,
which would leave a cached handle pointing at a detached model. Mirrors the
per-render lookup in ``flatten_plotter._make_hover_hook``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

Axis = Literal['x', 'y', 'c']
RangeTargets = dict[Axis, tuple[float, float]]


def _ordered_write(
    model: Any, lo: float, hi: float, lo_attr: str, hi_attr: str
) -> None:
    """Set ``(lo_attr, hi_attr)`` on ``model`` without ever inverting them.

    Bokeh's ``Document.hold`` is imperative (paired with ``unhold``), not a
    context manager, so we cannot batch the two writes into one PATCH-DOC
    dispatch without owning the document state. Instead, order the writes so
    ``lo <= hi`` holds after each individual property set: when the new high
    is above the current high, write the high first; otherwise write the low
    first.
    """
    current_hi = getattr(model, hi_attr, None)
    write_hi_first = current_hi is None or hi >= current_hi
    if write_hi_first:
        setattr(model, hi_attr, hi)
        setattr(model, lo_attr, lo)
    else:
        setattr(model, lo_attr, lo)
        setattr(model, hi_attr, hi)


class RangeHandles:
    """Per-render lookup of a plot's x/y ranges and color mapper.

    For Overlay plots (e.g., image + ROI rectangles), ``x_range``/``y_range``
    live on the top-level ``OverlayPlot`` while ``color_mapper`` lives on the
    image's sub-plot. ``color_mapper_plot`` searches sub-plots so the
    controller can set ``clim`` on the sub-plot -- the freeze mechanism (a
    finite ``self.clim`` causes HoloViews to skip data-range derivation on
    the next render).
    """

    @staticmethod
    def x_range(plot: Any) -> Any | None:
        """Return the current ``x_range`` handle, or ``None`` if not present."""
        return plot.handles.get('x_range')

    @staticmethod
    def y_range(plot: Any) -> Any | None:
        """Return the current ``y_range`` handle, or ``None`` if not present."""
        return plot.handles.get('y_range')

    @staticmethod
    def color_mapper(plot: Any) -> Any | None:
        """Return the current ``color_mapper`` handle, searching sub-plots."""
        mapper = plot.handles.get('color_mapper')
        if mapper is not None:
            return mapper
        for sub in (getattr(plot, 'subplots', None) or {}).values():
            sub_mapper = sub.handles.get('color_mapper')
            if sub_mapper is not None:
                return sub_mapper
        return None

    @staticmethod
    def color_mapper_plot(plot: Any) -> Any | None:
        """Return the sub-plot (or top-level plot) carrying the color mapper."""
        if plot.handles.get('color_mapper') is not None:
            return plot
        for sub in (getattr(plot, 'subplots', None) or {}).values():
            if sub.handles.get('color_mapper') is not None:
                return sub
        return None

    @classmethod
    def write(cls, plot: Any, axis: Axis, lo: float, hi: float) -> bool:
        """Write ``(lo, hi)`` to ``plot``'s handle for ``axis``.

        Returns ``True`` when a handle was found and written, ``False``
        when the required handle is missing (e.g., color_mapper on a
        non-image plot).
        """
        if axis == 'c':
            mapper = cls.color_mapper(plot)
            if mapper is None:
                return False
            _ordered_write(mapper, lo, hi, 'low', 'high')
            return True
        handle = cls.x_range(plot) if axis == 'x' else cls.y_range(plot)
        if handle is None:
            return False
        _ordered_write(handle, lo, hi, 'start', 'end')
        return True


def make_range_hook(
    axis: Axis,
    get_target: Callable[[], tuple[float, float] | None],
    get_enabled: Callable[[], bool],
) -> Callable[[Any, Any], None]:
    """Create a HoloViews hook that writes a per-axis range every render.

    Parameters
    ----------
    axis:
        Which axis this hook drives.
    get_target:
        Returns the current ``(lo, hi)`` target, or ``None`` when no target
        is available yet.
    get_enabled:
        Returns whether the per-axis autoscale toggle is currently on.

    Returns
    -------
    :
        A hook function compatible with ``hv.Element.opts(hooks=[...])``.
    """

    def hook(plot: Any, element: Any) -> None:
        del element
        if not get_enabled():
            return
        target = get_target()
        if target is None:
            return
        RangeHandles.write(plot, axis, *target)

    return hook
