# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Per-axis range and color-mapper handle access for HoloViews hooks.

:class:`RangeHandles` resolves a plot's ``Range1d`` / ``LinearColorMapper``
handles and writes ``(lo, hi)`` onto them. Handles are read from
``plot.handles`` on every render rather than cached: HoloViews swaps the Bokeh
figure on kdim changes and Layout transitions, which would leave a cached
handle pointing at a detached model. Mirrors the per-render lookup in
``flatten_plotter._make_hover_hook``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

Axis = Literal['x', 'y', 'c']
RangeTargets = dict[Axis, tuple[float, float]]


def _to_ns_float(value: Any) -> float:
    """Normalize a datetime range value to an epoch-nanosecond float.

    A datetime axis bound is ``np.datetime64`` on a fresh render but a plain
    float after an interactive zoom: Bokeh holds datetime axes in epoch
    *milliseconds* on the JS side, so a range that round-trips through the
    browser comes back as an epoch-ms float. Normalize both forms to epoch-ns
    floats so ordering comparisons never mix datetime64 with float (which
    raises ``UFuncTypeError``).
    """
    if isinstance(value, np.datetime64):
        return float(value.astype('datetime64[ns]').astype('int64'))
    return value * 1e6  # epoch-ms -> epoch-ns


def _ordered_write(
    model: Any, lo: float, hi: float, lo_attr: str, hi_attr: str, *, datetime: bool
) -> None:
    """Set ``(lo_attr, hi_attr)`` on ``model`` without ever inverting them.

    Bokeh's ``Document.hold`` is imperative (paired with ``unhold``), not a
    context manager, so we cannot batch the two writes into one PATCH-DOC
    dispatch without owning the document state. Instead, order the writes so
    ``lo <= hi`` holds after each individual property set: when the new high
    is above the current high, write the high first; otherwise write the low
    first.

    On a datetime axis the float targets are epoch *nanoseconds* by contract
    (see ``plots._finite_min_max``); they are cast back to ``np.datetime64[ns]``
    so Bokeh's range patch does not mix incompatible numpy dtypes. The unit is
    fixed at ns rather than copied from the current bound: the target is an
    ns count, so interpreting it in any other unit would corrupt the value.
    ``datetime`` is decided from the figure's axis type, not the current bound's
    dtype: after a zoom the bound is an epoch-ms float, yet the axis is still
    datetime.
    """
    current_hi = getattr(model, hi_attr, None)
    if datetime:
        lo_value: Any = np.datetime64(round(lo), 'ns')
        hi_value: Any = np.datetime64(round(hi), 'ns')
        write_hi_first = current_hi is None or _to_ns_float(hi_value) >= _to_ns_float(
            current_hi
        )
    else:
        lo_value, hi_value = lo, hi
        write_hi_first = current_hi is None or hi >= current_hi
    if write_hi_first:
        setattr(model, hi_attr, hi_value)
        setattr(model, lo_attr, lo_value)
    else:
        setattr(model, lo_attr, lo_value)
        setattr(model, hi_attr, hi_value)


def _axis_is_datetime(plot: Any, axis: Axis) -> bool:
    """Whether ``plot``'s ``x``/``y`` axis renders datetime values.

    Detected from the Bokeh figure's axis type rather than the current range
    bound's dtype: the bound is ``np.datetime64`` only until the first
    interactive zoom, after which Bokeh replaces it with an epoch-ms float.
    """
    from bokeh.models import DatetimeAxis

    state = getattr(plot, 'state', None)
    axes = getattr(state, 'xaxis' if axis == 'x' else 'yaxis', None)
    return any(isinstance(a, DatetimeAxis) for a in axes or ())


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
            _ordered_write(mapper, lo, hi, 'low', 'high', datetime=False)
            return True
        handle = cls.x_range(plot) if axis == 'x' else cls.y_range(plot)
        if handle is None:
            return False
        datetime = _axis_is_datetime(plot, axis) or isinstance(
            getattr(handle, 'end', None), np.datetime64
        )
        _ordered_write(handle, lo, hi, 'start', 'end', datetime=datetime)
        return True
