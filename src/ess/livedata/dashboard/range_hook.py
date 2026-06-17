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


def _coerce_to(value: float, reference: Any) -> Any:
    """Coerce a float target to ``reference``'s type for assignment.

    Bokeh stores datetime axis ranges as ``np.datetime64`` (matching the unit
    of the source coord). Range targets are computed in float space (epoch
    counts for datetime coords; see ``plots._finite_min_max``), so on a
    datetime axis a raw float would mix incompatible numpy dtypes -- both the
    ordering comparison below and Bokeh's range patch raise a ``UFuncTypeError``
    ('less_equal' loop missing for datetime64/float). Cast back to a datetime64
    of the same unit so the value round-trips and stays comparable.
    """
    if isinstance(reference, np.datetime64):
        return np.datetime64(round(value), np.datetime_data(reference)[0])
    return value


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
    lo = _coerce_to(lo, current_hi)
    hi = _coerce_to(hi, current_hi)
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
