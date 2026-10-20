# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""What a plotter computes once per update, for every session.

A :class:`Frame` splits a computed HoloViews element into what a figure is
*built* from and what an update *writes* into a built figure:

- ``spec`` is everything a figure is built from except the data arrays: per
  element its type, dimensions, labels and options. Equal specs mean a figure
  built from one frame can show the other by replacing the data alone.
- ``columns`` are the data-source columns of each element, in the order the
  elements appear in the frame, as HoloViews would write them into the
  element's Bokeh ``ColumnDataSource``.

Sessions write ``columns`` into figures they built from an earlier frame of the
same spec, which skips HoloViews' per-update machinery entirely (#1407). A frame
holding an element type without a column function here has no ``columns``; its
spec then is only the frame's layout shape, and sessions render it through a
HoloViews ``DynamicMap`` instead.

The arrays in ``columns`` are shared by every session's figure, so they must
never be modified after the frame is published.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable
from typing import Any

import holoviews as hv
import numpy as np

Columns = dict[str, Any]

# Option groups that shape a figure. 'norm' holds framewise/axiswise, which
# decide how HoloViews computes ranges at build time.
_OPTION_GROUPS = ('plot', 'style', 'norm')


def _image_columns(image: hv.Image) -> Columns:
    """Columns of ``RasterPlot.get_data`` for a single-valued, non-inverted image."""
    left, bottom, right, top = image.bounds.lbrt()
    values = image.dimension_values(2, flat=False)
    if 0 in values.shape:
        values = np.array([[np.nan]])
    return {
        'x': [left],
        'y': [bottom],
        'dw': [right - left],
        'dh': [top - bottom],
        'image': [values],
    }


def _curve_columns(curve: hv.Curve) -> Columns:
    """Columns of ``CurvePlot.get_data`` for a non-inverted, linear curve."""
    x, y = curve.dimensions()[:2]
    return {x.name: curve.dimension_values(0), y.name: curve.dimension_values(1)}


def _text_spec(text: hv.Text) -> Hashable:
    return (text.x, text.y, text.text)


# Element types whose updates can be written directly, with the function
# deriving their columns. Data the figure is built from goes into the spec
# instead (see ``_SPEC_ONLY``).
_COLUMNS: dict[type, Callable[[Any], Columns]] = {
    hv.Image: _image_columns,
    hv.Curve: _curve_columns,
}

# Element types drawn from their spec alone: their content is small and rarely
# changes (the "No data" and error placeholders), so a change rebuilds the
# figure rather than being written.
_SPEC_ONLY: dict[type, Callable[[Any], Hashable]] = {
    hv.Text: _text_spec,
}


@dataclasses.dataclass(frozen=True, eq=False)
class Frame:
    """A computed element with its spec and data-source columns.

    Parameters
    ----------
    element:
        The computed state: a HoloViews element for most plotters. Sessions
        build figures from it.
    spec:
        Everything a figure is built from except the data arrays (see the module
        docstring).
    columns:
        The columns of each element of ``element``, in order, or None if any of
        them has a type that cannot be written directly. Spec-only elements
        have empty columns.
    """

    element: Any
    spec: Hashable
    columns: tuple[Columns, ...] | None

    @classmethod
    def from_element(cls, element: Any) -> Frame:
        """Derive the spec and columns of ``element``."""
        leaves = _leaves(element)
        if leaves is None or not all(
            type(leaf) in _COLUMNS or type(leaf) in _SPEC_ONLY for leaf in leaves
        ):
            return cls(element=element, spec=layout_shape(element), columns=None)
        spec = (
            _options_spec(element) if isinstance(element, hv.Overlay) else None,
            tuple(_element_spec(leaf) for leaf in leaves),
        )
        columns = tuple(
            _COLUMNS[type(leaf)](leaf) if type(leaf) in _COLUMNS else {}
            for leaf in leaves
        )
        return cls(element=element, spec=spec, columns=columns)


def layout_shape(frame: Any) -> tuple[type, ...] | None:
    """Types of a layout frame's items in order, or None for any other frame.

    A DynamicMap returning ``hv.Layout`` is split into one sub-map per item of
    the first frame it renders. Every later frame must have the same number of
    items of each type; HoloViews logs any other frame as an error instead of
    raising, and the plot stops updating for good (#1301). Frames with a
    different shape therefore need a new DynamicMap, not a pipe send.
    """
    if not isinstance(frame, hv.Layout):
        return None
    return tuple(type(item) for item in frame)


def _leaves(element: Any) -> list[hv.Element] | None:
    """The elements a figure of ``element`` draws, or None if it is no figure."""
    if isinstance(element, hv.Overlay):
        return list(element.values())
    if isinstance(element, hv.Element):
        return [element]
    return None


def _element_spec(element: hv.Element) -> Hashable:
    spec_only = _SPEC_ONLY.get(type(element))
    dimensions = tuple(
        (dim.name, dim.label, dim.unit, dim.range, dim.soft_range)
        for dim in element.dimensions()
    )
    return (
        type(element),
        element.group,
        element.label,
        dimensions,
        _options_spec(element),
        _value_kinds(element) if spec_only is None else spec_only(element),
    )


def _value_kinds(element: hv.Element) -> Hashable:
    """The kind of axis the values of each dimension need.

    Bokeh picks the axis type (linear, datetime, categorical) when the figure is
    built. Integer and float values share an axis type.
    """
    kinds = (element.interface.dtype(element, dim).kind for dim in element.dimensions())
    return tuple(kind if kind in 'MmUSOb' else 'number' for kind in kinds)


def _options_spec(element: hv.Dimensioned) -> Hashable:
    return tuple(
        _freeze(hv.Store.lookup_options('bokeh', element, group).kwargs)
        for group in _OPTION_GROUPS
    )


def _freeze(value: Any) -> Hashable:
    """A hashable value comparing equal exactly when ``value`` does.

    Raises ``TypeError`` for a value that is neither hashable nor a container
    this knows how to convert: comparing it by identity would make every frame
    look different, and rebuild every figure on every update.
    """
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, hv.Cycle):
        return (hv.Cycle, _freeze(value.values))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple(_freeze(getattr(value, f.name)) for f in dataclasses.fields(value)),
        )
    hash(value)
    return value
