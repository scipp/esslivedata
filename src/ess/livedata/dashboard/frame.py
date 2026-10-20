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
holding an element type without a column function here, or with options its
column function cannot mirror (a log-y histogram), has no ``columns``; its spec
then is only the frame's layout shape, and sessions render it through a
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
from holoviews.core.util import dimension_sanitizer, isfinite

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


def _values(element: hv.Element, index: int) -> np.ndarray:
    """``element.dimension_values(index)`` without its per-call overhead.

    ``dimension_values`` runs through HoloViews' data pipeline, which creates
    param objects on every call and costs about 0.4 ms. It differs from this
    only for dimensions with ``nodata`` set, which no plotter sets.
    """
    return element.interface.values(element, element.get_dimension(index))


def _with_hover_columns(element: hv.Element, columns: Columns) -> Columns:
    """``columns`` plus a column per dimension, as added for the hover tool.

    Mirrors ``ElementPlot._get_hover_data`` for an element with a hover tool,
    drawn on its own (not as part of an ``NdOverlay``).
    """
    for index, dim in enumerate(element.dimensions()):
        name = dimension_sanitizer(dim.name)
        if name not in columns:
            columns[name] = _values(element, index)
    return columns


def _xy_columns(element: hv.Curve | hv.Scatter) -> Columns:
    """Columns of ``CurvePlot.get_data`` or ``PointPlot.get_data``.

    For a non-inverted element with a hover tool: a curve with linear
    interpolation, or a scatter without jitter.
    """
    x, y = element.dimensions()[:2]
    return _with_hover_columns(
        element, {x.name: _values(element, 0), y.name: _values(element, 1)}
    )


def _histogram_columns(histogram: hv.Histogram) -> Columns:
    """Columns of ``HistogramPlot.get_data`` for a non-inverted, linear-y histogram.

    For a histogram with a hover tool.
    """
    edges = histogram.interface.coords(histogram, histogram.kdims[0], edges=True)
    columns = {'top': _values(histogram, 1), 'left': edges[:-1], 'right': edges[1:]}
    return _with_hover_columns(histogram, columns)


def _error_bars_columns(error_bars: hv.ErrorBars) -> Columns:
    """Columns of ``ErrorPlot.get_data`` for vertical, symmetric error bars.

    For error bars on a numeric or datetime (not categorical) x-axis.
    """
    base, value, error = (_values(error_bars, i) for i in range(3))
    return {'base': base, 'lower': value - error, 'upper': value + error}


def _spread_columns(spread: hv.Spread) -> Columns:
    """Columns of ``SpreadPlot.get_data`` for a non-inverted, symmetric spread.

    The band is a polygon running along the lower edge and back along the upper
    one. As in ``SpreadPlot._split_area``, points with a non-finite coordinate,
    lower or upper value are dropped and split the band into one polygon per run
    of finite points, separated by a NaN (NaT for datetimes) point.
    """
    x, value, error = (_values(spread, i) for i in range(3))
    lower, upper = value - error, value + error
    finite = np.flatnonzero(isfinite(x) & isfinite(lower) & isfinite(upper))
    if len(finite) == 0:
        return {'x': [], 'y': []}
    x_gap = (
        np.array(['nat'], dtype=x.dtype) if x.dtype.kind in 'mM' else np.array([np.nan])
    )
    xs, ys = [], []
    for run in np.split(finite, np.flatnonzero(np.diff(finite) > 1) + 1):
        xs += [x[run], x[run][::-1], x_gap]
        ys += [lower[run], upper[run][::-1], [np.nan]]
    return {'x': np.concatenate(xs[:-1]), 'y': np.concatenate(ys[:-1])}


def _text_spec(text: hv.Text) -> Hashable:
    return (text.x, text.y, text.text)


# Element types whose updates can be written directly, with the function
# deriving their columns. Data the figure is built from goes into the spec
# instead (see ``_SPEC_ONLY``).
_COLUMNS: dict[type, Callable[[Any], Columns]] = {
    hv.Image: _image_columns,
    hv.Curve: _xy_columns,
    hv.Scatter: _xy_columns,
    hv.Histogram: _histogram_columns,
    hv.ErrorBars: _error_bars_columns,
    hv.Spread: _spread_columns,
}

# Plot option values the column functions assume. HoloViews derives other
# columns for other values, so an element with one cannot be written directly.
# On a log y-axis, for example, HoloViews adds a ``bottom`` column to a
# histogram when building the figure, filled with the start of the y-range at
# that time, which a frame cannot know.
_ASSUMED_OPTIONS: dict[type, dict[str, Any]] = {
    hv.Curve: {'interpolation': 'linear'},
    hv.Scatter: {'jitter': None},
    hv.Histogram: {'logy': False},
}
_ASSUMED_OPTIONS_ALL_TYPES: dict[str, Any] = {'invert_axes': False}

# Element types drawn from their spec alone: their content is small and rarely
# changes (the "No data" and error placeholders), so a change rebuilds the
# figure rather than being written.
_SPEC_ONLY: dict[type, Callable[[Any], Hashable]] = {
    hv.Text: _text_spec,
}

# Options left out of the spec. The colour range is written by the cell's
# autoscale controller on every update, and plotters set it from the data where
# HoloViews cannot pick one (a log scale with all-NaN or uniform values).
_UNSPECIFIED_OPTIONS = frozenset({'clim'})

Options = dict[str, dict[str, Any]]


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
        them cannot be written directly. Spec-only elements have empty columns.
    """

    element: Any
    spec: Hashable
    columns: tuple[Columns, ...] | None

    @classmethod
    def from_element(cls, element: Any) -> Frame:
        """Derive the spec and columns of ``element``.

        Raises for an option value that cannot be compared (see
        :func:`_freeze`): that is a fault of the plotter, not of the data.
        """
        leaves = _leaves(element)
        if leaves is not None:
            options = [_options(leaf) for leaf in leaves]
            columns = tuple(
                _columns(leaf, opts['plot'])
                for leaf, opts in zip(leaves, options, strict=True)
            )
            if None not in columns:
                spec = (
                    _options_spec(_options(element))
                    if isinstance(element, hv.Overlay)
                    else None,
                    tuple(
                        _element_spec(leaf, opts)
                        for leaf, opts in zip(leaves, options, strict=True)
                    ),
                )
                return cls(element=element, spec=spec, columns=columns)
        return cls(element=element, spec=_layout_shape(element), columns=None)


def _layout_shape(frame: Any) -> tuple[type, ...] | None:
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


def _options(element: hv.Dimensioned) -> Options:
    return {
        group: hv.Store.lookup_options('bokeh', element, group).kwargs
        for group in _OPTION_GROUPS
    }


def _columns(element: hv.Element, plot_options: dict[str, Any]) -> Columns | None:
    """Columns of ``element``, empty if spec-only, None if it cannot be written."""
    if type(element) in _SPEC_ONLY:
        return {}
    columns = _COLUMNS.get(type(element))
    if columns is None or not _is_mirrored(element, plot_options):
        return None
    return columns(element)


def _is_mirrored(element: hv.Element, plot_options: dict[str, Any]) -> bool:
    """Whether the column function of ``element`` reproduces HoloViews' columns."""
    assumed = {
        **_ASSUMED_OPTIONS_ALL_TYPES,
        **_ASSUMED_OPTIONS.get(type(element), {}),
    }
    if any(plot_options.get(name, value) != value for name, value in assumed.items()):
        return False
    # HoloViews sets the factors of a categorical axis on every update.
    if any(_kind(element, dim) in 'USO' for dim in element.kdims):
        return False
    if isinstance(element, hv.Image):
        # HoloViews draws further value dims in the hover only, and booleans as
        # integers.
        return len(element.vdims) == 1 and _kind(element, element.vdims[0]) != 'b'
    if isinstance(element, hv.ErrorBars):  # including Spread
        # Symmetric and vertical.
        return len(element.vdims) == 2 and not element.horizontal
    return True


def _element_spec(element: hv.Element, options: Options) -> Hashable:
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
        _options_spec(options),
        _value_kinds(element) if spec_only is None else spec_only(element),
    )


def _value_kinds(element: hv.Element) -> Hashable:
    """The kind of axis the values of each dimension need.

    Bokeh picks the axis type (linear, datetime, categorical) when the figure is
    built. Integer and float values share an axis type.
    """
    kinds = (_kind(element, dim) for dim in element.dimensions())
    return tuple(kind if kind in 'MmUSOb' else 'number' for kind in kinds)


def _kind(element: hv.Element, dim: hv.Dimension) -> str:
    return element.interface.dtype(element, dim).kind


def _options_spec(options: Options) -> Hashable:
    return tuple(
        _freeze(
            {k: v for k, v in options[group].items() if k not in _UNSPECIFIED_OPTIONS}
        )
        for group in _OPTION_GROUPS
    )


def _freeze(value: Any) -> Hashable:
    """A hashable value comparing equal when ``value`` does.

    Containers, arrays and dataclasses are compared by value, other hashable
    values as they hash, which for functions and most objects is by identity.
    Such an option must therefore be the same object in every frame: a function
    created per frame would rebuild the figure on every update. The registry
    test ``test_frame_spec_does_not_change_with_values`` guards each plotter
    against that, and the session pass logs each rebuild a spec change causes.

    Raises ``TypeError`` for a value that is neither hashable nor converted here.
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
