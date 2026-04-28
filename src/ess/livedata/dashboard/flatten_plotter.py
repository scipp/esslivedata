# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Plotter wrapping ImagePlotter with static (config-time) flattening of N-D data.

Each image axis is built from one or more input dims, flattened together in a
fixed order. A custom hover decomposes each axis cursor index back into
per-dim labels (with coord values when available) so the plot stays explorable
despite the synthetic axes.
"""

from __future__ import annotations

from enum import IntEnum

import holoviews as hv
import pydantic
import scipp as sc
from bokeh.models import CustomJSHover, HoverTool

from ess.livedata.config.workflow_spec import ResultKey

from .plot_params import PlotParams2d
from .plots import ImagePlotter


class FlattenAxisConfig(pydantic.BaseModel):
    """Static-flatten axis assignment.

    ``keep_dim`` is the *position* (0-based index) of the input dim that stays
    as one image axis. ``make_flatten_params`` narrows the field to an
    ``IntEnum`` whose member names are the actual dim names so the dashboard
    renders a dropdown of dim names while the saved value is just an integer
    — invariant under per-instrument dim renames.
    """

    keep_dim: int = pydantic.Field(
        default=0,
        title='Keep dim',
        description='Input dim (by position) that stays as one image axis. '
        'The other two dims are flattened together into the second axis.',
    )
    flatten_transposed: bool = pydantic.Field(
        default=False,
        title='Transpose flattened axes',
        description='Swap the order of the two flattened dims. By default '
        'they are flattened in their natural input order.',
    )
    transpose: bool = pydantic.Field(
        default=False,
        title='Transpose',
        description='Swap x and y axes of the result. When False the kept dim '
        'is on x and the flattened combination on y.',
    )


class FlattenParams(PlotParams2d):
    """Parameters for the static-flatten 3D-to-2D plotter."""

    flatten: FlattenAxisConfig = pydantic.Field(
        default_factory=FlattenAxisConfig,
        description='Selection of which input dim stays as one image axis '
        'and how the remaining two are flattened together.',
    )


def make_flatten_params(dims: tuple[str, ...]) -> type[FlattenParams]:
    """Create a FlattenParams subclass with ``keep_dim`` narrowed to ``dims``.

    Parameters
    ----------
    dims:
        Dim names of the workflow output template. When exactly three are
        provided, ``keep_dim`` is narrowed to an ``IntEnum`` mapping each
        dim name to its position so the UI dropdown shows dim names while
        the stored value is the integer position. For other arities the
        unmodified base ``FlattenParams`` is returned (the plotter rejects
        mismatches at plot time).

    Returns
    -------
    :
        A ``FlattenParams`` subclass.

    Notes
    -----
    Dim names must be valid Python identifiers since they become ``IntEnum``
    member names. All ESS dim names in this codebase satisfy that.
    """
    if len(dims) != 3:
        return FlattenParams

    KeepDim = IntEnum('KeepDim', [(d, i) for i, d in enumerate(dims)])

    class _AxisConfig(FlattenAxisConfig):
        keep_dim: KeepDim = pydantic.Field(  # type: ignore[valid-type]
            default=KeepDim(0),
            title='Keep dim',
            description='Input dim that stays as one image axis. The other '
            'two dims are flattened together into the second axis.',
        )

    class _FlattenParams(FlattenParams):
        flatten: _AxisConfig = pydantic.Field(  # type: ignore[valid-type]
            default_factory=_AxisConfig,
            description='Selection of which input dim stays as one image axis '
            'and how the remaining two are flattened together.',
        )

    return _FlattenParams


def _coord_1d(data: sc.DataArray, dim: str) -> sc.Variable | None:
    """1-D coord aligned with ``dim``, midpoints if bin-edged, else None."""
    if dim not in data.coords:
        return None
    coord = data.coords[dim]
    if coord.dims != (dim,):
        return None
    if data.coords.is_edges(dim):
        # float64 to dodge scipp/3765 mixed-precision issues with midpoints
        return sc.midpoints(coord.to(dtype='float64', copy=False), dim=dim)
    return coord


def _joined_dim_name(names: tuple[str, ...], existing: tuple[str, ...]) -> str:
    """Join dim names with '·', suffixing '_' until unique against ``existing``."""
    name = '·'.join(names)
    while name in existing:
        name = f'{name}_'
    return name


def _c_order_strides(sizes: tuple[int, ...]) -> list[int]:
    """C-order strides for ``sizes``: stride[k] = product of sizes[k+1:]."""
    strides = [1] * len(sizes)
    for k in range(len(sizes) - 2, -1, -1):
        strides[k] = strides[k + 1] * sizes[k + 1]
    return strides


def _build_axis_hover_formatter(
    names: tuple[str, ...],
    coords: tuple[sc.Variable | None, ...],
    sizes: tuple[int, ...],
) -> CustomJSHover:
    """Hover formatter for one image axis with one or more flattened input dims.

    Tooltip rows reference this formatter via ``${field}{dim_name}``; the JS
    side splits the rounded cursor index into per-dim indices via stride math
    and emits the coord value (or bare index) for the dim selected by
    ``format``.
    """
    strides = _c_order_strides(sizes)
    values_by_dim = [[] if c is None else [float(v) for v in c.values] for c in coords]
    return CustomJSHover(
        args={
            'names': list(names),
            'sizes': list(sizes),
            'strides': strides,
            'values_by_dim': values_by_dim,
        },
        code="""
        const k = names.indexOf(format);
        if (k < 0) return '';
        const idx = Math.round(value);
        if (idx < 0) return '';
        const size = sizes[k];
        const stride = strides[k];
        const i = ((Math.floor(idx / stride) % size) + size) % size;
        const vals = values_by_dim[k];
        if (vals.length === 0) return String(i);
        if (i >= vals.length) return '';
        return String(vals[i]);
        """,
    )


def _make_hover_hook(hover: HoverTool):
    """HoloViews hook replacing the default HoverTool with a custom one.

    Idempotent across re-renders.
    """

    def hook(plot, _element):
        if plot.handles.get('flatten_hover_installed'):
            return
        fig = plot.handles['plot']
        fig.toolbar.tools = [
            t for t in fig.toolbar.tools if not isinstance(t, HoverTool)
        ]
        fig.add_tools(hover)
        plot.handles['flatten_hover_installed'] = True

    return hook


class FlattenPlotter(ImagePlotter):
    """Image plotter with static flattening of N-D input to 2D.

    Each image axis is built from one or more input dims via ``x_dims`` and
    ``y_dims`` — ordered tuples of input-dim positions whose union covers all
    dims of the input. Axes with K ≥ 2 dims are flattened together (in the
    given order); axes with K = 1 pass through. A custom hover decomposes
    each axis cursor back into per-dim labels.
    """

    def __init__(
        self,
        *,
        x_dims: tuple[int, ...],
        y_dims: tuple[int, ...],
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not x_dims or not y_dims:
            raise ValueError("x_dims and y_dims must each contain at least one dim")
        positions = (*x_dims, *y_dims)
        if sorted(positions) != list(range(len(positions))):
            raise ValueError(
                f"x_dims/y_dims positions {positions} must be a permutation of "
                f"range({len(positions)}); some are out of range or duplicated"
            )
        self._x_dims = tuple(int(p) for p in x_dims)
        self._y_dims = tuple(int(p) for p in y_dims)

    @classmethod
    def from_axes(
        cls,
        params: PlotParams2d,
        *,
        x_dims: tuple[int, ...],
        y_dims: tuple[int, ...],
    ) -> FlattenPlotter:
        """Construct from a partition of input dims into two ordered axes.

        ``params`` supplies plot styling only; the axis spec comes from
        ``x_dims``/``y_dims`` directly. Useful for N-D cases not yet
        expressible through :class:`FlattenAxisConfig`.
        """
        rate = getattr(params, 'rate', None)
        return cls(
            grow_threshold=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
            tick_params=params.ticks,
            normalize_to_rate=rate.normalize_to_rate if rate is not None else False,
            x_dims=x_dims,
            y_dims=y_dims,
        )

    @classmethod
    def from_params(cls, params: FlattenParams) -> FlattenPlotter:
        """Construct from a 3D :class:`FlattenAxisConfig`.

        Translates ``keep_dim`` + ``flatten_transposed`` + ``transpose`` into
        the generic ``(x_dims, y_dims)`` partition.
        """
        keep = int(params.flatten.keep_dim)
        others = [i for i in (0, 1, 2) if i != keep]
        if params.flatten.flatten_transposed:
            others = others[::-1]
        if params.flatten.transpose:
            x_dims, y_dims = tuple(others), (keep,)
        else:
            x_dims, y_dims = (keep,), tuple(others)
        return cls.from_axes(params, x_dims=x_dims, y_dims=y_dims)

    @property
    def _ndim(self) -> int:
        return len(self._x_dims) + len(self._y_dims)

    def _resolve_dim_names(
        self, data: sc.DataArray
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if data.ndim != self._ndim:
            raise ValueError(
                f"FlattenPlotter requires {self._ndim}D input, got {data.ndim}D"
            )
        x_names = tuple(data.dims[p] for p in self._x_dims)
        y_names = tuple(data.dims[p] for p in self._y_dims)
        return x_names, y_names

    def _flatten_to_2d(
        self,
        data: sc.DataArray,
        x_names: tuple[str, ...],
        y_names: tuple[str, ...],
    ) -> sc.DataArray:
        """Flatten ``data`` to 2D with y as the slow (outer) axis.

        Per scipp/HoloViews convention the last (innermost) scipp dim becomes
        the x kdim and the first becomes y, so we transpose to ``y_names`` →
        ``x_names`` order before flattening each group.
        """
        out = data.transpose([*y_names, *x_names])
        if len(y_names) > 1:
            out = out.flatten(
                dims=list(y_names),
                to=_joined_dim_name(y_names, out.dims),
            )
        if len(x_names) > 1:
            out = out.flatten(
                dims=list(x_names),
                to=_joined_dim_name(x_names, out.dims),
            )
        return out

    def plot(
        self,
        data: sc.DataArray,
        data_key: ResultKey,
        *,
        label: str = '',
        output_display_name: str = '',
        **kwargs,
    ) -> hv.Image:
        x_names, y_names = self._resolve_dim_names(data)
        flat = self._flatten_to_2d(data, x_names, y_names)

        image = super().plot(
            flat,
            data_key,
            label=label,
            output_display_name=output_display_name,
            **kwargs,
        )

        tooltips, formatters = self._build_hover_spec(data, x_names, y_names)
        hover = HoverTool(tooltips=tooltips, formatters=formatters)
        return image.opts(hooks=[_make_hover_hook(hover)])

    def _build_hover_spec(
        self,
        data: sc.DataArray,
        x_names: tuple[str, ...],
        y_names: tuple[str, ...],
    ) -> tuple[list[tuple[str, str]], dict[str, CustomJSHover]]:
        """Tooltip rows + per-axis ``CustomJSHover`` dispatcher.

        Each row references its axis formatter via ``${field}{dim_name}``;
        the formatter looks up the dim by ``format`` and resolves the cursor
        index → per-dim coord value (or bare index when no coord is present).
        ``$x``/``$y`` always emit a float cursor position, so even single-dim
        axes go through a formatter — there is no float-fallback path.
        """
        tooltips: list[tuple[str, str]] = []
        formatters: dict[str, CustomJSHover] = {}
        for field, names in (('$x', x_names), ('$y', y_names)):
            coords = tuple(_coord_1d(data, n) for n in names)
            sizes = tuple(data.sizes[n] for n in names)
            formatters[field] = _build_axis_hover_formatter(names, coords, sizes)
            for name, coord in zip(names, coords, strict=True):
                tooltips.append((_dim_label(name, coord), f'{field}{{{name}}}'))
        tooltips.append(('value', '@image'))
        return tooltips, formatters


def _dim_label(name: str, coord: sc.Variable | None) -> str:
    if coord is not None and coord.unit is not None:
        return f'{name} [{coord.unit}]'
    return name
