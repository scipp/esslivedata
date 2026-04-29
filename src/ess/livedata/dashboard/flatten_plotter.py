# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Plotter wrapping ImagePlotter with static (config-time) flattening of N-D data.

The user partitions the input dims into two groups via ``axis_x_dims``; the
remaining dims form the Y group. Each group with K ≥ 2 dims is flattened
together in natural input order, optionally reversed by per-axis
``transpose_*_flatten`` flags. A custom hover decomposes each axis cursor
back into per-dim labels with coord values when available.
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
    """Static-flatten partition of input dims into X and Y groups.

    ``axis_x_dims`` holds *positions* (0-based input-dim indices) — saving
    integers rather than dim names keeps configs invariant under
    per-instrument dim renames. The Y group is always the complement.
    Ordering within each group is natural input order, with the
    ``transpose_*_flatten`` flags reversing it when the natural order is
    wrong; arbitrary K ≥ 3 permutations are not exposed at the params layer
    (use the plotter's direct constructor if needed).
    """

    axis_x_dims: set[int] = pydantic.Field(
        default_factory=lambda: {0},
        title='Dims on X axis',
        description='Input dims (by position) that combine into the X image '
        'axis. The remaining dims form the Y axis. Each group with K ≥ 2 '
        'dims is flattened together in natural input order.',
    )
    transpose_x_flatten: bool = pydantic.Field(
        default=False,
        title='Reverse X-axis flatten order',
        description='Reverse the order in which the X-axis dims are flattened. '
        'Has no effect when X has a single dim.',
    )
    transpose_y_flatten: bool = pydantic.Field(
        default=False,
        title='Reverse Y-axis flatten order',
        description='Reverse the order in which the Y-axis dims are flattened. '
        'Has no effect when Y has a single dim.',
    )
    transpose: bool = pydantic.Field(
        default=False,
        title='Transpose',
        description='Swap x and y axes of the result.',
    )


class FlattenParams(PlotParams2d):
    """Parameters for the static-flatten N-D-to-2D plotter."""

    flatten: FlattenAxisConfig = pydantic.Field(
        default_factory=FlattenAxisConfig,
        description='Partition of input dims into X- and Y-axis groups and '
        'their flatten ordering.',
    )


def make_flatten_params(dims: tuple[str, ...]) -> type[FlattenParams]:
    """Create a FlattenParams subclass with ``axis_x_dims`` narrowed to ``dims``.

    Parameters
    ----------
    dims:
        Dim names of the workflow output template. For ``len(dims) ≥ 2`` the
        ``axis_x_dims`` field is narrowed to ``set[Dim]`` where ``Dim`` is an
        ``IntEnum`` mapping each dim name to its position; the dashboard
        renders this as a multi-select of dim names while the stored value
        is a set of integer positions. For ``len(dims) < 2`` the unmodified
        base ``FlattenParams`` is returned (the plotter rejects mismatches
        at plot time).

    Returns
    -------
    :
        A ``FlattenParams`` subclass.
    """
    if len(dims) < 2:
        return FlattenParams

    n = len(dims)
    Dim = IntEnum(
        'Dim', [(d if d.isidentifier() else f'dim_{i}', i) for i, d in enumerate(dims)]
    )

    class _AxisConfig(FlattenAxisConfig):
        # min_length/max_length here are not just validation — ParamWidget
        # reads them off the field metadata to constrain the MultiSelect
        # itself, so the user never lands in an empty/full state.
        axis_x_dims: set[Dim] = pydantic.Field(  # type: ignore[valid-type]
            default_factory=lambda: {Dim(0)},
            min_length=1,
            max_length=n - 1,
            title='Dims on X axis',
            description='Input dims that combine into the X image axis. The '
            'remaining dims form the Y axis. Each group with K ≥ 2 dims is '
            'flattened together in natural input order.',
        )

    class _FlattenParams(FlattenParams):
        flatten: _AxisConfig = pydantic.Field(  # type: ignore[valid-type]
            default_factory=_AxisConfig,
            description='Partition of input dims into X- and Y-axis groups '
            'and their flatten ordering.',
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
    side either does a binary search (single non-flattened dim with physical
    coords) or splits the flat integer cursor index via stride math (flattened
    multi-dim axis where the image always carries integer indices).
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
        const vals = values_by_dim[k];
        if (names.length === 1) {
            // Single non-flattened dim: value is in image coordinate space
            // (physical units or integer indices depending on the coord).
            if (vals.length === 0) return String(Math.round(value));
            // Binary search for the nearest coordinate value.
            let lo = 0, hi = vals.length - 1;
            while (lo < hi) {
                const mid = lo + ((hi - lo + 1) >> 1);
                if (vals[mid] <= value) lo = mid; else hi = mid - 1;
            }
            if (lo + 1 < vals.length &&
                    Math.abs(vals[lo + 1] - value) < Math.abs(vals[lo] - value)) lo++;
            return String(vals[lo]);
        }
        // Flattened multi-dim axis: the image always carries integer indices
        // (0..N-1), so value is the flat integer position.
        const idx = Math.round(value);
        if (idx < 0) return '';
        const size = sizes[k];
        const stride = strides[k];
        const i = ((Math.floor(idx / stride) % size) + size) % size;
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

    The X axis takes the dims at positions in ``axis_x_dims``; Y takes the
    complement. Within each axis dims are flattened in natural input order,
    optionally reversed by the per-axis ``transpose_*_flatten`` flag. The
    global ``transpose`` swaps the resulting X and Y. A custom hover
    decomposes each axis cursor back into per-dim labels.
    """

    def __init__(
        self,
        *,
        axis_x_dims: frozenset[int] | set[int],
        transpose_x_flatten: bool = False,
        transpose_y_flatten: bool = False,
        transpose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        x = frozenset(int(p) for p in axis_x_dims)
        if not x:
            raise ValueError("axis_x_dims must contain at least one position")
        if any(p < 0 for p in x):
            raise ValueError(f"Negative position in axis_x_dims: {sorted(x)}")
        self._axis_x_dims = x
        self._transpose_x_flatten = transpose_x_flatten
        self._transpose_y_flatten = transpose_y_flatten
        self._transpose = transpose

    @classmethod
    def from_params(cls, params: FlattenParams) -> FlattenPlotter:
        cfg = params.flatten
        rate = getattr(params, 'rate', None)
        return cls(
            grow_threshold=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
            tick_params=params.ticks,
            normalize_to_rate=rate.normalize_to_rate if rate is not None else False,
            axis_x_dims=frozenset(int(p) for p in cfg.axis_x_dims),
            transpose_x_flatten=cfg.transpose_x_flatten,
            transpose_y_flatten=cfg.transpose_y_flatten,
            transpose=cfg.transpose,
        )

    def _resolve_dim_names(
        self, data: sc.DataArray
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        n = data.ndim
        if any(p >= n for p in self._axis_x_dims):
            raise ValueError(
                f"axis_x_dims {sorted(self._axis_x_dims)} contains positions "
                f"out of range for {n}D data"
            )
        x_positions = sorted(self._axis_x_dims)
        y_positions = [p for p in range(n) if p not in self._axis_x_dims]
        if not y_positions:
            raise ValueError(f"All {n} dims assigned to X axis; Y axis would be empty")
        if self._transpose_x_flatten:
            x_positions = x_positions[::-1]
        if self._transpose_y_flatten:
            y_positions = y_positions[::-1]
        if self._transpose:
            x_positions, y_positions = y_positions, x_positions
        x_names = tuple(data.dims[p] for p in x_positions)
        y_names = tuple(data.dims[p] for p in y_positions)
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
