# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Plotter wrapping ImagePlotter with static (config-time) flattening of 3D data.

The user picks which input dim stays as one image axis; the other two are
flattened together into the second image axis. A custom hover decomposes
the flat-axis cell index back into outer/inner labels (with coord values
when available) so the plot stays explorable despite the synthetic axis.
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


def _hover_part(coord: sc.Variable | None) -> tuple[str, list]:
    """JS-side ``(template, values)`` for one flat-dim slot in the hover.

    Hover tooltip rows already carry the dim name as a label, so the
    formatter emits only the value (with unit) or — when no coord is
    available — a bare index, avoiding ``blade: blade=...`` doubling.
    """
    if coord is None:
        return '%d', []
    values = [float(v) for v in coord.values]
    unit = '' if coord.unit is None else f' {coord.unit}'
    return f'%s{unit}', values


def _build_flat_hover_formatter(
    outer_coord: sc.Variable | None,
    inner_coord: sc.Variable | None,
    inner_size: int,
) -> CustomJSHover:
    """Hover formatter dispatching on ``format`` to outer/inner label.

    Used as ``@<flat_field>{outer}`` and ``@<flat_field>{inner}`` so a single
    formatter instance handles both rows of the decomposition. The flat-axis
    cursor coord is rounded to the cell index, then split into
    ``outer = idx // inner_size`` and ``inner = idx %% inner_size``.
    """
    outer_template, outer_values = _hover_part(outer_coord)
    inner_template, inner_values = _hover_part(inner_coord)
    return CustomJSHover(
        args={
            'outer_template': outer_template,
            'outer_values': outer_values,
            'inner_template': inner_template,
            'inner_values': inner_values,
            'inner_size': inner_size,
        },
        code="""
        const idx = Math.round(value);
        if (inner_size <= 0 || idx < 0) return '';
        let template, values, i;
        if (format === 'outer') {
            template = outer_template;
            values = outer_values;
            i = Math.floor(idx / inner_size);
        } else {
            template = inner_template;
            values = inner_values;
            i = ((idx % inner_size) + inner_size) % inner_size;
        }
        if (values.length === 0) return template.replace('%d', String(i));
        if (i >= values.length) return '';
        return template.replace('%s', String(values[i]));
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
    """Image plotter with static flattening of 3D input to 2D.

    Inherits ``compute()`` from ``Plotter``; only ``plot()`` is overridden to
    flatten the data and attach a hover that decomposes the synthetic flat
    axis back into outer/inner labels before delegating to ``ImagePlotter``.
    """

    def __init__(
        self,
        *,
        keep_dim: int,
        flatten_transposed: bool,
        transpose: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._keep_dim = int(keep_dim)
        self._flatten_transposed = flatten_transposed
        self._transpose = transpose

    @classmethod
    def from_params(cls, params: FlattenParams) -> FlattenPlotter:
        rate = getattr(params, 'rate', None)
        return cls(
            grow_threshold=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
            scale_opts=params.plot_scale,
            tick_params=params.ticks,
            normalize_to_rate=rate.normalize_to_rate if rate is not None else False,
            keep_dim=params.flatten.keep_dim,
            flatten_transposed=params.flatten.flatten_transposed,
            transpose=params.flatten.transpose,
        )

    def _resolve_axes(self, data: sc.DataArray) -> tuple[str, str, str]:
        """Return (keep, outer, inner) actual data dim names.

        ``keep_dim`` is a 0-based position into ``data.dims``. The two non-kept
        dims are taken in their natural input order as (outer, inner);
        ``flatten_transposed`` swaps them.
        """
        if data.ndim != 3:
            raise ValueError(f"FlattenPlotter requires 3D input, got {data.ndim}D")
        if not 0 <= self._keep_dim < data.ndim:
            raise ValueError(
                f"keep_dim={self._keep_dim} out of range for {data.ndim}D data"
            )
        keep = data.dims[self._keep_dim]
        others = [d for d in data.dims if d != keep]
        outer, inner = others if not self._flatten_transposed else others[::-1]
        return keep, outer, inner

    def _flatten_to_2d(
        self, data: sc.DataArray, keep: str, outer: str, inner: str
    ) -> sc.DataArray:
        flat_dim = f'{outer}·{inner}'
        if flat_dim in data.dims:
            flat_dim = f'{flat_dim}_'
        order = [outer, inner, keep] if not self._transpose else [keep, outer, inner]
        return data.transpose(order).flatten(dims=[outer, inner], to=flat_dim)

    def plot(
        self,
        data: sc.DataArray,
        data_key: ResultKey,
        *,
        label: str = '',
        output_display_name: str = '',
        **kwargs,
    ) -> hv.Image:
        keep, outer, inner = self._resolve_axes(data)
        flat = self._flatten_to_2d(data, keep, outer, inner)

        image = super().plot(
            flat,
            data_key,
            label=label,
            output_display_name=output_display_name,
            **kwargs,
        )

        # $x/$y refer to the cursor position in data space; @x/@y reference
        # the Image glyph's anchor (a constant) and would freeze the index at 0.
        flat_field = '$x' if self._transpose else '$y'
        kept_field = '$y' if self._transpose else '$x'
        hover_fmt = _build_flat_hover_formatter(
            _coord_1d(data, outer), _coord_1d(data, inner), data.sizes[inner]
        )
        kept_coord = _coord_1d(data, keep)
        kept_unit = kept_coord.unit if kept_coord is not None else None
        keep_label = f'{keep} [{kept_unit}]' if kept_unit is not None else keep
        # When there is no coord, $x/$y is in integer-index space [0, N-1], so
        # rounding gives the exact bin index. We do not extend this to int coords
        # with unit=None: there $x is in coord-value space and hovering between
        # bins would produce values not in the coord — correct display would
        # require a values-array nearest-neighbour lookup in JS.
        formatters = {flat_field: hover_fmt}
        if kept_coord is None:
            formatters[kept_field] = CustomJSHover(
                code='return String(Math.round(value));'
            )
        hover = HoverTool(
            tooltips=[
                (keep_label, kept_field),
                (outer, f'{flat_field}{{outer}}'),
                (inner, f'{flat_field}{{inner}}'),
                ('value', '@image'),
            ],
            formatters=formatters,
        )
        return image.opts(hooks=[_make_hover_hook(hover)])
