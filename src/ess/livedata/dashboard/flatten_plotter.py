# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Plotter wrapping ImagePlotter with static (config-time) flattening of 3D data.

The user picks which input dim stays as one image axis; the other two are
flattened together into the second image axis. The flattened axis carries two
stacked tick rows in the standard hierarchical convention: the inner-dim
ticks (zoom-aware via a Bokeh ``CustomJSTickFormatter``) sit closest to the
plot, and the outer-dim group labels at fixed group centres sit one row
further out. Bokeh's layout stacks newly added axes closer to the plot frame
than the primary, so the outer (fixed) ticks go on the primary axis and the
inner (zoom-aware) ticks on a secondary axis added via a hook.
"""

from __future__ import annotations

from enum import IntEnum

import holoviews as hv
import pydantic
import scipp as sc
from bokeh.models import CustomJSTickFormatter, LinearAxis

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


def _coord_for_ticks(data: sc.DataArray, dim: str) -> sc.Variable | None:
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


def _format_part(name: str, coord: sc.Variable | None) -> tuple[str, list]:
    """Return JS-side (label_for_missing, values) for one flattened-dim slot."""
    if coord is None:
        return name + '[%d]', []
    values = [float(v) for v in coord.values]
    unit = '' if coord.unit is None else f' {coord.unit}'
    return f'{name}=%s{unit}', values


def _format_value_static(v: float) -> str:
    """Mirror the JS formatter's per-value formatting for the outer-axis labels."""
    if 1e-3 <= abs(v) < 1e6:
        return f'{v:.4g}'
    return f'{v:.3e}'


def _label_part(name: str, idx: int, coord: sc.Variable | None) -> str:
    """Static-side label for one dim slot, mirroring the JS formatter output."""
    if coord is None:
        return f'{name}[{idx}]'
    unit = '' if coord.unit is None else f' {coord.unit}'
    return f'{name}={_format_value_static(float(coord.values[idx]))}{unit}'


def _build_inner_axis_formatter(
    name: str, coord: sc.Variable | None, inner_size: int
) -> CustomJSTickFormatter:
    """Bokeh formatter showing only the inner-dim value at each tick.

    ``Math.round(tick) % inner_size`` selects the inner index, so labels stay
    accurate at any zoom level. Falls back to ``name[idx]`` when no coord.
    """
    template, values = _format_part(name, coord)
    return CustomJSTickFormatter(
        args={'template': template, 'values': values, 'inner_size': inner_size},
        code="""
        const idx = Math.round(tick);
        if (inner_size <= 0 || idx < 0) return '';
        const i_idx = ((idx % inner_size) + inner_size) % inner_size;
        if (values.length === 0) {
            return template.replace('%d', String(i_idx));
        }
        if (i_idx >= values.length) return '';
        const v = values[i_idx];
        const s = (Math.abs(v) >= 1e-3 && Math.abs(v) < 1e6)
            ? v.toPrecision(4)
            : v.toExponential(3);
        return template.replace('%s', s);
        """,
    )


def _outer_axis_ticks(
    name: str, coord: sc.Variable | None, outer_size: int, inner_size: int
) -> list[tuple[float, str]]:
    """``(position, label)`` pairs for the outer-dim group axis.

    Each outer index ``i`` gets one tick at the centre of its group on the
    flattened axis (``i*inner_size + (inner_size-1)/2``), with a static label.
    Returned as a list so it can be passed straight to HoloViews'
    ``xticks``/``yticks`` opt.
    """
    return [
        (
            i * inner_size + (inner_size - 1) / 2.0,
            _label_part(name, i, coord),
        )
        for i in range(outer_size)
    ]


def _make_inner_axis_hook(side: str, formatter: CustomJSTickFormatter):
    """HoloViews hook adding a secondary axis carrying the inner-dim ticks.

    Bokeh stacks each newly added layout closer to the plot frame than the
    primary axis, so the secondary axis ends up between the plot and the
    primary outer-group ticks — i.e. inner-close, outer-far, the standard
    hierarchical convention. Idempotent across re-renders.
    """

    def hook(plot, _element):
        if plot.handles.get('flatten_inner_axis_installed'):
            return
        fig = plot.handles['plot']
        fig.add_layout(
            LinearAxis(formatter=formatter, axis_label=''),
            side,
        )
        plot.handles['flatten_inner_axis_installed'] = True

    return hook


class FlattenPlotter(ImagePlotter):
    """Image plotter with static flattening of 3D input to 2D.

    Inherits ``compute()`` from ``Plotter``; only ``plot()`` is overridden to
    flatten the data and apply a zoom-aware tick formatter to the flattened
    axis before delegating to ``ImagePlotter.plot()``.
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

        outer_coord = _coord_for_ticks(data, outer)
        inner_coord = _coord_for_ticks(data, inner)
        inner_formatter = _build_inner_axis_formatter(
            inner, inner_coord, data.sizes[inner]
        )
        outer_ticks = _outer_axis_ticks(
            outer, outer_coord, data.sizes[outer], data.sizes[inner]
        )
        # When transpose=True the kept dim is on y → flat axis is x (side='below');
        # otherwise flat axis is y (side='left').
        side = 'below' if self._transpose else 'left'
        ticks_opt = 'xticks' if self._transpose else 'yticks'
        label_opt = 'xlabel' if self._transpose else 'ylabel'
        # Outer goes on the primary axis (further from plot per Bokeh stacking),
        # inner on the secondary axis added via the hook (closer to plot). The
        # synthetic flat-dim name (e.g. "outer·inner") is suppressed since each
        # tick row carries its own dim name.
        return image.opts(
            **{ticks_opt: outer_ticks, label_opt: ''},
            hooks=[_make_inner_axis_hook(side, inner_formatter)],
        )
