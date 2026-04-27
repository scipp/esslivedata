# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Plotter wrapping ImagePlotter with static (config-time) flattening of 3D data.

The user picks which input dim stays as one image axis; the other two are
flattened together into the second image axis. Tick labels on the flattened
axis are computed dynamically (zoom-aware) from the input coords via a Bokeh
``CustomJSTickFormatter``.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import holoviews as hv
import pydantic
import scipp as sc
from bokeh.models import CustomJSTickFormatter

from ess.livedata.config.workflow_spec import ResultKey

from .plot_params import PlotParams2d
from .plots import ImagePlotter


class FlattenAxisConfig(pydantic.BaseModel):
    """Static-flatten axis assignment.

    The placeholder ``keep_dim: str`` is narrowed by ``make_flatten_params``
    to ``Literal[*dims]`` so the dashboard renders a dropdown of actual dim
    names. The other two dims are flattened in their natural input order
    (``flatten_transposed`` swaps that order).
    """

    keep_dim: str = pydantic.Field(
        default='',
        title='Keep dim',
        description='Input dim that stays as one image axis. The other two dims '
        'are flattened together into the second axis.',
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
    """Create a FlattenParams subclass narrowed to the given input dims.

    Parameters
    ----------
    dims:
        Dim names of the workflow output template. When exactly three are
        provided, the returned model narrows ``keep_dim`` to
        ``Literal[*dims]`` so the UI renders a dropdown of the actual dim
        names. For other arities the unmodified base ``FlattenParams`` is
        returned (the plotter rejects mismatches at plot time).

    Returns
    -------
    :
        A ``FlattenParams`` subclass.
    """
    if len(dims) != 3:
        return FlattenParams

    DimLit = Literal[*dims]  # type: ignore[valid-type]

    class _AxisConfig(FlattenAxisConfig):
        # Captured here so the plotter can fall back to positional binding
        # when the runtime data dim names differ from the template (e.g. the
        # detector_view abstraction declares dim_0/dim_1/... and per-instrument
        # transforms rename to tube/pixel/... preserving order).
        _template_dims: ClassVar[tuple[str, ...]] = dims

        keep_dim: DimLit = pydantic.Field(  # type: ignore[valid-type]
            default=dims[0],
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


def _build_flat_axis_formatter(
    outer_name: str,
    inner_name: str,
    outer_coord: sc.Variable | None,
    inner_coord: sc.Variable | None,
    inner_size: int,
) -> CustomJSTickFormatter:
    """Bokeh formatter mapping a flattened-axis position to a multi-index label.

    ``Math.round(tick)`` is unraveled into ``(o_idx, i_idx)`` against
    ``inner_size`` so labels remain accurate at any zoom level. When a coord is
    missing for one of the input dims, the label falls back to ``name[idx]``
    for that part.
    """
    outer_template, outer_values = _format_part(outer_name, outer_coord)
    inner_template, inner_values = _format_part(inner_name, inner_coord)
    return CustomJSTickFormatter(
        args={
            'outer_template': outer_template,
            'inner_template': inner_template,
            'outer_values': outer_values,
            'inner_values': inner_values,
            'inner_size': inner_size,
        },
        code="""
        const idx = Math.round(tick);
        const total = Math.max(1, inner_size);
        if (idx < 0) return '';
        const o_idx = Math.floor(idx / total);
        const i_idx = idx - o_idx * total;
        function fmt(template, values, fallback_idx) {
            if (values.length === 0) {
                return template.replace('%d', String(fallback_idx));
            }
            if (fallback_idx < 0 || fallback_idx >= values.length) return '';
            const v = values[fallback_idx];
            const s = (Math.abs(v) >= 1e-3 && Math.abs(v) < 1e6)
                ? v.toPrecision(4)
                : v.toExponential(3);
            return template.replace('%s', s);
        }
        const o = fmt(outer_template, outer_values, o_idx);
        const i = fmt(inner_template, inner_values, i_idx);
        return o + ', ' + i;
        """,
    )


class FlattenPlotter(ImagePlotter):
    """Image plotter with static flattening of 3D input to 2D.

    Inherits ``compute()`` from ``Plotter``; only ``plot()`` is overridden to
    flatten the data and apply a zoom-aware tick formatter to the flattened
    axis before delegating to ``ImagePlotter.plot()``.
    """

    def __init__(
        self,
        *,
        keep_dim: str,
        flatten_transposed: bool,
        transpose: bool,
        template_dims: tuple[str, ...] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._keep_dim = keep_dim
        self._flatten_transposed = flatten_transposed
        self._transpose = transpose
        self._template_dims = template_dims

    @classmethod
    def from_params(cls, params: FlattenParams) -> FlattenPlotter:
        rate = getattr(params, 'rate', None)
        template_dims = getattr(type(params.flatten), '_template_dims', None)
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
            template_dims=template_dims,
        )

    def _resolve_keep_dim(self, data: sc.DataArray) -> str:
        """Map the configured ``keep_dim`` to an actual data dim.

        Falls back to positional binding when the configured name doesn't
        match data.dims directly: this handles the common case where a shared
        template declares generic dims (``dim_0``, ``dim_1``, …) that are
        renamed by per-instrument transforms while preserving order.
        """
        configured = self._keep_dim
        if configured in data.dims:
            return configured
        if (
            self._template_dims is not None
            and configured in self._template_dims
            and len(self._template_dims) == data.ndim
        ):
            return data.dims[self._template_dims.index(configured)]
        raise ValueError(
            f"Configured keep_dim {configured!r} matches neither the data "
            f"dims {list(data.dims)} nor the template dims {self._template_dims}."
        )

    def _resolve_axes(self, data: sc.DataArray) -> tuple[str, str, str]:
        """Return (keep, outer, inner) actual data dim names.

        The two non-kept dims are taken in their natural input order to be
        (outer, inner); ``flatten_transposed`` swaps them.
        """
        if data.ndim != 3:
            raise ValueError(f"FlattenPlotter requires 3D input, got {data.ndim}D")
        keep = self._resolve_keep_dim(data)
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

        formatter = _build_flat_axis_formatter(
            outer_name=outer,
            inner_name=inner,
            outer_coord=_coord_for_ticks(data, outer),
            inner_coord=_coord_for_ticks(data, inner),
            inner_size=data.sizes[inner],
        )
        # When transpose=True the kept dim is on y → flat axis is x.
        axis_kwarg = 'xformatter' if self._transpose else 'yformatter'
        return image.opts(**{axis_kwarg: formatter})
