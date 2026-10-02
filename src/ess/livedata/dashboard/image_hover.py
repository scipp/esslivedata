# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Custom hover for 2-D images whose axes are indices or flattened dims.

``$x``/``$y`` of a Bokeh hover are the cursor position, a float. For an axis
that carries indices (no coord, or an integer coord) the hover must show the
index of the pixel under the cursor, so such axes are formatted by a
``CustomJSHover``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import holoviews as hv
import scipp as sc
from bokeh.models import CustomJSHover, HoverTool


def c_order_strides(sizes: tuple[int, ...]) -> list[int]:
    """C-order strides for ``sizes``: stride[k] = product of sizes[k+1:]."""
    strides = [1] * len(sizes)
    for k in range(len(sizes) - 2, -1, -1):
        strides[k] = strides[k + 1] * sizes[k + 1]
    return strides


# JS body of each axis's CustomJSHover, with args from axis_hover_formatter_args.
# Tooltip rows reference the formatter via ``${field}{dim_name}``. The code
# either picks the nearest value (single non-flattened dim with a coord)
# or splits the flat integer cursor index via stride math (flattened multi-dim
# axis, where the image always carries integer indices).
AXIS_HOVER_FORMATTER_JS = """
const k = names.indexOf(format);
if (k < 0) return '';
const vals = values_by_dim[k];
if (names.length === 1) {
    // Single non-flattened dim: value is in image coordinate space
    // (physical units or integer indices depending on the coord).
    if (vals.length === 0) {
        return String(Math.min(Math.max(Math.round(value), 0), sizes[k] - 1));
    }
    // Nearest coordinate value. Linear, since coords need not be sorted.
    let best = 0;
    for (let j = 1; j < vals.length; j++) {
        if (Math.abs(vals[j] - value) < Math.abs(vals[best] - value)) best = j;
    }
    return String(vals[best]);
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
"""


def axis_hover_formatter_args(
    names: tuple[str, ...],
    coords: tuple[sc.Variable | None, ...],
    sizes: tuple[int, ...],
) -> dict[str, list]:
    """``CustomJSHover`` args for one image axis with one or more flattened dims.

    Per input dim of the axis, in flattening order: its name, size, C-order
    stride, and coord values (empty when the dim has no coord).
    """
    return {
        'names': list(names),
        'sizes': list(sizes),
        'strides': c_order_strides(sizes),
        'values_by_dim': [
            [] if c is None else [float(v) for v in c.values] for c in coords
        ],
    }


def make_hover_hook(
    tooltips: list[tuple[str, str]], formatter_args: dict[str, dict[str, list]]
) -> Callable[[Any, hv.Element], None]:
    """HoloViews hook giving the element's renderer a custom HoverTool.

    The hook creates the HoverTool and its formatters itself, so each session's
    figure gets its own models (see :meth:`Plotter.style_opts`). The tool is
    scoped to the element's own renderer, so overlaid layers keep their hover.
    Idempotent across re-renders: the arguments of the first render stay.
    """

    def hook(plot: Any, _element: hv.Element) -> None:
        if plot.handles.get('image_hover_installed'):
            return
        fig = plot.handles['plot']
        renderer = plot.handles['glyph_renderer']
        # HoloViews merges the default hovers of overlaid images into one tool.
        for tool in [t for t in fig.toolbar.tools if isinstance(t, HoverTool)]:
            if isinstance(tool.renderers, list) and renderer in tool.renderers:
                tool.renderers = [r for r in tool.renderers if r is not renderer]
                if not tool.renderers:
                    fig.toolbar.tools = [t for t in fig.toolbar.tools if t is not tool]
        formatters = {
            field: CustomJSHover(args=args, code=AXIS_HOVER_FORMATTER_JS)
            for field, args in formatter_args.items()
        }
        fig.add_tools(
            HoverTool(tooltips=tooltips, formatters=formatters, renderers=[renderer])
        )
        plot.handles['image_hover_installed'] = True

    return hook


def is_index_axis(data: sc.DataArray, dim: str) -> bool:
    """Whether ``dim`` has no coord or point-wise integer coord values."""
    if dim not in data.coords:
        return True
    coord = data.coords[dim]
    return (
        coord.dims == (dim,)
        and not data.coords.is_edges(dim)
        and coord.dtype in (sc.DType.int32, sc.DType.int64)
    )


def index_axis_hover_spec(
    image: hv.Element, data: sc.DataArray
) -> tuple[list[tuple[str, str]], dict[str, dict[str, list]]] | None:
    """Tooltips and formatter args for a 2-D ``image`` of ``data``.

    Index axes are formatted by the nearest index. Other axes keep the cursor
    position. Returns ``None`` if neither axis is an index axis, in which case
    the default hover is kept. A ``QuadMesh`` also keeps it: that hover shows exact
    cell-centre coords.
    """
    if not isinstance(image, hv.Image):
        return None
    tooltips: list[tuple[str, str]] = []
    formatter_args: dict[str, dict[str, list]] = {}
    for field, kdim, dim in zip(
        ('$x', '$y'), image.kdims, reversed(data.dims), strict=True
    ):
        if is_index_axis(data, dim):
            coord = data.coords.get(dim)
            formatter_args[field] = axis_hover_formatter_args(
                (dim,), (coord,), (data.sizes[dim],)
            )
            tooltips.append((kdim.pprint_label, f'{field}{{{dim}}}'))
        else:
            tooltips.append((kdim.pprint_label, field))
    if not formatter_args:
        return None
    tooltips.append((image.vdims[0].pprint_label, '@image'))
    return tooltips, formatter_args
