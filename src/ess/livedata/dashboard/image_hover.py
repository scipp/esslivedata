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
# either does a binary search (single non-flattened dim with physical coords)
# or splits the flat integer cursor index via stride math (flattened multi-dim
# axis, where the image always carries integer indices).
AXIS_HOVER_FORMATTER_JS = """
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
    """HoloViews hook replacing the default HoverTool with a custom one.

    The hook creates the HoverTool and its formatters itself, so each session's
    figure gets its own models (see :meth:`Plotter.style_opts`). Idempotent
    across re-renders.
    """

    def hook(plot: Any, _element: hv.Element) -> None:
        if plot.handles.get('image_hover_installed'):
            return
        fig = plot.handles['plot']
        fig.toolbar.tools = [
            t for t in fig.toolbar.tools if not isinstance(t, HoverTool)
        ]
        formatters = {
            field: CustomJSHover(args=args, code=AXIS_HOVER_FORMATTER_JS)
            for field, args in formatter_args.items()
        }
        fig.add_tools(HoverTool(tooltips=tooltips, formatters=formatters))
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
    image: hv.Image | hv.QuadMesh, data: sc.DataArray
) -> tuple[list[tuple[str, str]], dict[str, dict[str, list]]] | None:
    """Tooltips and formatter args for a 2-D ``image`` of ``data``.

    Index axes are formatted by the nearest index. Other axes keep the cursor
    position. Returns ``None`` if neither axis is an index axis, in which case
    the default hover is kept.
    """
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
