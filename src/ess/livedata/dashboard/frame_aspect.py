# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""HoloViews hook that enforces a fixed data-aspect ratio on Bokeh plot frames.

HoloViews' ``data_aspect`` and ``aspect="equal"`` options do not produce
correctly shaped data areas (frames) when plots use ``responsive=True``
inside Panel containers.  This is an upstream bug spanning Bokeh, HoloViews,
and Panel.

The workaround uses a HoloViews *hook* that:

1. Switches the Bokeh figure to ``stretch_width`` or ``stretch_height``
   (chosen by the user via :class:`StretchMode`) so it fills the container
   along one axis.
2. Attaches a ``CustomJS`` callback that reads the current x/y ranges and
   adjusts the *other* dimension so the frame satisfies::

       pixels_per_x_unit / pixels_per_y_unit = data_aspect

   Because ``match_aspect`` is **not** set on the figure (that would cause
   Bokeh to pad ranges, creating a circular dependency), the ranges reflect
   the natural data extents and the callback can compute the correct frame
   shape in a single pass.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .plot_params import PlotAspect, PlotAspectType, StretchMode

_STRETCH_WIDTH_JS = """
    const iw = fig.inner_width;
    const ih = fig.inner_height;
    if (fig.outer_width < 50 || iw < 10) return;

    const x_span = Math.abs(fig.x_range.end - fig.x_range.start);
    const y_span = Math.abs(fig.y_range.end - fig.y_range.start);
    if (x_span < 1e-12 || y_span < 1e-12) return;

    const target_ratio = data_aspect * (x_span / y_span);
    const new_h = Math.round(iw / target_ratio + (fig.outer_height - ih));

    if (new_h > 50 && Math.abs(fig.height - new_h) > 2) {
        fig.height = new_h;
    }
"""

_STRETCH_HEIGHT_JS = """
    const iw = fig.inner_width;
    const ih = fig.inner_height;
    if (fig.outer_height < 50 || ih < 10) return;

    const x_span = Math.abs(fig.x_range.end - fig.x_range.start);
    const y_span = Math.abs(fig.y_range.end - fig.y_range.start);
    if (x_span < 1e-12 || y_span < 1e-12) return;

    const target_ratio = data_aspect * (x_span / y_span);
    const new_w = Math.round(ih * target_ratio + (fig.outer_width - iw));

    if (new_w > 50 && Math.abs(fig.width - new_w) > 2) {
        fig.width = new_w;
    }
"""


def make_data_aspect_hook(
    data_aspect: float, stretch: StretchMode
) -> Callable[[Any, Any], None]:
    """Create a HoloViews hook that enforces a fixed data-aspect ratio.

    Parameters
    ----------
    data_aspect:
        Ratio of pixels-per-x-unit to pixels-per-y-unit.
        1.0 gives equal scaling (same as ``aspect="equal"``).
    stretch:
        Which container axis to fill.  ``width`` fills the container
        width and computes height; ``height`` fills the container height
        and computes width.

    Returns
    -------
    :
        A hook function compatible with ``hv.Element.opts(hooks=[...])``.
    """
    fill_width = stretch == StretchMode.width

    def hook(plot: Any, element: Any) -> None:
        del element
        from bokeh.models import CustomJS

        fig = plot.handles['plot']
        if fill_width:
            fig.sizing_mode = "stretch_width"
            fig.height = 400
        else:
            fig.sizing_mode = "stretch_height"
            fig.width = 400

        callback = CustomJS(
            args={"fig": fig, "data_aspect": data_aspect},
            code=_STRETCH_WIDTH_JS if fill_width else _STRETCH_HEIGHT_JS,
        )
        fig.js_on_change("inner_width", callback)
        fig.js_on_change("inner_height", callback)
        fig.x_range.js_on_change("start", callback)
        fig.x_range.js_on_change("end", callback)
        fig.y_range.js_on_change("start", callback)
        fig.y_range.js_on_change("end", callback)

    return hook


def make_frame_aspect_hook_from_config(
    aspect: PlotAspect,
) -> Callable[[Any, Any], None] | None:
    """Create a frame-aspect hook if the config requires one.

    Returns ``None`` for aspect types that HoloViews handles correctly
    (``free``, ``square``, ``aspect``).  Returns a hook for ``equal``
    and ``data_aspect``, which need the JS workaround.

    Parameters
    ----------
    aspect:
        Plot aspect configuration.

    Returns
    -------
    :
        A hook function, or None if no hook is needed.
    """
    match aspect.aspect_type:
        case PlotAspectType.equal:
            return make_data_aspect_hook(1.0, aspect.stretch_mode)
        case PlotAspectType.data_aspect:
            return make_data_aspect_hook(aspect.ratio, aspect.stretch_mode)
        case _:
            return None
