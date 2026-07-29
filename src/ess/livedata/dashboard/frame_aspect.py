# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""HoloViews sizing opts that enforce frame aspect ratios on Bokeh plots.

HoloViews' aspect options (``aspect="square"``, ``data_aspect``, etc.) do not
produce correctly shaped data areas (frames) when plots use ``responsive=True``
inside Panel containers.  This is an upstream bug spanning Bokeh, HoloViews,
and Panel.

The workaround, assembled by :func:`make_frame_aspect_opts`, has two
cooperating parts:

1. Sizing opts: ``responsive=True`` plus a fixed cross dimension (an initial
   ``height`` for :attr:`StretchMode.width`, ``width`` for
   :attr:`StretchMode.height`).  HoloViews derives ``stretch_width`` (resp.
   ``stretch_height``) from this combination, so the figure fills the
   container along one axis.  Expressing the sizing mode through opts rather
   than writing it to the figure is essential: HoloViews recomputes plot
   properties from the opts whenever they change (``ElementPlot's
   _update_plot``), which would overwrite figure-level writes.
2. A hook attaching a ``CustomJS`` callback that adjusts the cross dimension
   in the browser so the frame has the correct shape.  HoloViews runs hooks
   from ``update_frame`` (i.e. on every data update), not only from
   ``initialize_plot``, so the hook tags the figure and only acts once per
   figure: repeated attaching would leak a callback set per update, and
   re-seeding the cross dimension would collapse the figure to the seed size
   for one layout frame per update.

Two hook variants exist:

- **Fixed frame ratio** (for ``square`` and ``aspect``): the frame's
  width/height ratio is a constant, independent of data ranges.
- **Data aspect** (for ``equal`` and ``data_aspect``): the frame shape
  depends on the visible x/y ranges so that
  ``pixels_per_x_unit / pixels_per_y_unit = data_aspect``.
  ``match_aspect`` is **not** set on the figure (that would cause Bokeh to
  pad ranges, creating a circular dependency).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .plot_params import PlotAspect, PlotAspectType, StretchMode

# Initial size of the JS-adjusted cross dimension.  Serves double duty: its
# presence makes HoloViews compute a stretched sizing mode (see module
# docstring), and it sizes the figure until the CustomJS first fires.
_INITIAL_CROSS_SIZE_PX = 400

# Marks figures whose CustomJS callback is already attached.
_HOOK_APPLIED_TAG = 'ess-livedata-frame-aspect'

# ---------------------------------------------------------------------------
# Fixed frame ratio JS (square, aspect=N) — no range reading needed
# ---------------------------------------------------------------------------

_FIXED_STRETCH_WIDTH_JS = """
    if (!fig.document) return;
    const iw = fig.inner_width;
    const ih = fig.inner_height;
    if (fig.outer_width < 50 || iw < 10) return;

    const new_h = Math.round(iw / frame_ratio + (fig.outer_height - ih));

    if (new_h > 50 && Math.abs(fig.height - new_h) > 2) {
        try { fig.height = new_h; } catch(e) {}
    }
"""

_FIXED_STRETCH_HEIGHT_JS = """
    if (!fig.document) return;
    const iw = fig.inner_width;
    const ih = fig.inner_height;
    if (fig.outer_height < 50 || ih < 10) return;

    const new_w = Math.round(ih * frame_ratio + (fig.outer_width - iw));

    if (new_w > 50 && Math.abs(fig.width - new_w) > 2) {
        try { fig.width = new_w; } catch(e) {}
    }
"""

# ---------------------------------------------------------------------------
# Data-aspect JS — reads x/y ranges to compute frame shape
# ---------------------------------------------------------------------------

_DATA_STRETCH_WIDTH_JS = """
    if (!fig.document) return;
    const iw = fig.inner_width;
    const ih = fig.inner_height;
    if (fig.outer_width < 50 || iw < 10) return;

    const x_span = Math.abs(fig.x_range.end - fig.x_range.start);
    const y_span = Math.abs(fig.y_range.end - fig.y_range.start);
    if (x_span < 1e-12 || y_span < 1e-12) return;

    const target_ratio = data_aspect * (x_span / y_span);
    const new_h = Math.round(iw / target_ratio + (fig.outer_height - ih));

    if (new_h > 50 && Math.abs(fig.height - new_h) > 2) {
        try { fig.height = new_h; } catch(e) {}
    }
"""

_DATA_STRETCH_HEIGHT_JS = """
    if (!fig.document) return;
    const iw = fig.inner_width;
    const ih = fig.inner_height;
    if (fig.outer_height < 50 || ih < 10) return;

    const x_span = Math.abs(fig.x_range.end - fig.x_range.start);
    const y_span = Math.abs(fig.y_range.end - fig.y_range.start);
    if (x_span < 1e-12 || y_span < 1e-12) return;

    const target_ratio = data_aspect * (x_span / y_span);
    const new_w = Math.round(ih * target_ratio + (fig.outer_width - iw));

    if (new_w > 50 && Math.abs(fig.width - new_w) > 2) {
        try { fig.width = new_w; } catch(e) {}
    }
"""


def _make_hook(
    stretch: StretchMode,
    js_args: dict[str, Any],
    code_width: str,
    code_height: str,
    *,
    listen_ranges: bool,
) -> Callable[[Any, Any], None]:
    """Build a HoloViews hook that attaches a CustomJS layout callback."""
    fill_width = stretch == StretchMode.width

    def hook(plot: Any, element: Any) -> None:
        del element
        from bokeh.models import CustomJS

        fig = plot.handles['plot']
        if _HOOK_APPLIED_TAG in fig.tags:
            return
        fig.tags.append(_HOOK_APPLIED_TAG)

        callback = CustomJS(
            args={"fig": fig, **js_args},
            code=code_width if fill_width else code_height,
        )
        fig.js_on_change("inner_width", callback)
        fig.js_on_change("inner_height", callback)
        if listen_ranges:
            fig.x_range.js_on_change("start", callback)
            fig.x_range.js_on_change("end", callback)
            fig.y_range.js_on_change("start", callback)
            fig.y_range.js_on_change("end", callback)

    return hook


def make_fixed_frame_ratio_hook(
    frame_ratio: float, stretch: StretchMode
) -> Callable[[Any, Any], None]:
    """Create a hook that enforces a fixed frame width/height ratio.

    Parameters
    ----------
    frame_ratio:
        Desired frame width / frame height.  1.0 gives a square frame.
    stretch:
        Which container axis to fill.

    Returns
    -------
    :
        A hook function compatible with ``hv.Element.opts(hooks=[...])``.
    """
    return _make_hook(
        stretch,
        js_args={"frame_ratio": frame_ratio},
        code_width=_FIXED_STRETCH_WIDTH_JS,
        code_height=_FIXED_STRETCH_HEIGHT_JS,
        listen_ranges=False,
    )


def make_data_aspect_hook(
    data_aspect: float, stretch: StretchMode
) -> Callable[[Any, Any], None]:
    """Create a hook that enforces a fixed data-aspect ratio.

    The frame shape adapts to the visible x/y ranges so that
    ``pixels_per_x_unit / pixels_per_y_unit = data_aspect``.

    Parameters
    ----------
    data_aspect:
        Ratio of pixels-per-x-unit to pixels-per-y-unit.
        1.0 gives equal scaling (same as ``aspect="equal"``).
    stretch:
        Which container axis to fill.

    Returns
    -------
    :
        A hook function compatible with ``hv.Element.opts(hooks=[...])``.
    """
    return _make_hook(
        stretch,
        js_args={"data_aspect": data_aspect},
        code_width=_DATA_STRETCH_WIDTH_JS,
        code_height=_DATA_STRETCH_HEIGHT_JS,
        listen_ranges=True,
    )


def make_frame_aspect_opts(aspect: PlotAspect) -> dict[str, Any]:
    """Create the HoloViews sizing opts enforcing the configured aspect.

    Returns plain ``{'responsive': True}`` for ``free`` (no aspect
    constraint).  Otherwise adds the initial cross dimension (from which
    HoloViews derives the stretched sizing mode) and a hook — a fixed-ratio
    hook for ``square`` and ``aspect``, or a data-aspect hook for ``equal``
    and ``data_aspect`` — that adjusts the cross dimension in the browser.

    Parameters
    ----------
    aspect:
        Plot aspect configuration.

    Returns
    -------
    :
        Opts dict suitable for any Bokeh-backed HoloViews element type.
    """
    match aspect.aspect_type:
        case PlotAspectType.square:
            hook = make_fixed_frame_ratio_hook(1.0, aspect.stretch_mode)
        case PlotAspectType.aspect:
            hook = make_fixed_frame_ratio_hook(aspect.ratio, aspect.stretch_mode)
        case PlotAspectType.equal:
            hook = make_data_aspect_hook(1.0, aspect.stretch_mode)
        case PlotAspectType.data_aspect:
            hook = make_data_aspect_hook(aspect.ratio, aspect.stretch_mode)
        case _:
            return {'responsive': True}
    cross = 'height' if aspect.stretch_mode == StretchMode.width else 'width'
    return {'responsive': True, cross: _INITIAL_CROSS_SIZE_PX, 'hooks': [hook]}
