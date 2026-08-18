# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""HoloViews sizing opts that enforce frame aspect ratios on Bokeh plots.

HoloViews' aspect options (``aspect="square"``, ``data_aspect``, etc.) do not
produce correctly shaped data areas (frames) when plots use ``responsive=True``
inside Panel containers.  This is an upstream bug spanning Bokeh, HoloViews,
and Panel.

The workaround, assembled by :func:`make_frame_aspect_opts`, has two
cooperating parts:

1. Sizing opts: ``responsive=True`` with no fixed dimension, from which
   HoloViews derives ``stretch_both``.  The figure therefore always fills its
   grid cell exactly and can never overflow it.  Expressing the sizing mode
   through opts rather than writing it to the figure is essential: HoloViews
   recomputes plot properties from the opts whenever they change
   (``ElementPlot._update_plot``), which would overwrite figure-level writes.
2. A hook attaching a ``CustomJS`` callback that shapes the *frame* in the
   browser, fitting the largest correctly shaped frame into the space the
   figure has (the letterbox rule).  Whichever dimension binds is decided per
   layout pass, so a plot stays correctly shaped in a cell of any shape.
   HoloViews runs hooks from ``update_frame`` (i.e. on every data update), not
   only from ``initialize_plot``, so the hook tags the figure and only acts
   once per figure: repeated attaching would leak a callback set per update.

   HoloViews also rewrites ``min_border_*`` from its own opts on every update,
   wiping the letterbox once per data frame.  The callback re-applies it: the
   reset changes the frame size, and the resulting ``inner_width`` /
   ``inner_height`` change re-triggers the callback within the same patch.
   This is why the rule must be idempotent and must derive everything it needs
   from the current layout.

Two hook variants exist:

- **Fixed frame ratio** (for ``square`` and ``aspect``): the frame's
  width/height ratio is a constant, independent of data ranges.
- **Data aspect** (for ``equal`` and ``data_aspect``): the frame shape
  depends on the visible x/y ranges so that
  ``pixels_per_x_unit / pixels_per_y_unit = data_aspect``.
  ``match_aspect`` is **not** set on the figure (that would cause Bokeh to
  pad ranges, creating a circular dependency).

The letterbox pads the right or bottom border, so the space it leaves over
shows up on the right of and below the plot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .plot_params import PlotAspect, PlotAspectType

# Marks figures whose CustomJS callback is already attached.
_HOOK_APPLIED_TAG = 'ess-livedata-frame-aspect'

# HoloViews' own ``border`` option, which it writes to every ``min_border_*`` on
# every update. A border at this value therefore means "no letterbox applied".
_HOLOVIEWS_MIN_BORDER = 10

# Letterbox the frame: shrink whichever dimension is too long by padding that
# side's border. Prefixed by a variant-specific prologue defining ``target``
# (the desired frame width/height ratio).
#
# Sizing the frame directly (``frame_width``/``frame_height``) does not work:
# Bokeh only honours those when they are set before the figure is first
# rendered, and ignores later writes. ``min_border_*`` is honoured at any time.
# It is a *minimum*, so it is expressed relative to the side's current size,
# and restoring it to HoloViews' own value removes the letterbox.
#
# Measuring and padding happen on separate passes (see the code): computing and
# applying in one pass makes the two branches chase each other forever, pegging
# the browser's main thread.
_FIT_FRAME_JS = """
    if (!fig.document) return;
    let bbox;
    try { bbox = Bokeh.index.find_one(fig).frame.bbox; } catch(e) { return; }
    const fw = bbox.width;
    const fh = bbox.height;
    const padded = fig.min_border_right > BASE_BORDER ||
                   fig.min_border_bottom > BASE_BORDER;

    const unpad = () => {
        fig.min_border_right = BASE_BORDER;
        fig.min_border_bottom = BASE_BORDER;
    };

    // A frame squeezed to nothing -- padding computed for a larger figure, left
    // over from before the window shrank -- has to get its space back first, or
    // the plot stays collapsed: every rule below needs a frame to measure.
    if (fw < 20 || fh < 20) {
        if (padded) unpad();
        return;
    }

    if (Math.abs(fw - fh * target) < 2) return;

    // Padding must be computed from an unpadded layout: a padded frame no
    // longer shows how much room the figure has. Unpad now and let the
    // resulting frame change re-trigger this callback to do the measuring.
    if (padded) {
        unpad();
        return;
    }

    // Keep a sliver of frame whatever happens, so a bad measurement cannot
    // collapse the plot into a state this callback can no longer see out of.
    if (fw > fh * target) {
        const border = fig.outer_width - bbox.x - fw;
        const room = fig.outer_width - bbox.x - 20;
        fig.min_border_right = Math.round(Math.min(border + fw - fh * target, room));
    } else {
        const border = fig.outer_height - bbox.y - fh;
        const room = fig.outer_height - bbox.y - 20;
        fig.min_border_bottom = Math.round(Math.min(border + fh - fw / target, room));
    }
"""

_FIXED_RATIO_PROLOGUE = """
    const target = frame_ratio;
"""

_DATA_ASPECT_PROLOGUE = """
    const x_span = Math.abs(fig.x_range.end - fig.x_range.start);
    const y_span = Math.abs(fig.y_range.end - fig.y_range.start);
    if (x_span < 1e-12 || y_span < 1e-12) return;
    const target = data_aspect * (x_span / y_span);
"""


def _make_hook(
    js_args: dict[str, Any], prologue: str, *, listen_ranges: bool
) -> Callable[[Any, Any], None]:
    """Build a HoloViews hook that attaches the letterbox CustomJS callback."""

    def hook(plot: Any, element: Any) -> None:
        del element
        from bokeh.models import CustomJS

        fig = plot.handles['plot']
        if _HOOK_APPLIED_TAG in fig.tags:
            return
        fig.tags.append(_HOOK_APPLIED_TAG)

        callback = CustomJS(
            args={"fig": fig, "BASE_BORDER": _HOLOVIEWS_MIN_BORDER, **js_args},
            code=prologue + _FIT_FRAME_JS,
        )
        fig.js_on_change("inner_width", callback)
        fig.js_on_change("inner_height", callback)
        if listen_ranges:
            fig.x_range.js_on_change("start", callback)
            fig.x_range.js_on_change("end", callback)
            fig.y_range.js_on_change("start", callback)
            fig.y_range.js_on_change("end", callback)

    return hook


def make_fixed_frame_ratio_hook(frame_ratio: float) -> Callable[[Any, Any], None]:
    """Create a hook that enforces a fixed frame width/height ratio.

    Parameters
    ----------
    frame_ratio:
        Desired frame width / frame height.  1.0 gives a square frame.

    Returns
    -------
    :
        A hook function compatible with ``hv.Element.opts(hooks=[...])``.
    """
    return _make_hook(
        {"frame_ratio": frame_ratio}, _FIXED_RATIO_PROLOGUE, listen_ranges=False
    )


def make_data_aspect_hook(data_aspect: float) -> Callable[[Any, Any], None]:
    """Create a hook that enforces a fixed data-aspect ratio.

    The frame shape adapts to the visible x/y ranges so that
    ``pixels_per_x_unit / pixels_per_y_unit = data_aspect``.

    Parameters
    ----------
    data_aspect:
        Ratio of pixels-per-x-unit to pixels-per-y-unit.
        1.0 gives equal scaling (same as ``aspect="equal"``).

    Returns
    -------
    :
        A hook function compatible with ``hv.Element.opts(hooks=[...])``.
    """
    return _make_hook(
        {"data_aspect": data_aspect}, _DATA_ASPECT_PROLOGUE, listen_ranges=True
    )


def make_frame_aspect_opts(aspect: PlotAspect) -> dict[str, Any]:
    """Create the HoloViews sizing opts enforcing the configured aspect.

    Returns plain ``{'responsive': True}`` for ``free`` (no aspect
    constraint).  Otherwise adds a hook — a fixed-ratio hook for ``square``
    and ``aspect``, or a data-aspect hook for ``equal`` and ``data_aspect`` —
    that letterboxes the frame in the browser.

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
            hook = make_fixed_frame_ratio_hook(1.0)
        case PlotAspectType.aspect:
            hook = make_fixed_frame_ratio_hook(aspect.ratio)
        case PlotAspectType.equal:
            hook = make_data_aspect_hook(1.0)
        case PlotAspectType.data_aspect:
            hook = make_data_aspect_hook(aspect.ratio)
        case _:
            return {'responsive': True}
    return {'responsive': True, 'hooks': [hook]}
