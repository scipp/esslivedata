#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Monkey-patch for HoloViews to fix line_dash rendering with WebGL backend.

This module provides a workaround for the Bokeh WebGL limitation where Quad glyphs
don't properly render line_dash styles. The patch converts HoloViews Rectangles to
use Patches glyphs instead of Quad glyphs, which correctly support dashed lines in
WebGL.

Background:
-----------
- Bokeh's WebGL renderer doesn't support line_dash on Quad glyphs (known limitation)
- Bokeh's Patches glyphs DO support line_dash in WebGL
- HoloViews uses Quad glyphs for Rectangles by default
- This patch makes Rectangles use Patches glyphs to enable dashed lines in WebGL

Usage:
------
Import this module early in your application, before creating any plots:

    from ess.livedata.dashboard import holoviews_webgl_patch
    holoviews_webgl_patch.apply()

Or use as a context manager for temporary patching:

    with holoviews_webgl_patch.patch():
        # Your plotting code here
        rects = hv.Rectangles(...).opts(line_dash='dashed')

The patch converts rectangle coordinates (left, right, bottom, top) to polygon
paths (xs, ys) which are functionally equivalent but render correctly with WebGL.

References:
-----------
- https://github.com/bokeh/bokeh/issues/... (known WebGL Quad limitation)
- Alternative: Use backend_opts={'plot.output_backend': 'canvas'} per-plot
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import holoviews as hv
from holoviews.plotting.bokeh.geometry import RectanglesPlot

logger = logging.getLogger(__name__)

# Store original class for restoration
_original_rectangles_plot: type[RectanglesPlot] | None = None
_patch_applied = False


class WebGLCompatibleRectanglesPlot(RectanglesPlot):
    """
    RectanglesPlot that uses Patches glyph instead of Quad for WebGL compatibility.

    This subclass overrides the default 'quad' plot method to use 'patches' instead,
    converting rectangle data (left, right, bottom, top) to polygon paths (xs, ys).
    This allows dashed lines to render correctly in WebGL.
    """

    _plot_methods = dict(single='patches')

    def get_data(
        self, element: Any, ranges: Any, style: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        """
        Convert rectangle data to patches format for WebGL compatibility.

        Quad format uses: left, right, bottom, top (four separate arrays)
        Patches format uses: xs, ys (arrays of coordinate lists)

        Each rectangle [left, right, bottom, top] becomes a closed polygon path:
        xs = [left, right, right, left]
        ys = [bottom, bottom, top, top]

        Parameters
        ----------
        element:
            HoloViews Rectangles element.
        ranges:
            Data ranges for the plot.
        style:
            Style dictionary.

        Returns
        -------
        :
            Tuple of (data, mapping, style) where data uses patches format.
        """
        # Get the original quad-format data from parent class
        data, mapping, style = super().get_data(element, ranges, style)

        # Extract quad coordinates
        left = data['left']
        right = data['right']
        bottom = data['bottom']
        top = data['top']

        # Convert to patches format (polygon paths)
        n = len(left)
        xs = [[left[i], right[i], right[i], left[i]] for i in range(n)]
        ys = [[bottom[i], bottom[i], top[i], top[i]] for i in range(n)]

        # Create patches data dictionary
        patches_data = {'xs': xs, 'ys': ys}
        patches_mapping = {'xs': 'xs', 'ys': 'ys'}

        # Preserve any additional data (hover, color dimensions, etc.)
        for key in data:
            if key not in ('left', 'right', 'bottom', 'top'):
                patches_data[key] = data[key]

        return patches_data, patches_mapping, style


def apply() -> None:
    """
    Apply the WebGL compatibility patch to HoloViews Rectangles.

    This replaces the Bokeh renderer's RectanglesPlot with WebGLCompatibleRectanglesPlot,
    enabling dashed lines to render correctly in WebGL mode.

    Can be called multiple times safely (idempotent).
    """
    global _original_rectangles_plot, _patch_applied

    if _patch_applied:
        logger.debug("HoloViews WebGL patch already applied")
        return

    # Save original plot class
    if _original_rectangles_plot is None:
        _original_rectangles_plot = hv.Store.registry['bokeh'].get(hv.Rectangles)

    # Apply patch
    hv.Store.registry['bokeh'][hv.Rectangles] = WebGLCompatibleRectanglesPlot
    _patch_applied = True

    logger.info(
        "Applied HoloViews WebGL patch: Rectangles will use Patches glyph for "
        "line_dash compatibility"
    )


def restore() -> None:
    """
    Restore the original HoloViews Rectangles behavior.

    Reverts the monkey-patch and restores the default Quad glyph rendering.
    """
    global _patch_applied

    if not _patch_applied or _original_rectangles_plot is None:
        logger.debug("HoloViews WebGL patch not applied, nothing to restore")
        return

    # Restore original
    hv.Store.registry['bokeh'][hv.Rectangles] = _original_rectangles_plot
    _patch_applied = False

    logger.info("Restored original HoloViews Rectangles behavior")


@contextmanager
def patch():
    """
    Context manager for temporarily applying the WebGL patch.

    Yields:
        None

    Example:
        >>> with holoviews_webgl_patch.patch():
        ...     rects = hv.Rectangles(...).opts(line_dash='dashed')
        ...     hv.render(rects)
    """
    apply()
    try:
        yield
    finally:
        restore()


def is_applied() -> bool:
    """
    Check if the patch is currently applied.

    Returns
    -------
    :
        True if patch is active, False otherwise.
    """
    return _patch_applied
