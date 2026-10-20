# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Figures built once by HoloViews and updated by writing their data sources.

Rendering a frame through HoloViews costs about 10 ms per image and 3 ms per
curve, per frame and per session: it re-applies options, computes ranges and
converts the data on every update (#1407). Most of that rebuilds what is
already there. A :class:`DirectFigure` lets HoloViews build the figure once,
from a static element, and then takes each frame's precomputed columns (see
:class:`~.frame.Frame`) straight into the glyphs' ``ColumnDataSource``.

Frames written into a figure must have the spec of the frame it was built from;
the cell is rebuilt when the spec changes. HoloViews' hooks run only at the
build. Hooks that must follow the data, such as autoscaling, are passed as
``update_hooks`` and called after each write.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import accumulate
from typing import Any

import holoviews as hv
from bokeh.document import Document
from holoviews.plotting.bokeh import BokehRenderer

from .frame import Frame

Hook = Callable[[Any, Any], None]


class DirectFigure:
    """A figure of one or more layers, updated by writing their data sources.

    Parameters
    ----------
    element:
        What to build the figure from: the cell's layers overlaid, with the
        cell's options applied.
    layer_sizes:
        Number of elements of each layer in ``element``, in order. HoloViews
        draws each element with its own subplot, in the order of the overlay.
    doc:
        The document the figure is built for.
    update_hooks:
        Called as ``hook(plot, element)`` after each write, like HoloViews hooks.
    """

    def __init__(
        self,
        element: hv.Element | hv.Overlay,
        layer_sizes: Sequence[int],
        *,
        doc: Document,
        update_hooks: Sequence[Hook] = (),
    ) -> None:
        self.plot = BokehRenderer.instance().get_plot(element, doc=doc)
        subplots = (
            list(self.plot.subplots.values())
            if isinstance(element, hv.Overlay)
            else [self.plot]
        )
        if len(subplots) != sum(layer_sizes):
            raise RuntimeError(
                f"Figure has {len(subplots)} subplots for {sum(layer_sizes)} elements"
            )
        sources = [subplot.handles.get('source') for subplot in subplots]
        ends = list(accumulate(layer_sizes))
        self._layer_sources = [
            sources[end - size : end]
            for size, end in zip(layer_sizes, ends, strict=True)
        ]
        self._update_hooks = list(update_hooks)

    @property
    def figure(self) -> Any:
        """The Bokeh figure."""
        return self.plot.state

    def write(self, layer: int, frame: Frame) -> None:
        """Show ``frame`` as the ``layer``-th layer, then run the update hooks.

        ``frame`` must have the spec of the layer's frame the figure was built
        from.
        """
        sources = self._layer_sources[layer]
        for source, columns in zip(sources, frame.columns, strict=True):
            if columns:
                # A new dict per figure: Bokeh keeps it as the source's data.
                source.data = dict(columns)
        for hook in self._update_hooks:
            hook(self.plot, frame.element)
