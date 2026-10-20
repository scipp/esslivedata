# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Figures built once by HoloViews and updated by writing their data sources.

Rendering a frame through HoloViews costs about 10 ms per image and 3 ms per
curve, per frame and per session: it re-applies options, computes ranges and
converts the data on every update (#1407). Most of that rebuilds what is
already there. A :class:`DirectFigure` lets HoloViews build the figure once,
from a static element, and then takes each frame's precomputed columns (see
:class:`~.frame.Frame`) straight into the glyphs' ``ColumnDataSource``.

The figure subscribes to the frames of the layers it shows. Each pushed frame
has the spec of the frame the figure was built from; the cell is rebuilt when
the spec changes. HoloViews' hooks run only at the build. Hooks that must
follow the data, such as autoscaling, are passed as ``update_hooks`` and called
after each write.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from itertools import accumulate
from typing import Any, Protocol

import holoviews as hv
from bokeh.document import Document

from .frame import Frame

Hook = Callable[[Any, Any], None]


class FrameSource(Protocol):
    """A layer's frames, as a session sees them (``SessionComponents``)."""

    frame: Frame

    def subscribe(self, callback: Callable[[Frame], None]) -> None: ...

    def unsubscribe(self, callback: Callable[[Frame], None]) -> None: ...


class DirectFigure:
    """A figure of one or more layers, updated by writing their data sources.

    Parameters
    ----------
    element:
        What to build the figure from: the current frames of ``layers``
        overlaid, in order, with the cell's options applied.
    layers:
        The layers drawn, whose later frames are written into the figure until
        :meth:`dispose`. HoloViews draws each element of a frame with its own
        subplot, in the order of the overlay.
    doc:
        The document the figure is built for.
    margin:
        The figure's margin, as Panel's ``margin``.
    update_hooks:
        Called as ``hook(plot, element)`` after each write, like HoloViews hooks.
    """

    def __init__(
        self,
        element: hv.Element | hv.Overlay,
        layers: Sequence[FrameSource],
        *,
        doc: Document,
        margin: int | tuple[int, ...],
        update_hooks: Sequence[Hook] = (),
    ) -> None:
        # The configured renderer (see ``reduction``), in the mode Panel's
        # HoloViews pane uses: only server mode attaches the plot to ``doc``.
        renderer = hv.renderer('bokeh').instance(mode='server')
        self.plot = renderer.get_plot(element, doc=doc, margin=margin)
        layer_sizes = [len(layer.frame.columns) for layer in layers]
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
        self._subscriptions = [
            (layer, functools.partial(self._write, index))
            for index, layer in enumerate(layers)
        ]
        for layer, callback in self._subscriptions:
            layer.subscribe(callback)

    @property
    def figure(self) -> Any:
        """The Bokeh figure."""
        return self.plot.state

    def dispose(self) -> None:
        """Stop showing the layers' frames, and release the figure.

        Building it registered the plot with its document, which holds it for
        the life of the session unless it is cleaned up.
        """
        for layer, callback in self._subscriptions:
            layer.unsubscribe(callback)
        self._subscriptions.clear()
        self.plot.cleanup()

    def _write(self, layer: int, frame: Frame) -> None:
        """Show ``frame`` as the ``layer``-th layer, then run the update hooks."""
        sources = self._layer_sources[layer]
        for source, columns in zip(sources, frame.columns, strict=True):
            if columns:
                # A new dict per figure: Bokeh keeps it as the source's data.
                source.data = dict(columns)
        for hook in self._update_hooks:
            hook(self.plot, frame.element)
