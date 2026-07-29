# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Pop-out windows: a cell's plot shown large in a draggable floating window.

A pop-out is a *temporary second view* of a cell, for inspecting detail that
the grid cell is too small to show. The cell itself is untouched: it keeps its
plot, its toolbars, and its place in the grid. The window renders the cell's
own composition a second time rather than composing its own, so the two views
repaint from one layer ``Pipe`` — the pop-out is live, not a snapshot — and
share one autoscale controller, so turning a toggle off in the window turns it
off in the cell. The cell widget owns that composition and disposes it; a
window only ever borrows it.

Deliberately a ``FloatPanel`` and not a ``pn.Modal``: a modal would block the
grid underneath, and several pop-outs can usefully be open at once (comparing
two detectors side by side). It is also why nothing here touches
``PlotGridTabs._current_modal``, whose presence suppresses the poll loop's data
flush.

A pop-out floats above the whole dashboard, so its cell must stay live even
when its grid is not the visible tab -- ``PlotGridTabs`` asks
:meth:`PlotPopoutManager.live_cells` which cells to keep computing and
flushing. Liveness is per cell, not per grid: the rest of a hidden grid still
sleeps. That is the cost of a pop-out: one extra rendered copy of the plot,
and one cell that goes on computing while its tab is away.

Liveness follows what the window *shows*, not that it exists, which is the
same rule a hidden tab already obeys. Minimizing (or smallifying) a pop-out
puts its cell back to sleep, so a user cannot accumulate cost by popping out
many cells and minimizing the windows: what is not rendered is not computed.
This is why there is no cap on the number of pop-outs -- comparing several
detectors side by side is the point, and the cost of that is bounded by the
screen it has to fit on.

Pop-outs are per session and non-persistent: they are not part of the plot
topology, so closing the browser tab discards them.
"""

from __future__ import annotations

import panel as pn

from ..plot_orchestrator import CellId
from .cell import CellWidget, ComposedPlot
from .styles import Colors

# Initial window size. Generous, but small enough that the grid stays partly
# visible behind it — the pop-out is a detail view of a cell, not a replacement
# for the dashboard. The user can resize and maximize from there.
#
# The height is a fraction of the viewport rather than a pixel count, so the
# window fits the screen it opens on. An aspect-locked plot still derives its
# height from the window's width (see ``frame_aspect``) and can exceed that; it
# scrolls rather than being clipped.
_POPOUT_WIDTH = 860
# Fallback for Panel's own model box; jsPanel opens at ``_CONTENT_HEIGHT``,
# which ``config`` below overrides it with.
_POPOUT_HEIGHT = 820

# Each further window is offset by this much so a second pop-out does not land
# exactly on the first, hiding it and its close button.
_CASCADE_STEP = 28
_CASCADE_WRAP = 6

# The window hangs from the top of the viewport rather than being centred in it,
# and is capped in viewport units so it can never grow past the bottom. Centring
# a fixed pixel height puts the title bar above y=0 on any screen shorter than
# the window -- and with it the close, minimize and maximize buttons, leaving no
# way to get rid of the window at all. Anchoring the top means the title bar is
# always on screen whatever the viewport, and the cap keeps the bottom edge (the
# resize handle) reachable too.
_TOP_MARGIN = 24
_CONTENT_HEIGHT = '78vh'

# jsPanel statuses in which the window actually renders its content. The others
# ('minimized', and the two 'smallified' variants, which collapse the window to
# its title bar) show no plot, so the cell behind them may sleep.
_VISIBLE_STATUSES = frozenset({'normalized', 'maximized'})


def _build_window(
    title: str, composed: ComposedPlot, css_classes: list[str], cascade: int
) -> pn.layout.FloatPanel:
    """Build the floating window for one cell's plot.

    Parameters
    ----------
    title:
        The cell's title, shown in the window header.
    composed:
        The cell's composed plot, rendered here a second time.
    css_classes:
        Automation hooks for the window element.
    cascade:
        How many windows are already open, offsetting this one so it does not
        land exactly on top of them.
    """
    offset = _CASCADE_STEP * (cascade % _CASCADE_WRAP)
    return pn.layout.FloatPanel(
        pn.Column(
            composed.build_pane(),
            sizing_mode='stretch_width',
            styles={
                # An explicit height, not ``stretch_both``: jsPanel moves the
                # window's DOM out of the Bokeh tree, leaving nothing for a
                # stretching child to measure itself against, and a free-aspect
                # plot then collapses to a sliver. Viewport units are explicit
                # in exactly that sense -- they need no parent to measure
                # against -- so they cap the height without reintroducing it.
                # The wrapper is also what gives the relocated content a layout
                # root of its own: a bare pane does not render once detached.
                'height': _CONTENT_HEIGHT,
                # A plot whose locked aspect makes it taller than the window
                # must stay reachable rather than be clipped.
                'overflow': 'auto',
            },
        ),
        name=title,
        # Free-floating rather than contained: the pop-out must be draggable
        # across the whole page, not clipped by the zero-height container that
        # roots it in the component tree.
        contained=False,
        position='center-top',
        offsetx=offset,
        offsety=_TOP_MARGIN + offset,
        width=_POPOUT_WIDTH,
        height=_POPOUT_HEIGHT,
        # ``config`` overrides the options Panel derives from the parameters
        # above, which is the only way to reach jsPanel's viewport-relative
        # sizing -- ``height`` is an int and cannot express it. (``maxSize``
        # looks like the natural fit but jsPanel applies it only to interactive
        # resizing, not to the size it opens at.)
        config={'contentSize': f'{_POPOUT_WIDTH} {_CONTENT_HEIGHT}'},
        theme=Colors.TAB_BORDER,
        css_classes=css_classes,
    )


class PlotPopoutManager:
    """Session-scoped registry of open plot pop-out windows.

    Holds the zero-height container that roots the windows in the component
    tree (the same trick the modal container uses) and keeps at most one
    pop-out per cell.
    """

    def __init__(self) -> None:
        # Zero-height so the container does not compete for vertical space;
        # the windows themselves render as free-floating overlays.
        self._container = pn.Column(height=0, sizing_mode='stretch_width')
        self._open: dict[CellId, pn.layout.FloatPanel] = {}

    @property
    def container(self) -> pn.Column:
        """The container to mount in the session's top-level layout."""
        return self._container

    def open_cells(self) -> frozenset[CellId]:
        """Cells that have a pop-out window, whether or not it is showing.

        Existence, not visibility -- a minimized window is still the user's
        window and must survive a rebuild of the cell behind it. Use
        :meth:`live_cells` to decide what to keep computing.
        """
        return frozenset(self._open)

    def live_cells(self) -> frozenset[CellId]:
        """Cells whose pop-out is actually rendering, and must stay computed.

        The poll loop treats these as live even when their grid is not the
        visible tab: watching a plot while working elsewhere in the dashboard
        is the point of popping it out, and a silently frozen window would
        misrepresent the data as current. A minimized window renders nothing,
        so it earns no such treatment and its cell sleeps like any other cell
        on a hidden tab.
        """
        return frozenset(
            cell_id
            for cell_id, window in self._open.items()
            if window.status in _VISIBLE_STATUSES
        )

    def open(self, cell_id: CellId, cell_widget: CellWidget) -> None:
        """Open (or re-open) the pop-out for a cell.

        Re-opening replaces an existing window for the same cell rather than
        stacking a second one — a cell has one pop-out, and a stale window
        hidden behind the new one would go on rendering unseen.

        A cell showing a status placeholder has no plot to show, so nothing
        opens.
        """
        composed = cell_widget.composed
        if composed is None:
            return
        self.close(cell_id)
        geometry = cell_widget.geometry
        window = _build_window(
            cell_widget.title,
            composed,
            # Per-cell automation hook, slugged by grid position like the cell
            # titlebar's — a CellId is a UUID, useless as a stable selector.
            css_classes=['lt-popout', f'lt-popout-r{geometry.row}c{geometry.col}'],
            cascade=len(self._open),
        )
        window.param.watch(lambda event: self._on_status(cell_id, event.new), 'status')
        self._open[cell_id] = window
        self._container.append(window)

    def close(self, cell_id: CellId) -> None:
        """Close the pop-out for a cell, if one is open.

        Popping the registry entry first makes the re-entrant call from the
        status watcher (closing the window round-trips a status change) a
        no-op. The cell's composed plot is left alone: the cell widget owns it.
        """
        window = self._open.pop(cell_id, None)
        if window is None:
            return
        window.status = 'closed'
        self._container.remove(window)

    def close_all(self) -> None:
        """Close every open pop-out."""
        for cell_id in list(self._open):
            self.close(cell_id)

    def _on_status(self, cell_id: CellId, status: str) -> None:
        """Drop the pop-out once the user closes its window.

        ``FloatPanel.status`` round-trips from jsPanel, so the window's own
        close button lands here. Minimize and restore need no handling: they
        change what :meth:`live_cells` reports, which the next poll pass reads
        for itself.
        """
        if status == 'closed':
            self.close(cell_id)
