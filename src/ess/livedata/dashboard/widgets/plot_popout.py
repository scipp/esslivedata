# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Pop-out windows: a cell's plot shown large in a draggable floating window.

A pop-out is a *temporary second view* of a cell, for inspecting detail that
the grid cell is too small to show. The cell itself is untouched: it keeps its
plot, its toolbars, and its place in the grid, and both views repaint from the
same layer ``Pipe`` — so the pop-out is live, not a snapshot.

Deliberately a ``FloatPanel`` and not a ``pn.Modal``: a modal would block the
grid underneath, and several pop-outs can usefully be open at once (comparing
two detectors side by side). It is also why nothing here touches
``PlotGridTabs._current_modal``, whose presence suppresses the poll loop's data
flush.

A pop-out floats above the whole dashboard, so its cell must stay live even
when its grid is not the visible tab -- ``PlotGridTabs`` asks
:meth:`PlotPopoutManager.open_cells` which cells to keep computing and
flushing. That is the cost of a pop-out: one extra rendered copy of the plot,
plus a hidden grid that no longer sleeps.

Pop-outs are per session and non-persistent: they are not part of the plot
topology, so closing the browser tab discards them.
"""

from __future__ import annotations

import panel as pn

from ..plot_orchestrator import CellId
from .cell import CellWidget, ComposedPlot
from .styles import Colors

# Initial window size in pixels. Generous, but small enough that the grid stays
# partly visible behind it — the pop-out is a detail view of a cell, not a
# replacement for the dashboard. The user can resize and maximize from there.
# Near-square on purpose: an aspect-locked plot derives its cross dimension from
# the stretched one (see ``frame_aspect``), so a wide window would push a square
# frame taller than the window. The content scrolls when it still does not fit.
_POPOUT_WIDTH = 860
_POPOUT_HEIGHT = 820

# Each further window is offset by this much so a second pop-out does not land
# exactly on the first, hiding it and its close button.
_CASCADE_STEP = 28
_CASCADE_WRAP = 6


class _Popout:
    """One open floating window showing an independent view of a cell's plot.

    Owns the view's autoscale controller (the in-grid view has its own; a
    controller binds to the first figure it renders into) and disposes it when
    the window closes.
    """

    def __init__(
        self,
        title: str,
        composed: ComposedPlot,
        css_classes: list[str],
        cascade: int,
    ) -> None:
        self._composed = composed
        offset = _CASCADE_STEP * (cascade % _CASCADE_WRAP)
        self.float_panel = pn.layout.FloatPanel(
            pn.Column(
                composed.build_pane(),
                sizing_mode='stretch_both',
                # A plot whose locked aspect makes it larger than the window
                # must stay reachable rather than be clipped. The wrapper is
                # also what gives the relocated content a layout root of its
                # own: jsPanel moves the window's DOM out of the Bokeh tree,
                # and the bare pane does not render once detached.
                styles={'overflow': 'auto'},
            ),
            name=title,
            # Free-floating rather than contained: the pop-out must be
            # draggable across the whole page, not clipped by the zero-height
            # container that roots it in the component tree.
            contained=False,
            position='center',
            offsetx=offset,
            offsety=offset,
            width=_POPOUT_WIDTH,
            height=_POPOUT_HEIGHT,
            theme=Colors.TAB_BORDER,
            css_classes=css_classes,
        )

    def dispose(self) -> None:
        """Close the window client-side and dispose its autoscale controller."""
        self.float_panel.status = 'closed'
        self._composed.dispose()


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
        self._open: dict[CellId, _Popout] = {}

    @property
    def container(self) -> pn.Column:
        """The container to mount in the session's top-level layout."""
        return self._container

    def open_cells(self) -> frozenset[CellId]:
        """Cells that currently have a pop-out window open.

        The poll loop treats these as live even when their grid is not the
        visible tab: watching a plot while working elsewhere in the dashboard
        is the point of popping it out, and a silently frozen window would
        misrepresent the data as current.
        """
        return frozenset(self._open)

    def open(self, cell_id: CellId, cell_widget: CellWidget) -> None:
        """Open (or re-open) the pop-out for a cell.

        Re-opening replaces an existing window for the same cell rather than
        stacking a second one — a cell has one pop-out, and a stale window
        hidden behind the new one would keep a disposed view alive.

        A cell showing a status placeholder has nothing to compose, so nothing
        opens.
        """
        composed = cell_widget.compose_detached_view()
        if composed is None:
            return
        self.close(cell_id)
        geometry = cell_widget.geometry
        popout = _Popout(
            cell_widget.title,
            composed,
            # Per-cell automation hook, slugged by grid position like the cell
            # titlebar's — a CellId is a UUID, useless as a stable selector.
            css_classes=['lt-popout', f'lt-popout-r{geometry.row}c{geometry.col}'],
            cascade=len(self._open),
        )
        popout.float_panel.param.watch(
            lambda event: self._on_status(cell_id, event.new), 'status'
        )
        self._open[cell_id] = popout
        self._container.append(popout.float_panel)

    def close(self, cell_id: CellId) -> None:
        """Close and dispose the pop-out for a cell, if one is open.

        Popping the registry entry first makes the re-entrant call from the
        status watcher (``dispose`` closes the window, which round-trips a
        status change) a no-op.
        """
        popout = self._open.pop(cell_id, None)
        if popout is None:
            return
        popout.dispose()
        self._container.remove(popout.float_panel)

    def close_all(self) -> None:
        """Close and dispose every open pop-out."""
        for cell_id in list(self._open):
            self.close(cell_id)

    def _on_status(self, cell_id: CellId, status: str) -> None:
        """Drop the pop-out once the user closes its window.

        ``FloatPanel.status`` round-trips from jsPanel, so the window's own
        close button lands here; minimize/maximize are left alone.
        """
        if status == 'closed':
            self.close(cell_id)
