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
off in the cell. The ``CellWidget`` owns both panes and severs both on
disposal; a window only ever borrows one.

That shared ownership is not a nicety: tearing down *any* plot on a layer's
pipe severs *every* plot on it (holoviews#6988, see ``CellWidget.dispose``).
Closing a window therefore stops the grid cell updating too, which is why
``PlotGridTabs`` rebuilds the cell behind a window it closes — see
``PlotGridTabs._close_popout``. Opening one needs no such repair: an extra
subscriber costs the existing ones nothing.

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

Known gap: the window shows the plot alone -- the freshness/lag pill stays in
the cell titlebar, which a hidden tab does not render. A stalled stream
therefore freezes a pop-out with no staleness cue: the cell keeps computing,
but nothing in the window marks the data as old.

Session ticks are gated on a ``has_work`` predicate over shared state (ADR
0007), and working a window's controls touches none of it: closing, minimizing
and restoring change only what this session shows. So the manager reports them
to its owner, which requests a tick -- the same treatment a tab switch gets.
The data path needs no such help: a new frame for the grid behind a live
pop-out is shared state, and ``PlotGridTabs`` widens its predicate to see it.

This module is deliberately passive: it builds, registers and tears down
windows, and reports what they show. Deciding which cells that keeps live, and
repairing the cell behind a closed window, belongs to ``PlotGridTabs`` and
``cell_plan.desired_cells``.

Pop-outs are per session and non-persistent: they are not part of the plot
topology, so closing the browser tab discards them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import panel as pn
from panel.reactive import ReactiveHTML

from ..plot_orchestrator import CellId
from .cell import CellWidget
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


class PopoutWindowFitter(ReactiveHTML):
    """Invisible widget that makes a pop-out's plot follow its window.

    Closes two gaps in Panel's ``FloatPanel``, both of which leave the plot at a
    size the window no longer has:

    - **The height chain is broken.** jsPanel sizes its own content element, but
      the wrappers Panel puts between that element and ours have no height, so
      nothing propagates the window's size inward. A stretching child then has
      nothing to stretch into and collapses to a sliver; a child with its own
      fixed height survives but can never follow the window, leaving whitespace
      once the window grows past it.
    - **Only drag-resize triggers a re-layout.** Panel listens for
      ``jspanelresizestop``, which a drag fires but maximize, normalize and
      smallify do not, so those change the window without resizing the plot in
      it. Re-dispatching that event on any status change hands the work to
      Panel's own handler rather than reimplementing it -- the ``panel``
      property is what its handler matches on.

    One instance per session, covering every pop-out however many are open.
    Mirrors the ``_scripts['render']`` pattern of
    :class:`~ess.livedata.dashboard.widgets.modal_escape_closer.ModalEscapeCloser`.
    """

    _template = """<div id="popout_fit" style="display:none;"></div>"""

    _scripts: ClassVar = {
        'render': """
            if (window.__esslivedataPopoutFit) { return; }
            window.__esslivedataPopoutFit = true;
            const style = document.createElement('style');
            style.id = 'esslivedata-popout-fit';
            style.textContent = `
                .jsPanel-content > .bk-root {
                    height: 100%;
                    box-sizing: border-box;
                }
                .jsPanel-content > .bk-root > div { height: 100%; }
            `;
            document.head.appendChild(style);
            state.handler = (event) => {
                const relayout = new Event('jspanelresizestop');
                relayout.panel = event.panel;
                document.dispatchEvent(relayout);
            };
            document.addEventListener('jspanelstatuschange', state.handler);
        """,
        'remove': """
            if (state.handler) {
                document.removeEventListener('jspanelstatuschange', state.handler);
                const style = document.getElementById('esslivedata-popout-fit');
                if (style) { style.remove(); }
                window.__esslivedataPopoutFit = false;
            }
        """,
    }

    def __init__(self, **params):
        params.setdefault('width', 0)
        params.setdefault('height', 0)
        params.setdefault('sizing_mode', 'fixed')
        params.setdefault('visible', False)
        super().__init__(**params)


# jsPanel statuses in which the window actually renders its content. The others
# ('minimized', and the two 'smallified' variants, which collapse the window to
# its title bar) show no plot, so the cell behind them may sleep.
_VISIBLE_STATUSES = frozenset({'normalized', 'maximized'})


def _build_window(
    title: str,
    pane: pn.viewable.Viewable,
    css_classes: list[str],
    cascade: int,
    status: str,
) -> pn.layout.FloatPanel:
    """Build the floating window for one cell's plot.

    Parameters
    ----------
    title:
        The cell's title, shown in the window header.
    pane:
        A second view of the cell's plot, built (and owned) by its widget.
    css_classes:
        Automation hooks for the window element.
    cascade:
        Cascade slot for this window, offsetting it so it does not land
        exactly on top of an already-open one.
    status:
        Initial jsPanel status. ``FloatPanel`` applies a non-default status on
        render, so a window rebuilt from a minimized one opens minimized.
    """
    offset = _CASCADE_STEP * (cascade % _CASCADE_WRAP)
    return pn.layout.FloatPanel(
        pn.Column(
            pane,
            # Stretches into the height ``_FILL_WINDOW`` gives the wrappers, so
            # the plot tracks the window at every size. The wrapper is also what
            # gives the relocated content a layout root of its own: a bare pane
            # does not render once jsPanel has moved it out of the Bokeh tree.
            sizing_mode='stretch_both',
            # A plot whose locked aspect makes it taller than the window must
            # stay reachable rather than be clipped.
            styles={'overflow': 'auto'},
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
        status=status,
    )


class PlotPopoutManager:
    """Session-scoped registry of open plot pop-out windows.

    Holds the zero-height container that roots the windows in the component
    tree (the same trick the modal container uses) and keeps at most one
    pop-out per cell.

    Parameters
    ----------
    on_window_change:
        Called with ``(cell_id, status)`` when the user works a window's own
        controls: closing, minimizing or restoring it. None of those moves
        shared state, so no version-gated predicate can see them and the owner
        has to run a pass of its own (the same treatment a tab switch gets);
        a close additionally needs the cell behind it rebuilt. Opening reports
        nothing: the pop-out button lives in a cell titlebar, so the cell is on
        the visible tab and already live and rendering.
    """

    def __init__(self, on_window_change: Callable[[CellId, str], None]) -> None:
        # Zero-height so the container does not compete for vertical space;
        # the windows themselves render as free-floating overlays. The fitter
        # is invisible and only installs document-level handlers, so it costs
        # nothing until a window exists.
        self._container = pn.Column(
            PopoutWindowFitter(), height=0, sizing_mode='stretch_width'
        )
        self._open: dict[CellId, pn.layout.FloatPanel] = {}
        # Monotone cascade slot counter. Counting open windows instead would
        # re-issue an occupied slot after a close (open A, open B, close A,
        # open C lands C exactly on B), hiding the covered window's close
        # button -- the very thing the cascade offset exists to avoid.
        self._cascade = 0
        self._on_window_change = on_window_change

    @property
    def container(self) -> pn.Column:
        """The container to mount in the session's top-level layout."""
        return self._container

    def status_of(self, cell_id: CellId) -> str | None:
        """The cell's window status ('normalized', 'minimized', ...), or None.

        None means no window exists; any status means one does, however it is
        showing -- a minimized window is still the user's window and must
        survive a rebuild of the cell behind it, minimized. Use
        :meth:`live_cells` to decide what to keep computing.
        """
        window = self._open.get(cell_id)
        return None if window is None else window.status

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

    def open(
        self, cell_id: CellId, cell_widget: CellWidget, status: str | None = None
    ) -> None:
        """Open (or re-open) the pop-out for a cell.

        Re-opening replaces an existing window for the same cell rather than
        stacking a second one — a cell has one pop-out, and a stale window
        hidden behind the new one would go on rendering unseen.

        A cell showing a status placeholder has no plot to show, so nothing
        opens.

        Parameters
        ----------
        cell_id:
            The cell to open a window for.
        cell_widget:
            The cell's widget, which builds (and owns) the pane rendered here.
        status:
            jsPanel status to open with; a cell rebuild passes the old
            window's status so a minimized window stays minimized (and its
            cell asleep). The window's position and size are not carried
            over. Defaults to a normal window.
        """
        if not cell_widget.has_plot:
            return
        self.close(cell_id)
        geometry = cell_widget.geometry
        window = _build_window(
            cell_widget.title,
            cell_widget.build_plot_pane(),
            # Per-cell automation hook, slugged by grid position like the cell
            # titlebar's — a CellId is a UUID, useless as a stable selector.
            css_classes=['lt-popout', f'lt-popout-r{geometry.row}c{geometry.col}'],
            cascade=self._cascade,
            status=status if status is not None else 'normalized',
        )
        self._cascade += 1
        window.param.watch(lambda event: self._on_status(cell_id, event.new), 'status')
        self._open[cell_id] = window
        self._container.append(window)

    def close(self, cell_id: CellId) -> None:
        """Close the pop-out for a cell, if one is open.

        Popping the registry entry first makes the re-entrant call from the
        owner's close handler a no-op. The window's pane is left to the cell
        widget, which owns and severs it -- but removing the window runs
        Panel's own pane cleanup, which severs the cell's plots along with it
        (see the module docstring), so every caller must be prepared to
        rebuild the cell.
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
        """Report the window's own controls: close, minimize, restore.

        ``FloatPanel.status`` round-trips from jsPanel, so the window's own
        buttons land here. A status this manager assigned itself does not:
        :meth:`close` drops the registry entry before assigning, so the
        unregistered cell returns early. That is what keeps the close/reopen
        of a cell rebuild from asking for a pass from inside the pass doing
        the rebuilding.

        Minimize and restore move nothing a predicate could watch -- only what
        :meth:`live_cells` reports -- so the owner must run the pass that acts
        on them, or a restored window would sit frozen at whatever it showed
        when it was minimized until the next full pass.
        """
        if cell_id not in self._open:
            return
        self._on_window_change(cell_id, status)
