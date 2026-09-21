# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Plot grids shown as a list of collapsible plots, for the phone layout.

A phone screen is too small for a grid of plots, so the phone layout
(``?layout=phone``) shows no grid tabs. Instead a single "Plots" tab lists
every cell of every enabled grid, one section per grid. At most one plot is
expanded at a time -- opening one closes the other -- since keeping several
plots updating is too heavy for a phone and drains its battery. The expanded
plot is rendered only while the list is visible; a collapsed plot costs what a
cell in a hidden grid tab costs, that is nothing unless another session is
viewing it. Which plot is expanded is decided by the owner of all sections,
since opening a plot closes one that may sit in another grid's section.

A :class:`PlotListSection` stands in for a :class:`~.plot_grid.PlotGrid`: the
tab widget places built cell widgets with ``insert_widget_at`` and takes them
out with ``remove_widget_at``, as it does for a grid. Unlike a grid, a section
must list cells that have no built widget yet, since building is what
expanding asks for. Its rows therefore come from topology (:meth:`sync`), and
a built widget is attached to the row of the cell at its geometry.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import panel as pn

from ..plot_orchestrator import CellGeometry, CellId, PlotCell, PlotGridConfig
from .styles import Colors

# Height of an expanded plot. A viewport fraction rather than pixels so the
# plot fills most of the screen while the next row's header stays in reach.
_EXPANDED_HEIGHT = '65dvh'

_ROW_BUTTON_CSS = """
    .bk-btn {
        justify-content: flex-start;
        text-align: left;
        font-size: 16px;
        font-weight: 600;
        padding: 10px 8px;
    }
"""


class _Row:
    """One cell: a header button toggling the plot below it."""

    def __init__(self, title: str, on_click: Callable[[], None]) -> None:
        self.expanded = False
        self.view: pn.viewable.Viewable | None = None
        self._title = title
        self.header = pn.widgets.Button(
            color='light',
            sizing_mode='stretch_width',
            stylesheets=[_ROW_BUTTON_CSS],
            margin=(2, 0),
        )
        self.header.on_click(lambda _: on_click())
        # A stretching child makes Panel give this column ``flex: 1 0 0``, and
        # a zero flex basis overrides any height, so both are pinned here.
        self.body = pn.Column(
            sizing_mode='stretch_width',
            stylesheets=[
                f':host {{ height: {_EXPANDED_HEIGHT}; flex: 0 0 auto !important; }}'
            ],
            visible=False,
            margin=0,
        )
        self.panel = pn.Column(self.header, self.body, sizing_mode='stretch_width')
        self._refresh()

    def set_title(self, title: str) -> None:
        self._title = title
        self._refresh()

    def set_view(self, view: pn.viewable.Viewable | None) -> None:
        self.view = view
        self._refresh()

    def set_expanded(self, expanded: bool) -> None:
        if expanded != self.expanded:
            self.expanded = expanded
            self._refresh()

    def _refresh(self) -> None:
        marker = '▾' if self.expanded else '▸'
        self.header.name = f'{marker}  {self._title}'
        self.body.visible = self.expanded
        # A collapsed row holds no view, so its plot has no Bokeh models.
        if not self.expanded:
            self.body.objects = []
        elif self.view is None:
            self.body.objects = [pn.pane.Markdown('*Loading…*')]
        else:
            self.body.objects = [self.view]


class PlotListSection:
    """One grid's cells as rows of collapsible plots.

    Parameters
    ----------
    title:
        The grid's title, shown as the section heading.
    title_of:
        Returns the title to show for a cell.
    on_tap:
        Called with the cell whose row the user tapped. The owner decides what
        is expanded and applies it with :meth:`show_expanded`.
    """

    def __init__(
        self,
        title: str,
        *,
        title_of: Callable[[PlotCell], str],
        on_tap: Callable[[CellId], None],
    ) -> None:
        self._title_of = title_of
        self._on_tap = on_tap
        self._rows: dict[CellId, _Row] = {}
        self._geometry: dict[CellId, CellGeometry] = {}
        self._composition: tuple | None = None
        self._heading = pn.pane.HTML(
            sizing_mode='stretch_width',
            styles={'color': Colors.TEXT_MUTED, 'font-size': '13px'},
            margin=(12, 0, 2, 0),
        )
        self._column = pn.Column(sizing_mode='stretch_width', margin=0)
        self._panel = pn.Column(
            self._heading, self._column, sizing_mode='stretch_width'
        )
        self.set_title(title)

    def set_title(self, title: str) -> None:
        self._heading.object = f'<b>{title.upper()}</b>'

    def sync(self, grid_config: PlotGridConfig) -> None:
        """Match the rows to the grid's cells, in reading order.

        Rows of surviving cells are kept, with their expanded state and view;
        new rows start collapsed.
        """
        cells = sorted(
            grid_config.cells.items(),
            key=lambda item: (item[1].geometry.row, item[1].geometry.col),
        )
        composition = tuple(
            (cell_id, cell.geometry, self._title_of(cell)) for cell_id, cell in cells
        )
        if composition == self._composition:
            return
        rows = {}
        for cell_id, _, title in composition:
            row = self._rows.get(cell_id)
            if row is None:
                row = _Row(title, on_click=partial(self.tap, cell_id))
            else:
                row.set_title(title)
            rows[cell_id] = row
        self._rows = rows
        self._geometry = {cell_id: geometry for cell_id, geometry, _ in composition}
        self._column.objects = [row.panel for row in rows.values()]
        self._composition = composition

    def tap(self, cell_id: CellId) -> None:
        """Report a tap on a cell's row to the owner."""
        self._on_tap(cell_id)

    def show_expanded(self, cell_id: CellId | None) -> None:
        """Expand the row of ``cell_id`` if this section has it; collapse others."""
        for row_cell_id, row in self._rows.items():
            row.set_expanded(row_cell_id == cell_id)

    def _row_at(self, geometry: CellGeometry) -> _Row | None:
        for cell_id, cell_geometry in self._geometry.items():
            if cell_geometry == geometry:
                return self._rows[cell_id]
        return None

    def insert_widget_at(self, geometry: CellGeometry, widget: pn.viewable.Viewable):
        """Attach a cell's built widget to the row of the cell at ``geometry``."""
        row = self._row_at(geometry)
        if row is not None:
            row.set_view(widget)

    def remove_widget_at(self, geometry: CellGeometry) -> None:
        row = self._row_at(geometry)
        if row is not None:
            row.set_view(None)

    @property
    def panel(self) -> pn.viewable.Viewable:
        return self._panel
