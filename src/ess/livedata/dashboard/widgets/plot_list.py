# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""The Plots tab of the phone layout: grid overviews, and one open plot.

A phone screen is too small for a grid of plots, so the phone layout
(``?layout=phone``) shows no grid tabs. Instead a single "Plots" tab shows a
preview of every enabled grid -- its cells as labelled boxes, without data --
and tapping a cell opens that plot in place of the previews, with a button
back to them. At most one plot is open, since keeping several plots updating
is too heavy for a phone and drains its battery. The open plot is rendered
only while the tab is visible; every other cell costs what a cell in a hidden
grid tab costs, that is nothing unless another session is viewing it.

A :class:`PlotListSection` stands in for a :class:`~.plot_grid.PlotGrid`: the
tab widget places built cell widgets with ``insert_widget_at`` and takes them
out with ``remove_widget_at``, as it does for a grid. Unlike a grid, a section
must show cells that have no built widget yet, since building is what opening
asks for. Its preview therefore comes from topology (:meth:`sync`). The widget
of the open cell is shown by the :class:`PlotList` that owns the sections.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import panel as pn

from ..plot_orchestrator import (
    CellGeometry,
    CellId,
    GridId,
    PlotCell,
    PlotGridConfig,
)
from .grid_preview import PreviewCell, create_grid_preview
from .icons import get_icon
from .styles import Colors

# Height of a grid preview per grid row: room for a two-line title in a cell
# of a phone-width grid.
_PREVIEW_ROW_HEIGHT = 64

# The open plot fills the screen below the back button and the page's top band.
_OPEN_PLOT_HEIGHT = 'calc(100dvh - 80px)'

_BACK_BUTTON_CSS = """
    .bk-btn {
        justify-content: flex-start;
        text-align: left;
        font-size: 16px;
        font-weight: 600;
        padding: 10px 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
"""


class PlotListSection:
    """One grid's preview in the Plots tab.

    Parameters
    ----------
    owner:
        The list this section belongs to, which shows the open plot.
    grid_id:
        The grid this section previews.
    title:
        The grid's title, shown as the section heading.
    """

    def __init__(self, owner: PlotList, grid_id: GridId, title: str) -> None:
        self._owner = owner
        self._grid_id = grid_id
        self._title = title
        # Built cell widgets, by the geometry they were inserted at.
        self._views: dict[CellGeometry, pn.viewable.Viewable] = {}
        self._geometry: dict[CellId, CellGeometry] = {}
        self._titles: dict[CellId, str] = {}
        self._composition: tuple | None = None
        self._heading = pn.pane.HTML(
            sizing_mode='stretch_width',
            styles={'color': Colors.TEXT_MUTED, 'font-size': '13px'},
            margin=(12, 0, 2, 0),
        )
        self._preview = pn.Column(sizing_mode='stretch_width', margin=0)
        self._panel = pn.Column(
            self._heading, self._preview, sizing_mode='stretch_width'
        )
        self.set_title(title)

    @property
    def title(self) -> str:
        return self._title

    def set_title(self, title: str) -> None:
        self._title = title
        self._heading.object = f'<b>{title.upper()}</b>'

    def sync(self, grid_config: PlotGridConfig) -> None:
        """Match the preview to the grid's cells."""
        title_of = self._owner.title_of
        cells = tuple(
            (cell_id, cell.geometry, title_of(cell))
            for cell_id, cell in grid_config.cells.items()
        )
        composition = (grid_config.nrows, grid_config.ncols, cells)
        if composition == self._composition:
            return
        self._owner.drop_if_open(self._grid_id, {cell_id for cell_id, _, _ in cells})
        self._geometry = {cell_id: geometry for cell_id, geometry, _ in cells}
        self._titles = {cell_id: title for cell_id, _, title in cells}
        self._preview.objects = [
            create_grid_preview(
                grid_config.nrows,
                grid_config.ncols,
                [
                    PreviewCell(geometry, title, on_click=partial(self.tap, cell_id))
                    for cell_id, geometry, title in cells
                ],
                width=None,
                height=grid_config.nrows * _PREVIEW_ROW_HEIGHT,
            )
        ]
        self._composition = composition

    def tap(self, cell_id: CellId) -> None:
        """Open a cell's plot, as tapping its box does."""
        self._owner.open(self._grid_id, cell_id)

    def cell_title(self, cell_id: CellId) -> str:
        return self._titles.get(cell_id, '')

    def view_of(self, cell_id: CellId) -> pn.viewable.Viewable | None:
        geometry = self._geometry.get(cell_id)
        return None if geometry is None else self._views.get(geometry)

    def insert_widget_at(self, geometry: CellGeometry, widget: pn.viewable.Viewable):
        """Record a cell's built widget, showing it if its plot is open."""
        self._views[geometry] = widget
        self._owner.refresh_open_plot()

    def remove_widget_at(self, geometry: CellGeometry) -> None:
        self._views.pop(geometry, None)
        self._owner.refresh_open_plot()

    @property
    def panel(self) -> pn.viewable.Viewable:
        return self._panel


class PlotList:
    """The Plots tab: grid previews, or the one open plot.

    Parameters
    ----------
    title_of:
        Returns the title to show for a cell.
    on_open_changed:
        Called after a plot was opened or closed, so the owner can render or
        release it.
    """

    def __init__(
        self,
        *,
        title_of: Callable[[PlotCell], str],
        on_open_changed: Callable[[], None],
    ) -> None:
        self.title_of = title_of
        self._on_open_changed = on_open_changed
        self._sections: dict[GridId, PlotListSection] = {}
        self._listed: list[GridId] = []
        self._open: tuple[GridId, CellId] | None = None
        self._back = pn.widgets.Button(
            icon=get_icon('chevron-left'),
            color='light',
            sizing_mode='stretch_width',
            stylesheets=[_BACK_BUTTON_CSS],
            margin=(2, 0),
        )
        self._back.on_click(lambda _: self.close())
        # A stretching child makes Panel give this column ``flex: 1 0 0``, and
        # a zero flex basis overrides any height, so both are pinned here.
        self._plot = pn.Column(
            sizing_mode='stretch_width',
            stylesheets=[
                f':host {{ height: {_OPEN_PLOT_HEIGHT}; flex: 0 0 auto !important; }}'
            ],
            margin=0,
        )
        self._panel = pn.Column(sizing_mode='stretch_both', scroll=True)

    def section(self, grid_id: GridId, title: str) -> PlotListSection:
        """Create the section previewing a grid."""
        section = PlotListSection(self, grid_id, title)
        self._sections[grid_id] = section
        return section

    def set_listed(self, grid_ids: Sequence[GridId]) -> None:
        """Show the sections of these grids, in order: the enabled grids.

        An open plot of a grid no longer listed is closed.
        """
        self._listed = list(grid_ids)
        if self._open is not None and self._open[0] not in self._listed:
            self._open = None
        self._render()

    def drop_if_open(self, grid_id: GridId, cell_ids: set[CellId]) -> None:
        """Close the open plot if it is of ``grid_id`` but not among ``cell_ids``.

        For a cell that left the topology. Called from the tab widget's pass,
        so the owner is not notified: the pass itself reads the open cell.
        """
        if (
            self._open is not None
            and self._open[0] == grid_id
            and self._open[1] not in cell_ids
        ):
            self._open = None
            self._render()

    @property
    def open_cell(self) -> tuple[GridId, CellId] | None:
        """The grid and cell whose plot is open, if any."""
        return self._open

    def open(self, grid_id: GridId, cell_id: CellId) -> None:
        self._open = (grid_id, cell_id)
        self._render()
        self._on_open_changed()

    def close(self) -> None:
        self._open = None
        self._render()
        self._on_open_changed()

    def refresh_open_plot(self) -> None:
        """Show the open cell's current widget, which a pass may have rebuilt."""
        if self._open is None:
            return
        grid_id, cell_id = self._open
        view = self._sections[grid_id].view_of(cell_id)
        if view is None:
            self._plot.objects = [pn.pane.Markdown('*Loading…*')]
        elif list(self._plot.objects) != [view]:
            self._plot.objects = [view]

    def _render(self) -> None:
        with pn.io.hold():
            if self._open is None:
                # Taking the plot off the page removes its Bokeh models.
                self._plot.objects = []
                self._panel.objects = [
                    self._sections[grid_id].panel for grid_id in self._listed
                ]
                return
            grid_id, cell_id = self._open
            section = self._sections[grid_id]
            self._back.label = f'{section.title}: {section.cell_title(cell_id)}'
            self.refresh_open_plot()
            self._panel.objects = [self._back, self._plot]

    @property
    def panel(self) -> pn.viewable.Viewable:
        return self._panel
