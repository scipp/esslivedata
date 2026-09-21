# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""The Plots tab of the phone layout: grid overviews, and one open plot.

A phone screen is too small for a grid of plots, so the phone layout
(``?layout=phone``) shows no grid tabs. Instead a single "Plots" tab shows a
preview of every enabled grid -- its cells as labelled boxes, without data --
and tapping a cell opens that plot in place of the previews, with a button
back to them and buttons stepping to the previous and next plot, in reading
order across grids. At most one plot is open, since keeping several plots updating
is too heavy for a phone and drains its battery. The open plot is rendered
only while the tab is visible; every other cell costs what a cell in a hidden
grid tab costs, that is nothing unless another session is viewing it.

A :class:`PlotOverviewSection` stands in for a :class:`~.plot_grid.PlotGrid`: the
tab widget places built cell widgets with ``insert_widget_at`` and takes them
out with ``remove_widget_at``, as it does for a grid. Unlike a grid, a section
must show cells that have no built widget yet, since building is what opening
asks for. Its preview therefore comes from topology (:meth:`sync`). The widget
of the open cell is shown by the :class:`PlotOverview` that owns the sections.
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
from .styles import Colors, PhoneLayout

# Height of a grid preview per grid row:
_PREVIEW_ROW_HEIGHT = 96

# Height of the navigation bar above an open plot, in pixels: the smallest
# comfortable fingertip target.
_NAV_HEIGHT = 40

# The open plot fills the screen below the navigation bar. What else takes
# height: the page's top band, the tab content's padding above and below, and
# 2 px to spare for rounding.
_OPEN_PLOT_HEIGHT = (
    'calc(100dvh - '
    f'{_NAV_HEIGHT + PhoneLayout.TOP_BAND + 2 * PhoneLayout.TAB_CONTENT_PADDING + 2}'
    'px)'
)

# Scrolling the previews comes to rest with a grid's heading at the top when it
# ends near one. ``proximity`` rather than ``mandatory``: a preview taller than
# the screen would otherwise have parts that scrolling cannot reach.
_SNAP_CONTAINER_CSS = ':host { scroll-snap-type: y proximity; }'
_SNAP_TARGET_CSS = ':host { scroll-snap-align: start; }'

# ``!important`` outranks the design's button padding.
_BACK_BUTTON_CSS = """
    .bk-btn {
        justify-content: flex-start;
        text-align: left;
        font-size: 15px;
        font-weight: 600;
        padding: 0 6px !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
"""
_STEP_BUTTON_CSS = '.bk-btn { padding: 0 !important; }'


class PlotOverviewSection:
    """One grid's preview in the Plots tab.

    Parameters
    ----------
    owner:
        The overview this section belongs to, which shows the open plot.
    grid_id:
        The grid this section previews.
    title:
        The grid's title, shown as the section heading.
    """

    def __init__(self, owner: PlotOverview, grid_id: GridId, title: str) -> None:
        self._owner = owner
        self._grid_id = grid_id
        self._title = title
        # Built cell widgets, by the geometry they were inserted at.
        self._views: dict[CellGeometry, pn.viewable.Viewable] = {}
        self._geometry: dict[CellId, CellGeometry] = {}
        self._titles: dict[CellId, str] = {}
        self._order: list[CellId] = []
        self._composition: tuple | None = None
        self._heading = pn.pane.HTML(
            sizing_mode='stretch_width',
            styles={'color': Colors.TEXT_MUTED, 'font-size': '13px'},
            margin=(12, 0, 2, 0),
        )
        self._preview = pn.Column(sizing_mode='stretch_width', margin=0)
        self._panel = pn.Column(
            self._heading,
            self._preview,
            sizing_mode='stretch_width',
            stylesheets=[_SNAP_TARGET_CSS],
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
        self._order = sorted(
            self._geometry, key=lambda c: (self._geometry[c].row, self._geometry[c].col)
        )
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

    @property
    def cell_ids(self) -> list[CellId]:
        """The grid's cells in reading order."""
        return self._order

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


class PlotOverview:
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
        self._sections: dict[GridId, PlotOverviewSection] = {}
        self._listed: list[GridId] = []
        self._open: tuple[GridId, CellId] | None = None
        self._back = pn.widgets.Button(
            icon=get_icon('layout-grid'),
            color='light',
            sizing_mode='stretch_width',
            height=_NAV_HEIGHT,
            stylesheets=[_BACK_BUTTON_CSS],
            margin=0,
            css_classes=['lt-plot-back'],
        )
        self._back.on_click(lambda _: self.close())
        self._prev = self._step_button('chevron-left', -1, 'lt-plot-prev')
        self._next = self._step_button('chevron-right', 1, 'lt-plot-next')
        self._nav = pn.Row(
            self._back, self._prev, self._next, sizing_mode='stretch_width', margin=0
        )
        # A stretching child makes Panel give this column ``flex: 1 0 0``, and
        # a zero flex basis overrides any height, so both are pinned here.
        self._plot = pn.Column(
            sizing_mode='stretch_width',
            stylesheets=[
                f':host {{ height: {_OPEN_PLOT_HEIGHT}; flex: 0 0 auto !important; }}'
            ],
            margin=0,
        )
        self._panel = pn.Column(
            sizing_mode='stretch_both',
            scroll=True,
            stylesheets=[_SNAP_CONTAINER_CSS],
        )

    def _step_button(self, icon: str, step: int, css_class: str) -> pn.widgets.Button:
        button = pn.widgets.Button(
            icon=get_icon(icon),
            icon_size='1.6em',
            color='light',
            width=_NAV_HEIGHT,
            height=_NAV_HEIGHT,
            margin=(0, 0, 0, 4),
            stylesheets=[_STEP_BUTTON_CSS],
            css_classes=[css_class],
        )
        button.on_click(lambda _: self.step(step))
        return button

    def section(self, grid_id: GridId, title: str) -> PlotOverviewSection:
        """Create the section previewing a grid."""
        section = PlotOverviewSection(self, grid_id, title)
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

    def _sequence(self) -> list[tuple[GridId, CellId]]:
        """All listed plots, in reading order, grid after grid."""
        return [
            (grid_id, cell_id)
            for grid_id in self._listed
            for cell_id in self._sections[grid_id].cell_ids
        ]

    def step(self, step: int) -> None:
        """Open the plot ``step`` places after the open one; none past the ends."""
        sequence = self._sequence()
        if self._open not in sequence:
            return
        index = sequence.index(self._open) + step
        if 0 <= index < len(sequence):
            self.open(*sequence[index])

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
            sequence = self._sequence()
            here = sequence.index(self._open) if self._open in sequence else None
            self._prev.disabled = here is None or here == 0
            self._next.disabled = here is None or here == len(sequence) - 1
            self.refresh_open_plot()
            self._panel.objects = [self._nav, self._plot]

    @property
    def panel(self) -> pn.viewable.Viewable:
        return self._panel
