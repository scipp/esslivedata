# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""A miniature of a plot grid: its cells as labelled boxes, without data.

Shown by the grid manager for a grid being created, and by the phone layout's
Plots tab as the map from which a plot is opened.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import panel as pn

from ..plot_orchestrator import CellGeometry

# Cell colors, cycled through in cell order. Local to the preview.
_CELL_COLORS = [
    '#e3f2fd',  # light blue
    '#f3e5f5',  # light purple
    '#e8f5e9',  # light green
    '#fff3e0',  # light orange
    '#fce4ec',  # light pink
    '#e0f7fa',  # light cyan
]
_CELL_BORDER = '2px solid #1976d2'


class PreviewCell(NamedTuple):
    """One cell of a preview.

    ``on_click`` makes the cell a button; a cell with one shows its title only.
    """

    geometry: CellGeometry
    title: str
    subtitle: str = ''
    on_click: Callable[[], None] | None = None


def _label(cell: PreviewCell, color: str) -> pn.viewable.Viewable:
    if cell.on_click is None:
        return pn.pane.HTML(
            f'<div style="font-size: 10px; font-weight: 500;">{cell.title}</div>'
            f'<div style="font-size: 9px; color: #666;">{cell.subtitle}</div>',
            styles={
                'background-color': color,
                'border': _CELL_BORDER,
                'border-radius': '4px',
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'center',
                'justify-content': 'center',
                'text-align': 'center',
                'box-sizing': 'border-box',
            },
            sizing_mode='stretch_both',
            margin=1,
        )
    # ``!important`` outranks the design's button colors.
    button = pn.widgets.Button(
        label=cell.title,
        sizing_mode='stretch_both',
        margin=1,
        stylesheets=[
            f"""
            .bk-btn {{
                height: 100%;
                background-color: {color} !important;
                border: {_CELL_BORDER} !important;
                border-radius: 4px;
                white-space: normal;
                overflow: hidden;
                font-size: 13px;
                font-weight: 500;
                line-height: 1.2;
                padding: 2px;
            }}
            """
        ],
    )
    on_click = cell.on_click
    button.on_click(lambda _: on_click())
    return button


def create_grid_preview(
    nrows: int,
    ncols: int,
    cells: Sequence[PreviewCell],
    *,
    width: int | None,
    height: int,
) -> pn.Column:
    """Build a preview of a grid layout.

    Parameters
    ----------
    nrows:
        Number of rows in the grid.
    ncols:
        Number of columns in the grid.
    cells:
        The cells to show. Cells that do not fit in ``nrows`` x ``ncols`` are
        skipped; positions no cell covers are drawn as empty boxes.
    width:
        Width of the grid in pixels, or None to stretch to the available width.
    height:
        Height of the grid in pixels.
    """
    grid = pn.GridSpec(
        width=width,
        height=height,
        sizing_mode='fixed' if width is not None else 'stretch_width',
    )

    def fits(geometry: CellGeometry) -> bool:
        return (
            geometry.row + geometry.row_span <= nrows
            and geometry.col + geometry.col_span <= ncols
        )

    covered: set[tuple[int, int]] = set()
    for cell in cells:
        geometry = cell.geometry
        if fits(geometry):
            for r in range(geometry.row, geometry.row + geometry.row_span):
                for c in range(geometry.col, geometry.col + geometry.col_span):
                    covered.add((r, c))

    for row in range(nrows):
        for col in range(ncols):
            if (row, col) not in covered:
                grid[row, col] = pn.pane.HTML(
                    '',
                    styles={
                        'background-color': '#f5f5f5',
                        'border': '1px dashed #ccc',
                        'box-sizing': 'border-box',
                    },
                    sizing_mode='stretch_both',
                    margin=1,
                )

    for i, cell in enumerate(cells):
        geometry = cell.geometry
        if not fits(geometry):
            continue
        grid[
            geometry.row : geometry.row + geometry.row_span,
            geometry.col : geometry.col + geometry.col_span,
        ] = _label(cell, _CELL_COLORS[i % len(_CELL_COLORS)])

    return pn.Column(
        grid,
        width=None if width is None else width + 24,
        sizing_mode='fixed' if width is not None else 'stretch_width',
        styles={
            'background-color': '#fafafa',
            'border': '1px solid #e0e0e0',
            'border-radius': '4px',
            'padding': '10px',
        },
    )
