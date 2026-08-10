# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Topology of the plot hierarchy: grids, cells, layers, and their configs.

The *topology* (see the dashboard glossary) is the shared arrangement of
plots — which grids exist, which cells each grid holds, which layers each
cell holds — owned and versioned by ``PlotOrchestrator``. This module holds
the plain data types describing it; it must stay free of UI-framework
imports, direct or transitive, so that policy code (``cell_plan``) and its
tests can consume topology without loading Panel/Bokeh/HoloViews.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NewType
from uuid import UUID

import pydantic

from ess.livedata.config.workflow_spec import WorkflowId

from .data_roles import PRIMARY
from .plot_data_service import LayerId

GridId = NewType('GridId', UUID)
CellId = NewType('CellId', UUID)


@dataclass(frozen=True)
class CellGeometry:
    """
    Grid cell geometry (position and size).

    Defines the location and span of a cell in a plot grid.
    """

    row: int
    col: int
    row_span: int
    col_span: int

    def overlaps(self, other: CellGeometry) -> bool:
        """Return True if this cell shares any grid slot with ``other``."""
        return (
            self.row < other.row + other.row_span
            and other.row < self.row + self.row_span
            and self.col < other.col + other.col_span
            and other.col < self.col + self.col_span
        )


def reject_overlapping_cells(geometries: Iterable[CellGeometry]) -> None:
    """Raise ValueError if any two cell geometries overlap.

    Grid cells must tile without overlap; overlapping cells claim the same
    slot for two plots. This guards the collection-level entry points (config
    load, file upload), which build a full cell set at once and so have to
    decide before applying any of it: relying on ``add_cell`` alone would raise
    partway through, leaving a half-built grid behind and reporting the fault
    only once the user had committed to the import.
    """
    seen: list[CellGeometry] = []
    for geometry in geometries:
        for other in seen:
            if geometry.overlaps(other):
                raise ValueError(f'Cell geometry {geometry} overlaps {other}')
        seen.append(geometry)


@dataclass
class DataSourceConfig:
    """Configuration for a single data source in a plot layer.

    This defines how to connect a layer to a workflow's user-facing output
    view. The backend pydantic field name (used in ``DataKey``) is
    resolved at subscription time from the view name plus the current
    window mode.
    """

    workflow_id: WorkflowId
    source_names: list[str]
    view_name: str = 'result'


@dataclass
class PlotConfig:
    """Configuration for a single plot layer.

    The data_sources dict maps role names to DataSourceConfig:

    - **"primary"**: The main data source (required). For standard plots, this is
      the only entry. For correlation histograms, this is the data to be histogrammed.
    - **"x_axis"**: X-axis correlation data (optional). Used by correlation histograms.
    - **"y_axis"**: Y-axis correlation data (optional). For 2D correlation histograms.

    Static overlays (e.g., geometric shapes) have a primary source with empty
    source_names, a synthetic workflow ID, and store a user-defined name in output_name.

    Convenience properties (workflow_id, source_names, output_name) provide direct
    access to the primary data source.
    """

    data_sources: dict[str, DataSourceConfig]
    plot_name: str
    params: pydantic.BaseModel

    @property
    def workflow_id(self) -> WorkflowId:
        """Workflow ID from the primary data source."""
        if PRIMARY not in self.data_sources:
            raise ValueError("Cannot access workflow_id: no primary data source")
        return self.data_sources[PRIMARY].workflow_id

    @property
    def source_names(self) -> list[str]:
        """Source names from the primary data source."""
        if PRIMARY not in self.data_sources:
            raise ValueError("Cannot access source_names: no primary data source")
        return self.data_sources[PRIMARY].source_names

    @property
    def view_name(self) -> str:
        """Output view name from the primary data source."""
        if PRIMARY not in self.data_sources:
            raise ValueError("Cannot access view_name: no primary data source")
        return self.data_sources[PRIMARY].view_name

    def is_static(self) -> bool:
        """Return True if this is a static overlay (no workflow subscription needed).

        Static overlays have only a primary data source with empty source_names.
        They use a synthetic workflow ID and store the user-defined overlay name
        in view_name.
        """
        if PRIMARY not in self.data_sources or len(self.data_sources) != 1:
            return False
        return len(self.data_sources[PRIMARY].source_names) == 0


@dataclass
class Layer:
    """A layer within a plot cell, combining identity with configuration."""

    layer_id: LayerId
    config: PlotConfig


@dataclass
class PlotCell:
    """
    Configuration for a plot cell (position, size, and layers to plot).

    The plots are placed in the given row and col of a :py:class:`PlotGrid`, spanning
    the given number of rows and columns. A cell can contain multiple layers that
    are composed via hv.Overlay.

    ``user_title`` is an optional user-defined cell title shown in the cell
    titlebar. When ``None`` the titlebar shows a title derived from the layers.
    """

    geometry: CellGeometry
    layers: list[Layer]
    user_title: str | None = None


@dataclass
class PlotGridConfig:
    """A plot grid tab configuration."""

    title: str = ""
    nrows: int = 3
    ncols: int = 3
    cells: dict[CellId, PlotCell] = field(default_factory=dict)
    enabled: bool = True
