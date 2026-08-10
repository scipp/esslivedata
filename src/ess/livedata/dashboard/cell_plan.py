# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Desired per-session widget state for plot-grid cells.

:func:`desired_cells` is the *policy* half of a session's reconcile pass: a
pure function from shared-state snapshots and this session's view to the
target widget tree. The *mechanism* half — diffing the result against the
widgets that exist and building or disposing them — lives with the widgets
(``widgets/plot_grid_tabs.py``) and never changes when policy does.

Purity rule: this module must not import Panel, Bokeh, or HoloViews, and
:func:`desired_cells` must not mutate anything it reads. Policy questions
("should hidden grids pre-warm?", "should a modal suspend materialization?")
are answered here, as reviewable diffs to one function, and unit-tested with
plain data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .plot_data_service import LayerId, LayerSnapshot
from .plot_topology import (
    CellGeometry,
    CellId,
    GridId,
    PlotCell,
    PlotGridConfig,
)


@dataclass(frozen=True, slots=True)
class LayerBuildInput:
    """What a cell widget is built from, for one layer.

    ``snapshot`` pins the layer's lifecycle state (state, version, plotter
    identity, error message); snapshots are immutable and replaced wholesale,
    so equality means "no transition took effect in between".

    ``has_plot`` is tracked separately because the plotter's computed state is
    mutable *behind* a snapshot: the first viewer's activation computes a plot
    without a lifecycle transition when the layer is STOPPED (retained data).
    A widget built from "no computed plot" renders a placeholder, so the input
    must change when a real plot appears.
    """

    layer_id: LayerId
    snapshot: LayerSnapshot | None
    has_plot: bool


@dataclass(frozen=True, slots=True)
class CellBuildInputs:
    """Everything a cell widget's content is derived from.

    A built widget records the instance it was built from; the differ rebuilds
    exactly when the current inputs no longer compare equal. There is no
    record to update by hand, so a record cannot go stale: a failed build
    leaves the previous record (or none) in place, and the next pass retries.
    """

    geometry: CellGeometry
    user_title: str | None
    layers: tuple[LayerBuildInput, ...]


def cell_build_inputs(
    cell: PlotCell, layer_snapshot: Callable[[LayerId], LayerSnapshot | None]
) -> CellBuildInputs:
    """Sample a cell's build inputs as of right now.

    Called by :func:`desired_cells` for the plan, and again by the applier
    immediately before constructing a widget — the widget then records what
    the build actually saw, so a lifecycle transition landing between plan and
    build surfaces as an input difference on the next pass instead of being
    absorbed unrendered.
    """
    layers = []
    for layer in cell.layers:
        snapshot = layer_snapshot(layer.layer_id)
        has_plot = snapshot is not None and snapshot.has_displayable_plot()
        layers.append(
            LayerBuildInput(
                layer_id=layer.layer_id, snapshot=snapshot, has_plot=has_plot
            )
        )
    return CellBuildInputs(
        geometry=cell.geometry, user_title=cell.user_title, layers=tuple(layers)
    )


@dataclass(frozen=True, slots=True)
class CellPlan:
    """Target state of one materialized cell, from one session's point of view.

    ``inputs`` is what the session's widget for the cell must be built from.
    A cell present in topology but absent from the plans is *deferred*: any
    number of input changes while deferred coalesce into zero builds, and its
    inputs are not even sampled.
    """

    grid_id: GridId
    inputs: CellBuildInputs


@dataclass(frozen=True, slots=True)
class SessionView:
    """This session's contribution to the materialization decision.

    ``active_grid_id`` is None when no grid tab is visible — including while a
    modal is open, which obscures the plots.

    ``live_cell_ids`` are cells this session renders outside the visible grid:
    those behind a showing pop-out window (``widgets/plot_popout.py``), which
    floats above whatever tab is up. A minimized window renders nothing and
    contributes no cell, so parking a pop-out costs exactly what a hidden tab
    does.
    """

    active_grid_id: GridId | None
    live_cell_ids: frozenset[CellId] = frozenset()


def desired_cells(
    grids: Mapping[GridId, PlotGridConfig],
    layer_snapshot: Callable[[LayerId], LayerSnapshot | None],
    view: SessionView,
    watched: Callable[[LayerId], bool],
) -> dict[CellId, CellPlan]:
    """Compute the cells one session should hold built widgets for.

    Cells of disabled grids and *deferred* cells are omitted: they are not
    part of the target tree, but their already-built widgets survive (the
    applier disposes only cells that left the topology, so a re-enable or
    reveal finds them intact).

    A cell is in the plans when its grid is the one this session displays,
    when this session renders it anyway (``view.live_cell_ids``), or when any
    of its layers is watched (holds a viewer token — another session's, or
    this session's own on a live pop-out, since the caller releases this
    session's tokens on hidden layers before asking). A watched layer's plot
    is computed centrally anyway, so the build is a real plot pre-warming this
    session's tab switch; an unwatched hidden cell's build would be a
    placeholder that the reveal's first-viewer activation immediately
    invalidates (#1216).

    Parameters
    ----------
    grids:
        Topology view of the grids this session holds tab widgets for
        (enabled or disabled); the caller narrows the full topology, so a
        grid whose tab does not exist yet has no plans until the tab
        reconcile creates it.
    layer_snapshot:
        Accessor for per-layer lifecycle snapshots
        (:meth:`PlotDataService.get`).
    view:
        This session's view state: the grid tab it shows, and any further
        cells it renders in floating windows.
    watched:
        Whether any session holds a viewer token on a layer (the caller's
        one-shot read of :meth:`PlotDataService.viewed_layers`).

    Returns
    -------
    :
        Plan per materialized cell, insertion-ordered by grid then cell.
    """
    plans: dict[CellId, CellPlan] = {}
    for grid_id, grid in grids.items():
        if not grid.enabled:
            continue
        is_active = grid_id == view.active_grid_id
        for cell_id, cell in grid.cells.items():
            # A cell always has >=1 layer while it exists in topology (the
            # last layer's removal removes the cell); skip the transient
            # empty state defensively.
            if not cell.layers:
                continue
            if not (
                is_active
                or cell_id in view.live_cell_ids
                or any(watched(layer.layer_id) for layer in cell.layers)
            ):
                continue
            plans[cell_id] = CellPlan(
                grid_id=grid_id,
                inputs=cell_build_inputs(cell, layer_snapshot),
            )
    return plans
