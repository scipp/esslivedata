# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Plain-data tests of the session's desired-state policy.

No Panel, Bokeh, or HoloViews anywhere: topology and layer snapshots are
constructed as literals and the returned plans compared directly. This is
the home of every materialization-policy scenario, including the #1216 gate.
"""

from __future__ import annotations

from uuid import uuid4

from ess.livedata.dashboard.cell_plan import (
    SessionView,
    cell_build_inputs,
    desired_cells,
)
from ess.livedata.dashboard.plot_data_service import (
    LayerId,
    LayerSnapshot,
    LayerState,
)
from ess.livedata.dashboard.plot_orchestrator import (
    CellGeometry,
    CellId,
    GridId,
    Layer,
    PlotCell,
    PlotGridConfig,
)


class _Plotter:
    """Sentinel plotter: identity plus a settable cached-state flag."""

    def __init__(self, *, cached: bool = True) -> None:
        self.cached = cached

    def has_cached_state(self) -> bool:
        return self.cached


def _grid_id() -> GridId:
    return GridId(uuid4())


def _cell(*layer_ids: LayerId, user_title: str | None = None) -> PlotCell:
    return PlotCell(
        geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
        layers=[Layer(layer_id=lid, config=None) for lid in layer_ids],
        user_title=user_title,
    )


def _topology(
    grid_id: GridId, cells: dict[CellId, PlotCell], *, enabled: bool = True
) -> dict[GridId, PlotGridConfig]:
    return {
        grid_id: PlotGridConfig(
            title='G', nrows=2, ncols=2, cells=cells, enabled=enabled
        )
    }


class TestMaterialization:
    def test_active_grid_cell_is_materialized(self):
        grid_id, layer_id, cell_id = _grid_id(), LayerId(uuid4()), CellId(uuid4())
        snapshot = LayerSnapshot(state=LayerState.READY, version=3, plotter=_Plotter())
        plans = desired_cells(
            _topology(grid_id, {cell_id: _cell(layer_id)}),
            {layer_id: snapshot}.get,
            SessionView(active_grid_id=grid_id),
            watched=lambda _: False,
        )
        assert cell_id in plans
        assert plans[cell_id].grid_id == grid_id

    def test_hidden_unwatched_cell_is_deferred(self):
        """#1216: nobody watching, nothing built — whatever the versions do."""
        grid_id, layer_id, cell_id = _grid_id(), LayerId(uuid4()), CellId(uuid4())
        snapshot = LayerSnapshot(state=LayerState.READY, version=7, plotter=_Plotter())
        plans = desired_cells(
            _topology(grid_id, {cell_id: _cell(layer_id)}),
            {layer_id: snapshot}.get,
            SessionView(active_grid_id=None),
            watched=lambda _: False,
        )
        assert cell_id not in plans

    def test_hidden_watched_cell_is_materialized(self):
        """Pre-warm: another session's viewer keeps the plot computed, so the
        build is real and survives this session's reveal."""
        grid_id, layer_id, cell_id = _grid_id(), LayerId(uuid4()), CellId(uuid4())
        snapshot = LayerSnapshot(state=LayerState.READY, version=1, plotter=_Plotter())
        plans = desired_cells(
            _topology(grid_id, {cell_id: _cell(layer_id)}),
            {layer_id: snapshot}.get,
            SessionView(active_grid_id=None),
            watched=lambda lid: lid == layer_id,
        )
        assert cell_id in plans

    def test_modal_defers_like_a_hidden_tab(self):
        """An open modal reports no active grid; unwatched cells defer."""
        grid_id, layer_id, cell_id = _grid_id(), LayerId(uuid4()), CellId(uuid4())
        snapshot = LayerSnapshot(state=LayerState.READY, version=1, plotter=_Plotter())
        plans = desired_cells(
            _topology(grid_id, {cell_id: _cell(layer_id)}),
            {layer_id: snapshot}.get,
            SessionView(active_grid_id=None),
            watched=lambda _: False,
        )
        assert cell_id not in plans

    def test_hidden_cell_with_a_showing_popout_is_materialized(self):
        """A pop-out floats above whatever tab is up, so its cell is rendered
        wherever it sits and must track its inputs like a visible one."""
        grid_id, layer_id, cell_id = _grid_id(), LayerId(uuid4()), CellId(uuid4())
        snapshot = LayerSnapshot(state=LayerState.READY, version=1, plotter=_Plotter())
        plans = desired_cells(
            _topology(grid_id, {cell_id: _cell(layer_id)}),
            {layer_id: snapshot}.get,
            SessionView(active_grid_id=None, live_cell_ids=frozenset({cell_id})),
            watched=lambda _: False,
        )
        assert cell_id in plans

    def test_a_live_popout_does_not_materialize_its_neighbours(self):
        """Liveness is per cell: the rest of the hidden grid still sleeps."""
        grid_id, layer_id = _grid_id(), LayerId(uuid4())
        popped_out, neighbour = CellId(uuid4()), CellId(uuid4())
        snapshot = LayerSnapshot(state=LayerState.READY, version=1, plotter=_Plotter())
        plans = desired_cells(
            _topology(
                grid_id,
                {popped_out: _cell(layer_id), neighbour: _cell(LayerId(uuid4()))},
            ),
            {layer_id: snapshot}.get,
            SessionView(active_grid_id=None, live_cell_ids=frozenset({popped_out})),
            watched=lambda _: False,
        )
        assert set(plans) == {popped_out}

    def test_multi_layer_cell_materializes_if_any_layer_watched(self):
        grid_id = _grid_id()
        watched_layer, other_layer = LayerId(uuid4()), LayerId(uuid4())
        cell_id = CellId(uuid4())
        snapshot = LayerSnapshot(state=LayerState.READY, version=1, plotter=_Plotter())
        plans = desired_cells(
            _topology(grid_id, {cell_id: _cell(watched_layer, other_layer)}),
            {watched_layer: snapshot, other_layer: snapshot}.get,
            SessionView(active_grid_id=None),
            watched=lambda lid: lid == watched_layer,
        )
        assert cell_id in plans


class TestTopologyFiltering:
    def test_disabled_grid_cells_are_omitted(self):
        grid_id, layer_id, cell_id = _grid_id(), LayerId(uuid4()), CellId(uuid4())
        plans = desired_cells(
            _topology(grid_id, {cell_id: _cell(layer_id)}, enabled=False),
            {}.get,
            SessionView(active_grid_id=grid_id),
            watched=lambda _: True,
        )
        assert plans == {}

    def test_empty_cell_is_omitted(self):
        grid_id, cell_id = _grid_id(), CellId(uuid4())
        plans = desired_cells(
            _topology(grid_id, {cell_id: _cell()}),
            {}.get,
            SessionView(active_grid_id=grid_id),
            watched=lambda _: False,
        )
        assert plans == {}


class TestBuildInputs:
    def test_equal_state_gives_equal_inputs(self):
        layer_id = LayerId(uuid4())
        snapshot = LayerSnapshot(state=LayerState.READY, version=2, plotter=_Plotter())
        cell = _cell(layer_id)
        first = cell_build_inputs(cell, {layer_id: snapshot}.get)
        second = cell_build_inputs(cell, {layer_id: snapshot}.get)
        assert first == second

    def test_lifecycle_transition_changes_inputs(self):
        layer_id = LayerId(uuid4())
        plotter = _Plotter()
        before = LayerSnapshot(state=LayerState.READY, version=2, plotter=plotter)
        after = before.job_stopped()
        cell = _cell(layer_id)
        assert cell_build_inputs(cell, {layer_id: before}.get) != cell_build_inputs(
            cell, {layer_id: after}.get
        )

    def test_plotter_swap_changes_inputs(self):
        """A job restart replaces the plotter; same state name, new identity."""
        layer_id = LayerId(uuid4())
        cell = _cell(layer_id)
        first = LayerSnapshot(
            state=LayerState.WAITING_FOR_DATA, version=1, plotter=_Plotter()
        )
        second = first.job_started(_Plotter())
        assert cell_build_inputs(cell, {layer_id: first}.get) != cell_build_inputs(
            cell, {layer_id: second}.get
        )

    def test_computed_plot_changes_inputs_without_a_transition(self):
        """STOPPED retained data: the reveal's first-viewer activation computes
        a plot without a version bump; the input must still change so the
        placeholder widget is rebuilt with the real plot."""
        layer_id = LayerId(uuid4())
        plotter = _Plotter(cached=False)
        snapshot = LayerSnapshot(state=LayerState.STOPPED, version=4, plotter=plotter)
        cell = _cell(layer_id)
        before = cell_build_inputs(cell, {layer_id: snapshot}.get)
        plotter.cached = True
        after = cell_build_inputs(cell, {layer_id: snapshot}.get)
        assert before != after

    def test_title_change_changes_inputs(self):
        layer_id = LayerId(uuid4())
        snapshot = LayerSnapshot(state=LayerState.READY, version=1, plotter=_Plotter())
        assert cell_build_inputs(
            _cell(layer_id), {layer_id: snapshot}.get
        ) != cell_build_inputs(
            _cell(layer_id, user_title='Renamed'), {layer_id: snapshot}.get
        )

    def test_layer_set_change_changes_inputs(self):
        first_layer, second_layer = LayerId(uuid4()), LayerId(uuid4())
        snapshot = LayerSnapshot(state=LayerState.READY, version=1, plotter=_Plotter())
        states = {first_layer: snapshot, second_layer: snapshot}
        assert cell_build_inputs(_cell(first_layer), states.get) != cell_build_inputs(
            _cell(first_layer, second_layer), states.get
        )

    def test_missing_snapshot_differs_from_registered_one(self):
        layer_id = LayerId(uuid4())
        snapshot = LayerSnapshot(state=LayerState.READY, version=1, plotter=_Plotter())
        cell = _cell(layer_id)
        assert cell_build_inputs(cell, {}.get) != cell_build_inputs(
            cell, {layer_id: snapshot}.get
        )


def test_module_stays_pure():
    """The policy module must not load any UI framework (see its docstring).

    Runs in a subprocess: in-process the check is worthless, since the test
    session has long imported Panel and HoloViews. Transitive imports count —
    that is the realistic way purity would be lost.
    """
    import subprocess
    import sys

    script = (
        "import sys;"
        "import ess.livedata.dashboard.cell_plan;"
        "leaked = sorted("
        "    m for m in sys.modules"
        "    if m.split('.')[0] in ('panel', 'holoviews', 'bokeh')"
        ");"
        "assert not leaked, f'cell_plan pulled in UI frameworks: {leaked}'"
    )
    subprocess.run([sys.executable, '-c', script], check=True)  # noqa: S603
