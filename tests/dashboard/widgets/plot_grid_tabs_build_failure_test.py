# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
A cell build that raises must cost one cell, not the session (#1276).

The poll pass builds every cell of the session in one loop and pushes the
session's frame data alongside it, so an unguarded plotter failure aborts the
remaining builds *and* the data flush -- every plot in the session goes silent
while looking perfectly healthy.
"""

from __future__ import annotations

from uuid import uuid4

import holoviews as hv
import panel as pn
import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.plot_data_service import LayerId, PlotDataService
from ess.livedata.dashboard.plot_orchestrator import (
    CellGeometry,
    CellId,
    DataSourceConfig,
    Layer,
    PlotCell,
    PlotConfig,
    PlotOrchestrator,
)
from tests.helpers.plot_fakes import EmptyParams, FakePlotter

hv.extension('bokeh')


class RaisingPlotter(FakePlotter):
    """Plotter whose ``legend_position`` raises, as in the wild #1276 case."""

    @property
    def legend_position(self):
        raise AttributeError("'RaisingPlotter' object has no attribute 'legend_pos'")


def _make_config() -> PlotConfig:
    return PlotConfig(
        data_sources={
            'primary': DataSourceConfig(
                workflow_id=WorkflowId(instrument='test', name='wf', version=1),
                source_names=['src'],
                view_name='result',
            )
        },
        plot_name='image',
        params=EmptyParams(),
    )


def _add_cell(
    plot_orchestrator: PlotOrchestrator,
    plot_data_service: PlotDataService,
    grid_id,
    plotter: FakePlotter | None,
    *,
    col: int,
) -> tuple[CellId, LayerId]:
    """Add a one-layer cell to a grid, registering ``plotter`` unless None."""
    layer_id = LayerId(uuid4())
    cell_id = CellId(uuid4())
    cell = PlotCell(
        geometry=CellGeometry(row=0, col=col, row_span=1, col_span=1),
        layers=[Layer(layer_id=layer_id, config=_make_config())],
    )
    plot_orchestrator.peek_grid(grid_id).cells[cell_id] = cell
    if plotter is not None:
        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)
    return cell_id, layer_id


def _markdown_text(view: pn.viewable.Viewable) -> str:
    """All Markdown content rendered anywhere in a cell widget."""
    return '\n'.join(pane.object for pane in view.select(pn.pane.Markdown))


class _Grid:
    """A grid whose first cell raises on build and whose second one is fine."""

    def __init__(self, plot_orchestrator, plot_data_service, plot_grid_tabs):
        self.grid_id = plot_orchestrator.add_grid(title='Test', nrows=1, ncols=2)
        self.broken_plotter = RaisingPlotter(cached_state=hv.Curve([1, 2, 3]))
        self.healthy_plotter = FakePlotter(cached_state=hv.Curve([4, 5, 6]))
        self.broken_cell, self.broken_layer = _add_cell(
            plot_orchestrator,
            plot_data_service,
            self.grid_id,
            self.broken_plotter,
            col=0,
        )
        self.healthy_cell, self.healthy_layer = _add_cell(
            plot_orchestrator,
            plot_data_service,
            self.grid_id,
            self.healthy_plotter,
            col=1,
        )
        self._tabs = plot_grid_tabs
        self._orchestrator = plot_orchestrator

    def reveal(self) -> None:
        """Show the grid's tab and run a full poll pass over it."""
        self._tabs._poll_for_plot_updates()
        self._tabs.tabs.active = self._tabs._static_tabs_count
        self._tabs._poll_for_plot_updates()

    def new_frame(self) -> None:
        """Advance the grid's frame generation, arming the data flush."""
        self._orchestrator._frame_clock.commit(self.grid_id)


@pytest.fixture
def grid(plot_orchestrator, plot_data_service, plot_grid_tabs):
    return _Grid(plot_orchestrator, plot_data_service, plot_grid_tabs)


class TestBuildFailureIsolation:
    def test_cells_after_the_broken_one_are_still_built(self, plot_grid_tabs, grid):
        grid.reveal()

        assert grid.healthy_cell in plot_grid_tabs._cells

    def test_broken_cell_keeps_a_widget_reporting_the_failure(
        self, plot_grid_tabs, grid
    ):
        grid.reveal()

        cell_widget = plot_grid_tabs._cells[grid.broken_cell]
        assert not cell_widget.has_plot
        assert 'AttributeError' in _markdown_text(cell_widget.view)

    def test_broken_cell_is_not_rebuilt_until_its_inputs_change(
        self, plot_grid_tabs, plot_data_service, grid
    ):
        grid.reveal()
        original = plot_grid_tabs._cells[grid.broken_cell]

        plot_grid_tabs._poll_for_plot_updates()
        plot_grid_tabs._poll_for_plot_updates()
        assert plot_grid_tabs._cells[grid.broken_cell] is original

        # A reconfigured (here: restarted) layer is a changed input, so the
        # session tries again rather than latching on a stale failure.
        plot_data_service.job_started(
            grid.broken_layer, FakePlotter(cached_state=hv.Curve([1, 2, 3]))
        )
        plot_data_service.data_arrived(grid.broken_layer)
        plot_grid_tabs._poll_for_plot_updates()

        rebuilt = plot_grid_tabs._cells[grid.broken_cell]
        assert rebuilt is not original
        assert rebuilt.has_plot

    def test_chrome_failure_falls_back_to_an_error_panel(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs, grid
    ):
        """A layer missing from PlotDataService breaks the cell past composition."""
        unbuildable, _ = _add_cell(
            plot_orchestrator, plot_data_service, grid.grid_id, None, col=1
        )
        grid.reveal()

        assert 'failed to build' in _markdown_text(
            plot_grid_tabs._cells[unbuildable].view
        )


class TestBuildFailureDoesNotStarveData:
    def test_healthy_layers_keep_receiving_frames(self, plot_grid_tabs, grid):
        grid.reveal()
        session_layer = plot_grid_tabs._session_layers[grid.healthy_layer]
        assert session_layer.components is not None

        grid.healthy_plotter.compute(hv.Curve([7, 8, 9]))
        grid.new_frame()
        plot_grid_tabs._poll_for_plot_updates()

        assert not session_layer.components.presenter.has_pending_update()

    def test_a_disposal_cannot_stand_between_a_frame_and_the_pipes(
        self, plot_orchestrator, plot_grid_tabs, grid
    ):
        """The flush runs ahead of widget surgery, so a raising dispose is moot."""
        grid.reveal()
        session_layer = plot_grid_tabs._session_layers[grid.healthy_layer]
        broken_widget = plot_grid_tabs._cells[grid.broken_cell]

        def raising_dispose() -> None:
            raise RuntimeError('teardown blew up')

        broken_widget.dispose = raising_dispose
        del plot_orchestrator.peek_grid(grid.grid_id).cells[grid.broken_cell]
        grid.healthy_plotter.compute(hv.Curve([7, 8, 9]))
        grid.new_frame()
        with pytest.raises(RuntimeError):
            plot_grid_tabs._poll_for_plot_updates()

        assert not session_layer.components.presenter.has_pending_update()
