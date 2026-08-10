# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Behavior tests for the PlotGridTabs poll loop: cell (re)builds for
Layout-producing plotters (issue #805), the titlebar status pill, and
per-layer time panes.
"""

from __future__ import annotations

import time
from uuid import uuid4

import holoviews as hv

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.timestamp import Timestamp
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
from ess.livedata.dashboard.plots import TimeBounds
from ess.livedata.dashboard.widgets.styles import StatusPill
from tests.helpers.panel_ui import click_tool
from tests.helpers.plot_fakes import EmptyParams, FakePlotter, ViewerToken

hv.extension('bokeh')


# -- Helpers ---------------------------------------------------------------


def _make_layout() -> hv.Layout:
    return hv.Layout(
        [hv.Curve([1, 2, 3]).relabel('A'), hv.Curve([4, 5, 6]).relabel('B')]
    )


def _make_bounds(age_seconds: float) -> TimeBounds:
    """Bounds whose data age is ``age_seconds`` relative to now."""
    now_ns = time.time_ns()
    end_ns = now_ns - int(age_seconds * 1e9)
    return TimeBounds(
        min_end=Timestamp.from_ns(end_ns),
        created_at=Timestamp.from_ns(now_ns),
        min_start=Timestamp.from_ns(end_ns - int(1e9)),
        max_end=Timestamp.from_ns(end_ns),
    )


def _inject_layer(
    plot_orchestrator: PlotOrchestrator,
    plot_data_service: PlotDataService,
    grid_id,
    plotter: FakePlotter,
) -> LayerId:
    """
    Add a cell+layer to a grid and register the plotter in PlotDataService.

    Bypasses workflow subscription (not needed for poll tests) by writing
    directly into the orchestrator's grid config and PlotDataService.
    """
    layer_id = LayerId(uuid4())
    cell_id = CellId(uuid4())
    config = PlotConfig(
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
    cell = PlotCell(
        geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
        layers=[Layer(layer_id=layer_id, config=config)],
    )
    grid = plot_orchestrator.peek_grid(grid_id)
    grid.cells[cell_id] = cell

    plot_data_service.job_started(layer_id, plotter)
    plot_data_service.data_arrived(layer_id)
    return layer_id


def _inject_two_layer_cell(
    plot_orchestrator: PlotOrchestrator,
    plot_data_service: PlotDataService,
    grid_id,
    plotters: tuple[FakePlotter, FakePlotter],
) -> tuple[LayerId, LayerId]:
    """Add one cell with two layers, registering one plotter per layer."""
    cell_id = CellId(uuid4())
    layer_ids = (LayerId(uuid4()), LayerId(uuid4()))

    def cfg(view):
        return PlotConfig(
            data_sources={
                'primary': DataSourceConfig(
                    workflow_id=WorkflowId(instrument='test', name='wf', version=1),
                    source_names=['src'],
                    view_name=view,
                )
            },
            plot_name='image',
            params=EmptyParams(),
        )

    cell = PlotCell(
        geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
        layers=[
            Layer(layer_id=lid, config=cfg(view))
            for lid, view in zip(layer_ids, ('a', 'b'), strict=True)
        ],
    )
    plot_orchestrator.peek_grid(grid_id).cells[cell_id] = cell
    for lid, plotter in zip(layer_ids, plotters, strict=True):
        plot_data_service.job_started(lid, plotter)
        plot_data_service.data_arrived(lid)
    return layer_ids


# -- Tests -----------------------------------------------------------------


class TestPollHandlesLayoutPlotters:
    """
    _poll_for_plot_updates must not raise when a layer's DynamicMap has
    been evaluated as Layout by Bokeh, even if a subsequent version change
    triggers a rebuild.
    """

    def test_poll_creates_session_layer_for_layout_plotter(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs
    ):
        """First poll with a Layout plotter creates a session layer with components."""
        plotter = FakePlotter(cached_state=_make_layout())
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=2, ncols=2)
        layer_id = _inject_layer(plot_orchestrator, plot_data_service, grid_id, plotter)

        # First poll adds the grid tab; revealing it builds the cell/session
        # layer (the switch runs the pass synchronously in tests).
        plot_grid_tabs._poll_for_plot_updates()
        plot_grid_tabs.tabs.active = plot_grid_tabs._static_tabs_count

        session_layer = plot_grid_tabs._session_layers.get(layer_id)
        assert session_layer is not None
        assert session_layer.dmap is not None

    def test_plotter_swap_rebuilds_a_rendered_layout_cell(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs
    ):
        """The rebuild path must survive a Layout that Bokeh already rendered.

        The generic differ rule (changed inputs replace the widget) is covered
        in ``plot_grid_tabs_test.py``; what is under test here is that doing so
        for a Layout-valued DynamicMap does not raise (#805).
        """
        plotter = FakePlotter(cached_state=_make_layout())
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=2, ncols=2)
        layer_id = _inject_layer(plot_orchestrator, plot_data_service, grid_id, plotter)

        plot_grid_tabs._poll_for_plot_updates()
        plot_grid_tabs.tabs.active = plot_grid_tabs._static_tabs_count
        cell_id = next(iter(plot_orchestrator.peek_grid(grid_id).cells))
        original_widget = plot_grid_tabs._cells[cell_id]
        original_widget.view.get_root()

        plot_data_service.job_started(
            layer_id, FakePlotter(cached_state=_make_layout())
        )
        plot_data_service.data_arrived(layer_id)
        plot_grid_tabs._poll_for_plot_updates()

        assert plot_grid_tabs._cells[cell_id] is not original_widget


class TestFreshnessIndicator:
    """The titlebar freshness pane reflects the plotter's time bounds."""

    def test_poll_populates_freshness_pane(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs
    ):
        """A poll on the active grid fills the cell's freshness pane with lag."""
        plotter = FakePlotter(
            cached_state=_make_layout(), time_bounds=_make_bounds(2.0)
        )
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=2, ncols=2)
        _inject_layer(plot_orchestrator, plot_data_service, grid_id, plotter)

        # First poll adds the tab and builds the cell; activate then poll again
        # so the freshness pane fills for the now-active grid. The per-layer
        # time panes exist only once their toolbars are revealed.
        plot_grid_tabs._poll_for_plot_updates()
        plot_grid_tabs.tabs.active = plot_grid_tabs._static_tabs_count
        for cell_widget in plot_grid_tabs._cells.values():
            click_tool(cell_widget.view, 'lt-tool-layer-details')
        plot_grid_tabs._poll_for_plot_updates()

        panes = [cw.freshness_pane for cw in plot_grid_tabs._cells.values()]
        assert len(panes) == 1
        # Pill shows the compact data age with band styling, no hover tooltip.
        # Age is computed against the poll's wall clock, so assert structure
        # rather than an exact value.
        assert 'border-radius' in panes[0].object
        assert 's</span>' in panes[0].object  # compact age label, e.g. "2.0s"
        assert 'title=' not in panes[0].object

        # The per-layer time pane shows the full range + lag.
        layer_panes = [
            pane
            for cw in plot_grid_tabs._cells.values()
            for pane in cw.layer_time_panes.values()
        ]
        assert len(layer_panes) == 1
        assert 'Lag:' in layer_panes[0].object

    def test_multilayer_each_layer_pane_populated(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs
    ):
        """Both layers in a multi-layer cell get their time pane populated."""
        bounds = _make_bounds(2.0)
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=2, ncols=2)
        l1, l2 = _inject_two_layer_cell(
            plot_orchestrator,
            plot_data_service,
            grid_id,
            (
                FakePlotter(cached_state=_make_layout(), time_bounds=bounds),
                FakePlotter(cached_state=_make_layout(), time_bounds=bounds),
            ),
        )

        # First poll adds the tab and builds the cell; activate then poll again
        # so the per-layer time panes fill for the now-active grid. The panes
        # exist only once their toolbars are revealed.
        plot_grid_tabs._poll_for_plot_updates()
        plot_grid_tabs.tabs.active = plot_grid_tabs._static_tabs_count
        for cell_widget in plot_grid_tabs._cells.values():
            click_tool(cell_widget.view, 'lt-tool-layer-details')
        plot_grid_tabs._poll_for_plot_updates()

        (cell_widget,) = plot_grid_tabs._cells.values()
        assert 'Lag:' in cell_widget.layer_time_panes[l1].object
        assert 'Lag:' in cell_widget.layer_time_panes[l2].object
        # The cell pill must still show with two layers.
        assert 'border-radius' in cell_widget.freshness_pane.object

    def test_rebuilt_cell_refills_its_time_panes_on_the_same_poll(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs
    ):
        """A rebuilt cell must show its age and time range straight away.

        A rebuild mints blank panes, so the poll that rebuilds refills them (a
        revealed cell's panes are additionally seeded from current bounds as
        they are built). Without either, the cell shows no age and no time
        range until the next frame or the stall tick, however good the data
        behind it is.
        """
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=2, ncols=2)
        layer_id = _inject_layer(
            plot_orchestrator,
            plot_data_service,
            grid_id,
            FakePlotter(cached_state=_make_layout(), time_bounds=_make_bounds(2.0)),
        )
        plot_grid_tabs._poll_for_plot_updates()
        plot_grid_tabs.tabs.active = plot_grid_tabs._static_tabs_count
        for cell_widget in plot_grid_tabs._cells.values():
            click_tool(cell_widget.view, 'lt-tool-layer-details')
        plot_grid_tabs._poll_for_plot_updates()

        # A new plotter bumps the layer version, so the next poll rebuilds the
        # cell -- without a new frame, which would have refreshed it anyway.
        plot_data_service.job_started(
            layer_id,
            FakePlotter(cached_state=_make_layout(), time_bounds=_make_bounds(3.0)),
        )
        plot_data_service.data_arrived(layer_id)
        plot_grid_tabs._poll_for_plot_updates()

        (cell_widget,) = plot_grid_tabs._cells.values()
        assert 'border-radius' in cell_widget.freshness_pane.object
        assert 'Lag:' in cell_widget.layer_time_panes[layer_id].object

    def test_poll_updating_nothing_does_not_consume_the_stall_interval(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs
    ):
        """The stall clock budgets the next update; it is not a poll counter.

        Polls over a hidden grid update no cell. Restarting the interval on
        them would make a cell revealed just afterwards wait a further full
        interval before the stall path could fill its pill.
        """
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=2, ncols=2)
        _inject_layer(
            plot_orchestrator,
            plot_data_service,
            grid_id,
            FakePlotter(cached_state=_make_layout(), time_bounds=_make_bounds(2.0)),
        )
        # Leave a non-plot static tab active so the grid is hidden.
        plot_grid_tabs.tabs.active = 0
        stall_clock = plot_grid_tabs._last_freshness_update

        plot_grid_tabs._poll_for_plot_updates()
        plot_grid_tabs._poll_for_plot_updates()

        assert plot_grid_tabs._last_freshness_update == stall_clock

    def test_stall_rerenders_pill_into_older_band(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs
    ):
        """With no new frame, only an elapsed stall interval re-renders the
        pill; the aged bounds then move it to an older band."""
        from ess.livedata.dashboard.widgets.plot_grid_tabs import (
            _FRESHNESS_STALL_INTERVAL_S,
        )

        plotter = FakePlotter(
            cached_state=_make_layout(), time_bounds=_make_bounds(2.0)
        )
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=2, ncols=2)
        _inject_layer(plot_orchestrator, plot_data_service, grid_id, plotter)
        plot_grid_tabs._poll_for_plot_updates()
        plot_grid_tabs.tabs.active = plot_grid_tabs._static_tabs_count
        plot_grid_tabs._poll_for_plot_updates()
        (cell_widget,) = plot_grid_tabs._cells.values()
        assert StatusPill.FRESH[0] in cell_widget.freshness_pane.object

        # Ages the stream without a wall-clock sleep; no version moves, so a
        # poll before the stall interval elapses leaves the pill untouched.
        plotter.time_bounds = _make_bounds(60.0)
        plot_grid_tabs._poll_for_plot_updates()
        assert StatusPill.FRESH[0] in cell_widget.freshness_pane.object

        plot_grid_tabs._last_freshness_update -= _FRESHNESS_STALL_INTERVAL_S
        assert plot_grid_tabs._has_pending_work()
        plot_grid_tabs._poll_for_plot_updates()

        pill = cell_widget.freshness_pane.object
        assert StatusPill.OLD[0] in pill
        assert StatusPill.FRESH[0] not in pill

    def test_hidden_grid_does_not_update_freshness(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs
    ):
        """Freshness is only refreshed for the active grid tab."""
        plotter = FakePlotter(cached_state=_make_layout(), time_bounds=None)
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=2, ncols=2)
        # Leave a non-plot static tab active so the grid is hidden.
        plot_grid_tabs.tabs.active = 0
        layer_id = _inject_layer(plot_orchestrator, plot_data_service, grid_id, plotter)

        # Another session's viewer token makes the hidden cell build (#1216).
        token = ViewerToken()
        plot_data_service.set_active(layer_id, token, True)

        plot_grid_tabs._poll_for_plot_updates()

        assert plot_grid_tabs._cells
        for cell_widget in plot_grid_tabs._cells.values():
            assert cell_widget.freshness_pane.object in ('', None)


class TestStoppedJobIndication:
    """The titlebar pill freezes to a status when jobs stop or error (#1120)."""

    def _poll_active(self, plot_grid_tabs):
        """Poll once to build, activate the grid tab, poll again to refresh."""
        plot_grid_tabs._poll_for_plot_updates()
        plot_grid_tabs.tabs.active = plot_grid_tabs._static_tabs_count
        plot_grid_tabs._poll_for_plot_updates()
        (cell_widget,) = plot_grid_tabs._cells.values()
        return cell_widget

    def test_stopped_job_shows_stopped_pill_despite_fresh_bounds(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs
    ):
        plotter = FakePlotter(
            cached_state=_make_layout(), time_bounds=_make_bounds(2.0)
        )
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=2, ncols=2)
        layer_id = _inject_layer(plot_orchestrator, plot_data_service, grid_id, plotter)
        plot_data_service.job_stopped(layer_id)

        cell_widget = self._poll_active(plot_grid_tabs)

        # Frozen "stopped" pill; the fresh bounds seen by the freshness update
        # above must not overwrite it with an age band.
        pill = cell_widget.freshness_pane.object
        assert 'stopped' in pill
        assert StatusPill.STOPPED[0] in pill
        assert StatusPill.FRESH[0] not in pill

    def test_errored_layer_shows_error_pill(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs
    ):
        plotter = FakePlotter(
            cached_state=_make_layout(), time_bounds=_make_bounds(2.0)
        )
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=2, ncols=2)
        layer_id = _inject_layer(plot_orchestrator, plot_data_service, grid_id, plotter)
        plot_data_service.error_occurred(layer_id, 'boom')

        cell_widget = self._poll_active(plot_grid_tabs)

        assert 'error' in cell_widget.freshness_pane.object

    def test_partial_stop_excludes_stopped_layer_from_freshness(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs
    ):
        """A stopped layer's growing age must not drag the live pill to red."""
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=2, ncols=2)
        stopped_layer, _live_layer = _inject_two_layer_cell(
            plot_orchestrator,
            plot_data_service,
            grid_id,
            (
                FakePlotter(cached_state=_make_layout(), time_bounds=_make_bounds(120)),
                FakePlotter(cached_state=_make_layout(), time_bounds=_make_bounds(2.0)),
            ),
        )
        plot_data_service.job_stopped(stopped_layer)

        cell_widget = self._poll_active(plot_grid_tabs)

        pill = cell_widget.freshness_pane.object
        assert StatusPill.FRESH[0] in pill
        assert StatusPill.OLD[0] not in pill
        assert 'stopped' not in pill
