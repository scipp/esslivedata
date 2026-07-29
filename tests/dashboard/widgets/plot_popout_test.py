# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Behavior tests for plot pop-out windows.

A pop-out is a second, independent view of a cell's plot in a floating window.
The tests cover the two properties that make it useful -- the cell is left
alone, and both views stay live off the same data pipe -- plus the lifecycle
rules that keep a window from outliving the cell it shows.
"""

from __future__ import annotations

import holoviews as hv
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.dashboard.data_roles import PRIMARY
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.notification_queue import NotificationQueue
from ess.livedata.dashboard.plot_data_service import PlotDataService
from ess.livedata.dashboard.plot_orchestrator import (
    CellGeometry,
    DataSourceConfig,
    PlotConfig,
    PlotOrchestrator,
)
from ess.livedata.dashboard.plot_params import PlotParams1d
from ess.livedata.dashboard.plots import LinePlotter
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.session_layer import SessionLayer
from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry
from ess.livedata.dashboard.session_updater import SessionUpdater
from ess.livedata.dashboard.static_plots import LinesCoordinates, VLinesParams
from ess.livedata.dashboard.stream_manager import StreamManager
from ess.livedata.dashboard.widgets.plot_grid_tabs import PlotGridTabs
from ess.livedata.dashboard.widgets.workflow_status_widget import (
    WorkflowStatusListWidget,
)

hv.extension('bokeh')

_WORKFLOW = WorkflowId(instrument='test', name='wf', version=1)
_GEOMETRY = CellGeometry(row=0, col=0, row_span=1, col_span=1)


@pytest.fixture
def plot_data_service():
    return PlotDataService()


@pytest.fixture
def data_service():
    return DataService()


@pytest.fixture
def job_service():
    return JobService()


@pytest.fixture
def plot_orchestrator(job_orchestrator, data_service, plot_data_service):
    stream_manager = StreamManager(data_service=data_service)
    return PlotOrchestrator(
        plotting_controller=PlottingController(stream_manager=stream_manager),
        job_orchestrator=job_orchestrator,
        data_service=data_service,
        instrument='dummy',
        plot_data_service=plot_data_service,
    )


@pytest.fixture
def plot_grid_tabs(
    plot_orchestrator,
    workflow_registry,
    plot_data_service,
    job_orchestrator,
    job_service,
):
    stream_manager = StreamManager(data_service=DataService())
    return PlotGridTabs(
        plot_orchestrator=plot_orchestrator,
        workflow_registry=workflow_registry,
        plotting_controller=PlottingController(stream_manager=stream_manager),
        workflow_status_widget=WorkflowStatusListWidget(
            orchestrator=job_orchestrator, job_service=job_service
        ),
        plot_data_service=plot_data_service,
        session_updater=SessionUpdater(
            session_id=SessionId('test'),
            session_registry=SessionRegistry(),
            notification_queue=NotificationQueue(),
        ),
    )


def _curve(values: list[float]) -> sc.DataArray:
    return sc.DataArray(
        sc.array(dims=['x'], values=values, unit='counts'),
        coords={'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0], unit='m')},
    )


def _line_config() -> PlotConfig:
    return PlotConfig(
        data_sources={
            PRIMARY: DataSourceConfig(
                workflow_id=_WORKFLOW, source_names=['s1'], view_name='out'
            )
        },
        plot_name='lines',
        params=PlotParams1d(),
    )


def _static_config() -> PlotConfig:
    return PlotConfig(
        data_sources={
            PRIMARY: DataSourceConfig(
                workflow_id=_WORKFLOW, source_names=[], view_name='guides'
            )
        },
        plot_name='vlines',
        params=VLinesParams(geometry=LinesCoordinates(positions='10, 20')),
    )


@pytest.fixture
def line_cell(plot_orchestrator, plot_grid_tabs, plot_data_service):
    """A built cell showing a live 1-D line plot, plus its plotter and pipe.

    Uses a real ``LinePlotter``: the pop-out's defining properties (a private
    autoscale controller, a shared data pipe) only exist for a plot that
    actually autoscales and actually updates.
    """
    grid_id = plot_orchestrator.add_grid(title='G', nrows=1, ncols=1)
    cell_id = plot_orchestrator.add_cell(grid_id, _GEOMETRY)
    layer_id = plot_orchestrator.add_layer(cell_id, _line_config())

    plotter = LinePlotter.from_params(PlotParams1d())
    key = DataKey(workflow_id=_WORKFLOW, source_name='s1', output_name='out')
    plotter.compute({PRIMARY: {key: _curve([1.0, 2.0, 3.0])}})
    plot_data_service.job_started(layer_id, plotter)
    plot_data_service.data_arrived(layer_id)

    state = plot_data_service.get(layer_id)
    session_layer = SessionLayer(layer_id=layer_id, last_seen_version=state.version)
    session_layer.ensure_components(state)
    plot_grid_tabs._session_layers[layer_id] = session_layer

    plot_grid_tabs._poll_for_plot_updates()
    return cell_id


def _open_windows(plot_grid_tabs) -> list:
    return list(plot_grid_tabs._popouts.container.objects)


class TestPopoutOpensAnIndependentView:
    def test_popout_opens_one_floating_window(self, plot_grid_tabs, line_cell):
        assert _open_windows(plot_grid_tabs) == []

        plot_grid_tabs._show_popout(line_cell)

        windows = _open_windows(plot_grid_tabs)
        assert len(windows) == 1
        assert 'lt-popout-r0c0' in windows[0].css_classes

    def test_window_title_is_the_cell_title(
        self, plot_grid_tabs, plot_orchestrator, line_cell
    ):
        plot_orchestrator.set_cell_title(line_cell, 'Popped')
        plot_grid_tabs._poll_for_plot_updates()

        plot_grid_tabs._show_popout(line_cell)

        assert _open_windows(plot_grid_tabs)[0].name == 'Popped'

    def test_cell_keeps_its_own_plot_and_autoscale_controller(
        self, plot_grid_tabs, line_cell
    ):
        cell_widget = plot_grid_tabs._cells[line_cell]
        cell_controller = cell_widget.autoscale_controller
        assert cell_controller is not None

        plot_grid_tabs._show_popout(line_cell)

        # The cell is untouched: same widget, same plot, same controller.
        assert plot_grid_tabs._cells[line_cell] is cell_widget
        assert cell_widget.autoscale_controller is cell_controller

    def test_each_view_gets_its_own_autoscale_controller(
        self, plot_grid_tabs, line_cell
    ):
        cell_widget = plot_grid_tabs._cells[line_cell]

        detached = cell_widget.compose_detached_view()

        # A controller binds its Bokeh tools to the first figure it renders
        # into, so a shared one would leave the pop-out without toggles.
        assert detached.autoscale is not None
        assert detached.autoscale is not cell_widget.autoscale_controller

    def test_both_views_render_their_own_autoscale_tools(
        self, plot_grid_tabs, line_cell
    ):
        """Rendering one view must not strip the hooks off the other.

        HoloViews ``.opts()`` mutates in place unless cloned, so composing a
        second view of the same single-layer cell can silently replace the
        first view's hooks -- the cell would lose its autoscale toggles the
        moment it was popped out.
        """
        cell_widget = plot_grid_tabs._cells[line_cell]
        detached = cell_widget.compose_detached_view()
        renderer = hv.renderer('bokeh')

        cell_figure = renderer.get_plot(cell_widget.compose_detached_view().plot)
        popout_figure = renderer.get_plot(detached.plot)

        for figure in (cell_figure, popout_figure):
            descriptions = {tool.description for tool in figure.state.toolbar.tools}
            assert 'Fit ranges to current data' in descriptions

    def test_both_views_repaint_from_the_same_pipe(self, plot_grid_tabs, line_cell):
        """The pop-out is live, not a snapshot: one pipe drives both figures."""
        cell_widget = plot_grid_tabs._cells[line_cell]
        detached = cell_widget.compose_detached_view()

        session_layer = next(iter(plot_grid_tabs._session_layers.values()))
        pipe = session_layer.components.pipe
        cell_plot = cell_widget.compose_detached_view().plot

        assert pipe in cell_plot.streams
        assert pipe in detached.plot.streams


class TestPopoutLifecycle:
    def test_reopening_replaces_rather_than_stacks(self, plot_grid_tabs, line_cell):
        plot_grid_tabs._show_popout(line_cell)
        plot_grid_tabs._show_popout(line_cell)

        assert len(_open_windows(plot_grid_tabs)) == 1

    def test_closing_the_window_drops_it(self, plot_grid_tabs, line_cell):
        plot_grid_tabs._show_popout(line_cell)

        # jsPanel round-trips the user's click on the window's close button.
        _open_windows(plot_grid_tabs)[0].status = 'closed'

        assert _open_windows(plot_grid_tabs) == []

    def test_minimizing_the_window_keeps_it(self, plot_grid_tabs, line_cell):
        plot_grid_tabs._show_popout(line_cell)

        _open_windows(plot_grid_tabs)[0].status = 'minimized'

        assert len(_open_windows(plot_grid_tabs)) == 1

    def test_cell_rebuild_carries_the_window_along(
        self, plot_grid_tabs, plot_orchestrator, line_cell
    ):
        """A rebuilt cell composes fresh plots, so the window is rebuilt too.

        It must not simply vanish: rebuilds follow ordinary user actions
        elsewhere in the dashboard (a rename, a job restart), and a floating
        view disappearing in response would read as a crash.
        """
        plot_grid_tabs._show_popout(line_cell)
        window_before = _open_windows(plot_grid_tabs)[0]

        plot_orchestrator.set_cell_title(line_cell, 'Renamed')
        plot_grid_tabs._poll_for_plot_updates()

        windows = _open_windows(plot_grid_tabs)
        assert len(windows) == 1
        assert windows[0] is not window_before
        assert windows[0].name == 'Renamed'

    def test_cell_removal_closes_the_window(
        self, plot_grid_tabs, plot_orchestrator, line_cell
    ):
        plot_grid_tabs._show_popout(line_cell)
        layer_id = plot_orchestrator.get_cell(line_cell).layers[0].layer_id

        plot_orchestrator.remove_layer(layer_id)
        plot_grid_tabs._poll_for_plot_updates()

        assert _open_windows(plot_grid_tabs) == []

    def test_session_teardown_closes_all_windows(self, plot_grid_tabs, line_cell):
        plot_grid_tabs._show_popout(line_cell)

        plot_grid_tabs.dispose_widgets()

        assert _open_windows(plot_grid_tabs) == []

    def test_placeholder_cell_pops_out_nothing(self, plot_grid_tabs, plot_orchestrator):
        """A cell waiting for data has no plot to show in a window."""
        grid_id = plot_orchestrator.add_grid(title='G', nrows=1, ncols=1)
        cell_id = plot_orchestrator.add_cell(grid_id, _GEOMETRY)
        plot_orchestrator.add_layer(cell_id, _line_config())
        plot_grid_tabs._poll_for_plot_updates()
        assert not plot_grid_tabs._cells[cell_id].has_plot

        plot_grid_tabs._show_popout(cell_id)

        assert _open_windows(plot_grid_tabs) == []


class TestPopoutKeepsItsCellLive:
    """A window floating above another tab must not show frozen data."""

    def test_hidden_grid_sleeps_without_a_popout(
        self, plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
    ):
        plot_grid_tabs.tabs.active = 0  # Workflows: no grid is visible

        plot_grid_tabs._poll_for_plot_updates()

        assert plot_grid_tabs._popouts.open_cells() == frozenset()
        assert not _layer_is_active(
            plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
        )

    def test_popped_out_cell_stays_active_on_another_tab(
        self, plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
    ):
        plot_grid_tabs._show_popout(line_cell)
        plot_grid_tabs.tabs.active = 0

        plot_grid_tabs._poll_for_plot_updates()

        assert plot_grid_tabs._popouts.open_cells() == frozenset({line_cell})
        assert _layer_is_active(
            plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
        )

    def test_closing_the_window_lets_the_cell_sleep_again(
        self, plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
    ):
        plot_grid_tabs._show_popout(line_cell)
        plot_grid_tabs.tabs.active = 0
        plot_grid_tabs._poll_for_plot_updates()

        plot_grid_tabs._popouts.close(line_cell)
        plot_grid_tabs._poll_for_plot_updates()

        assert not _layer_is_active(
            plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
        )


def _layer_is_active(plot_grid_tabs, plot_orchestrator, plot_data_service, cell_id):
    """Whether a viewer still holds interest in the cell's layer.

    The interest token is what keeps a layer computing; a hidden grid's layers
    lose it, which is exactly what would freeze a pop-out.
    """
    layer_id = plot_orchestrator.get_cell(cell_id).layers[0].layer_id
    return plot_data_service.get(layer_id).has_viewers


class TestPopoutDoesNotBlockDataFlow:
    def test_open_popout_leaves_the_active_grid_resolvable(
        self, plot_grid_tabs, line_cell
    ):
        """A pop-out must not masquerade as a modal.

        ``_get_active_grid_id`` returns None while a config modal is open,
        which suppresses the poll loop's data flush. A pop-out that routed
        through the modal container would therefore freeze every plot.
        """
        active_before = plot_grid_tabs._get_active_grid_id()

        plot_grid_tabs._show_popout(line_cell)

        assert plot_grid_tabs._get_active_grid_id() == active_before
        assert plot_grid_tabs._current_modal is None


class TestPopoutButton:
    def test_button_disabled_without_a_plot(self, plot_grid_tabs, plot_orchestrator):
        grid_id = plot_orchestrator.add_grid(title='G', nrows=1, ncols=1)
        cell_id = plot_orchestrator.add_cell(grid_id, _GEOMETRY)
        plot_orchestrator.add_layer(cell_id, _line_config())
        plot_grid_tabs._poll_for_plot_updates()

        assert _popout_button(plot_grid_tabs._cells[cell_id]).disabled

    def test_button_enabled_with_a_plot(self, plot_grid_tabs, line_cell):
        assert not _popout_button(plot_grid_tabs._cells[line_cell]).disabled

    def test_button_click_opens_the_window(self, plot_grid_tabs, line_cell):
        button = _popout_button(plot_grid_tabs._cells[line_cell])

        button.clicks += 1

        assert len(_open_windows(plot_grid_tabs)) == 1


def _popout_button(cell_widget):
    titlebar = cell_widget.view[0]
    (button,) = [
        obj
        for obj in titlebar.objects
        if 'lt-tool-arrows-maximize' in (getattr(obj, 'css_classes', None) or [])
    ]
    return button


class TestStaticOnlyCell:
    """A static-overlay-only cell has no autoscale axes but is still poppable."""

    def test_static_cell_pops_out_without_a_controller(
        self, plot_grid_tabs, plot_orchestrator
    ):
        grid_id = plot_orchestrator.add_grid(title='G', nrows=1, ncols=1)
        cell_id = plot_orchestrator.add_cell(grid_id, _GEOMETRY)
        plot_orchestrator.add_layer(cell_id, _static_config())
        plot_grid_tabs._poll_for_plot_updates()

        plot_grid_tabs._show_popout(cell_id)

        assert len(_open_windows(plot_grid_tabs)) == 1
        assert plot_grid_tabs._cells[cell_id].autoscale_controller is None
