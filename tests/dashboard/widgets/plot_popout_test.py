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
import panel as pn
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.dashboard.data_roles import PRIMARY
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.frame_clock import FrameClock
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
# Tooltip of the controller's one-shot Fit tool, its handle in a Bokeh toolbar.
_FIT = 'Fit ranges to current data'


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
def frame_clock():
    """The orchestrator's frame clock, shared with the tests.

    A committed frame is the signal a session's tick predicate reacts to, so
    holding the clock lets a test raise it for a chosen grid without standing
    up the ingestion path that would normally do the committing.
    """
    return FrameClock()


@pytest.fixture
def plot_orchestrator(job_orchestrator, data_service, plot_data_service, frame_clock):
    stream_manager = StreamManager(data_service=data_service)
    return PlotOrchestrator(
        plotting_controller=PlottingController(stream_manager=stream_manager),
        job_orchestrator=job_orchestrator,
        data_service=data_service,
        instrument='dummy',
        plot_data_service=plot_data_service,
        frame_clock=frame_clock,
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
def line_plotter():
    """A real ``LinePlotter`` holding one computed curve.

    Real rather than a stub: the properties under test — a shared autoscale
    controller, a shared data pipe — only exist for a plot that actually
    autoscales and actually updates.
    """
    plotter = LinePlotter.from_params(PlotParams1d())
    key = DataKey(workflow_id=_WORKFLOW, source_name='s1', output_name='out')
    plotter.compute({PRIMARY: {key: _curve([1.0, 2.0, 3.0])}})
    return plotter


@pytest.fixture
def line_cell(plot_orchestrator, plot_grid_tabs, plot_data_service, line_plotter):
    """A built cell showing a live 1-D line plot."""
    grid_id = plot_orchestrator.add_grid(title='G', nrows=1, ncols=1)
    cell_id = plot_orchestrator.add_cell(grid_id, _GEOMETRY)
    layer_id = plot_orchestrator.add_layer(cell_id, _line_config())

    plotter = line_plotter
    plot_data_service.job_started(layer_id, plotter)
    plot_data_service.data_arrived(layer_id)

    state = plot_data_service.get(layer_id)
    session_layer = SessionLayer(layer_id=layer_id, last_seen_version=state.version)
    session_layer.ensure_components(state)
    plot_grid_tabs._session_layers[layer_id] = session_layer

    plot_grid_tabs._poll_for_plot_updates()
    return cell_id


def _open_windows(plot_grid_tabs) -> list:
    """The floating windows in the manager's container.

    The container also holds the invisible fitter that installs the pop-out's
    document-level handlers; only the windows are of interest here.
    """
    return [
        obj
        for obj in plot_grid_tabs._popouts.container.objects
        if isinstance(obj, pn.layout.FloatPanel)
    ]


def _tick(plot_grid_tabs) -> None:
    """Drive one gated poll cycle, as SessionUpdater's wake ticks do.

    Session ticks run the poll pass only when ``_has_pending_work`` fires, so
    driving the pass directly would let a pop-out that the predicate cannot
    see pass these tests while freezing in the browser.
    """
    if plot_grid_tabs._has_pending_work():
        plot_grid_tabs._poll_for_plot_updates()


class TestPopoutRendersTheCellsPlotAgain:
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

    def test_both_views_show_the_same_composed_plot(self, plot_grid_tabs, line_cell):
        """One composition, rendered twice — not two compositions.

        Sharing is what links the two views: one ``Pipe`` feeds both figures,
        and one autoscale controller drives both toolbars.
        """
        cell_widget = plot_grid_tabs._cells[line_cell]

        plot_grid_tabs._show_popout(line_cell)

        window = _open_windows(plot_grid_tabs)[0]
        assert _rendered_plot(window) is cell_widget.composed.plot

    def test_both_toolbars_get_the_same_autoscale_tools(
        self, plot_grid_tabs, line_cell
    ):
        """Tools install per figure, so neither toolbar goes bare.

        Both toolbars get the *same* tool models: that identity is what makes
        a toggle flipped in the window show as flipped in the cell.
        """
        composed = plot_grid_tabs._cells[line_cell].composed

        cell_tools = _autoscale_tools(_render(composed.plot))
        popout_tools = _autoscale_tools(_render(composed.plot))

        assert _FIT in cell_tools
        assert cell_tools.keys() > {_FIT}  # at least one axis toggle too
        assert cell_tools == popout_tools

    def test_toggling_autoscale_in_one_view_shows_in_the_other(
        self, plot_grid_tabs, line_cell
    ):
        composed = plot_grid_tabs._cells[line_cell].composed
        cell_tools = _autoscale_tools(_render(composed.plot))
        popout_tools = _autoscale_tools(_render(composed.plot))
        toggle = next(name for name in cell_tools if name != _FIT)
        assert popout_tools[toggle].active

        cell_tools[toggle].active = False

        assert not popout_tools[toggle].active

    def test_fit_reaches_every_rendered_figure(
        self, plot_grid_tabs, line_cell, line_plotter
    ):
        """A Fit click must fit both views, not whichever renders first.

        Both figures repaint from one pipe push, so tracking the pending fit
        as a single flag would let the first figure consume it and leave the
        second showing a stale range.
        """
        composed = plot_grid_tabs._cells[line_cell].composed
        figures = [_render(composed.plot).state for _ in range(2)]
        tools = _autoscale_tools_of(figures[0])
        for name, tool in tools.items():
            if name != _FIT:
                tool.active = False  # ranges now only move on an explicit Fit
        for figure in figures:
            figure.x_range.start, figure.x_range.end = -999.0, 999.0

        tools[_FIT].active = True  # the user clicks Fit
        _repaint(plot_grid_tabs, line_plotter)

        for figure in figures:
            assert (figure.x_range.start, figure.x_range.end) != (-999.0, 999.0)

    def test_both_views_repaint_from_the_same_pipe(self, plot_grid_tabs, line_cell):
        """The pop-out is live, not a snapshot: one pipe drives both figures."""
        cell_widget = plot_grid_tabs._cells[line_cell]
        session_layer = next(iter(plot_grid_tabs._session_layers.values()))

        plot_grid_tabs._show_popout(line_cell)

        pipe = session_layer.components.pipe
        assert pipe in cell_widget.composed.plot.streams
        assert pipe in _rendered_plot(_open_windows(plot_grid_tabs)[0]).streams


def _render(plot):
    """Render a composed plot into a fresh Bokeh figure, as a view would."""
    return hv.renderer('bokeh').get_plot(plot)


def _autoscale_tools(rendered) -> dict[str, object]:
    """The controller's toolbar tools on a rendered plot, keyed by tooltip."""
    return _autoscale_tools_of(rendered.state)


def _autoscale_tools_of(figure) -> dict[str, object]:
    from bokeh.models import CustomAction

    return {
        tool.description: tool
        for tool in figure.toolbar.tools
        if isinstance(tool, CustomAction)
    }


def _repaint(plot_grid_tabs, plotter) -> None:
    """Push a frame through the layer's pipe, repainting every rendered view."""
    session_layer = next(iter(plot_grid_tabs._session_layers.values()))
    session_layer.components.pipe.send(plotter.get_cached_state())


def _rendered_plot(window):
    """The HoloViews object a pop-out window renders."""
    column = window[0]
    return column[0][0].object


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

        _tick(plot_grid_tabs)

        assert plot_grid_tabs._popouts.open_cells() == frozenset()
        assert not _layer_is_active(
            plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
        )

    def test_popped_out_cell_stays_active_on_another_tab(
        self, plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
    ):
        plot_grid_tabs._show_popout(line_cell)
        plot_grid_tabs.tabs.active = 0

        _tick(plot_grid_tabs)

        assert plot_grid_tabs._popouts.open_cells() == frozenset({line_cell})
        assert _layer_is_active(
            plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
        )

    def test_closing_the_window_lets_the_cell_sleep_again(
        self, plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
    ):
        plot_grid_tabs._show_popout(line_cell)
        plot_grid_tabs.tabs.active = 0
        _tick(plot_grid_tabs)

        # jsPanel round-trips the user's click on the window's close button.
        _open_windows(plot_grid_tabs)[0].status = 'closed'

        assert not _layer_is_active(
            plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
        )

    @pytest.mark.parametrize('status', ['minimized', 'smallified', 'smallifiedmax'])
    def test_window_showing_no_plot_lets_the_cell_sleep(
        self, plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell, status
    ):
        """Liveness follows what a window renders, not that it exists.

        Otherwise popping out many cells and minimizing the windows would pin
        every one of those cells live, on every grid, for nothing on screen.
        """
        plot_grid_tabs._show_popout(line_cell)
        plot_grid_tabs.tabs.active = 0
        _tick(plot_grid_tabs)
        assert _layer_is_active(
            plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
        )

        _open_windows(plot_grid_tabs)[0].status = status

        assert not _layer_is_active(
            plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
        )

    def test_restoring_the_window_wakes_its_cell(
        self, plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
    ):
        """Restoring must reach the cell without waiting for a full pass.

        Nothing shared moves when a window is restored, so no ``has_work``
        predicate can see it; the manager asks for the pass itself. Hence no
        tick here -- an assertion that survives only if it did.
        """
        plot_grid_tabs._show_popout(line_cell)
        plot_grid_tabs.tabs.active = 0
        window = _open_windows(plot_grid_tabs)[0]
        window.status = 'minimized'

        window.status = 'normalized'

        assert _layer_is_active(
            plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
        )

    def test_minimized_window_still_survives_a_cell_rebuild(
        self, plot_grid_tabs, plot_orchestrator, line_cell
    ):
        """Sleeping is not closing: the user's window must still be there."""
        plot_grid_tabs._show_popout(line_cell)
        _open_windows(plot_grid_tabs)[0].status = 'minimized'

        plot_orchestrator.set_cell_title(line_cell, 'Renamed')
        _tick(plot_grid_tabs)

        assert len(_open_windows(plot_grid_tabs)) == 1


class TestPopoutWakesTheSessionItFloatsOver:
    """Ticks are predicate-gated, so a pop-out must be visible to the gate.

    A session parked on another tab is woken by every data burst, but only
    runs the poll pass when ``_has_pending_work`` fires. If that predicate
    stayed scoped to the visible grid, the pass feeding the pop-out would
    never run and the window would silently freeze.
    """

    def test_frame_for_a_popped_out_grid_wakes_a_session_on_another_tab(
        self, plot_grid_tabs, frame_clock, line_cell
    ):
        plot_grid_tabs._show_popout(line_cell)
        plot_grid_tabs.tabs.active = 0
        _tick(plot_grid_tabs)
        assert not plot_grid_tabs._has_pending_work()

        _new_frame(frame_clock, plot_grid_tabs, line_cell)

        assert plot_grid_tabs._has_pending_work()

    def test_frame_for_a_hidden_grid_alone_does_not_wake_the_session(
        self, plot_grid_tabs, frame_clock, line_cell
    ):
        """The widening must not become "wake on any grid".

        Scoping the gate per grid is what keeps another session's tab from
        costing this one a hold+freeze pass; a pop-out buys in one more grid,
        not all of them.
        """
        plot_grid_tabs.tabs.active = 0
        _tick(plot_grid_tabs)
        assert not plot_grid_tabs._has_pending_work()

        _new_frame(frame_clock, plot_grid_tabs, line_cell)

        assert not plot_grid_tabs._has_pending_work()

    def test_minimized_window_does_not_wake_the_session(
        self, plot_grid_tabs, frame_clock, line_cell
    ):
        plot_grid_tabs._show_popout(line_cell)
        plot_grid_tabs.tabs.active = 0
        _open_windows(plot_grid_tabs)[0].status = 'minimized'
        _tick(plot_grid_tabs)
        assert not plot_grid_tabs._has_pending_work()

        _new_frame(frame_clock, plot_grid_tabs, line_cell)

        assert not plot_grid_tabs._has_pending_work()


def _new_frame(frame_clock, plot_grid_tabs, cell_id) -> None:
    """Signal a completed data-burst frame for the grid holding a cell.

    What the ingestion thread does at the end of a burst, and what a live
    pop-out on a hidden tab must still react to.
    """
    frame_clock.commit(plot_grid_tabs._cell_grid[cell_id])


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
