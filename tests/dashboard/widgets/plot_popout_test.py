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
from ess.livedata.dashboard.frame_clock import FrameClock
from ess.livedata.dashboard.plot_orchestrator import (
    CellGeometry,
    DataSourceConfig,
    PlotConfig,
    PlotOrchestrator,
)
from ess.livedata.dashboard.plot_params import PlotParams1d
from ess.livedata.dashboard.plots import LinePlotter
from ess.livedata.dashboard.session_layer import SessionLayer
from ess.livedata.dashboard.static_plots import LinesCoordinates, VLinesParams
from ess.livedata.dashboard.theme import DEFAULT_THEME

hv.extension('bokeh')

_WORKFLOW = WorkflowId(instrument='test', name='wf', version=1)
_GEOMETRY = CellGeometry(row=0, col=0, row_span=1, col_span=1)
# Tooltip of the controller's one-shot Fit tool, its handle in a Bokeh toolbar.
_FIT = 'Fit ranges to current data'


@pytest.fixture
def frame_clock():
    """The orchestrator's frame clock, shared with the tests.

    A committed frame is the signal a session's tick predicate reacts to, so
    holding the clock lets a test raise it for a chosen grid without standing
    up the ingestion path that would normally do the committing.
    """
    return FrameClock()


@pytest.fixture
def plot_orchestrator(
    plotting_controller, job_orchestrator, data_service, plot_data_service, frame_clock
):
    """The shared fixture, with the frame clock exposed to the test."""
    return PlotOrchestrator(
        plotting_controller=plotting_controller,
        job_orchestrator=job_orchestrator,
        data_service=data_service,
        instrument='dummy',
        plot_data_service=plot_data_service,
        frame_clock=frame_clock,
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


def _make_line_plotter() -> LinePlotter:
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
def line_plotter():
    return _make_line_plotter()


def _add_line_cell(
    plot_orchestrator, plot_grid_tabs, plot_data_service, plotter, *, title
):
    """Add a 1x1 grid holding one built cell showing a live 1-D line plot."""
    grid_id = plot_orchestrator.add_grid(title=title, nrows=1, ncols=1)
    cell_id = plot_orchestrator.add_cell(grid_id, _GEOMETRY)
    layer_id = plot_orchestrator.add_layer(cell_id, _line_config())

    plot_data_service.job_started(layer_id, plotter)
    plot_data_service.data_arrived(layer_id)

    state = plot_data_service.get(layer_id)
    session_layer = SessionLayer(layer_id=layer_id)
    session_layer.ensure_components(state)
    plot_grid_tabs._session_layers[layer_id] = session_layer

    _show_grid(plot_grid_tabs, grid_id)
    return cell_id


def _show_grid(plot_grid_tabs, grid_id) -> None:
    """Make a grid's tab the visible one, building its cells.

    A session materializes cells only for the grid it displays, so a test
    wanting a built widget has to show its tab; the first pass is what creates
    that tab in the first place.
    """
    plot_grid_tabs._poll_for_plot_updates()
    tabbed = plot_grid_tabs._tabbed_grid_ids()
    plot_grid_tabs.tabs.active = plot_grid_tabs._static_tabs_count + tabbed.index(
        grid_id
    )
    plot_grid_tabs._poll_for_plot_updates()


@pytest.fixture
def line_cell(plot_orchestrator, plot_grid_tabs, plot_data_service, line_plotter):
    """A built cell showing a live 1-D line plot."""
    return _add_line_cell(
        plot_orchestrator, plot_grid_tabs, plot_data_service, line_plotter, title='G'
    )


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
        assert _rendered_plot(window) is cell_widget._plot

    def test_both_toolbars_get_the_same_autoscale_tools(
        self, plot_grid_tabs, line_cell
    ):
        """Tools install per figure, so neither toolbar goes bare.

        Both toolbars get the *same* tool models: that identity is what makes
        a toggle flipped in the window show as flipped in the cell.
        """
        plot = plot_grid_tabs._cells[line_cell]._plot

        cell_tools = _autoscale_tools(_render(plot))
        popout_tools = _autoscale_tools(_render(plot))

        assert _FIT in cell_tools
        assert cell_tools.keys() > {_FIT}  # at least one axis toggle too
        assert cell_tools == popout_tools

    def test_toggling_autoscale_in_one_view_shows_in_the_other(
        self, plot_grid_tabs, line_cell
    ):
        plot = plot_grid_tabs._cells[line_cell]._plot
        cell_tools = _autoscale_tools(_render(plot))
        popout_tools = _autoscale_tools(_render(plot))
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
        plot = plot_grid_tabs._cells[line_cell]._plot
        figures = [_render(plot).state for _ in range(2)]
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
        assert pipe in cell_widget._plot.streams
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

    def test_the_window_wears_the_shell_header_color(self, plot_grid_tabs, line_cell):
        """A window floating over the dashboard is part of the shell, and the
        shell's chrome color is the theme's to pick."""
        plot_grid_tabs._show_popout(line_cell)

        (window,) = _open_windows(plot_grid_tabs)
        assert window.theme == DEFAULT_THEME.floating_header_background

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
        """A rebuilt cell composes fresh plots; the window must show those.

        The window itself has to be the same one: its size and position live
        in jsPanel and never round-trip to Python, so a replacement can only
        come back at the default size in the next cascade slot -- a rename
        would fling the window across the screen and shrink it.
        """
        plot_grid_tabs._show_popout(line_cell)
        window_before = _open_windows(plot_grid_tabs)[0]

        plot_orchestrator.set_cell_title(line_cell, 'Renamed')
        plot_grid_tabs._poll_for_plot_updates()

        (window,) = _open_windows(plot_grid_tabs)
        assert window is window_before
        assert window.name == 'Renamed'
        assert _rendered_plot(window) is plot_grid_tabs._cells[line_cell]._plot

    def test_cell_removal_closes_the_window(
        self, plot_grid_tabs, plot_orchestrator, line_cell
    ):
        plot_grid_tabs._show_popout(line_cell)
        layer_id = plot_orchestrator.get_cell(line_cell).layers[0].layer_id

        plot_orchestrator.remove_layer(layer_id)
        plot_grid_tabs._poll_for_plot_updates()

        assert _open_windows(plot_grid_tabs) == []

    def test_disabling_the_grid_closes_the_window(
        self, plot_grid_tabs, plot_orchestrator, line_cell
    ):
        """A disabled grid's layers stop computing (they lose their viewers),
        so its pop-out would float on frozen; it is closed instead. The cell
        itself survives for a re-enable."""
        plot_grid_tabs._show_popout(line_cell)

        plot_orchestrator.set_grid_enabled(
            plot_grid_tabs._cells[line_cell].grid_id, enabled=False
        )
        _tick(plot_grid_tabs)

        assert _open_windows(plot_grid_tabs) == []
        assert line_cell in plot_grid_tabs._cells

    def test_reopening_after_a_close_does_not_cover_an_open_window(
        self, plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
    ):
        """Cascade slots are never reused: counting open windows instead
        would land the third window exactly on the second after the first
        closed, hiding the covered window's close button."""
        second = _add_line_cell(
            plot_orchestrator,
            plot_grid_tabs,
            plot_data_service,
            _make_line_plotter(),
            title='H',
        )
        third = _add_line_cell(
            plot_orchestrator,
            plot_grid_tabs,
            plot_data_service,
            _make_line_plotter(),
            title='I',
        )
        plot_grid_tabs._show_popout(line_cell)
        plot_grid_tabs._show_popout(second)
        _open_windows(plot_grid_tabs)[0].status = 'closed'
        plot_grid_tabs._show_popout(third)

        offsets = {(w.offsetx, w.offsety) for w in _open_windows(plot_grid_tabs)}
        assert len(offsets) == 2

    def test_session_teardown_closes_all_windows(self, plot_grid_tabs, line_cell):
        plot_grid_tabs._show_popout(line_cell)

        plot_grid_tabs.dispose_widgets()

        assert _open_windows(plot_grid_tabs) == []

    def test_placeholder_cell_pops_out_nothing(self, plot_grid_tabs, plot_orchestrator):
        """A cell waiting for data has no plot to show in a window."""
        grid_id = plot_orchestrator.add_grid(title='G', nrows=1, ncols=1)
        cell_id = plot_orchestrator.add_cell(grid_id, _GEOMETRY)
        plot_orchestrator.add_layer(cell_id, _line_config())
        _show_grid(plot_grid_tabs, grid_id)
        assert not plot_grid_tabs._cells[cell_id].has_plot

        plot_grid_tabs._show_popout(cell_id)

        assert _open_windows(plot_grid_tabs) == []

    def test_a_rebuild_into_a_placeholder_closes_the_window(
        self, plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
    ):
        """A restarted job drops the cell back to a placeholder until its
        first frame arrives, and a placeholder has nothing to show."""
        plot_grid_tabs._show_popout(line_cell)
        layer_id = plot_orchestrator.get_cell(line_cell).layers[0].layer_id

        plot_data_service.job_started(layer_id, LinePlotter.from_params(PlotParams1d()))
        _tick(plot_grid_tabs)

        assert not plot_grid_tabs._cells[line_cell].has_plot
        assert _open_windows(plot_grid_tabs) == []


class TestPopoutKeepsItsCellLive:
    """A window floating above another tab must not show frozen data."""

    def test_hidden_grid_sleeps_without_a_popout(
        self, plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
    ):
        plot_grid_tabs.tabs.active = 0  # Workflows: no grid is visible

        _tick(plot_grid_tabs)

        assert _open_windows(plot_grid_tabs) == []
        assert not _layer_is_active(
            plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
        )

    def test_popped_out_cell_stays_active_on_another_tab(
        self, plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
    ):
        plot_grid_tabs._show_popout(line_cell)
        plot_grid_tabs.tabs.active = 0

        _tick(plot_grid_tabs)

        assert plot_grid_tabs._popouts.status_of(line_cell) == 'normalized'
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
        self, plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
    ):
        """Sleeping is not closing: the user's window must still be there.

        And still be *sleeping*, or a job restart would pop every parked
        window open and wake every cell behind them, letting rebuilds defeat
        the minimize-to-sleep handle.
        """
        plot_grid_tabs._show_popout(line_cell)
        plot_grid_tabs.tabs.active = 0
        _open_windows(plot_grid_tabs)[0].status = 'minimized'
        _tick(plot_grid_tabs)

        plot_orchestrator.set_cell_title(line_cell, 'Renamed')
        _tick(plot_grid_tabs)

        (window,) = _open_windows(plot_grid_tabs)
        assert window.status == 'minimized'
        assert not _layer_is_active(
            plot_grid_tabs, plot_orchestrator, plot_data_service, line_cell
        )


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
    frame_clock.commit(plot_grid_tabs._cells[cell_id].grid_id)


def _layer_is_active(plot_grid_tabs, plot_orchestrator, plot_data_service, cell_id):
    """Whether a viewer still holds interest in the cell's layer.

    The interest token is what keeps a layer computing; a hidden grid's layers
    lose it, which is exactly what would freeze a pop-out.
    """
    layer_id = plot_orchestrator.get_cell(cell_id).layers[0].layer_id
    return plot_data_service.has_viewers(layer_id)


class TestPerGridFrameFlush:
    """The flush gate is per grid, not one flag for the session.

    A presenter holds pending data as soon as the ingestion thread builds its
    layer, *before* the grid's frame commits. With a pop-out from another grid
    live, a session renders two grids; a shared gate would let a frame for
    either push the other's half-built burst, staggering layers of one burst
    across repaints -- the very thing the frame clock exists to prevent.
    """

    @pytest.fixture
    def popped_out_cell(self, plot_orchestrator, plot_grid_tabs, plot_data_service):
        """A second grid's cell, popped out into a floating window."""
        cell_id = _add_line_cell(
            plot_orchestrator,
            plot_grid_tabs,
            plot_data_service,
            _make_line_plotter(),
            title='H',
        )
        plot_grid_tabs._show_popout(cell_id)
        return cell_id

    def test_frame_for_the_popout_grid_leaves_the_visible_burst_pending(
        self,
        plot_grid_tabs,
        plot_orchestrator,
        plot_data_service,
        frame_clock,
        line_cell,
        popped_out_cell,
    ):
        _show_grid_tab(plot_grid_tabs, line_cell)
        _tick(plot_grid_tabs)

        _build_layer(plot_orchestrator, plot_data_service, line_cell)
        _new_frame(frame_clock, plot_grid_tabs, popped_out_cell)
        _tick(plot_grid_tabs)

        assert _has_pending(plot_grid_tabs, plot_orchestrator, line_cell)

    def test_frame_for_the_own_grid_flushes_the_visible_burst(
        self,
        plot_grid_tabs,
        plot_orchestrator,
        plot_data_service,
        frame_clock,
        line_cell,
        popped_out_cell,
    ):
        _show_grid_tab(plot_grid_tabs, line_cell)
        _tick(plot_grid_tabs)

        _build_layer(plot_orchestrator, plot_data_service, line_cell)
        _new_frame(frame_clock, plot_grid_tabs, line_cell)
        _tick(plot_grid_tabs)

        assert not _has_pending(plot_grid_tabs, plot_orchestrator, line_cell)

    def test_frame_for_the_popout_grid_flushes_the_window(
        self,
        plot_grid_tabs,
        plot_orchestrator,
        plot_data_service,
        frame_clock,
        line_cell,
        popped_out_cell,
    ):
        _show_grid_tab(plot_grid_tabs, line_cell)
        _tick(plot_grid_tabs)

        _build_layer(plot_orchestrator, plot_data_service, popped_out_cell)
        _new_frame(frame_clock, plot_grid_tabs, popped_out_cell)
        _tick(plot_grid_tabs)

        assert not _has_pending(plot_grid_tabs, plot_orchestrator, popped_out_cell)


def _show_grid_tab(plot_grid_tabs, cell_id) -> None:
    """Make the tab of the grid holding a cell the visible one."""
    _show_grid(plot_grid_tabs, plot_grid_tabs._cells[cell_id].grid_id)


def _build_layer(plot_orchestrator, plot_data_service, cell_id) -> None:
    """Recompute the cell's plotter, as the ingestion thread's layer build does.

    Its presenters now hold a pending update, ahead of the grid's frame
    commit -- the mid-burst state a flush for another grid must not push.
    """
    layer_id = plot_orchestrator.get_cell(cell_id).layers[0].layer_id
    plotter = plot_data_service.get(layer_id).plotter
    key = DataKey(workflow_id=_WORKFLOW, source_name='s1', output_name='out')
    plotter.compute({PRIMARY: {key: _curve([4.0, 5.0, 6.0])}})


def _has_pending(plot_grid_tabs, plot_orchestrator, cell_id) -> bool:
    """Whether the cell's layer still holds an unflushed update."""
    layer_id = plot_orchestrator.get_cell(cell_id).layers[0].layer_id
    session_layer = plot_grid_tabs._session_layers[layer_id]
    return session_layer.components.presenter.has_pending_update()


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
        _show_grid(plot_grid_tabs, grid_id)

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
        _show_grid(plot_grid_tabs, grid_id)

        plot_grid_tabs._show_popout(cell_id)

        assert len(_open_windows(plot_grid_tabs)) == 1
        assert plot_grid_tabs._cells[cell_id].autoscale_controller is None


class TestClosingAWindowKeepsTheCellRendering:
    """Tearing down one view of a cell tears down every view of it.

    ``Plot.cleanup`` drops *all* weakly-held plot-refresh subscribers on the
    streams it touches, not just its own (holoviews#6988), so removing a
    pop-out window unsubscribes the grid cell's plot along with the window's.
    Rebuilding the cell is what puts a live plot back; without it the cell
    would sit frozen for the rest of the session, showing data that looks
    current.
    """

    def test_cell_plot_stays_subscribed_after_its_window_closes(
        self, plot_grid_tabs, line_cell
    ):
        pipe = _pipe(plot_grid_tabs)
        plot_grid_tabs._cells[line_cell].view.get_root()
        subscribed = len(pipe.subscribers)

        plot_grid_tabs._show_popout(line_cell)
        _open_windows(plot_grid_tabs)[0].get_root()
        _open_windows(plot_grid_tabs)[0].status = 'closed'
        plot_grid_tabs._cells[line_cell].view.get_root()

        assert len(pipe.subscribers) == subscribed

    def test_closing_a_window_rebuilds_the_cell(self, plot_grid_tabs, line_cell):
        """The visible half of the sever repair, stated directly."""
        plot_grid_tabs._show_popout(line_cell)
        before = plot_grid_tabs._cells[line_cell]

        _open_windows(plot_grid_tabs)[0].status = 'closed'

        assert plot_grid_tabs._cells[line_cell] is not before

    def test_a_windowless_cell_is_left_alone(self, plot_grid_tabs, line_cell):
        """No window, no sever, no rebuild -- an idle pass costs nothing."""
        before = plot_grid_tabs._cells[line_cell]

        plot_grid_tabs._close_popout(line_cell)

        assert plot_grid_tabs._cells[line_cell] is before


def _pipe(plot_grid_tabs):
    """The one session layer's data pipe, which every view subscribes to."""
    return next(iter(plot_grid_tabs._session_layers.values())).components.pipe
