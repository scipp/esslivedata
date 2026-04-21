# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Regression tests for issue #805: polling must handle Layout-producing plotters
even when their DynamicMap has already been evaluated by Bokeh.
"""

from __future__ import annotations

from uuid import uuid4

import holoviews as hv
import pydantic
import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.notification_queue import NotificationQueue
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
from ess.livedata.dashboard.plots import PresenterBase
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry
from ess.livedata.dashboard.session_updater import SessionUpdater
from ess.livedata.dashboard.stream_manager import StreamManager
from ess.livedata.dashboard.widgets.plot_grid_tabs import PlotGridTabs
from ess.livedata.dashboard.widgets.workflow_status_widget import (
    WorkflowStatusListWidget,
)

hv.extension('bokeh')


# -- Fakes -----------------------------------------------------------------


class FakePlotter:
    """Minimal plotter whose cached state can be set to any HoloViews object."""

    def __init__(self, cached_state=None):
        self._cached_state = cached_state
        self._presenters: list[FakePresenter] = []
        self.canvas_spec = None

    def get_cached_state(self):
        return self._cached_state

    def has_cached_state(self):
        return self._cached_state is not None

    def create_presenter(self, *, owner=None):
        presenter = FakePresenter(self, owner=owner)
        self._presenters.append(presenter)
        return presenter

    def mark_presenters_dirty(self):
        for p in self._presenters:
            p._mark_dirty()


class FakePresenter(PresenterBase):
    def present(self, pipe):
        return hv.DynamicMap(lambda data: data, streams=[pipe], cache_size=1)


class _Params(pydantic.BaseModel):
    pass


# -- Fixtures --------------------------------------------------------------


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


# -- Helpers ---------------------------------------------------------------


def _make_layout() -> hv.Layout:
    return hv.Layout(
        [hv.Curve([1, 2, 3]).relabel('A'), hv.Curve([4, 5, 6]).relabel('B')]
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
                workflow_id=WorkflowId(
                    instrument='test', namespace='test', name='wf', version=1
                ),
                source_names=['src'],
                output_name='result',
            )
        },
        plot_name='image',
        params=_Params(),
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
        plot_grid_tabs.tabs.active = plot_grid_tabs._static_tabs_count
        layer_id = _inject_layer(plot_orchestrator, plot_data_service, grid_id, plotter)

        plot_grid_tabs._poll_for_plot_updates()

        session_layer = plot_grid_tabs._session_layers.get(layer_id)
        assert session_layer is not None
        assert session_layer.dmap is not None

    def test_failed_rebuild_does_not_bump_version(
        self, plot_orchestrator, plot_data_service, plot_grid_tabs
    ):
        """
        If a rebuild raises, the session layer's version must stay stale
        so the next poll cycle retries.
        """
        plotter = FakePlotter(cached_state=_make_layout())
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=2, ncols=2)
        plot_grid_tabs.tabs.active = plot_grid_tabs._static_tabs_count
        layer_id = _inject_layer(plot_orchestrator, plot_data_service, grid_id, plotter)

        plot_grid_tabs._poll_for_plot_updates()
        version_after_first_poll = plot_grid_tabs._session_layers[
            layer_id
        ].last_seen_version

        # Bump version
        plot_data_service.job_started(
            layer_id, FakePlotter(cached_state=_make_layout())
        )
        plot_data_service.data_arrived(layer_id)
        new_version = plot_data_service.get(layer_id).version
        assert new_version != version_after_first_poll

        # Inject a transient failure into the rebuild path
        original = plot_grid_tabs._get_session_composed_plot

        def _raise(cell):
            raise RuntimeError("injected failure")

        plot_grid_tabs._get_session_composed_plot = _raise
        try:
            try:
                plot_grid_tabs._poll_for_plot_updates()
            except RuntimeError:
                pass

            session_layer = plot_grid_tabs._session_layers[layer_id]
            assert session_layer.last_seen_version != new_version
        finally:
            plot_grid_tabs._get_session_composed_plot = original
