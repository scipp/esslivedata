# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import holoviews as hv
import panel as pn
import pytest

from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.notification_queue import NotificationQueue
from ess.livedata.dashboard.plot_data_service import PlotDataService
from ess.livedata.dashboard.plot_orchestrator import PlotOrchestrator
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry
from ess.livedata.dashboard.session_updater import SessionUpdater
from ess.livedata.dashboard.stream_manager import StreamManager
from ess.livedata.dashboard.widgets.plot_grid_tabs import PlotGridTabs
from ess.livedata.dashboard.widgets.workflow_status_widget import (
    WorkflowStatusListWidget,
)

hv.extension('bokeh')


@pytest.fixture
def data_service():
    """Create a DataService for testing."""
    return DataService()


@pytest.fixture
def job_service():
    """Create a JobService for testing."""
    return JobService()


@pytest.fixture
def stream_manager(data_service):
    """Create a StreamManager for testing."""
    return StreamManager(data_service=data_service)


@pytest.fixture
def plotting_controller(stream_manager):
    """Create a PlottingController for testing."""
    return PlottingController(
        stream_manager=stream_manager,
    )


@pytest.fixture
def plot_orchestrator(
    plotting_controller, job_orchestrator, data_service, plot_data_service
):
    """Create a PlotOrchestrator for testing."""
    return PlotOrchestrator(
        plotting_controller=plotting_controller,
        job_orchestrator=job_orchestrator,
        data_service=data_service,
        instrument='dummy',
        plot_data_service=plot_data_service,
    )


@pytest.fixture
def workflow_status_widget(job_orchestrator, job_service):
    """Create a WorkflowStatusListWidget for testing."""
    return WorkflowStatusListWidget(
        orchestrator=job_orchestrator,
        job_service=job_service,
    )


@pytest.fixture
def plot_data_service():
    """Create a PlotDataService for testing."""
    return PlotDataService()


@pytest.fixture
def session_updater():
    """Create a SessionUpdater for testing."""
    registry = SessionRegistry()
    return SessionUpdater(
        session_id=SessionId('test-session'),
        session_registry=registry,
        notification_queue=NotificationQueue(),
    )


@pytest.fixture
def plot_grid_tabs(
    plot_orchestrator,
    workflow_registry,
    plotting_controller,
    workflow_status_widget,
    plot_data_service,
    session_updater,
):
    """Create a PlotGridTabs widget for testing."""
    return PlotGridTabs(
        plot_orchestrator=plot_orchestrator,
        workflow_registry=workflow_registry,
        plotting_controller=plotting_controller,
        workflow_status_widget=workflow_status_widget,
        plot_data_service=plot_data_service,
        session_updater=session_updater,
    )


class TestPlotGridTabsInitialization:
    """Tests for PlotGridTabs initialization."""

    def test_creates_panel_tabs_widget(self, plot_grid_tabs):
        """Test that widget creates a Panel Tabs object."""
        assert isinstance(plot_grid_tabs.tabs, pn.Tabs)

    def test_starts_with_two_tabs_when_no_grids(self, plot_grid_tabs):
        """Test that widget starts with two static tabs when no grids exist."""
        # Should have exactly two tabs (Workflows and Manage Plots)
        assert len(plot_grid_tabs.tabs) == 2

    def test_initializes_from_existing_grids(
        self,
        plot_orchestrator,
        workflow_registry,
        plotting_controller,
        workflow_status_widget,
        plot_data_service,
        session_updater,
    ):
        """Test that widget creates tabs for existing grids."""
        # Add grids before creating widget
        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)

        # Create widget
        widget = PlotGridTabs(
            plot_orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            workflow_status_widget=workflow_status_widget,
            plot_data_service=plot_data_service,
            session_updater=session_updater,
        )

        # Should have 4 tabs: Workflows + Manage + 2 grids
        assert len(widget.tabs) == 4

    def test_subscribes_to_lifecycle_events(self, plot_orchestrator, plot_grid_tabs):
        """Test that widget subscribes to orchestrator lifecycle events."""
        # Verify subscription by adding a grid and checking it appears
        plot_orchestrator.add_grid(title='New Grid', nrows=3, ncols=3)

        # Should now have 3 tabs: Workflows + Manage + New Grid
        assert len(plot_grid_tabs.tabs) == 3


class TestGridTabManagement:
    """Tests for adding and removing grid tabs."""

    def test_on_grid_created_adds_tab(self, plot_orchestrator, plot_grid_tabs):
        """Test that creating a grid adds a new tab."""
        initial_count = len(plot_grid_tabs.tabs)

        plot_orchestrator.add_grid(title='Test Grid', nrows=4, ncols=4)

        # Should have one more tab
        assert len(plot_grid_tabs.tabs) == initial_count + 1

    def test_on_grid_created_switches_to_new_tab(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """Test that creating a grid auto-switches to that tab."""
        plot_orchestrator.add_grid(title='Auto Switch', nrows=3, ncols=3)

        # Active tab should be newly created grid
        # (Workflows=0, Manage=1, grid=2)
        assert plot_grid_tabs.tabs.active == 2

    def test_on_grid_removed_removes_tab(self, plot_orchestrator, plot_grid_tabs):
        """Test that removing a grid removes its tab."""
        grid_id = plot_orchestrator.add_grid(title='To Remove', nrows=3, ncols=3)
        assert len(plot_grid_tabs.tabs) == 3  # Workflows + Manage + Grid

        plot_orchestrator.remove_grid(grid_id)

        # Should only have two static tabs left (Workflows + Manage)
        assert len(plot_grid_tabs.tabs) == 2

    def test_removing_grid_updates_correctly(self, plot_orchestrator, plot_grid_tabs):
        """Test that removing a grid from the middle works correctly."""
        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        grid_id_2 = plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        plot_orchestrator.add_grid(title='Grid 3', nrows=4, ncols=4)

        # Remove middle grid
        plot_orchestrator.remove_grid(grid_id_2)

        # Should have Workflows + Manage + 2 remaining grids
        assert len(plot_grid_tabs.tabs) == 4

    def test_multiple_widget_instances_stay_synchronized(
        self,
        plot_orchestrator,
        workflow_registry,
        plotting_controller,
        job_service,
        job_orchestrator,
        plot_data_service,
    ):
        """Test that multiple widgets sharing same orchestrator stay in sync."""
        # Create separate session updaters for each widget (simulating different
        # sessions)
        registry = SessionRegistry()
        session_updater1 = SessionUpdater(
            session_id=SessionId('session-1'),
            session_registry=registry,
            notification_queue=NotificationQueue(),
        )
        session_updater2 = SessionUpdater(
            session_id=SessionId('session-2'),
            session_registry=registry,
            notification_queue=NotificationQueue(),
        )

        # Create separate workflow status widgets for each instance
        workflow_status_widget1 = WorkflowStatusListWidget(
            orchestrator=job_orchestrator,
            job_service=job_service,
        )
        workflow_status_widget2 = WorkflowStatusListWidget(
            orchestrator=job_orchestrator,
            job_service=job_service,
        )

        widget1 = PlotGridTabs(
            plot_orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            workflow_status_widget=workflow_status_widget1,
            plot_data_service=plot_data_service,
            session_updater=session_updater1,
        )
        widget2 = PlotGridTabs(
            plot_orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            workflow_status_widget=workflow_status_widget2,
            plot_data_service=plot_data_service,
            session_updater=session_updater2,
        )

        # Add grid via orchestrator
        grid_id = plot_orchestrator.add_grid(title='Shared Grid', nrows=3, ncols=3)

        # Both widgets should have the new tab
        assert len(widget1.tabs) == 3  # Workflows + Manage + Shared Grid
        assert len(widget2.tabs) == 3

        # Remove grid
        plot_orchestrator.remove_grid(grid_id)

        # Both widgets should reflect removal
        assert len(widget1.tabs) == 2  # Workflows + Manage
        assert len(widget2.tabs) == 2


class TestManageTab:
    """Tests for the Manage tab functionality."""

    def test_manage_tab_count_stable_when_grids_added(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """Test that adding grids doesn't remove or duplicate the Manage tab."""
        initial_count = len(plot_grid_tabs.tabs)
        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        # Should have exactly one more tab
        assert len(plot_grid_tabs.tabs) == initial_count + 1

        plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        # Should have exactly one more tab again
        assert len(plot_grid_tabs.tabs) == initial_count + 2


class TestShutdown:
    """Tests for widget shutdown and cleanup."""

    def test_sever_unsubscribes_from_lifecycle(self, plot_orchestrator, plot_grid_tabs):
        """Test that tier-2 sever unsubscribes from orchestrator lifecycle."""
        plot_grid_tabs.sever()

        # Adding a grid should not affect the widget anymore
        initial_count = len(plot_grid_tabs.tabs)
        plot_orchestrator.add_grid(title='After Shutdown', nrows=3, ncols=3)

        # Tab count should not change
        assert len(plot_grid_tabs.tabs) == initial_count

    def test_sever_is_idempotent(self, plot_grid_tabs):
        """Test that tier-2 sever can be called multiple times."""
        plot_grid_tabs.sever()
        plot_grid_tabs.sever()  # Should not raise

    def test_dispose_widgets_is_idempotent(self, plot_grid_tabs):
        """Test that tier-1 dispose_widgets can be called multiple times."""
        plot_grid_tabs.dispose_widgets()
        plot_grid_tabs.dispose_widgets()  # Should not raise


class _FakeDocument:
    """Captures next-tick callbacks scheduled by the reaper path."""

    def __init__(self):
        self.next_tick_callbacks = []

    def add_next_tick_callback(self, callback):
        self.next_tick_callbacks.append(callback)


class _FakeCallback:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class TestReaperTeardown:
    """#955: stale-session reaper severs shared state without touching the
    Bokeh document on the reaper thread."""

    def _make_widget(
        self,
        plot_orchestrator,
        workflow_registry,
        plotting_controller,
        workflow_status_widget,
        plot_data_service,
        registry,
        document,
    ):
        updater = SessionUpdater(
            session_id=SessionId('stale-session'),
            session_registry=registry,
            notification_queue=NotificationQueue(),
            document=document,
        )
        callback = _FakeCallback()
        updater.set_periodic_callback(callback)
        widget = PlotGridTabs(
            plot_orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            workflow_status_widget=workflow_status_widget,
            plot_data_service=plot_data_service,
            session_updater=updater,
        )
        return widget, callback

    def test_reaper_severs_lifecycle_without_touching_document(
        self,
        plot_orchestrator,
        workflow_registry,
        plotting_controller,
        workflow_status_widget,
        plot_data_service,
    ):
        registry = SessionRegistry(stale_timeout_seconds=-1.0)
        document = _FakeDocument()
        widget, callback = self._make_widget(
            plot_orchestrator,
            workflow_registry,
            plotting_controller,
            workflow_status_widget,
            plot_data_service,
            registry,
            document,
        )

        cleaned = registry.cleanup_stale_sessions()

        assert SessionId('stale-session') in cleaned
        # Tier 2 ran inline on the reaper thread: lifecycle severed, so a new
        # grid does not add a tab to this session's widget.
        initial_count = len(widget.tabs)
        plot_orchestrator.add_grid(title='After Reaper', nrows=2, ncols=2)
        assert len(widget.tabs) == initial_count
        # Tier 1 was NOT run on the reaper thread: the periodic callback is not
        # stopped here, it is scheduled onto the session's IOLoop instead.
        assert not callback.stopped
        assert len(document.next_tick_callbacks) == 1


class TestOverlayFiltering:
    """Tests for overlay suggestion filtering."""

    class MockConfig:
        """Simple test double for PlotConfig."""

        def __init__(self, plot_name):
            self.plot_name = plot_name

    class MockLayer:
        """Simple test double for a layer with config."""

        def __init__(self, plot_name):
            self.config = TestOverlayFiltering.MockConfig(plot_name)

    def test_existing_overlays_filtered_from_suggestions(self):
        """Test that overlays already in the cell are not suggested again."""
        # Simulate the filtering logic used in _create_layer_toolbars
        # This tests the filtering independently of the full widget setup

        # Mock cell layers: image + rectangles_readback already added
        cell_layers = [
            self.MockLayer('image'),
            self.MockLayer('rectangles_readback'),
        ]

        # Collect existing plotter names (same as in _create_layer_toolbars)
        existing_plotter_names = {layer.config.plot_name for layer in cell_layers}

        # Simulate available overlays for an image layer
        # (normally returned by _get_available_overlays_for_layer)
        available_overlays_for_image = [
            ('roi_rectangle', 'rectangles_readback', 'ROI Rectangles (Readback)'),
            ('roi_polygon', 'polygons_readback', 'ROI Polygons (Readback)'),
        ]

        # Apply filtering (same logic as in _create_layer_toolbars)
        filtered_overlays = [
            overlay
            for overlay in available_overlays_for_image
            if overlay[1] not in existing_plotter_names
        ]

        # rectangles_readback should be filtered out (already exists)
        # polygons_readback should remain
        assert len(filtered_overlays) == 1
        assert filtered_overlays[0][1] == 'polygons_readback'

    def test_no_overlays_filtered_when_none_exist(self):
        """Test that all overlays are available when none have been added."""
        # Only image layer exists
        cell_layers = [self.MockLayer('image')]
        existing_plotter_names = {layer.config.plot_name for layer in cell_layers}

        available_overlays_for_image = [
            ('roi_rectangle', 'rectangles_readback', 'ROI Rectangles (Readback)'),
            ('roi_polygon', 'polygons_readback', 'ROI Polygons (Readback)'),
        ]

        filtered_overlays = [
            overlay
            for overlay in available_overlays_for_image
            if overlay[1] not in existing_plotter_names
        ]

        # Both should be available
        assert len(filtered_overlays) == 2

    def test_all_overlays_filtered_when_all_exist(self):
        """Test that no overlays suggested when all have been added."""
        # All layers exist
        cell_layers = [
            self.MockLayer('image'),
            self.MockLayer('rectangles_readback'),
            self.MockLayer('polygons_readback'),
        ]
        existing_plotter_names = {layer.config.plot_name for layer in cell_layers}

        available_overlays_for_image = [
            ('roi_rectangle', 'rectangles_readback', 'ROI Rectangles (Readback)'),
            ('roi_polygon', 'polygons_readback', 'ROI Polygons (Readback)'),
        ]

        filtered_overlays = [
            overlay
            for overlay in available_overlays_for_image
            if overlay[1] not in existing_plotter_names
        ]

        # None should be available
        assert len(filtered_overlays) == 0


class TestPollForPlotUpdates:
    """Tests for the version-based polling mechanism in _poll_for_plot_updates."""

    def test_version_change_triggers_session_layer_recreation(
        self, plot_data_service, plot_orchestrator
    ):
        """Test version change from plotter replacement triggers recreation.

        This tests the invariant that the version mechanism is sufficient to
        trigger rebuilds when plotters change, without needing explicit
        plotter identity checks in the polling loop.
        """
        from uuid import uuid4

        import holoviews as hv

        from ess.livedata.dashboard.plot_data_service import LayerId
        from ess.livedata.dashboard.plots import PresenterBase
        from ess.livedata.dashboard.session_layer import SessionLayer

        # Create fake plotters
        class FakePlotter:
            def __init__(self, name):
                self.name = name
                self._cached_state = None
                self._presenters = []

            def compute(self, data):
                self._cached_state = data
                for p in self._presenters:
                    p._mark_dirty()

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
                return hv.DynamicMap(lambda data: hv.Curve([]), streams=[pipe])

        layer_id = LayerId(uuid4())

        # Setup initial state: plotter A with data
        plotter_a = FakePlotter('A')
        plotter_a.compute({'value': 1})
        plot_data_service.job_started(layer_id, plotter_a)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)
        initial_version = state.version

        # Create session layer with components for plotter A
        session_layer = SessionLayer(
            layer_id=layer_id, last_seen_version=initial_version
        )
        session_layer.ensure_components(state)

        assert session_layer.components is not None
        original_components = session_layer.components
        assert session_layer.components.is_valid_for(plotter_a)

        # Simulate workflow restart: job_started with new plotter B
        plotter_b = FakePlotter('B')
        plotter_b.compute({'value': 2})
        plot_data_service.job_started(layer_id, plotter_b)
        plot_data_service.data_arrived(layer_id)

        new_state = plot_data_service.get(layer_id)

        # Version must have changed (this is the key invariant)
        assert new_state.version != initial_version
        assert new_state.version != session_layer.last_seen_version

        # Simulate what _poll_for_plot_updates does:
        # It detects version change and triggers rebuild via ensure_components
        session_layer.last_seen_version = new_state.version
        session_layer.ensure_components(new_state)

        # Components should be recreated for the new plotter
        assert session_layer.components is not original_components
        assert session_layer.components.is_valid_for(plotter_b)
        assert not original_components.is_valid_for(plotter_b)

    def test_version_unchanged_preserves_components(self, plot_data_service):
        """Test that components are preserved when version hasn't changed."""
        from uuid import uuid4

        import holoviews as hv

        from ess.livedata.dashboard.plot_data_service import LayerId
        from ess.livedata.dashboard.plots import PresenterBase
        from ess.livedata.dashboard.session_layer import SessionLayer

        class FakePlotter:
            def __init__(self):
                self._cached_state = None
                self._presenters = []

            def compute(self, data):
                self._cached_state = data
                for p in self._presenters:
                    p._mark_dirty()

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
                return hv.DynamicMap(lambda data: hv.Curve([]), streams=[pipe])

        layer_id = LayerId(uuid4())
        plotter = FakePlotter()
        plotter.compute({'value': 1})

        plot_data_service.job_started(layer_id, plotter)
        plot_data_service.data_arrived(layer_id)

        state = plot_data_service.get(layer_id)

        session_layer = SessionLayer(layer_id=layer_id, last_seen_version=state.version)
        session_layer.ensure_components(state)
        original_components = session_layer.components

        # Simulate polling with no version change
        # (same state, version matches last_seen_version)
        session_layer.ensure_components(state)

        # Components should be preserved
        assert session_layer.components is original_components


class TestComposeMixedLayers:
    """Composition of cells mixing dynamic and static (overlay) layers."""

    def test_static_layer_does_not_break_autoscale_controller(
        self, plot_grid_tabs, plot_data_service
    ):
        """A static overlay layer must not be fed to the autoscale controller.

        Static plotters are not ``Plotter`` subclasses and lack
        ``AUTOSCALE_AXES``; mixing one with a dynamic layer previously raised
        ``AttributeError`` while building the cell's ``CellAutoscaleController``.
        """
        from uuid import uuid4

        import holoviews as hv

        from ess.livedata.config.workflow_spec import WorkflowId
        from ess.livedata.dashboard.data_roles import PRIMARY
        from ess.livedata.dashboard.plot_data_service import LayerId
        from ess.livedata.dashboard.plot_orchestrator import (
            CellGeometry,
            DataSourceConfig,
            Layer,
            PlotCell,
            PlotConfig,
        )
        from ess.livedata.dashboard.plot_params import TimeWindowParams
        from ess.livedata.dashboard.plots import PresenterBase
        from ess.livedata.dashboard.range_hook import Axis
        from ess.livedata.dashboard.session_layer import SessionLayer
        from ess.livedata.dashboard.static_plots import (
            LinesCoordinates,
            LinesPlotter,
            VLinesParams,
        )
        from ess.livedata.dashboard.widgets.plot_grid_tabs import CellId

        class _DynamicPresenter(PresenterBase):
            def present(self, pipe):
                return hv.DynamicMap(lambda data: hv.Curve([]), streams=[pipe])

        class _DynamicPlotter:
            AUTOSCALE_AXES: frozenset[Axis] = frozenset({'x', 'y'})

            def __init__(self):
                self._cached_state = None
                self._presenters = []

            @property
            def autoscale_axes(self):
                return self.AUTOSCALE_AXES

            def compute(self, data):
                self._cached_state = data

            def get_cached_state(self):
                return self._cached_state

            def has_cached_state(self):
                return self._cached_state is not None

            def create_presenter(self):
                presenter = _DynamicPresenter(self)
                self._presenters.append(presenter)
                return presenter

            def iter_range_targets(self):
                return iter(())

        wf = WorkflowId(instrument='test', name='wf', version=1)
        geo = CellGeometry(row=0, col=0, row_span=1, col_span=1)

        dynamic_config = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=wf, source_names=['s1'], view_name='result'
                )
            },
            plot_name='lines',
            params=TimeWindowParams(),
        )
        static_config = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=wf, source_names=[], view_name='guides'
                )
            },
            plot_name='vlines',
            params=TimeWindowParams(),
        )
        assert static_config.is_static()

        dynamic_layer = Layer(layer_id=LayerId(uuid4()), config=dynamic_config)
        static_layer = Layer(layer_id=LayerId(uuid4()), config=static_config)
        cell = PlotCell(geometry=geo, layers=[dynamic_layer, static_layer])

        dynamic_plotter = _DynamicPlotter()
        dynamic_plotter.compute(hv.Curve([1, 2, 3]))
        static_plotter = LinesPlotter.vlines(
            VLinesParams(geometry=LinesCoordinates(positions='10, 20'))
        )
        static_plotter.compute({})

        for layer, plotter in (
            (dynamic_layer, dynamic_plotter),
            (static_layer, static_plotter),
        ):
            plot_data_service.job_started(layer.layer_id, plotter)
            plot_data_service.data_arrived(layer.layer_id)
            state = plot_data_service.get(layer.layer_id)
            session_layer = SessionLayer(
                layer_id=layer.layer_id, last_seen_version=state.version
            )
            session_layer.ensure_components(state)
            plot_grid_tabs._session_layers[layer.layer_id] = session_layer

        cell_id = CellId(uuid4())
        cell_widget = plot_grid_tabs._build_cell(cell_id, cell)

        assert cell_widget.has_plot
        # The dynamic layer still drives autoscale; controller was built.
        assert cell_widget.autoscale_controller is not None


class TestComposeTableLayer:
    """Composition of a cell containing a table layer.

    Tables render as a DataTable widget, not a plain Bokeh figure, so they
    cannot be fused via ``hv.Overlay``. Such a layer is forbidden from sharing a
    cell with other layers (enforced in ``PlotOrchestrator``); the cell widget
    therefore only ever renders a table as the sole layer.
    """

    @staticmethod
    def _add_table_layer(plot_grid_tabs, plot_data_service, *, output: str):
        import uuid

        import holoviews as hv
        import scipp as sc

        from ess.livedata.config.workflow_spec import (
            DataKey,
            WorkflowId,
        )
        from ess.livedata.dashboard.data_roles import PRIMARY
        from ess.livedata.dashboard.plot_data_service import LayerId
        from ess.livedata.dashboard.plot_orchestrator import (
            DataSourceConfig,
            Layer,
            PlotConfig,
        )
        from ess.livedata.dashboard.plot_params import PlotParamsTable
        from ess.livedata.dashboard.session_layer import SessionLayer
        from ess.livedata.dashboard.table_plotter import TablePlotter

        wf = WorkflowId(instrument='test', name='wf', version=1)
        config = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=wf, source_names=['bank0'], view_name=output
                )
            },
            plot_name='table',
            params=PlotParamsTable(),
        )
        layer = Layer(layer_id=LayerId(uuid.uuid4()), config=config)

        plotter = TablePlotter.from_params(PlotParamsTable())
        key = DataKey(
            workflow_id=wf,
            source_name='bank0',
            output_name=output,
        )
        plotter.compute({PRIMARY: {key: sc.DataArray(sc.scalar(1.0, unit='counts'))}})
        assert isinstance(plotter.get_cached_state(), hv.Table)

        plot_data_service.job_started(layer.layer_id, plotter)
        plot_data_service.data_arrived(layer.layer_id)
        state = plot_data_service.get(layer.layer_id)
        session_layer = SessionLayer(
            layer_id=layer.layer_id, last_seen_version=state.version
        )
        session_layer.ensure_components(state)
        plot_grid_tabs._session_layers[layer.layer_id] = session_layer
        return layer

    def test_single_table_layer_is_not_a_layout(
        self, plot_grid_tabs, plot_data_service
    ):
        from uuid import uuid4

        import holoviews as hv

        from ess.livedata.dashboard.plot_orchestrator import CellGeometry, PlotCell
        from ess.livedata.dashboard.widgets.plot_grid_tabs import CellId

        layer = self._add_table_layer(
            plot_grid_tabs, plot_data_service, output='counts'
        )
        cell = PlotCell(
            geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
            layers=[layer],
        )
        cell_widget = plot_grid_tabs._build_cell(CellId(uuid4()), cell)

        assert cell_widget.has_plot
        assert not isinstance(cell_widget._plot, hv.Layout)


class TestDisabledGridTabs:
    """Tests for disabled grid handling in PlotGridTabs."""

    def test_disabled_grid_not_shown_on_init(
        self,
        plot_orchestrator,
        workflow_registry,
        plotting_controller,
        workflow_status_widget,
        plot_data_service,
        session_updater,
    ):
        """Disabled grids are not added as tabs during initialization."""
        plot_orchestrator.add_grid(title='Visible', nrows=2, ncols=2)
        grid_id = plot_orchestrator.add_grid(title='Hidden', nrows=2, ncols=2)
        plot_orchestrator.set_grid_enabled(grid_id, enabled=False)

        widget = PlotGridTabs(
            plot_orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            workflow_status_widget=workflow_status_widget,
            plot_data_service=plot_data_service,
            session_updater=session_updater,
        )

        # 2 static tabs + 1 visible grid = 3
        assert len(widget.tabs) == 3
        assert 'Visible' in widget.tabs._names
        assert 'Hidden' not in widget.tabs._names

    def test_disabling_grid_removes_tab(self, plot_orchestrator, plot_grid_tabs):
        """Disabling a grid removes its tab."""
        grid_id = plot_orchestrator.add_grid(title='Will Hide', nrows=2, ncols=2)
        tabs_before = len(plot_grid_tabs.tabs)

        plot_orchestrator.set_grid_enabled(grid_id, enabled=False)

        assert len(plot_grid_tabs.tabs) == tabs_before - 1

    def test_re_enabling_grid_adds_tab_back(self, plot_orchestrator, plot_grid_tabs):
        """Re-enabling a disabled grid adds its tab back."""
        grid_id = plot_orchestrator.add_grid(title='Toggle', nrows=2, ncols=2)
        plot_orchestrator.set_grid_enabled(grid_id, enabled=False)
        tabs_after_disable = len(plot_grid_tabs.tabs)

        plot_orchestrator.set_grid_enabled(grid_id, enabled=True)

        assert len(plot_grid_tabs.tabs) == tabs_after_disable + 1

    def test_poll_skips_disabled_grids(self, plot_orchestrator, plot_grid_tabs):
        """Polling skips disabled grids and cleans up their session layers."""
        from uuid import uuid4

        from ess.livedata.dashboard.plot_data_service import LayerId
        from ess.livedata.dashboard.session_layer import SessionLayer

        grid_id = plot_orchestrator.add_grid(title='Will Disable', nrows=2, ncols=2)

        # Simulate a session layer that was created while the grid was active
        fake_layer_id = LayerId(uuid4())
        plot_grid_tabs._session_layers[fake_layer_id] = SessionLayer(
            layer_id=fake_layer_id, last_seen_version=0
        )

        # Disable the grid — poll should not visit it, so session layer
        # becomes orphaned and gets cleaned up
        plot_orchestrator.set_grid_enabled(grid_id, enabled=False)
        plot_grid_tabs._poll_for_plot_updates()

        assert fake_layer_id not in plot_grid_tabs._session_layers

    def test_rename_updates_tab_title(self, plot_orchestrator, plot_grid_tabs):
        """Renaming a grid updates the corresponding tab title."""
        grid_id = plot_orchestrator.add_grid(title='Old Name', nrows=2, ncols=2)

        plot_orchestrator.rename_grid(grid_id, 'New Name')

        assert 'New Name' in plot_grid_tabs.tabs._names
        assert 'Old Name' not in plot_grid_tabs.tabs._names

    def test_reorder_updates_tab_order(self, plot_orchestrator, plot_grid_tabs):
        """Moving a grid updates tab order."""
        plot_orchestrator.add_grid(title='Alpha', nrows=2, ncols=2)
        id_b = plot_orchestrator.add_grid(title='Beta', nrows=2, ncols=2)

        plot_orchestrator.move_grid(id_b, -1)

        static = plot_grid_tabs._static_tabs_count
        grid_titles = plot_grid_tabs.tabs._names[static:]
        assert grid_titles == ['Beta', 'Alpha']

    def test_active_grid_id_resolves_correctly_with_first_grid_disabled(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """Active tab maps to the right grid when the first grid is disabled.

        A disabled grid preceding enabled ones must not skew the tab->grid
        mapping: it has no tab, so the first visible grid tab is the first
        *enabled* grid, not the first entry in ``_grid_widgets``.
        """
        id_a = plot_orchestrator.add_grid(title='A', nrows=2, ncols=2)
        id_b = plot_orchestrator.add_grid(title='B', nrows=2, ncols=2)
        id_c = plot_orchestrator.add_grid(title='C', nrows=2, ncols=2)

        plot_orchestrator.set_grid_enabled(id_a, enabled=False)

        static = plot_grid_tabs._static_tabs_count
        # Two visible grid tabs: B then C.
        assert plot_grid_tabs.tabs._names[static:] == ['B', 'C']

        plot_grid_tabs.tabs.active = static
        assert plot_grid_tabs._get_active_grid_id() == id_b

        plot_grid_tabs.tabs.active = static + 1
        assert plot_grid_tabs._get_active_grid_id() == id_c

    def test_active_grid_id_resolves_correctly_with_middle_grid_disabled(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """Active tab maps to the right grid when a middle grid is disabled."""
        id_a = plot_orchestrator.add_grid(title='A', nrows=2, ncols=2)
        id_b = plot_orchestrator.add_grid(title='B', nrows=2, ncols=2)
        id_c = plot_orchestrator.add_grid(title='C', nrows=2, ncols=2)

        plot_orchestrator.set_grid_enabled(id_b, enabled=False)

        static = plot_grid_tabs._static_tabs_count
        assert plot_grid_tabs.tabs._names[static:] == ['A', 'C']

        plot_grid_tabs.tabs.active = static
        assert plot_grid_tabs._get_active_grid_id() == id_a

        plot_grid_tabs.tabs.active = static + 1
        assert plot_grid_tabs._get_active_grid_id() == id_c

    def test_removing_grid_after_disabled_grid_removes_right_tab(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """Removing an enabled grid preceded by a disabled one pops its tab.

        With the first grid disabled, the positional tab index of a later grid
        differs from its index in ``_grid_widgets``; removal must pop the tab
        that actually belongs to the removed grid.
        """
        id_a = plot_orchestrator.add_grid(title='A', nrows=2, ncols=2)
        plot_orchestrator.add_grid(title='B', nrows=2, ncols=2)
        id_c = plot_orchestrator.add_grid(title='C', nrows=2, ncols=2)

        plot_orchestrator.set_grid_enabled(id_a, enabled=False)

        static = plot_grid_tabs._static_tabs_count
        assert plot_grid_tabs.tabs._names[static:] == ['B', 'C']

        plot_orchestrator.remove_grid(id_c)

        # C's tab is gone; B remains.
        assert plot_grid_tabs.tabs._names[static:] == ['B']
