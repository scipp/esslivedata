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


def _tick(*widgets: PlotGridTabs) -> None:
    """Drive one gated poll cycle for each widget.

    Runs the poll pass only when the widget's ``_has_pending_work`` predicate
    fires, mirroring SessionUpdater's wake/housekeeping ticks. This makes every
    test asserting a visible effect after ``_tick`` also guard the predicate:
    a change source the pass handles but the predicate misses fails here
    instead of silently lagging until the periodic full pass in production.
    Tests that rely on that full pass (documented predicate holes, e.g.
    plotter swaps) call ``_poll_for_plot_updates`` directly.
    """
    for widget in widgets:
        if widget._has_pending_work():
            widget._poll_for_plot_updates()


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

    def test_new_grid_appears_on_poll(self, plot_orchestrator, plot_grid_tabs):
        """A grid added after construction appears as a tab on the next poll."""
        plot_orchestrator.add_grid(title='New Grid', nrows=3, ncols=3)
        _tick(plot_grid_tabs)

        # Should now have 3 tabs: Workflows + Manage + New Grid
        assert len(plot_grid_tabs.tabs) == 3


class TestGridTabManagement:
    """Tests for adding and removing grid tabs (driven by polling)."""

    def test_add_grid_adds_tab_on_poll(self, plot_orchestrator, plot_grid_tabs):
        """Test that creating a grid adds a new tab after a poll."""
        initial_count = len(plot_grid_tabs.tabs)

        plot_orchestrator.add_grid(title='Test Grid', nrows=4, ncols=4)
        _tick(plot_grid_tabs)

        # Should have one more tab
        assert len(plot_grid_tabs.tabs) == initial_count + 1

    def test_grid_created_directly_does_not_switch_tab(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """A grid created directly on the orchestrator (no local flag) appears
        but does not steal focus -- the "other session" view."""
        plot_orchestrator.add_grid(title='Other Session Grid', nrows=3, ncols=3)
        _tick(plot_grid_tabs)

        # Tab exists but active is unchanged (still Workflows at index 0).
        assert len(plot_grid_tabs.tabs) == 3
        assert plot_grid_tabs.tabs.active == 0

    def test_local_grid_creation_switches_to_new_tab(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """A grid created via this session's manager focuses its tab on poll."""
        plot_grid_tabs._grid_manager._on_add_grid(None)
        _tick(plot_grid_tabs)

        # Active tab should be the newly created grid (Workflows=0, Manage=1,
        # grid=2).
        assert plot_grid_tabs.tabs.active == 2

    def test_remove_grid_removes_tab_on_poll(self, plot_orchestrator, plot_grid_tabs):
        """Test that removing a grid removes its tab after a poll."""
        grid_id = plot_orchestrator.add_grid(title='To Remove', nrows=3, ncols=3)
        _tick(plot_grid_tabs)
        assert len(plot_grid_tabs.tabs) == 3  # Workflows + Manage + Grid

        plot_orchestrator.remove_grid(grid_id)
        _tick(plot_grid_tabs)

        # Should only have two static tabs left (Workflows + Manage)
        assert len(plot_grid_tabs.tabs) == 2

    def test_removing_grid_updates_correctly(self, plot_orchestrator, plot_grid_tabs):
        """Test that removing a grid from the middle works correctly."""
        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        grid_id_2 = plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        plot_orchestrator.add_grid(title='Grid 3', nrows=4, ncols=4)
        _tick(plot_grid_tabs)

        # Remove middle grid
        plot_orchestrator.remove_grid(grid_id_2)
        _tick(plot_grid_tabs)

        # Should have Workflows + Manage + 2 remaining grids
        assert len(plot_grid_tabs.tabs) == 4

    def _make_widget(
        self,
        plot_orchestrator,
        workflow_registry,
        plotting_controller,
        job_service,
        job_orchestrator,
        plot_data_service,
        session_id,
    ):
        registry = SessionRegistry()
        session_updater = SessionUpdater(
            session_id=SessionId(session_id),
            session_registry=registry,
            notification_queue=NotificationQueue(),
        )
        workflow_status_widget = WorkflowStatusListWidget(
            orchestrator=job_orchestrator,
            job_service=job_service,
        )
        return PlotGridTabs(
            plot_orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            workflow_status_widget=workflow_status_widget,
            plot_data_service=plot_data_service,
            session_updater=session_updater,
        )

    def test_multiple_widget_instances_stay_synchronized(
        self,
        plot_orchestrator,
        workflow_registry,
        plotting_controller,
        job_service,
        job_orchestrator,
        plot_data_service,
    ):
        """Two widgets sharing an orchestrator both reconcile on their polls."""
        widget1 = self._make_widget(
            plot_orchestrator,
            workflow_registry,
            plotting_controller,
            job_service,
            job_orchestrator,
            plot_data_service,
            'session-1',
        )
        widget2 = self._make_widget(
            plot_orchestrator,
            workflow_registry,
            plotting_controller,
            job_service,
            job_orchestrator,
            plot_data_service,
            'session-2',
        )

        # Add grid via orchestrator
        grid_id = plot_orchestrator.add_grid(title='Shared Grid', nrows=3, ncols=3)
        _tick(widget1, widget2)

        # Both widgets should have the new tab
        assert len(widget1.tabs) == 3  # Workflows + Manage + Shared Grid
        assert len(widget2.tabs) == 3

        # Remove grid
        plot_orchestrator.remove_grid(grid_id)
        _tick(widget1, widget2)

        # Both widgets should reflect removal
        assert len(widget1.tabs) == 2  # Workflows + Manage
        assert len(widget2.tabs) == 2

    def test_local_creation_focuses_only_creating_session(
        self,
        plot_orchestrator,
        workflow_registry,
        plotting_controller,
        job_service,
        job_orchestrator,
        plot_data_service,
    ):
        """Headline cross-session regression: a grid created via session A's
        local manager path focuses the tab in A only; B gains the tab but its
        active tab is unchanged. On the pre-change push code this failed because
        ``_on_grid_created`` set ``tabs.active`` in every session."""
        widget_a = self._make_widget(
            plot_orchestrator,
            workflow_registry,
            plotting_controller,
            job_service,
            job_orchestrator,
            plot_data_service,
            'session-a',
        )
        widget_b = self._make_widget(
            plot_orchestrator,
            workflow_registry,
            plotting_controller,
            job_service,
            job_orchestrator,
            plot_data_service,
            'session-b',
        )
        assert widget_a.tabs.active == 0
        assert widget_b.tabs.active == 0

        # A creates a grid through its own manager (local path sets A's pending
        # focus); B never learns it was local.
        widget_a._grid_manager._on_add_grid(None)
        _tick(widget_a, widget_b)

        # Both gained the tab.
        assert len(widget_a.tabs) == 3
        assert len(widget_b.tabs) == 3
        # A focused the new grid; B's active tab is unchanged.
        assert widget_a.tabs.active == 2
        assert widget_b.tabs.active == 0

    def test_replace_grid_keeps_position_without_stealing_focus(
        self,
        plot_orchestrator,
        workflow_registry,
        plotting_controller,
        job_service,
        job_orchestrator,
        plot_data_service,
    ):
        """replace_grid keeps the tab position; a non-creating session viewing
        the replaced grid stays on that position and is not thrown elsewhere."""
        widget_b = self._make_widget(
            plot_orchestrator,
            workflow_registry,
            plotting_controller,
            job_service,
            job_orchestrator,
            plot_data_service,
            'session-b',
        )
        grid_id = plot_orchestrator.add_grid(title='G', nrows=2, ncols=2)
        _tick(widget_b)
        # B views the grid tab (index 2: Workflows, Manage, G).
        widget_b.tabs.active = 2

        # Another session replaces the grid; B carries no local focus flag.
        plot_orchestrator.replace_grid(grid_id, 'G2', nrows=3, ncols=3)
        _tick(widget_b)

        static = widget_b._static_tabs_count
        assert widget_b.tabs._names[static:] == ['G2']
        # Same position; focus not stolen to a static tab.
        assert widget_b.tabs.active == 2


class TestManageTab:
    """Tests for the Manage tab functionality."""

    def test_manage_tab_count_stable_when_grids_added(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """Test that adding grids doesn't remove or duplicate the Manage tab."""
        initial_count = len(plot_grid_tabs.tabs)
        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        _tick(plot_grid_tabs)
        # Should have exactly one more tab
        assert len(plot_grid_tabs.tabs) == initial_count + 1

        plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        _tick(plot_grid_tabs)
        # Should have exactly one more tab again
        assert len(plot_grid_tabs.tabs) == initial_count + 2


def _register_active_layer(plot_orchestrator, plot_data_service, plot_grid_tabs):
    """Register a layer with an active viewer token held by the session.

    Returns the layer id. Used to assert tier-2 sever releases the token.
    """
    from uuid import uuid4

    from ess.livedata.dashboard.plot_data_service import LayerId
    from ess.livedata.dashboard.session_layer import SessionLayer

    layer_id = LayerId(uuid4())
    plot_data_service.job_started(layer_id, object())
    state = plot_data_service.get(layer_id)
    session_layer = SessionLayer(layer_id=layer_id, last_seen_version=state.version)
    plot_grid_tabs._session_layers[layer_id] = session_layer
    plot_orchestrator.activate_layer(layer_id, session_layer, True)
    return layer_id


class TestShutdown:
    """Tests for widget shutdown and cleanup."""

    def test_sever_releases_viewer_tokens(
        self, plot_orchestrator, plot_grid_tabs, plot_data_service
    ):
        """Tier-2 sever releases the session's viewer/interest tokens and drops
        its session-layer records. There is no lifecycle subscription to sever:
        topology changes stop being observed once polling stops."""
        layer_id = _register_active_layer(
            plot_orchestrator, plot_data_service, plot_grid_tabs
        )
        assert plot_data_service.get(layer_id).has_viewers

        plot_grid_tabs.sever()

        assert not plot_data_service.get(layer_id).has_viewers
        assert plot_grid_tabs._session_layers == {}

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

    def test_reaper_severs_shared_state_without_touching_document(
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
        # Hold an active viewer token so we can observe tier-2 release it.
        layer_id = _register_active_layer(plot_orchestrator, plot_data_service, widget)
        assert plot_data_service.get(layer_id).has_viewers

        cleaned = registry.cleanup_stale_sessions()

        assert SessionId('stale-session') in cleaned
        # Tier 2 ran inline on the reaper thread: the session's viewer token was
        # released and its session-layer records cleared. There is no lifecycle
        # subscription to sever under polling.
        assert not plot_data_service.get(layer_id).has_viewers
        assert widget._session_layers == {}
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
        _tick(plot_grid_tabs)
        tabs_before = len(plot_grid_tabs.tabs)

        plot_orchestrator.set_grid_enabled(grid_id, enabled=False)
        _tick(plot_grid_tabs)

        assert len(plot_grid_tabs.tabs) == tabs_before - 1

    def test_re_enabling_grid_adds_tab_back(self, plot_orchestrator, plot_grid_tabs):
        """Re-enabling a disabled grid adds its tab back."""
        grid_id = plot_orchestrator.add_grid(title='Toggle', nrows=2, ncols=2)
        plot_orchestrator.set_grid_enabled(grid_id, enabled=False)
        _tick(plot_grid_tabs)
        tabs_after_disable = len(plot_grid_tabs.tabs)

        plot_orchestrator.set_grid_enabled(grid_id, enabled=True)
        _tick(plot_grid_tabs)

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
        _tick(plot_grid_tabs)

        plot_orchestrator.rename_grid(grid_id, 'New Name')
        _tick(plot_grid_tabs)

        assert 'New Name' in plot_grid_tabs.tabs._names
        assert 'Old Name' not in plot_grid_tabs.tabs._names

    def test_reorder_updates_tab_order(self, plot_orchestrator, plot_grid_tabs):
        """Moving a grid updates tab order."""
        plot_orchestrator.add_grid(title='Alpha', nrows=2, ncols=2)
        id_b = plot_orchestrator.add_grid(title='Beta', nrows=2, ncols=2)

        plot_orchestrator.move_grid(id_b, -1)
        _tick(plot_grid_tabs)

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
        _tick(plot_grid_tabs)

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
        _tick(plot_grid_tabs)

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
        _tick(plot_grid_tabs)

        static = plot_grid_tabs._static_tabs_count
        assert plot_grid_tabs.tabs._names[static:] == ['B', 'C']

        plot_orchestrator.remove_grid(id_c)
        _tick(plot_grid_tabs)

        # C's tab is gone; B remains.
        assert plot_grid_tabs.tabs._names[static:] == ['B']


def _add_static_cell(plot_orchestrator, grid_id, geometry, *, positions='10, 20'):
    """Add a cell with a single static (no-workflow) vlines layer.

    Static overlays compute from params alone, so they let cell-reconcile tests
    run without workflow data.
    """
    from ess.livedata.config.workflow_spec import WorkflowId
    from ess.livedata.dashboard.data_roles import PRIMARY
    from ess.livedata.dashboard.plot_orchestrator import DataSourceConfig, PlotConfig
    from ess.livedata.dashboard.static_plots import LinesCoordinates, VLinesParams

    config = PlotConfig(
        data_sources={
            PRIMARY: DataSourceConfig(
                workflow_id=WorkflowId(instrument='test', name='wf', version=1),
                source_names=[],
                view_name='guides',
            )
        },
        plot_name='vlines',
        params=VLinesParams(geometry=LinesCoordinates(positions=positions)),
    )
    cell_id = plot_orchestrator.add_cell(grid_id, geometry)
    plot_orchestrator.add_layer(cell_id, config)
    return cell_id


class TestCellReconcile:
    """Poll-driven cell reconcile (replaces the former push cell callbacks)."""

    _GEO = None  # set in setup

    @staticmethod
    def _geometry():
        from ess.livedata.dashboard.plot_orchestrator import CellGeometry

        return CellGeometry(row=0, col=0, row_span=1, col_span=1)

    def test_new_cell_built_on_poll(self, plot_orchestrator, plot_grid_tabs):
        grid_id = plot_orchestrator.add_grid(title='G', nrows=2, ncols=2)
        cell_id = _add_static_cell(plot_orchestrator, grid_id, self._geometry())

        assert cell_id not in plot_grid_tabs._cells
        _tick(plot_grid_tabs)
        assert cell_id in plot_grid_tabs._cells

    def test_remove_last_layer_removes_and_disposes_cell(
        self, plot_orchestrator, plot_grid_tabs
    ):
        grid_id = plot_orchestrator.add_grid(title='G', nrows=2, ncols=2)
        cell_id = _add_static_cell(plot_orchestrator, grid_id, self._geometry())
        _tick(plot_grid_tabs)
        assert cell_id in plot_grid_tabs._cells
        layer_id = plot_orchestrator.get_cell(cell_id).layers[0].layer_id

        # Removing the only layer removes the whole cell from topology.
        plot_orchestrator.remove_layer(layer_id)
        _tick(plot_grid_tabs)

        assert cell_id not in plot_grid_tabs._cells
        assert cell_id not in plot_grid_tabs._cell_grid
        assert cell_id not in plot_grid_tabs._cell_signatures

    def test_set_cell_title_rebuilds_cell(self, plot_orchestrator, plot_grid_tabs):
        grid_id = plot_orchestrator.add_grid(title='G', nrows=2, ncols=2)
        cell_id = _add_static_cell(plot_orchestrator, grid_id, self._geometry())
        _tick(plot_grid_tabs)
        widget_before = plot_grid_tabs._cells[cell_id]

        plot_orchestrator.set_cell_title(cell_id, 'Renamed')
        _tick(plot_grid_tabs)

        # Signature changed (user_title) -> the cell widget was rebuilt.
        assert plot_grid_tabs._cells[cell_id] is not widget_before

    def test_disabling_grid_does_not_dispose_cell(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """Disabling a grid must not dispose its cells: the grid still exists in
        topology (only hidden), so the removal sweep keeps the cell widget even
        though the poll loop skips disabled grids. Re-enabling restores it."""
        grid_id = plot_orchestrator.add_grid(title='G', nrows=2, ncols=2)
        cell_id = _add_static_cell(plot_orchestrator, grid_id, self._geometry())
        _tick(plot_grid_tabs)
        widget = plot_grid_tabs._cells[cell_id]

        plot_orchestrator.set_grid_enabled(grid_id, enabled=False)
        _tick(plot_grid_tabs)

        # Same instance: preserved across disable, not swept/disposed.
        assert plot_grid_tabs._cells.get(cell_id) is widget

        plot_orchestrator.set_grid_enabled(grid_id, enabled=True)
        _tick(plot_grid_tabs)
        # Still present after re-enable (rebuilt from a fresh session layer).
        assert cell_id in plot_grid_tabs._cells

    def test_cell_edit_does_not_churn_tabs(self, plot_orchestrator, plot_grid_tabs):
        """Cell-level changes bump the topology version but must not tear down
        and re-append the Tabs entries -- that would discard the active tab's
        Bokeh models (flicker) on every layer edit in every session. Grid-level
        changes still rebuild the tabs."""
        grid_id = plot_orchestrator.add_grid(title='G', nrows=2, ncols=2)
        cell_id = _add_static_cell(plot_orchestrator, grid_id, self._geometry())
        _tick(plot_grid_tabs)

        events: list = []
        plot_grid_tabs.tabs.param.watch(events.append, 'objects')

        plot_orchestrator.set_cell_title(cell_id, 'Renamed')
        _tick(plot_grid_tabs)
        assert events == []

        plot_orchestrator.rename_grid(grid_id, 'G2')
        _tick(plot_grid_tabs)
        assert events

    def test_reconfigure_vanished_layer_shows_error_not_modal(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """A gear click racing a removal in another session (poll window) must
        not raise; no modal opens (an error notification is shown instead)."""
        from uuid import uuid4

        from ess.livedata.dashboard.plot_data_service import LayerId

        plot_grid_tabs._on_reconfigure_layer(LayerId(uuid4()))

        assert plot_grid_tabs._current_modal is None

    def test_update_layer_config_rebuilds_cell(self, plot_orchestrator, plot_grid_tabs):
        grid_id = plot_orchestrator.add_grid(title='G', nrows=2, ncols=2)
        cell_id = _add_static_cell(plot_orchestrator, grid_id, self._geometry())
        _tick(plot_grid_tabs)
        widget_before = plot_grid_tabs._cells[cell_id]
        layer_id = plot_orchestrator.get_cell(cell_id).layers[0].layer_id

        from ess.livedata.config.workflow_spec import WorkflowId
        from ess.livedata.dashboard.data_roles import PRIMARY
        from ess.livedata.dashboard.plot_orchestrator import (
            DataSourceConfig,
            PlotConfig,
        )
        from ess.livedata.dashboard.static_plots import LinesCoordinates, VLinesParams

        new_config = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=WorkflowId(instrument='test', name='wf', version=1),
                    source_names=[],
                    view_name='guides',
                )
            },
            plot_name='vlines',
            params=VLinesParams(geometry=LinesCoordinates(positions='30, 40')),
        )
        plot_orchestrator.update_layer_config(layer_id, new_config)
        _tick(plot_grid_tabs)

        # Reconfigure mints a fresh LayerId -> signature changed -> rebuilt.
        assert plot_grid_tabs._cells[cell_id] is not widget_before


class TestWakeGateContract:
    """Contract of ``_has_pending_work``, the gate registered with SessionUpdater.

    The gate must go quiet once a pass ran (a stuck-True gate makes every
    housekeeping tick pay the full hold+freeze pass in every session) and
    re-arm for each change source the pass renders (a missed source lags
    until the periodic full pass). The mutate->tick->assert tests above cover
    the re-arm direction implicitly via the gated ``_tick``; this test pins
    the quiescence direction explicitly.
    """

    def test_gate_quiet_after_pass_and_rearms_per_source(
        self, plot_orchestrator, plot_grid_tabs
    ):
        from ess.livedata.dashboard.plot_orchestrator import CellGeometry
        from ess.livedata.dashboard.widgets.plot_grid_tabs import (
            _FRESHNESS_STALL_INTERVAL_S,
        )

        _tick(plot_grid_tabs)
        assert not plot_grid_tabs._has_pending_work()

        # Topology change arms the gate; the pass clears it.
        grid_id = plot_orchestrator.add_grid(title='G', nrows=2, ncols=2)
        geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)
        _add_static_cell(plot_orchestrator, grid_id, geometry)
        assert plot_grid_tabs._has_pending_work()
        _tick(plot_grid_tabs)
        assert not plot_grid_tabs._has_pending_work()

        # A tab switch wakes synchronously (no document in tests), so the
        # flush must have passed through the gate; quiet again afterwards.
        plot_grid_tabs.tabs.active = 2
        assert plot_grid_tabs._last_active_grid_id == grid_id
        assert not plot_grid_tabs._has_pending_work()

        # Stall aging: with cells on the active grid and no data events, the
        # timer alone re-arms the gate; the pass resets it.
        plot_grid_tabs._last_freshness_update -= _FRESHNESS_STALL_INTERVAL_S
        assert plot_grid_tabs._has_pending_work()
        _tick(plot_grid_tabs)
        assert not plot_grid_tabs._has_pending_work()

    def _quiescent_active_layer(self, plot_orchestrator, plot_grid_tabs):
        """Put a static layer on the active grid and drive the gate quiet."""
        from ess.livedata.dashboard.plot_orchestrator import CellGeometry

        grid_id = plot_orchestrator.add_grid(title='G', nrows=2, ncols=2)
        geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)
        cell_id = _add_static_cell(plot_orchestrator, grid_id, geometry)
        _tick(plot_grid_tabs)
        plot_grid_tabs.tabs.active = 2
        assert plot_grid_tabs._last_active_grid_id == grid_id
        assert not plot_grid_tabs._has_pending_work()
        return plot_orchestrator.get_cell(cell_id).layers[0].layer_id

    @staticmethod
    def _assert_only_layer_version_term_armed(plot_orchestrator, plot_grid_tabs):
        """Pin that the gate fired on the layer-version term, not by accident.

        Topology and frame generation are the other event-driven terms; if
        either moved too, the assertion on ``_has_pending_work`` would pass
        even with the layer-version term removed.
        """
        assert plot_orchestrator.topology_version() == (
            plot_grid_tabs._last_topology_version
        )
        assert plot_orchestrator.frame_generation(
            plot_grid_tabs._last_active_grid_id
        ) == (plot_grid_tabs._last_flushed_generation)

    def test_job_stopped_rearms_gate(
        self, plot_orchestrator, plot_grid_tabs, plot_data_service
    ):
        """Stopping a workflow must arm the gate through the layer-version term.

        Unlike a restart, a stop produces no further frame, so the gate's
        frame-generation term can never cover it: without the layer-version
        term the plots keep their live rendering until the next unconditional
        full pass, i.e. up to _FULL_PASS_INTERVAL_S after the workflow stopped.
        """
        layer_id = self._quiescent_active_layer(plot_orchestrator, plot_grid_tabs)

        plot_data_service.job_stopped(layer_id)

        self._assert_only_layer_version_term_armed(plot_orchestrator, plot_grid_tabs)
        assert plot_grid_tabs._has_pending_work()
        _tick(plot_grid_tabs)
        assert not plot_grid_tabs._has_pending_work()

    def test_plotter_swap_rearms_gate(
        self,
        plot_orchestrator,
        plot_grid_tabs,
        plot_data_service,
        plotting_controller,
    ):
        """A plotter swap keeps the LayerId, so the cell signature is unchanged
        and only the layer-version term can arm the gate."""
        from ess.livedata.dashboard.static_plots import LinesCoordinates, VLinesParams

        layer_id = self._quiescent_active_layer(plot_orchestrator, plot_grid_tabs)
        plotter = plotting_controller.create_plotter(
            'vlines', params=VLinesParams(geometry=LinesCoordinates(positions='30'))
        )

        plot_data_service.job_started(layer_id, plotter)

        self._assert_only_layer_version_term_armed(plot_orchestrator, plot_grid_tabs)
        assert plot_grid_tabs._has_pending_work()
        _tick(plot_grid_tabs)
        assert not plot_grid_tabs._has_pending_work()
