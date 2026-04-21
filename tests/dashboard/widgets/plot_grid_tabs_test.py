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
def workflow_controller(job_orchestrator, workflow_registry, data_service):
    """Create a WorkflowController for testing."""
    from ess.livedata.dashboard.workflow_controller import WorkflowController

    return WorkflowController(
        job_orchestrator=job_orchestrator,
        workflow_registry=workflow_registry,
        data_service=data_service,
        correlation_histogram_controller=None,
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

    def test_shutdown_unsubscribes_from_lifecycle(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """Test that shutdown unsubscribes from orchestrator lifecycle."""
        # Shutdown the widget
        plot_grid_tabs.shutdown()

        # Adding a grid should not affect the widget anymore
        initial_count = len(plot_grid_tabs.tabs)
        plot_orchestrator.add_grid(title='After Shutdown', nrows=3, ncols=3)

        # Tab count should not change
        assert len(plot_grid_tabs.tabs) == initial_count

    def test_shutdown_can_be_called_multiple_times(self, plot_grid_tabs):
        """Test that shutdown is idempotent."""
        plot_grid_tabs.shutdown()
        plot_grid_tabs.shutdown()  # Should not raise


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
