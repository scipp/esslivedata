# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import holoviews as hv
import panel as pn
import pytest

from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plot_orchestrator import PlotOrchestrator, SubscriptionId
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.stream_manager import StreamManager
from ess.livedata.dashboard.widgets.plot_grid_tabs import PlotGridTabs

hv.extension('bokeh')


class FakeJobOrchestrator:
    """Fake JobOrchestrator for testing PlotOrchestrator."""

    def __init__(self):
        self._subscriptions: dict = {}

    def subscribe_to_workflow(self, workflow_id, callback) -> SubscriptionId:
        """Register a callback for workflow availability."""
        subscription_id = SubscriptionId(uuid.uuid4())
        self._subscriptions[subscription_id] = (workflow_id, callback)
        return subscription_id

    def unsubscribe(self, subscription_id: SubscriptionId) -> None:
        """Unsubscribe from workflow availability notifications."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]


@pytest.fixture
def data_service():
    """Create a DataService for testing."""
    return DataService()


@pytest.fixture
def job_service(data_service):
    """Create a JobService for testing."""
    return JobService(data_service=data_service)


@pytest.fixture
def stream_manager(data_service):
    """Create a StreamManager for testing."""
    return StreamManager(data_service=data_service, pipe_factory=hv.streams.Pipe)


@pytest.fixture
def plotting_controller(job_service, stream_manager):
    """Create a PlottingController for testing."""
    return PlottingController(
        job_service=job_service,
        stream_manager=stream_manager,
    )


@pytest.fixture
def fake_job_orchestrator():
    """Create a fake JobOrchestrator for testing."""
    return FakeJobOrchestrator()


@pytest.fixture
def plot_orchestrator(plotting_controller, fake_job_orchestrator):
    """Create a PlotOrchestrator for testing."""
    return PlotOrchestrator(
        plotting_controller=plotting_controller,
        job_orchestrator=fake_job_orchestrator,
    )


@pytest.fixture
def plot_grid_tabs(plot_orchestrator):
    """Create a PlotGridTabs widget for testing."""
    return PlotGridTabs(plot_orchestrator=plot_orchestrator)


class TestPlotGridTabsInitialization:
    """Tests for PlotGridTabs initialization."""

    def test_creates_panel_tabs_widget(self, plot_grid_tabs):
        """Test that widget creates a Panel Tabs object."""
        assert isinstance(plot_grid_tabs.panel, pn.Tabs)

    def test_starts_with_manage_tab_only_when_no_grids(self, plot_grid_tabs):
        """Test that widget starts with only Manage tab when no grids exist."""
        # Should have exactly one tab (Manage)
        assert len(plot_grid_tabs.panel) == 1
        # The tab should be named "Manage"
        assert plot_grid_tabs.panel._names[0] == 'Manage'

    def test_initializes_from_existing_grids(self, plot_orchestrator):
        """Test that widget creates tabs for existing grids."""
        # Add grids before creating widget
        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)

        # Create widget
        widget = PlotGridTabs(plot_orchestrator=plot_orchestrator)

        # Should have 3 tabs: Grid 1, Grid 2, Manage
        assert len(widget.panel) == 3
        assert widget.panel._names[0] == 'Grid 1'
        assert widget.panel._names[1] == 'Grid 2'
        assert widget.panel._names[2] == 'Manage'

    def test_subscribes_to_lifecycle_events(self, plot_orchestrator, plot_grid_tabs):
        """Test that widget subscribes to orchestrator lifecycle events."""
        # Verify subscription by adding a grid and checking it appears
        plot_orchestrator.add_grid(title='New Grid', nrows=3, ncols=3)

        # Should now have 2 tabs: New Grid, Manage
        assert len(plot_grid_tabs.panel) == 2
        assert plot_grid_tabs.panel._names[0] == 'New Grid'


class TestGridTabManagement:
    """Tests for adding and removing grid tabs."""

    def test_on_grid_created_adds_tab(self, plot_orchestrator, plot_grid_tabs):
        """Test that creating a grid adds a new tab."""
        initial_count = len(plot_grid_tabs.panel)

        plot_orchestrator.add_grid(title='Test Grid', nrows=4, ncols=4)

        # Should have one more tab
        assert len(plot_grid_tabs.panel) == initial_count + 1
        # New tab should be before Manage tab
        assert plot_grid_tabs.panel._names[-2] == 'Test Grid'
        assert plot_grid_tabs.panel._names[-1] == 'Manage'

    def test_on_grid_created_switches_to_new_tab(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """Test that creating a grid auto-switches to that tab."""
        plot_orchestrator.add_grid(title='Auto Switch', nrows=3, ncols=3)

        # Active tab should be the newly created grid (index 0, since it's the first)
        assert plot_grid_tabs.panel.active == 0

    def test_on_grid_removed_removes_tab(self, plot_orchestrator, plot_grid_tabs):
        """Test that removing a grid removes its tab."""
        grid_id = plot_orchestrator.add_grid(title='To Remove', nrows=3, ncols=3)
        assert len(plot_grid_tabs.panel) == 2  # Grid + Manage

        plot_orchestrator.remove_grid(grid_id)

        # Should only have Manage tab left
        assert len(plot_grid_tabs.panel) == 1
        assert plot_grid_tabs.panel._names[0] == 'Manage'

    def test_removing_grid_updates_tab_indices(self, plot_orchestrator, plot_grid_tabs):
        """Test that removing a grid updates internal tab index mapping."""
        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        grid_id_2 = plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        plot_orchestrator.add_grid(title='Grid 3', nrows=4, ncols=4)

        # Remove middle grid
        plot_orchestrator.remove_grid(grid_id_2)

        # Should have Grid 1, Grid 3, Manage
        assert len(plot_grid_tabs.panel) == 3
        assert plot_grid_tabs.panel._names[0] == 'Grid 1'
        assert plot_grid_tabs.panel._names[1] == 'Grid 3'
        assert plot_grid_tabs.panel._names[2] == 'Manage'

    def test_multiple_widget_instances_stay_synchronized(self, plot_orchestrator):
        """Test that multiple widgets sharing same orchestrator stay in sync."""
        widget1 = PlotGridTabs(plot_orchestrator=plot_orchestrator)
        widget2 = PlotGridTabs(plot_orchestrator=plot_orchestrator)

        # Add grid via orchestrator
        grid_id = plot_orchestrator.add_grid(title='Shared Grid', nrows=3, ncols=3)

        # Both widgets should have the new tab
        assert len(widget1.panel) == 2  # Shared Grid + Manage
        assert len(widget2.panel) == 2
        assert widget1.panel._names[0] == 'Shared Grid'
        assert widget2.panel._names[0] == 'Shared Grid'

        # Remove grid
        plot_orchestrator.remove_grid(grid_id)

        # Both widgets should reflect removal
        assert len(widget1.panel) == 1  # Only Manage
        assert len(widget2.panel) == 1


class TestManageTab:
    """Tests for the Manage tab functionality."""

    def test_manage_tab_is_always_last(self, plot_orchestrator, plot_grid_tabs):
        """Test that Manage tab stays at the end when grids are added."""
        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        assert plot_grid_tabs.panel._names[-1] == 'Manage'

        plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        assert plot_grid_tabs.panel._names[-1] == 'Manage'

    def test_manage_tab_exists_on_initialization(self, plot_grid_tabs):
        """Test that Manage tab is created during initialization."""
        # Find Manage tab in names list
        manage_tabs = [name for name in plot_grid_tabs.panel._names if name == 'Manage']
        assert len(manage_tabs) == 1


class TestShutdown:
    """Tests for widget shutdown and cleanup."""

    def test_shutdown_unsubscribes_from_lifecycle(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """Test that shutdown unsubscribes from orchestrator lifecycle."""
        # Shutdown the widget
        plot_grid_tabs.shutdown()

        # Adding a grid should not affect the widget anymore
        initial_count = len(plot_grid_tabs.panel)
        plot_orchestrator.add_grid(title='After Shutdown', nrows=3, ncols=3)

        # Tab count should not change
        assert len(plot_grid_tabs.panel) == initial_count

    def test_shutdown_can_be_called_multiple_times(self, plot_grid_tabs):
        """Test that shutdown is idempotent."""
        plot_grid_tabs.shutdown()
        plot_grid_tabs.shutdown()  # Should not raise


class TestPlotRequestCallback:
    """Tests for plot request handling (placeholder for future implementation)."""

    def test_plot_request_shows_not_implemented_notification(
        self, plot_grid_tabs, plot_orchestrator
    ):
        """Test that clicking grid cells shows 'not implemented' message."""
        # This test verifies the placeholder behavior until plot management is added
        # We're just checking that the callback exists and can be called without error
        plot_grid_tabs._on_plot_requested()
        # If no exception is raised, the placeholder callback works
