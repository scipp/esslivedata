# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import holoviews as hv
import panel as pn
import pytest

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plot_orchestrator import (
    PlotOrchestrator,
    StubJobOrchestrator,
)
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.stream_manager import StreamManager
from ess.livedata.dashboard.widgets.plot_grid_tabs import PlotGridTabs

hv.extension('bokeh')


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
    """Create a stub JobOrchestrator for testing."""
    return StubJobOrchestrator()


@pytest.fixture
def plot_orchestrator(plotting_controller, fake_job_orchestrator):
    """Create a PlotOrchestrator for testing."""
    return PlotOrchestrator(
        plotting_controller=plotting_controller,
        job_orchestrator=fake_job_orchestrator,
    )


@pytest.fixture
def workflow_registry():
    """Create a minimal workflow registry for testing."""
    # Create a simple test workflow spec
    from pydantic import BaseModel, Field

    class TestOutputs(BaseModel):
        test_output: str = Field(title="Test Output")

    workflow_id = WorkflowId(
        instrument='test', namespace='data_reduction', name='test_workflow', version=1
    )

    workflow_spec = WorkflowSpec(
        instrument='test',
        namespace='data_reduction',
        name='test_workflow',
        version=1,
        title='Test Workflow',
        description='A test workflow',
        source_names=['source1', 'source2'],
        outputs=TestOutputs,
        params=None,
    )

    return {workflow_id: workflow_spec}


@pytest.fixture
def plot_grid_tabs(plot_orchestrator, workflow_registry, plotting_controller):
    """Create a PlotGridTabs widget for testing."""
    return PlotGridTabs(
        plot_orchestrator=plot_orchestrator,
        workflow_registry=workflow_registry,
        plotting_controller=plotting_controller,
    )


class TestPlotGridTabsInitialization:
    """Tests for PlotGridTabs initialization."""

    def test_creates_panel_tabs_widget(self, plot_grid_tabs):
        """Test that widget creates a Panel Tabs object."""
        assert isinstance(plot_grid_tabs.panel, pn.Tabs)

    def test_starts_with_one_tab_when_no_grids(self, plot_grid_tabs):
        """Test that widget starts with only one tab when no grids exist."""
        # Should have exactly one tab (the Manage tab)
        assert len(plot_grid_tabs.panel) == 1

    def test_initializes_from_existing_grids(
        self, plot_orchestrator, workflow_registry, plotting_controller
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
        )

        # Should have 3 tabs: Manage + 2 grids
        assert len(widget.panel) == 3

    def test_subscribes_to_lifecycle_events(self, plot_orchestrator, plot_grid_tabs):
        """Test that widget subscribes to orchestrator lifecycle events."""
        # Verify subscription by adding a grid and checking it appears
        plot_orchestrator.add_grid(title='New Grid', nrows=3, ncols=3)

        # Should now have 2 tabs: Manage + New Grid
        assert len(plot_grid_tabs.panel) == 2


class TestGridTabManagement:
    """Tests for adding and removing grid tabs."""

    def test_on_grid_created_adds_tab(self, plot_orchestrator, plot_grid_tabs):
        """Test that creating a grid adds a new tab."""
        initial_count = len(plot_grid_tabs.panel)

        plot_orchestrator.add_grid(title='Test Grid', nrows=4, ncols=4)

        # Should have one more tab
        assert len(plot_grid_tabs.panel) == initial_count + 1

    def test_on_grid_created_switches_to_new_tab(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """Test that creating a grid auto-switches to that tab."""
        plot_orchestrator.add_grid(title='Auto Switch', nrows=3, ncols=3)

        # Active tab should be the newly created grid (index 1, since Manage is at 0)
        assert plot_grid_tabs.panel.active == 1

    def test_on_grid_removed_removes_tab(self, plot_orchestrator, plot_grid_tabs):
        """Test that removing a grid removes its tab."""
        grid_id = plot_orchestrator.add_grid(title='To Remove', nrows=3, ncols=3)
        assert len(plot_grid_tabs.panel) == 2  # Grid + Manage

        plot_orchestrator.remove_grid(grid_id)

        # Should only have one tab left
        assert len(plot_grid_tabs.panel) == 1

    def test_removing_grid_updates_correctly(self, plot_orchestrator, plot_grid_tabs):
        """Test that removing a grid from the middle works correctly."""
        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        grid_id_2 = plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        plot_orchestrator.add_grid(title='Grid 3', nrows=4, ncols=4)

        # Remove middle grid
        plot_orchestrator.remove_grid(grid_id_2)

        # Should have Manage + 2 remaining grids
        assert len(plot_grid_tabs.panel) == 3

    def test_multiple_widget_instances_stay_synchronized(
        self, plot_orchestrator, workflow_registry, plotting_controller
    ):
        """Test that multiple widgets sharing same orchestrator stay in sync."""
        widget1 = PlotGridTabs(
            plot_orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
        )
        widget2 = PlotGridTabs(
            plot_orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
        )

        # Add grid via orchestrator
        grid_id = plot_orchestrator.add_grid(title='Shared Grid', nrows=3, ncols=3)

        # Both widgets should have the new tab
        assert len(widget1.panel) == 2  # Manage + Shared Grid
        assert len(widget2.panel) == 2

        # Remove grid
        plot_orchestrator.remove_grid(grid_id)

        # Both widgets should reflect removal
        assert len(widget1.panel) == 1  # Only Manage
        assert len(widget2.panel) == 1


class TestManageTab:
    """Tests for the Manage tab functionality."""

    def test_manage_tab_count_stable_when_grids_added(
        self, plot_orchestrator, plot_grid_tabs
    ):
        """Test that adding grids doesn't remove or duplicate the Manage tab."""
        initial_count = len(plot_grid_tabs.panel)
        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        # Should have exactly one more tab
        assert len(plot_grid_tabs.panel) == initial_count + 1

        plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        # Should have exactly one more tab again
        assert len(plot_grid_tabs.panel) == initial_count + 2


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
