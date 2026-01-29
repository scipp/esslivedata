# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import holoviews as hv
import panel as pn
import pytest

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plot_data_service import PlotDataService
from ess.livedata.dashboard.plot_orchestrator import PlotOrchestrator
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.session_registry import SessionId, SessionRegistry
from ess.livedata.dashboard.session_updater import SessionUpdater
from ess.livedata.dashboard.stream_manager import StreamManager
from ess.livedata.dashboard.widgets.job_status_widget import JobStatusListWidget
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
def plotting_controller(job_service, stream_manager):
    """Create a PlottingController for testing."""
    return PlottingController(
        job_service=job_service,
        stream_manager=stream_manager,
    )


@pytest.fixture
def fake_data_service():
    """Create a fake DataService."""
    from ess.livedata.dashboard.data_service import DataService

    return DataService()


@pytest.fixture
def fake_job_service():
    """Create a fake JobService."""
    from ess.livedata.dashboard.job_service import JobService

    return JobService()


@pytest.fixture
def plot_orchestrator(
    plotting_controller, job_orchestrator, fake_data_service, plot_data_service
):
    """Create a PlotOrchestrator for testing."""
    return PlotOrchestrator(
        plotting_controller=plotting_controller,
        job_orchestrator=job_orchestrator,
        data_service=fake_data_service,
        instrument='dummy',
        plot_data_service=plot_data_service,
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
def command_service():
    """Create a CommandService for testing."""
    from ess.livedata.dashboard.command_service import CommandService
    from ess.livedata.fakes import FakeMessageSink

    return CommandService(sink=FakeMessageSink())


@pytest.fixture
def job_controller(command_service, fake_job_service):
    """Create a JobController for testing."""
    from ess.livedata.dashboard.job_controller import JobController

    return JobController(command_service=command_service, job_service=fake_job_service)


@pytest.fixture
def job_status_widget(fake_job_service, job_controller):
    """Create a JobStatusListWidget for testing."""
    return JobStatusListWidget(
        job_service=fake_job_service, job_controller=job_controller
    )


@pytest.fixture
def workflow_controller(job_orchestrator, workflow_registry, fake_data_service):
    """Create a WorkflowController for testing."""
    from ess.livedata.dashboard.workflow_controller import WorkflowController

    return WorkflowController(
        job_orchestrator=job_orchestrator,
        workflow_registry=workflow_registry,
        data_service=fake_data_service,
        correlation_histogram_controller=None,
    )


@pytest.fixture
def workflow_status_widget(job_orchestrator, fake_job_service):
    """Create a WorkflowStatusListWidget for testing."""
    return WorkflowStatusListWidget(
        orchestrator=job_orchestrator,
        job_service=fake_job_service,
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
    )


@pytest.fixture
def plot_grid_tabs(
    plot_orchestrator,
    workflow_registry,
    plotting_controller,
    job_status_widget,
    workflow_status_widget,
    plot_data_service,
    session_updater,
):
    """Create a PlotGridTabs widget for testing."""
    return PlotGridTabs(
        plot_orchestrator=plot_orchestrator,
        workflow_registry=workflow_registry,
        plotting_controller=plotting_controller,
        job_status_widget=job_status_widget,
        workflow_status_widget=workflow_status_widget,
        plot_data_service=plot_data_service,
        session_updater=session_updater,
    )


class TestPlotGridTabsInitialization:
    """Tests for PlotGridTabs initialization."""

    def test_creates_panel_tabs_widget(self, plot_grid_tabs):
        """Test that widget creates a Panel Tabs object."""
        assert isinstance(plot_grid_tabs.tabs, pn.Tabs)

    def test_starts_with_three_tabs_when_no_grids(self, plot_grid_tabs):
        """Test that widget starts with three static tabs when no grids exist."""
        # Should have exactly three tabs (Jobs, Workflows, and Manage Plots)
        assert len(plot_grid_tabs.tabs) == 3

    def test_initializes_from_existing_grids(
        self,
        plot_orchestrator,
        workflow_registry,
        plotting_controller,
        job_status_widget,
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
            job_status_widget=job_status_widget,
            workflow_status_widget=workflow_status_widget,
            plot_data_service=plot_data_service,
            session_updater=session_updater,
        )

        # Should have 5 tabs: Jobs + Workflows + Manage + 2 grids
        assert len(widget.tabs) == 5

    def test_subscribes_to_lifecycle_events(self, plot_orchestrator, plot_grid_tabs):
        """Test that widget subscribes to orchestrator lifecycle events."""
        # Verify subscription by adding a grid and checking it appears
        plot_orchestrator.add_grid(title='New Grid', nrows=3, ncols=3)

        # Should now have 4 tabs: Jobs + Workflows + Manage + New Grid
        assert len(plot_grid_tabs.tabs) == 4


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
        # (Jobs=0, Workflows=1, Manage=2, grid=3)
        assert plot_grid_tabs.tabs.active == 3

    def test_on_grid_removed_removes_tab(self, plot_orchestrator, plot_grid_tabs):
        """Test that removing a grid removes its tab."""
        grid_id = plot_orchestrator.add_grid(title='To Remove', nrows=3, ncols=3)
        assert len(plot_grid_tabs.tabs) == 4  # Jobs + Workflows + Manage + Grid

        plot_orchestrator.remove_grid(grid_id)

        # Should only have three static tabs left (Jobs + Workflows + Manage)
        assert len(plot_grid_tabs.tabs) == 3

    def test_removing_grid_updates_correctly(self, plot_orchestrator, plot_grid_tabs):
        """Test that removing a grid from the middle works correctly."""
        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        grid_id_2 = plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        plot_orchestrator.add_grid(title='Grid 3', nrows=4, ncols=4)

        # Remove middle grid
        plot_orchestrator.remove_grid(grid_id_2)

        # Should have Jobs + Workflows + Manage + 2 remaining grids
        assert len(plot_grid_tabs.tabs) == 5

    def test_multiple_widget_instances_stay_synchronized(
        self,
        plot_orchestrator,
        workflow_registry,
        plotting_controller,
        fake_job_service,
        job_controller,
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
        )
        session_updater2 = SessionUpdater(
            session_id=SessionId('session-2'),
            session_registry=registry,
        )

        # Create separate job status widgets for each instance
        job_status_widget1 = JobStatusListWidget(
            job_service=fake_job_service, job_controller=job_controller
        )
        job_status_widget2 = JobStatusListWidget(
            job_service=fake_job_service, job_controller=job_controller
        )

        # Create separate workflow status widgets for each instance
        workflow_status_widget1 = WorkflowStatusListWidget(
            orchestrator=job_orchestrator,
            job_service=fake_job_service,
        )
        workflow_status_widget2 = WorkflowStatusListWidget(
            orchestrator=job_orchestrator,
            job_service=fake_job_service,
        )

        widget1 = PlotGridTabs(
            plot_orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            job_status_widget=job_status_widget1,
            workflow_status_widget=workflow_status_widget1,
            plot_data_service=plot_data_service,
            session_updater=session_updater1,
        )
        widget2 = PlotGridTabs(
            plot_orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            plotting_controller=plotting_controller,
            job_status_widget=job_status_widget2,
            workflow_status_widget=workflow_status_widget2,
            plot_data_service=plot_data_service,
            session_updater=session_updater2,
        )

        # Add grid via orchestrator
        grid_id = plot_orchestrator.add_grid(title='Shared Grid', nrows=3, ncols=3)

        # Both widgets should have the new tab
        assert len(widget1.tabs) == 4  # Jobs + Workflows + Manage + Shared Grid
        assert len(widget2.tabs) == 4

        # Remove grid
        plot_orchestrator.remove_grid(grid_id)

        # Both widgets should reflect removal
        assert len(widget1.tabs) == 3  # Jobs + Workflows + Manage
        assert len(widget2.tabs) == 3


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
