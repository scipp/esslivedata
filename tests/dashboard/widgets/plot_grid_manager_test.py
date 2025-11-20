# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import holoviews as hv
import panel as pn
import pytest

from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plot_orchestrator import PlotOrchestrator
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.stream_manager import StreamManager
from ess.livedata.dashboard.widgets.plot_grid_manager import PlotGridManager
from tests.dashboard.fakes import FakeJobOrchestrator

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
def grid_manager(plot_orchestrator):
    """Create a PlotGridManager for testing."""
    return PlotGridManager(orchestrator=plot_orchestrator)


class TestPlotGridManagerInitialization:
    """Tests for PlotGridManager initialization."""

    def test_creates_panel_widget(self, grid_manager):
        """Test that manager creates a Panel widget."""
        assert isinstance(grid_manager.panel, pn.Column)


class TestMultipleManagers:
    """Tests for multiple manager instances (multi-user scenario)."""

    def test_multiple_managers_stay_synchronized(self, plot_orchestrator):
        """Test that multiple managers sharing same orchestrator stay in sync."""
        # Create managers which register as callbacks with orchestrator
        _manager1 = PlotGridManager(orchestrator=plot_orchestrator)
        _manager2 = PlotGridManager(orchestrator=plot_orchestrator)

        # Add grid via orchestrator
        grid_id = plot_orchestrator.add_grid(title='Shared Grid', nrows=3, ncols=3)

        # Both managers should react - verify via orchestrator
        assert len(plot_orchestrator.get_all_grids()) == 1
        assert grid_id in plot_orchestrator.get_all_grids()

        # Remove grid via orchestrator
        plot_orchestrator.remove_grid(grid_id)

        # Both managers should reflect removal - verify via orchestrator
        assert len(plot_orchestrator.get_all_grids()) == 0


class TestShutdown:
    """Tests for manager shutdown and cleanup."""

    def test_shutdown_unsubscribes_from_lifecycle(
        self, plot_orchestrator, grid_manager
    ):
        """Test that shutdown unsubscribes from orchestrator lifecycle."""
        # Shutdown the manager
        grid_manager.shutdown()

        # Adding a grid should not cause errors even though manager is shut down
        plot_orchestrator.add_grid(title='After Shutdown', nrows=3, ncols=3)

        # Verify grid was added to orchestrator (manager is just unsubscribed)
        assert len(plot_orchestrator.get_all_grids()) == 1

    def test_shutdown_can_be_called_multiple_times(self, grid_manager):
        """Test that shutdown is idempotent."""
        grid_manager.shutdown()
        grid_manager.shutdown()  # Should not raise
