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
from ess.livedata.dashboard.widgets.plot_grid_manager import PlotGridManager

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
def grid_manager(plot_orchestrator):
    """Create a PlotGridManager for testing."""
    return PlotGridManager(orchestrator=plot_orchestrator)


class TestPlotGridManagerInitialization:
    """Tests for PlotGridManager initialization."""

    def test_creates_panel_widget(self, grid_manager):
        """Test that manager creates a Panel widget."""
        assert isinstance(grid_manager.panel, pn.Column)

    def test_initializes_with_default_values(self, grid_manager):
        """Test that input fields have sensible defaults."""
        # Access private attributes for testing
        assert grid_manager._title_input.value == 'New Grid'
        assert grid_manager._nrows_input.value == 3
        assert grid_manager._ncols_input.value == 3

    def test_subscribes_to_lifecycle_events(self, plot_orchestrator, grid_manager):
        """Test that manager subscribes to orchestrator lifecycle events."""
        # Add a grid and verify the list updates
        plot_orchestrator.add_grid(title='Test Grid', nrows=2, ncols=2)
        # Grid list should have one entry
        assert len(grid_manager._grid_list) == 1


class TestAddGrid:
    """Tests for adding grids through the manager."""

    def test_clicking_add_button_creates_grid(self, plot_orchestrator, grid_manager):
        """Test that clicking Add Grid button creates a new grid."""
        initial_count = len(plot_orchestrator.get_all_grids())

        # Set input values
        grid_manager._title_input.value = 'Custom Grid'
        grid_manager._nrows_input.value = 4
        grid_manager._ncols_input.value = 5

        # Trigger add button click
        grid_manager._add_button.param.trigger('clicks')

        # Verify grid was created
        assert len(plot_orchestrator.get_all_grids()) == initial_count + 1

        # Find the new grid
        grids = plot_orchestrator.get_all_grids()
        new_grid = list(grids.values())[-1]
        assert new_grid.title == 'Custom Grid'
        assert new_grid.nrows == 4
        assert new_grid.ncols == 5

    def test_add_button_resets_inputs(self, plot_orchestrator, grid_manager):
        """Test that adding a grid resets input fields."""
        grid_manager._title_input.value = 'Custom Grid'
        grid_manager._nrows_input.value = 4
        grid_manager._ncols_input.value = 5

        grid_manager._add_button.param.trigger('clicks')

        # Inputs should be reset
        assert grid_manager._title_input.value == 'New Grid'
        assert grid_manager._nrows_input.value == 3
        assert grid_manager._ncols_input.value == 3


class TestGridList:
    """Tests for the grid list display."""

    def test_grid_list_updates_when_grid_added(self, plot_orchestrator, grid_manager):
        """Test that grid list updates when grids are added."""
        assert len(grid_manager._grid_list) == 0

        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        assert len(grid_manager._grid_list) == 1

        plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        assert len(grid_manager._grid_list) == 2

    def test_grid_list_updates_when_grid_removed(self, plot_orchestrator, grid_manager):
        """Test that grid list updates when grids are removed."""
        grid_id = plot_orchestrator.add_grid(title='To Remove', nrows=2, ncols=2)
        assert len(grid_manager._grid_list) == 1

        plot_orchestrator.remove_grid(grid_id)
        assert len(grid_manager._grid_list) == 0

    def test_grid_list_displays_grid_info(self, plot_orchestrator, grid_manager):
        """Test that grid list shows title and dimensions."""
        plot_orchestrator.add_grid(title='Test Grid', nrows=4, ncols=5)

        # Grid list should have one row
        assert len(grid_manager._grid_list) == 1

        # Check that the row contains grid information
        grid_row = grid_manager._grid_list[0]
        assert isinstance(grid_row, pn.Row)
        # First item should be the grid info string
        info_pane = grid_row[0]
        assert isinstance(info_pane, pn.pane.Str)
        assert 'Test Grid' in str(info_pane.object)
        assert '4x5' in str(info_pane.object)


class TestRemoveGrid:
    """Tests for removing grids through the manager."""

    def test_clicking_remove_button_removes_grid(self, plot_orchestrator, grid_manager):
        """Test that clicking remove button removes the grid."""
        plot_orchestrator.add_grid(title='To Remove', nrows=2, ncols=2)
        assert len(plot_orchestrator.get_all_grids()) == 1

        # Find and click the remove button
        grid_row = grid_manager._grid_list[0]
        remove_button = grid_row[2]  # Third item (after info and spacer)
        assert isinstance(remove_button, pn.widgets.Button)
        remove_button.param.trigger('clicks')

        # Grid should be removed
        assert len(plot_orchestrator.get_all_grids()) == 0

    def test_remove_buttons_work_for_multiple_grids(
        self, plot_orchestrator, grid_manager
    ):
        """Test that each grid has a working remove button."""
        plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        grid_id_2 = plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        plot_orchestrator.add_grid(title='Grid 3', nrows=4, ncols=4)

        # Remove second grid
        grid_row = grid_manager._grid_list[1]
        remove_button = grid_row[2]
        remove_button.param.trigger('clicks')

        # Should have 2 grids left
        assert len(plot_orchestrator.get_all_grids()) == 2
        # Grid 2 should be gone
        remaining_grids = plot_orchestrator.get_all_grids()
        assert grid_id_2 not in remaining_grids


class TestMultipleManagers:
    """Tests for multiple manager instances."""

    def test_multiple_managers_stay_synchronized(self, plot_orchestrator):
        """Test that multiple managers sharing same orchestrator stay in sync."""
        manager1 = PlotGridManager(orchestrator=plot_orchestrator)
        manager2 = PlotGridManager(orchestrator=plot_orchestrator)

        # Add grid via orchestrator
        plot_orchestrator.add_grid(title='Shared Grid', nrows=3, ncols=3)

        # Both managers should show the grid
        assert len(manager1._grid_list) == 1
        assert len(manager2._grid_list) == 1

        # Remove grid via orchestrator
        grid_id = next(iter(plot_orchestrator.get_all_grids().keys()))
        plot_orchestrator.remove_grid(grid_id)

        # Both managers should reflect removal
        assert len(manager1._grid_list) == 0
        assert len(manager2._grid_list) == 0


class TestShutdown:
    """Tests for manager shutdown and cleanup."""

    def test_shutdown_unsubscribes_from_lifecycle(
        self, plot_orchestrator, grid_manager
    ):
        """Test that shutdown unsubscribes from orchestrator lifecycle."""
        # Shutdown the manager
        grid_manager.shutdown()

        # Adding a grid should not affect the manager anymore
        initial_list_len = len(grid_manager._grid_list)
        plot_orchestrator.add_grid(title='After Shutdown', nrows=3, ncols=3)

        # Grid list should not update
        assert len(grid_manager._grid_list) == initial_list_len

    def test_shutdown_can_be_called_multiple_times(self, grid_manager):
        """Test that shutdown is idempotent."""
        grid_manager.shutdown()
        grid_manager.shutdown()  # Should not raise
