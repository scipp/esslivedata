# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid
from collections.abc import Callable

import holoviews as hv
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    ResultKey,
    WorkflowId,
)
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plot_orchestrator import (
    PlotCell,
    PlotConfig,
    PlotOrchestrator,
    SubscriptionId,
)
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.stream_manager import StreamManager

hv.extension('bokeh')


class FakeJobOrchestrator:
    """Fake JobOrchestrator for testing PlotOrchestrator."""

    def __init__(self):
        self._subscriptions: dict[SubscriptionId, tuple[WorkflowId, Callable]] = {}

    def subscribe_to_workflow(
        self, workflow_id: WorkflowId, callback: Callable[[JobNumber], None]
    ) -> SubscriptionId:
        """Register a callback for workflow availability."""
        subscription_id = SubscriptionId(uuid.uuid4())
        self._subscriptions[subscription_id] = (workflow_id, callback)
        return subscription_id

    def unsubscribe(self, subscription_id: SubscriptionId) -> None:
        """Unsubscribe from workflow availability notifications."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]

    def simulate_workflow_commit(
        self, workflow_id: WorkflowId, job_number: JobNumber
    ) -> None:
        """Test helper to simulate workflow commit and notify subscribers."""
        for _sub_id, (wf_id, callback) in list(self._subscriptions.items()):
            if wf_id == workflow_id:
                callback(job_number)


@pytest.fixture
def workflow_id():
    """Create a test WorkflowId."""
    return WorkflowId(
        instrument='test_instrument',
        namespace='test_namespace',
        name='test_workflow',
        version=1,
    )


@pytest.fixture
def fake_job_orchestrator():
    """Create a fake JobOrchestrator for testing."""
    return FakeJobOrchestrator()


@pytest.fixture
def job_number() -> JobNumber:
    """Create a test job number."""
    return uuid.uuid4()


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
def plot_orchestrator(plotting_controller, fake_job_orchestrator):
    """Create a PlotOrchestrator for testing."""
    return PlotOrchestrator(
        plotting_controller=plotting_controller,
        job_orchestrator=fake_job_orchestrator,
    )


@pytest.fixture
def detector_data():
    """Create simple 1D detector data for testing."""
    x = sc.arange('x', 10, dtype='float64')
    data = sc.arange('x', 10, dtype='float64')
    return sc.DataArray(data, coords={'x': x})


class TestPlotOrchestrator:
    """Tests for PlotOrchestrator."""

    def test_add_grid(self, plot_orchestrator):
        """Test adding a plot grid."""
        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)
        assert grid_id is not None
        grid = plot_orchestrator.get_grid(grid_id)
        assert grid is not None
        assert grid.title == "Test Grid"
        assert grid.nrows == 3
        assert grid.ncols == 3
        assert len(grid.cells) == 0

    def test_remove_grid(self, plot_orchestrator):
        """Test removing a plot grid."""
        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)
        plot_orchestrator.remove_grid(grid_id)
        assert plot_orchestrator.get_grid(grid_id) is None

    def test_add_plot_without_job(self, plot_orchestrator, workflow_id):
        """Test adding a plot configuration when no matching job exists."""
        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)

        plot_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='intensity',
            source_names=['detector_1'],
            plot_name='lines',
            params={},
        )

        cell = PlotCell(
            row=0,
            col=0,
            row_span=1,
            col_span=1,
            config=plot_config,
        )

        cell_id = plot_orchestrator.add_plot(grid_id, cell)

        assert cell_id is not None
        grid = plot_orchestrator.get_grid(grid_id)
        assert len(grid.cells) == 1
        added_cell = grid.cells[cell_id]
        assert added_cell.config == plot_config

    def test_add_plot_with_matching_job(
        self,
        plot_orchestrator,
        fake_job_orchestrator,
        job_service,
        data_service,
        plotting_controller,
        workflow_id,
        job_number,
        detector_data,
    ):
        """Test adding a plot and receiving workflow commit notification."""
        job_id = JobId(source_name='detector_1', job_number=job_number)
        result_key = ResultKey(
            workflow_id=workflow_id,
            job_id=job_id,
            output_name='intensity',
        )

        # Adding data to data_service automatically registers the job
        # via data_updated callback
        data_service[result_key] = detector_data

        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)

        plot_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='intensity',
            source_names=['detector_1'],
            plot_name='lines',
            params={},
        )

        # Subscribe to lifecycle events
        received_plots = []

        def on_cell_updated(gid, c, plot=None, error=None):
            received_plots.append((gid, c, plot, error))

        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=on_cell_updated)

        cell = PlotCell(
            row=0,
            col=0,
            row_span=1,
            col_span=1,
            config=plot_config,
        )

        plot_orchestrator.add_plot(grid_id, cell)

        # Simulate workflow commit from JobOrchestrator
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

        # Verify plot was created and callback was called (twice: once on add,
        # once on plot ready)
        assert len(received_plots) == 2
        # First call: cell added, no plot yet
        assert received_plots[0][2] is None  # plot
        assert received_plots[0][3] is None  # error
        # Second call: plot ready
        assert received_plots[1][2] is not None  # plot
        assert isinstance(received_plots[1][2], hv.DynamicMap)
        assert received_plots[1][3] is None  # error

    def test_workflow_commit_triggers_plot_creation(
        self,
        plot_orchestrator,
        fake_job_orchestrator,
        job_service,
        data_service,
        plotting_controller,
        workflow_id,
        job_number,
        detector_data,
    ):
        """Test that workflow commit triggers plot creation for waiting plots."""
        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)

        plot_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='intensity',
            source_names=['detector_1'],
            plot_name='lines',
            params={},
        )

        # Subscribe to lifecycle events
        received_plots = []

        def on_cell_updated(gid, c, plot=None, error=None):
            received_plots.append((gid, c, plot, error))

        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=on_cell_updated)

        cell = PlotCell(
            row=0,
            col=0,
            row_span=1,
            col_span=1,
            config=plot_config,
        )

        plot_orchestrator.add_plot(grid_id, cell)

        # Add data to data service
        job_id = JobId(source_name='detector_1', job_number=job_number)
        result_key = ResultKey(
            workflow_id=workflow_id,
            job_id=job_id,
            output_name='intensity',
        )
        data_service[result_key] = detector_data

        # Simulate workflow commit
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

        # Plot should now be created (2 calls: once on add, once on plot ready)
        assert len(received_plots) == 2
        # First call: cell added, no plot yet
        assert received_plots[0][2] is None
        # Second call: plot ready
        assert received_plots[1][2] is not None
        assert isinstance(received_plots[1][2], hv.DynamicMap)
        assert received_plots[1][3] is None

    def test_remove_plot(self, plot_orchestrator, workflow_id):
        """Test removing a plot from a grid."""
        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)

        plot_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='intensity',
            source_names=['detector_1'],
            plot_name='lines',
            params={},
        )

        cell = PlotCell(
            row=0,
            col=0,
            row_span=1,
            col_span=1,
            config=plot_config,
        )

        cell_id = plot_orchestrator.add_plot(grid_id, cell)
        grid = plot_orchestrator.get_grid(grid_id)
        assert len(grid.cells) == 1

        plot_orchestrator.remove_plot(cell_id)
        grid = plot_orchestrator.get_grid(grid_id)
        assert len(grid.cells) == 0

    def test_get_all_grids(self, plot_orchestrator):
        """Test getting all grids."""
        grid_id_1 = plot_orchestrator.add_grid(title="Grid 1", nrows=2, ncols=2)
        grid_id_2 = plot_orchestrator.add_grid(title="Grid 2", nrows=3, ncols=3)

        all_grids = plot_orchestrator.get_all_grids()
        assert len(all_grids) == 2
        assert grid_id_1 in all_grids
        assert grid_id_2 in all_grids
        assert all_grids[grid_id_1].title == "Grid 1"
        assert all_grids[grid_id_2].title == "Grid 2"

    def test_get_plot_config(self, plot_orchestrator, workflow_id):
        """Test getting plot configuration by cell ID."""
        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)

        plot_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='intensity',
            source_names=['detector_1'],
            plot_name='lines',
            params={'window': 10},
        )

        cell = PlotCell(
            row=0,
            col=0,
            row_span=1,
            col_span=1,
            config=plot_config,
        )

        cell_id = plot_orchestrator.add_plot(grid_id, cell)

        retrieved_config = plot_orchestrator.get_plot_config(cell_id)
        assert retrieved_config == plot_config
        assert retrieved_config.workflow_id == workflow_id
        assert retrieved_config.output_name == 'intensity'
        assert retrieved_config.params == {'window': 10}

    def test_update_plot_config(
        self,
        plot_orchestrator,
        fake_job_orchestrator,
        job_service,
        data_service,
        plotting_controller,
        workflow_id,
        job_number,
        detector_data,
    ):
        """Test updating plot configuration."""
        # Set up job and data
        job_id = JobId(source_name='detector_1', job_number=job_number)
        result_key = ResultKey(
            workflow_id=workflow_id,
            job_id=job_id,
            output_name='intensity',
        )
        data_service[result_key] = detector_data

        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)

        # Subscribe to lifecycle events
        received_plots = []

        def on_cell_updated(gid, c, plot=None, error=None):
            received_plots.append((gid, c, plot, error))

        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=on_cell_updated)

        # Add initial plot
        plot_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='intensity',
            source_names=['detector_1'],
            plot_name='lines',
            params={},
        )

        cell = PlotCell(
            row=0,
            col=0,
            row_span=1,
            col_span=1,
            config=plot_config,
        )

        cell_id = plot_orchestrator.add_plot(grid_id, cell)

        # Trigger workflow commit to create initial plot
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

        # Verify initial plot was created (2 events: add and plot ready)
        assert len(received_plots) == 2
        assert received_plots[1][2] is not None  # plot ready

        # Update configuration with same workflow but different output
        new_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='spectrum',  # Changed output
            source_names=['detector_1'],
            plot_name='lines',
            params={},
        )

        plot_orchestrator.update_plot_config(cell_id, new_config)

        # Verify config was updated (3rd event: cell updated with new config)
        assert len(received_plots) == 3
        updated_config = plot_orchestrator.get_plot_config(cell_id)
        assert updated_config.output_name == 'spectrum'
        assert updated_config.source_names == ['detector_1']

    def test_remove_plot_unsubscribes_from_workflow(
        self, plot_orchestrator, fake_job_orchestrator, workflow_id, job_number
    ):
        """Test that removing a plot unsubscribes from workflow notifications."""
        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)

        plot_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='intensity',
            source_names=['detector_1'],
            plot_name='lines',
            params={},
        )

        cell = PlotCell(
            row=0,
            col=0,
            row_span=1,
            col_span=1,
            config=plot_config,
        )

        cell_id = plot_orchestrator.add_plot(grid_id, cell)

        # Verify subscription was created
        assert len(fake_job_orchestrator._subscriptions) == 1

        # Remove the plot
        plot_orchestrator.remove_plot(cell_id)

        # Verify subscription was removed
        assert len(fake_job_orchestrator._subscriptions) == 0

        # Verify workflow commit doesn't trigger callback for removed plot
        # This should not raise any errors
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

    def test_update_plot_config_resubscribes(
        self, plot_orchestrator, fake_job_orchestrator, workflow_id
    ):
        """Test updating plot config unsubscribes old and creates new subscription."""
        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)

        plot_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='intensity',
            source_names=['detector_1'],
            plot_name='lines',
            params={},
        )

        cell = PlotCell(
            row=0,
            col=0,
            row_span=1,
            col_span=1,
            config=plot_config,
        )

        cell_id = plot_orchestrator.add_plot(grid_id, cell)

        # Get initial subscription ID from orchestrator's tracking dict
        initial_sub_id = plot_orchestrator._cell_to_subscription[cell_id]
        assert initial_sub_id is not None

        # Verify one subscription exists
        assert len(fake_job_orchestrator._subscriptions) == 1

        # Update configuration
        new_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='spectrum',
            source_names=['detector_1'],
            plot_name='lines',
            params={},
        )

        plot_orchestrator.update_plot_config(cell_id, new_config)

        # Get new subscription ID from tracking dict
        new_sub_id = plot_orchestrator._cell_to_subscription[cell_id]
        assert new_sub_id is not None

        # Verify subscription ID changed
        assert new_sub_id != initial_sub_id

        # Verify still only one subscription (old was removed, new was added)
        assert len(fake_job_orchestrator._subscriptions) == 1

        # Verify the old subscription was removed
        assert initial_sub_id not in fake_job_orchestrator._subscriptions

        # Verify the new subscription exists
        assert new_sub_id in fake_job_orchestrator._subscriptions

    def test_remove_grid_unsubscribes_all_plots(
        self, plot_orchestrator, fake_job_orchestrator, workflow_id, job_number
    ):
        """Test that removing a grid unsubscribes all plots in that grid."""
        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)

        # Add multiple plots to the grid
        plot_configs = [
            PlotConfig(
                workflow_id=workflow_id,
                output_name=f'output_{i}',
                source_names=['detector_1'],
                plot_name='lines',
                params={},
            )
            for i in range(3)
        ]

        cell_ids = []
        for i, config in enumerate(plot_configs):
            cell = PlotCell(
                row=i,
                col=0,
                row_span=1,
                col_span=1,
                config=config,
            )
            cell_id = plot_orchestrator.add_plot(grid_id, cell)
            cell_ids.append(cell_id)

        # Verify 3 subscriptions were created
        assert len(fake_job_orchestrator._subscriptions) == 3

        # Remove the entire grid
        plot_orchestrator.remove_grid(grid_id)

        # Verify all subscriptions were removed
        assert len(fake_job_orchestrator._subscriptions) == 0

        # Verify grid is gone
        assert plot_orchestrator.get_grid(grid_id) is None

        # Verify workflow commits don't trigger callbacks for removed plots
        # This should not raise any errors
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

    def test_shutdown_unsubscribes_all_plots_in_all_grids(
        self, plot_orchestrator, fake_job_orchestrator, workflow_id, job_number
    ):
        """Test that shutdown unsubscribes all plots across all grids."""
        # Create two grids with multiple plots each
        grid_id_1 = plot_orchestrator.add_grid(title="Grid 1", nrows=2, ncols=2)
        grid_id_2 = plot_orchestrator.add_grid(title="Grid 2", nrows=2, ncols=2)

        # Add 2 plots to grid 1
        for i in range(2):
            cell = PlotCell(
                row=i,
                col=0,
                row_span=1,
                col_span=1,
                config=PlotConfig(
                    workflow_id=workflow_id,
                    output_name=f'output_1_{i}',
                    source_names=['detector_1'],
                    plot_name='lines',
                    params={},
                ),
            )
            plot_orchestrator.add_plot(grid_id_1, cell)

        # Add 3 plots to grid 2
        for i in range(3):
            cell = PlotCell(
                row=i,
                col=0,
                row_span=1,
                col_span=1,
                config=PlotConfig(
                    workflow_id=workflow_id,
                    output_name=f'output_2_{i}',
                    source_names=['detector_1'],
                    plot_name='lines',
                    params={},
                ),
            )
            plot_orchestrator.add_plot(grid_id_2, cell)

        # Verify 5 total subscriptions (2 + 3)
        assert len(fake_job_orchestrator._subscriptions) == 5

        # Shutdown the orchestrator
        plot_orchestrator.shutdown()

        # Verify all subscriptions were removed
        assert len(fake_job_orchestrator._subscriptions) == 0

        # Verify workflow commits don't trigger callbacks
        # This should not raise any errors
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

    def test_lifecycle_subscriptions(self, plot_orchestrator, workflow_id):
        """Test lifecycle event subscriptions."""
        # Track lifecycle events
        events = []

        def on_grid_created(grid_id, grid_config):
            events.append(('grid_created', grid_id, grid_config))

        def on_grid_removed(grid_id):
            events.append(('grid_removed', grid_id))

        def on_cell_updated(grid_id, cell, plot=None, error=None):
            events.append(('cell_updated', grid_id, cell.row, cell.col, plot, error))

        def on_cell_removed(grid_id, cell):
            events.append(('cell_removed', grid_id, cell.row, cell.col))

        # Subscribe to lifecycle events
        subscription_id = plot_orchestrator.subscribe_to_lifecycle(
            on_grid_created=on_grid_created,
            on_grid_removed=on_grid_removed,
            on_cell_updated=on_cell_updated,
            on_cell_removed=on_cell_removed,
        )

        # Add a grid - should trigger grid_created
        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)
        assert len(events) == 1
        assert events[0][0] == 'grid_created'
        assert events[0][1] == grid_id
        assert events[0][2].title == "Test Grid"
        assert events[0][2].nrows == 3
        assert events[0][2].ncols == 3

        # Add a plot - should trigger cell_updated (no plot yet)
        cell = PlotCell(
            row=1,
            col=2,
            row_span=2,
            col_span=3,
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='intensity',
                source_names=['detector_1'],
                plot_name='lines',
                params={},
            ),
        )
        cell_id = plot_orchestrator.add_plot(grid_id, cell)
        assert len(events) == 2
        assert events[1] == ('cell_updated', grid_id, 1, 2, None, None)

        # Update plot config - should trigger cell_updated (no plot yet)
        new_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='spectrum',
            source_names=['detector_1'],
            plot_name='lines',
            params={},
        )
        plot_orchestrator.update_plot_config(cell_id, new_config)
        assert len(events) == 3
        assert events[2] == ('cell_updated', grid_id, 1, 2, None, None)

        # Remove the plot - should trigger cell_removed
        plot_orchestrator.remove_plot(cell_id)
        assert len(events) == 4
        assert events[3] == ('cell_removed', grid_id, 1, 2)

        # Remove the grid - should trigger grid_removed
        plot_orchestrator.remove_grid(grid_id)
        assert len(events) == 5
        assert events[4] == ('grid_removed', grid_id)

        # Unsubscribe
        plot_orchestrator.unsubscribe_from_lifecycle(subscription_id)

        # Further operations should not trigger events
        plot_orchestrator.add_grid(title="Another Grid", nrows=2, ncols=2)
        assert len(events) == 5  # No new events

    def test_partial_lifecycle_subscriptions(self, plot_orchestrator, workflow_id):
        """Test subscribing to only some lifecycle events."""
        # Track only cell events
        cell_events = []

        def on_cell_updated(grid_id, cell, plot=None, error=None):
            cell_events.append(('updated', grid_id, cell, plot, error))

        def on_cell_removed(grid_id, cell):
            cell_events.append(('removed', grid_id, cell))

        # Subscribe only to cell events
        plot_orchestrator.subscribe_to_lifecycle(
            on_cell_updated=on_cell_updated,
            on_cell_removed=on_cell_removed,
        )

        # Add grid - should not trigger anything (not subscribed to grid events)
        grid_id = plot_orchestrator.add_grid(title="Test Grid", nrows=3, ncols=3)
        assert len(cell_events) == 0

        # Add cell - should trigger cell_updated
        cell = PlotCell(
            row=0,
            col=0,
            row_span=1,
            col_span=1,
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='intensity',
                source_names=['detector_1'],
                plot_name='lines',
                params={},
            ),
        )
        cell_id = plot_orchestrator.add_plot(grid_id, cell)
        assert len(cell_events) == 1
        assert cell_events[0][0] == 'updated'
        assert cell_events[0][2].row == 0
        assert cell_events[0][2].col == 0
        assert cell_events[0][3] is None  # plot
        assert cell_events[0][4] is None  # error

        # Remove cell - should trigger cell_removed
        plot_orchestrator.remove_plot(cell_id)
        assert len(cell_events) == 2
        assert cell_events[1][0] == 'removed'
        assert cell_events[1][2].row == 0
        assert cell_events[1][2].col == 0
