# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid
from collections.abc import Callable

import pytest

from ess.livedata.config.workflow_spec import JobNumber, WorkflowId
from ess.livedata.dashboard.plot_orchestrator import (
    GridId,
    PlotCell,
    PlotConfig,
    PlotGridConfig,
    PlotOrchestrator,
    SubscriptionId,
)


class FakePlot:
    """Dummy plot object for testing."""

    pass


class CallbackCapture:
    """Captures callback invocations for testing."""

    def __init__(self, side_effect: BaseException | None = None):
        """Initialize callback capture.

        Parameters
        ----------
        side_effect:
            Optional exception to raise when called.
        """
        self._calls: list[tuple] = []
        self._side_effect = side_effect

    def __call__(self, *args, **kwargs) -> None:
        """Record call and optionally raise exception."""
        self._calls.append((args, kwargs))
        if self._side_effect is not None:
            raise self._side_effect

    @property
    def call_count(self) -> int:
        """Return the number of calls."""
        return len(self._calls)

    @property
    def call_args(self) -> tuple:
        """Return the arguments of the last call.

        Returns a tuple of (args, kwargs) tuples for Mock-like compatibility.
        """
        if not self._calls:
            raise AssertionError("No calls recorded")
        args, kwargs = self._calls[-1]
        # Return in Mock-compatible format: call_args[0] gets positional args
        return (args, kwargs)

    def assert_called_once(self) -> None:
        """Assert that callback was called exactly once."""
        assert self.call_count == 1, f"Expected 1 call, got {self.call_count}"

    def assert_called_once_with(self, *args, **kwargs) -> None:
        """Assert that callback was called once with specific arguments."""
        self.assert_called_once()
        last_args, last_kwargs = self._calls[-1]
        assert last_args == args, f"Expected args {args}, got {last_args}"
        assert last_kwargs == kwargs, f"Expected kwargs {kwargs}, got {last_kwargs}"

    def assert_called_with(self, *args, **kwargs) -> None:
        """Assert that callback was called with specific arguments."""
        if not any(
            call_args == args and call_kwargs == kwargs
            for call_args, call_kwargs in self._calls
        ):
            raise AssertionError(
                f"Not called with args {args}, kwargs {kwargs}. Calls: {self._calls}"
            )

    def assert_not_called(self) -> None:
        """Assert that callback was not called."""
        assert self.call_count == 0, f"Expected 0 calls, got {self.call_count}"


class FakeJobOrchestrator:
    """Fake JobOrchestrator for testing."""

    def __init__(self):
        self._subscriptions: dict[SubscriptionId, Callable[[JobNumber], None]] = {}
        self._workflow_subscriptions: dict[WorkflowId, set[SubscriptionId]] = {}

    def subscribe_to_workflow(
        self, workflow_id: WorkflowId, callback: Callable[[JobNumber], None]
    ) -> SubscriptionId:
        """Subscribe to workflow availability notifications."""
        subscription_id = SubscriptionId(uuid.uuid4())
        self._subscriptions[subscription_id] = callback

        # Track which workflows have subscriptions
        if workflow_id not in self._workflow_subscriptions:
            self._workflow_subscriptions[workflow_id] = set()
        self._workflow_subscriptions[workflow_id].add(subscription_id)

        return subscription_id

    def unsubscribe(self, subscription_id: SubscriptionId) -> None:
        """Unsubscribe from workflow availability notifications."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            # Remove from workflow tracking
            for workflow_subs in self._workflow_subscriptions.values():
                workflow_subs.discard(subscription_id)

    def simulate_workflow_commit(
        self, workflow_id: WorkflowId, job_number: JobNumber
    ) -> None:
        """Simulate a workflow commit by calling all callbacks for that workflow."""
        if workflow_id not in self._workflow_subscriptions:
            return

        for subscription_id in self._workflow_subscriptions[workflow_id]:
            if subscription_id in self._subscriptions:
                self._subscriptions[subscription_id](job_number)

    @property
    def subscription_count(self) -> int:
        """Return the total number of active subscriptions."""
        return len(self._subscriptions)


class FakePlottingController:
    """Fake PlottingController for testing."""

    def __init__(self):
        self._should_raise = False
        self._exception_to_raise = None
        self._plot_object = FakePlot()
        self._calls: list[dict] = []

    def create_plot(
        self,
        job_number: JobNumber,
        source_names: list[str],
        output_name: str | None,
        plot_name: str,
        params: dict,
    ):
        """Create a fake plot or raise an exception if configured to do so."""
        self._calls.append(
            {
                'job_number': job_number,
                'source_names': source_names,
                'output_name': output_name,
                'plot_name': plot_name,
                'params': params,
            }
        )
        if self._should_raise:
            raise self._exception_to_raise
        return self._plot_object

    def configure_to_raise(self, exception: Exception) -> None:
        """Configure the controller to raise an exception on create_plot."""
        self._should_raise = True
        self._exception_to_raise = exception

    def reset(self) -> None:
        """Reset the controller to normal behavior."""
        self._should_raise = False
        self._exception_to_raise = None
        self._calls.clear()

    def call_count(self) -> int:
        """Return the number of create_plot calls."""
        return len(self._calls)

    def get_calls(self) -> list[dict]:
        """Return all recorded create_plot calls."""
        return self._calls.copy()


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
def workflow_id_2():
    """Create a second test WorkflowId."""
    return WorkflowId(
        instrument='test_instrument',
        namespace='test_namespace',
        name='test_workflow_2',
        version=1,
    )


@pytest.fixture
def job_number():
    """Create a test job number."""
    return uuid.uuid4()


@pytest.fixture
def plot_config(workflow_id):
    """Create a basic PlotConfig."""
    return PlotConfig(
        workflow_id=workflow_id,
        output_name='test_output',
        source_names=['source1', 'source2'],
        plot_name='test_plot',
        params={'param1': 'value1'},
    )


@pytest.fixture
def plot_cell(plot_config):
    """Create a basic PlotCell."""
    return PlotCell(
        row=0,
        col=0,
        row_span=1,
        col_span=1,
        config=plot_config,
    )


@pytest.fixture
def fake_job_orchestrator():
    """Create a FakeJobOrchestrator."""
    return FakeJobOrchestrator()


@pytest.fixture
def fake_plotting_controller():
    """Create a FakePlottingController."""
    return FakePlottingController()


@pytest.fixture
def plot_orchestrator(fake_job_orchestrator, fake_plotting_controller):
    """Create a PlotOrchestrator with fakes."""
    return PlotOrchestrator(
        plotting_controller=fake_plotting_controller,
        job_orchestrator=fake_job_orchestrator,
    )


class TestGridManagement:
    """Tests for basic grid creation and removal."""

    def test_add_grid_creates_retrievable_grid(self, plot_orchestrator):
        """Add grid creates retrievable grid with correct config."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=4)

        grid = plot_orchestrator.get_grid(grid_id)
        assert grid is not None
        assert grid.title == 'Test Grid'
        assert grid.nrows == 3
        assert grid.ncols == 4
        assert len(grid.cells) == 0

    def test_remove_grid_makes_it_unretrievable(self, plot_orchestrator):
        """Remove grid makes it unretrievable."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        plot_orchestrator.remove_grid(grid_id)

        assert plot_orchestrator.get_grid(grid_id) is None

    def test_get_all_grids_returns_all_added_grids(self, plot_orchestrator):
        """Get all grids returns all added grids."""
        grid_id_1 = plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        grid_id_2 = plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)

        all_grids = plot_orchestrator.get_all_grids()

        assert len(all_grids) == 2
        assert grid_id_1 in all_grids
        assert grid_id_2 in all_grids
        assert all_grids[grid_id_1].title == 'Grid 1'
        assert all_grids[grid_id_2].title == 'Grid 2'

    def test_get_non_existent_grid_returns_none(self, plot_orchestrator):
        """Get non-existent grid returns None."""
        fake_grid_id = GridId(uuid.uuid4())
        assert plot_orchestrator.get_grid(fake_grid_id) is None

    def test_get_grid_returns_copy_that_can_be_modified_safely(
        self, plot_orchestrator, plot_cell
    ):
        """Modifying returned grid from get_grid does not affect internal state."""
        grid_id = plot_orchestrator.add_grid(title='Original Title', nrows=3, ncols=3)
        plot_orchestrator.add_plot(grid_id, plot_cell)

        # Get grid and modify it
        grid = plot_orchestrator.get_grid(grid_id)
        grid.title = 'Modified Title'
        grid.nrows = 99
        grid.ncols = 99

        # Verify internal state unchanged
        internal_grid = plot_orchestrator.get_grid(grid_id)
        assert internal_grid.title == 'Original Title'
        assert internal_grid.nrows == 3
        assert internal_grid.ncols == 3

    def test_get_grid_returns_copy_with_cells_that_can_be_modified_safely(
        self, plot_orchestrator, plot_cell, plot_config
    ):
        """Modifying cells in returned grid from get_grid does not affect internal
        state."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Get grid and modify a cell
        grid = plot_orchestrator.get_grid(grid_id)
        grid.cells[cell_id].config.params['new_param'] = 'new_value'
        grid.cells[cell_id].row = 99

        # Verify internal state unchanged
        internal_grid = plot_orchestrator.get_grid(grid_id)
        assert 'new_param' not in internal_grid.cells[cell_id].config.params
        assert internal_grid.cells[cell_id].row == 0

    def test_get_all_grids_returns_copy_that_can_be_modified_safely(
        self, plot_orchestrator
    ):
        """Modifying returned dict from get_all_grids does not affect internal state."""
        grid_id_1 = plot_orchestrator.add_grid(title='Grid 1', nrows=2, ncols=2)
        grid_id_2 = plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)

        # Get all grids and modify the dict and a grid
        all_grids = plot_orchestrator.get_all_grids()
        all_grids[grid_id_1].title = 'Modified Grid 1'
        del all_grids[grid_id_2]

        # Verify internal state unchanged
        internal_grids = plot_orchestrator.get_all_grids()
        assert len(internal_grids) == 2
        assert internal_grids[grid_id_1].title == 'Grid 1'
        assert grid_id_2 in internal_grids

    def test_get_all_grids_returns_copy_with_cells_that_can_be_modified_safely(
        self, plot_orchestrator, plot_cell
    ):
        """Modifying cells in grids from get_all_grids does not affect internal
        state."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Get all grids and modify a cell
        all_grids = plot_orchestrator.get_all_grids()
        all_grids[grid_id].cells[cell_id].config.params['new_param'] = 'new_value'
        all_grids[grid_id].cells[cell_id].row = 99

        # Verify internal state unchanged
        internal_grid = plot_orchestrator.get_grid(grid_id)
        assert 'new_param' not in internal_grid.cells[cell_id].config.params
        assert internal_grid.cells[cell_id].row == 0


class TestCellManagement:
    """Tests for cell operations without triggering plot creation."""

    def test_add_cell_to_grid_makes_it_retrievable_in_grid_config(
        self, plot_orchestrator, plot_cell
    ):
        """Add cell to grid makes it retrievable in grid config."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        grid = plot_orchestrator.get_grid(grid_id)
        assert cell_id in grid.cells
        assert grid.cells[cell_id] == plot_cell

    def test_remove_cell_from_grid_removes_it_from_grid_config(
        self, plot_orchestrator, plot_cell
    ):
        """Remove cell from grid removes it from grid config."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        plot_orchestrator.remove_plot(cell_id)

        grid = plot_orchestrator.get_grid(grid_id)
        assert cell_id not in grid.cells

    def test_get_plot_config_returns_correct_config(
        self, plot_orchestrator, plot_cell, plot_config
    ):
        """Get plot config returns correct config."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        retrieved_config = plot_orchestrator.get_plot_config(cell_id)

        assert retrieved_config == plot_config

    def test_update_plot_config_changes_the_config(
        self, plot_orchestrator, plot_cell, plot_config, workflow_id
    ):
        """Update plot config changes the config."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        new_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='new_output',
            source_names=['new_source'],
            plot_name='new_plot',
            params={'new_param': 'new_value'},
        )
        plot_orchestrator.update_plot_config(cell_id, new_config)

        retrieved_config = plot_orchestrator.get_plot_config(cell_id)
        assert retrieved_config == new_config
        assert retrieved_config.output_name == 'new_output'

    def test_add_multiple_cells_to_same_grid(self, plot_orchestrator, workflow_id):
        """Add multiple cells to same grid."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        cell_1 = PlotCell(
            row=0,
            col=0,
            row_span=1,
            col_span=1,
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='out1',
                source_names=['src1'],
                plot_name='plot1',
                params={},
            ),
        )
        cell_2 = PlotCell(
            row=1,
            col=1,
            row_span=1,
            col_span=1,
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='out2',
                source_names=['src2'],
                plot_name='plot2',
                params={},
            ),
        )

        cell_id_1 = plot_orchestrator.add_plot(grid_id, cell_1)
        cell_id_2 = plot_orchestrator.add_plot(grid_id, cell_2)

        grid = plot_orchestrator.get_grid(grid_id)
        assert len(grid.cells) == 2
        assert cell_id_1 in grid.cells
        assert cell_id_2 in grid.cells

    def test_add_cells_to_multiple_grids(self, plot_orchestrator, plot_cell):
        """Add cells to multiple grids."""
        grid_id_1 = plot_orchestrator.add_grid(title='Grid 1', nrows=3, ncols=3)
        grid_id_2 = plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)

        cell_id_1 = plot_orchestrator.add_plot(grid_id_1, plot_cell)
        cell_id_2 = plot_orchestrator.add_plot(grid_id_2, plot_cell)

        grid_1 = plot_orchestrator.get_grid(grid_id_1)
        grid_2 = plot_orchestrator.get_grid(grid_id_2)
        assert cell_id_1 in grid_1.cells
        assert cell_id_2 in grid_2.cells

    def test_add_cell_subscribes_to_workflow(
        self, plot_orchestrator, plot_cell, fake_job_orchestrator
    ):
        """Each add should subscribe appropriately."""
        initial_count = fake_job_orchestrator.subscription_count
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        plot_orchestrator.add_plot(grid_id, plot_cell)

        assert fake_job_orchestrator.subscription_count == initial_count + 1

    def test_remove_cell_unsubscribes_from_workflow(
        self, plot_orchestrator, plot_cell, fake_job_orchestrator
    ):
        """Remove cell should unsubscribe."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)
        count_after_add = fake_job_orchestrator.subscription_count

        plot_orchestrator.remove_plot(cell_id)

        assert fake_job_orchestrator.subscription_count == count_after_add - 1

    def test_update_config_resubscribes(
        self, plot_orchestrator, plot_cell, workflow_id, fake_job_orchestrator
    ):
        """Update config should unsubscribe and resubscribe."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)
        count_after_add = fake_job_orchestrator.subscription_count

        new_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='new_output',
            source_names=['new_source'],
            plot_name='new_plot',
            params={},
        )
        plot_orchestrator.update_plot_config(cell_id, new_config)

        # Should still have same number of subscriptions (unsubscribe + resubscribe)
        assert fake_job_orchestrator.subscription_count == count_after_add


class TestWorkflowIntegrationAndPlotCreationTiming:
    """Tests for workflow integration and plot creation timing."""

    def test_workflow_commit_after_cell_added_creates_plot(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        job_number,
        fake_job_orchestrator,
        fake_plotting_controller,
    ):
        """Workflow commit AFTER cell added should create plot."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        _ = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Simulate workflow commit
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

        # Verify plot was created
        assert fake_plotting_controller.call_count() == 1
        calls = fake_plotting_controller.get_calls()
        assert calls[0]['job_number'] == job_number
        assert calls[0]['source_names'] == plot_cell.config.source_names
        assert calls[0]['output_name'] == plot_cell.config.output_name
        assert calls[0]['plot_name'] == plot_cell.config.plot_name

    def test_workflow_commit_before_cell_added_creates_plot_when_cell_added(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        job_number,
        fake_job_orchestrator,
        fake_plotting_controller,
    ):
        """Adding cell subscribes to workflow, then workflow commit creates plot."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        # Add cell (subscribes to workflow)
        _ = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Commit workflow - should trigger plot creation
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

        # Verify plot was created
        assert fake_plotting_controller.call_count() == 1

    def test_multiple_cells_subscribed_to_same_workflow_all_receive_plots(
        self,
        plot_orchestrator,
        workflow_id,
        job_number,
        fake_job_orchestrator,
        fake_plotting_controller,
    ):
        """Multiple cells subscribed to same workflow should all receive plots."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        # Add multiple cells with same workflow_id
        cell_1 = PlotCell(
            row=0,
            col=0,
            row_span=1,
            col_span=1,
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='out1',
                source_names=['src1'],
                plot_name='plot1',
                params={},
            ),
        )
        cell_2 = PlotCell(
            row=1,
            col=1,
            row_span=1,
            col_span=1,
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='out2',
                source_names=['src2'],
                plot_name='plot2',
                params={},
            ),
        )

        plot_orchestrator.add_plot(grid_id, cell_1)
        plot_orchestrator.add_plot(grid_id, cell_2)

        # Both should be subscribed
        assert fake_job_orchestrator.subscription_count == 2

        # Commit workflow - both should receive notification
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

        # Both plots should be created
        assert fake_plotting_controller.call_count() == 2
        calls = fake_plotting_controller.get_calls()
        assert {c['plot_name'] for c in calls} == {'plot1', 'plot2'}

    def test_cell_removed_before_workflow_commit_no_plot_created(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        job_number,
        fake_job_orchestrator,
        fake_plotting_controller,
    ):
        """Cell removed before workflow commit should not create plot."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Remove cell before workflow commits
        plot_orchestrator.remove_plot(cell_id)

        # Commit workflow - should not create plot
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

        # Verify no plot was created
        assert fake_plotting_controller.call_count() == 0

    def test_update_config_resubscribes_and_new_workflow_commit_creates_plot(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_id_2,
        job_number,
        fake_job_orchestrator,
        fake_plotting_controller,
    ):
        """
        Update config resubscribes to new workflow and creates plot.

        When config is updated with new workflow, the new commit creates plot.
        """
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Update to different workflow_id
        new_config = PlotConfig(
            workflow_id=workflow_id_2,
            output_name='new_output',
            source_names=['new_source'],
            plot_name='new_plot',
            params={},
        )
        plot_orchestrator.update_plot_config(cell_id, new_config)

        # Commit old workflow - should not create plot
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)
        assert fake_plotting_controller.call_count() == 0

        # Commit new workflow - should create plot
        fake_job_orchestrator.simulate_workflow_commit(workflow_id_2, job_number)
        assert fake_plotting_controller.call_count() == 1
        calls = fake_plotting_controller.get_calls()
        assert calls[0]['plot_name'] == 'new_plot'


class TestLifecycleEventNotifications:
    """Tests for lifecycle event notifications."""

    def test_on_grid_created_called_with_correct_grid_id_and_config(
        self, plot_orchestrator
    ):
        """on_grid_created called with correct grid_id and config."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_grid_created=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=4)

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == grid_id
        assert isinstance(call_args[1], PlotGridConfig)
        assert call_args[1].title == 'Test Grid'
        assert call_args[1].nrows == 3
        assert call_args[1].ncols == 4

    def test_on_grid_removed_called_when_grid_removed(self, plot_orchestrator):
        """on_grid_removed called when grid removed."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_grid_removed=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        plot_orchestrator.remove_grid(grid_id)

        callback.assert_called_once_with(grid_id)

    def test_on_cell_updated_called_when_cell_added_no_plot_yet(
        self, plot_orchestrator, plot_cell
    ):
        """on_cell_updated called when cell added (no plot yet)."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        plot_orchestrator.add_plot(grid_id, plot_cell)

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == grid_id
        assert call_args[1] == plot_cell
        assert call_args[2] is None  # No plot yet
        assert len(callback.call_args[0]) == 4
        # Fourth argument is error, should be None
        assert call_args[3] is None

    def test_on_cell_updated_called_when_plot_created_with_plot_object(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        job_number,
        fake_job_orchestrator,
        fake_plotting_controller,
    ):
        """on_cell_updated called when plot created (with plot object)."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        plot_orchestrator.add_plot(grid_id, plot_cell)

        # Should have been called once when cell was added
        assert callback.call_count == 1

        # Simulate workflow commit
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

        # Should be called again with plot object
        assert callback.call_count == 2
        call_args = callback.call_args[0]
        assert call_args[0] == grid_id
        assert call_args[1] == plot_cell
        assert call_args[2] == fake_plotting_controller._plot_object
        assert call_args[3] is None  # No error

    def test_on_cell_updated_called_when_plot_fails_with_error_message(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        job_number,
        fake_job_orchestrator,
        fake_plotting_controller,
    ):
        """on_cell_updated called when plot fails (with error message)."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        plot_orchestrator.add_plot(grid_id, plot_cell)

        # Configure controller to raise exception
        fake_plotting_controller.configure_to_raise(ValueError('Test error'))

        # Simulate workflow commit
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

        # Should be called with error
        assert callback.call_count == 2
        call_args = callback.call_args[0]
        assert call_args[0] == grid_id
        assert call_args[1] == plot_cell
        assert call_args[2] is None  # No plot
        assert 'Test error' in call_args[3]  # Error message

    def test_on_cell_updated_called_when_config_updated_no_plot_yet(
        self, plot_orchestrator, plot_cell, workflow_id
    ):
        """on_cell_updated called when config updated (no plot yet)."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Update config
        new_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='new_output',
            source_names=['new_source'],
            plot_name='new_plot',
            params={},
        )
        plot_orchestrator.update_plot_config(cell_id, new_config)

        # Should have been called twice (add + update)
        assert callback.call_count == 2
        call_args = callback.call_args[0]
        assert call_args[2] is None  # No plot yet after update

    def test_on_cell_removed_called_when_cell_removed(
        self, plot_orchestrator, plot_cell
    ):
        """on_cell_removed called when cell removed."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_removed=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)
        plot_orchestrator.remove_plot(cell_id)

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == grid_id
        assert call_args[1] == plot_cell

    def test_multiple_subscribers_all_receive_notifications(self, plot_orchestrator):
        """Multiple subscribers all receive notifications."""
        callback_1 = CallbackCapture()
        callback_2 = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_grid_created=callback_1)
        plot_orchestrator.subscribe_to_lifecycle(on_grid_created=callback_2)

        plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        callback_1.assert_called_once()
        callback_2.assert_called_once()

    def test_unsubscribe_prevents_further_notifications(self, plot_orchestrator):
        """Unsubscribe prevents further notifications."""
        callback = CallbackCapture()
        subscription_id = plot_orchestrator.subscribe_to_lifecycle(
            on_grid_created=callback
        )

        plot_orchestrator.add_grid(title='Grid 1', nrows=3, ncols=3)
        assert callback.call_count == 1

        # Unsubscribe
        plot_orchestrator.unsubscribe_from_lifecycle(subscription_id)

        plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        # Should still be 1 (not called for Grid 2)
        assert callback.call_count == 1

    def test_late_subscriber_does_not_receive_events_for_existing_grids(
        self, plot_orchestrator
    ):
        """Late subscriber doesn't receive events for existing grids."""
        # Add grid before subscribing
        plot_orchestrator.add_grid(title='Existing Grid', nrows=3, ncols=3)

        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_grid_created=callback)

        # Should not have been called for existing grid
        callback.assert_not_called()

        # Should be called for new grids
        plot_orchestrator.add_grid(title='New Grid', nrows=3, ncols=3)
        callback.assert_called_once()


class TestErrorHandling:
    """Tests for error handling."""

    def test_plotting_controller_raises_exception_calls_on_cell_updated_with_error(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        job_number,
        fake_job_orchestrator,
        fake_plotting_controller,
    ):
        """PlottingController exception calls on_cell_updated with error."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        plot_orchestrator.add_plot(grid_id, plot_cell)

        fake_plotting_controller.configure_to_raise(
            RuntimeError('Plot creation failed')
        )
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

        # Should have been called with error
        assert callback.call_count == 2
        call_args = callback.call_args[0]
        assert 'Plot creation failed' in call_args[3]

    def test_plotting_controller_raises_exception_orchestrator_remains_usable(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        job_number,
        fake_job_orchestrator,
        fake_plotting_controller,
    ):
        """PlottingController raises exception but orchestrator remains usable."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        fake_plotting_controller.configure_to_raise(
            RuntimeError('Plot creation failed')
        )
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

        # Orchestrator should still be usable
        fake_plotting_controller.reset()
        plot_orchestrator.remove_plot(cell_id)
        assert plot_orchestrator.get_grid(grid_id) is not None

    def test_lifecycle_callback_raises_exception_other_callbacks_still_invoked(
        self, plot_orchestrator
    ):
        """Lifecycle callback raises exception but other callbacks still invoked."""
        failing_callback = CallbackCapture(side_effect=RuntimeError('Callback failed'))
        working_callback = CallbackCapture()

        plot_orchestrator.subscribe_to_lifecycle(on_grid_created=failing_callback)
        plot_orchestrator.subscribe_to_lifecycle(on_grid_created=working_callback)

        plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        # Both should have been called
        failing_callback.assert_called_once()
        working_callback.assert_called_once()

    def test_lifecycle_callback_raises_exception_orchestrator_remains_usable(
        self, plot_orchestrator
    ):
        """Lifecycle callback raises exception but orchestrator remains usable."""
        failing_callback = CallbackCapture(side_effect=RuntimeError('Callback failed'))
        plot_orchestrator.subscribe_to_lifecycle(on_grid_created=failing_callback)

        plot_orchestrator.add_grid(title='Grid 1', nrows=3, ncols=3)

        # Should still be able to add more grids
        grid_id = plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)
        assert plot_orchestrator.get_grid(grid_id) is not None


class TestCleanupAndResourceManagement:
    """Tests for cleanup and resource management."""

    def test_remove_grid_with_multiple_cells_all_unsubscribed(
        self, plot_orchestrator, workflow_id, fake_job_orchestrator
    ):
        """Remove grid with multiple cells unsubscribes all from JobOrchestrator."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        # Add multiple cells
        for i in range(3):
            cell = PlotCell(
                row=i,
                col=0,
                row_span=1,
                col_span=1,
                config=PlotConfig(
                    workflow_id=workflow_id,
                    output_name=f'out{i}',
                    source_names=[f'src{i}'],
                    plot_name=f'plot{i}',
                    params={},
                ),
            )
            plot_orchestrator.add_plot(grid_id, cell)

        assert fake_job_orchestrator.subscription_count == 3

        # Remove grid
        plot_orchestrator.remove_grid(grid_id)

        # All subscriptions should be removed
        assert fake_job_orchestrator.subscription_count == 0

    def test_shutdown_with_multiple_grids_all_unsubscribed(
        self, plot_orchestrator, workflow_id, fake_job_orchestrator
    ):
        """Shutdown with multiple grids should unsubscribe all."""
        # Add multiple grids with cells
        for i in range(2):
            grid_id = plot_orchestrator.add_grid(title=f'Grid {i}', nrows=3, ncols=3)
            cell = PlotCell(
                row=0,
                col=0,
                row_span=1,
                col_span=1,
                config=PlotConfig(
                    workflow_id=workflow_id,
                    output_name=f'out{i}',
                    source_names=[f'src{i}'],
                    plot_name=f'plot{i}',
                    params={},
                ),
            )
            plot_orchestrator.add_plot(grid_id, cell)

        assert fake_job_orchestrator.subscription_count == 2

        # Shutdown
        plot_orchestrator.shutdown()

        # All subscriptions should be removed
        assert fake_job_orchestrator.subscription_count == 0

    def test_shutdown_all_grids_removed(self, plot_orchestrator):
        """Shutdown should remove all grids."""
        plot_orchestrator.add_grid(title='Grid 1', nrows=3, ncols=3)
        plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)

        plot_orchestrator.shutdown()

        assert len(plot_orchestrator.get_all_grids()) == 0

    def test_remove_last_plot_from_grid_grid_still_exists(
        self, plot_orchestrator, plot_cell
    ):
        """Remove last plot from grid should not remove the grid itself."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        plot_orchestrator.remove_plot(cell_id)

        # Grid should still exist
        grid = plot_orchestrator.get_grid(grid_id)
        assert grid is not None
        assert len(grid.cells) == 0


class TestEdgeCasesAndComplexScenarios:
    """Tests for edge cases and complex scenarios."""

    def test_update_config_to_different_workflow_id_unsubscribe_old_subscribe_new(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_id_2,
        fake_job_orchestrator,
    ):
        """Update config to different workflow_id unsubscribes old, subscribes new."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Should be subscribed to workflow_id
        assert len(fake_job_orchestrator._workflow_subscriptions[workflow_id]) == 1

        # Update to different workflow_id
        new_config = PlotConfig(
            workflow_id=workflow_id_2,
            output_name='new_output',
            source_names=['new_source'],
            plot_name='new_plot',
            params={},
        )
        plot_orchestrator.update_plot_config(cell_id, new_config)

        # Should no longer be subscribed to old workflow
        assert len(fake_job_orchestrator._workflow_subscriptions[workflow_id]) == 0
        # Should be subscribed to new workflow
        assert len(fake_job_orchestrator._workflow_subscriptions[workflow_id_2]) == 1

    def test_multiple_updates_to_same_cell_config_subscription_management_correct(
        self, plot_orchestrator, plot_cell, workflow_id, fake_job_orchestrator
    ):
        """
        Multiple updates to same cell config maintain correct subscriptions.

        Updating the same cell multiple times should not leak subscriptions.
        """
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        initial_count = fake_job_orchestrator.subscription_count

        # Multiple updates with same workflow_id
        for i in range(3):
            new_config = PlotConfig(
                workflow_id=workflow_id,
                output_name=f'output_{i}',
                source_names=[f'source_{i}'],
                plot_name=f'plot_{i}',
                params={},
            )
            plot_orchestrator.update_plot_config(cell_id, new_config)

        # Should still have same subscription count
        assert fake_job_orchestrator.subscription_count == initial_count

    def test_remove_grid_with_cells_that_have_pending_plot_creation(
        self, plot_orchestrator, plot_cell, workflow_id, fake_job_orchestrator
    ):
        """Remove grid with cells that have pending plot creation."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        plot_orchestrator.add_plot(grid_id, plot_cell)

        # Remove grid before workflow commits
        plot_orchestrator.remove_grid(grid_id)

        # Now commit workflow - should not crash
        job_number = uuid.uuid4()
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)

        # No exception should be raised

    def test_add_cell_immediately_remove_it_then_workflow_commits_no_error(
        self, plot_orchestrator, plot_cell, workflow_id, fake_job_orchestrator
    ):
        """Add cell, immediately remove it, then workflow commits - no error."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Immediately remove
        plot_orchestrator.remove_plot(cell_id)

        # Workflow commits
        job_number = uuid.uuid4()
        fake_job_orchestrator.simulate_workflow_commit(workflow_id, job_number)
