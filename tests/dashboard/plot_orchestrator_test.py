# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pytest

from ess.livedata.config.workflow_spec import JobNumber
from ess.livedata.dashboard.plot_orchestrator import (
    CellGeometry,
    GridId,
    PlotCell,
    PlotConfig,
    PlotGridConfig,
    PlotOrchestrator,
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

    def setup_data_pipeline(
        self,
        job_number,
        workflow_id,
        source_names: list[str],
        output_name: str | None,
        plot_name: str,
        params: dict,
    ):
        """Fake setup_data_pipeline for testing two-phase plot creation."""
        from ess.livedata.config.workflow_spec import JobId, ResultKey
        from ess.livedata.dashboard.extractors import LatestValueExtractor

        # Create fake subscriber and pipe
        keys = [
            ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(job_number=job_number, source_name=source_name),
                output_name=output_name,
            )
            for source_name in source_names
        ]

        class FakeSubscriber:
            """Fake data subscriber for testing."""

            def __init__(self, keys_to_monitor):
                self._keys = keys_to_monitor
                self._extractors = {
                    key: LatestValueExtractor() for key in keys_to_monitor
                }

            @property
            def keys(self):
                return set(self._keys)

            @property
            def extractors(self):
                return self._extractors

            def trigger(self, store):
                pass

        class FakePipe:
            """Fake pipe with data dictionary."""

            @property
            def data(self):
                return {}

        return FakeSubscriber(keys), FakePipe(), keys

    def create_plot_from_pipeline(
        self,
        plot_name: str,
        params: dict,
        pipe,
    ):
        """Fake create_plot_from_pipeline for testing two-phase plot creation."""
        # Record call as if it were create_plot for backward compatibility with tests
        self._calls.append(
            {
                'plot_name': plot_name,
                'params': params,
                '_from_pipeline': True,
            }
        )
        if self._should_raise:
            raise self._exception_to_raise
        return self._plot_object


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
    geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)
    return PlotCell(geometry=geometry, config=plot_config)


@pytest.fixture
def fake_plotting_controller():
    """Create a FakePlottingController."""
    return FakePlottingController()


@pytest.fixture
def fake_data_service():
    """Create a fake DataService."""
    from ess.livedata.dashboard.data_service import DataService

    return DataService()


@pytest.fixture
def fake_job_service(fake_data_service):
    """Create a fake JobService."""
    from ess.livedata.dashboard.job_service import JobService

    return JobService(data_service=fake_data_service)


@pytest.fixture
def plot_orchestrator(job_orchestrator, fake_plotting_controller, fake_data_service):
    """Create a PlotOrchestrator with real JobOrchestrator."""
    return PlotOrchestrator(
        plotting_controller=fake_plotting_controller,
        job_orchestrator=job_orchestrator,
        data_service=fake_data_service,
    )


def commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec):
    """Helper to stage and commit a workflow for testing."""
    # Ensure workflow is initialized (register it if needed)
    if workflow_id not in job_orchestrator._workflows:
        from ess.livedata.dashboard.job_orchestrator import JobConfig, WorkflowState

        state = WorkflowState()
        for source_name in workflow_spec.source_names:
            state.staged_jobs[source_name] = JobConfig(params={}, aux_source_names={})
        job_orchestrator._workflows[workflow_id] = state

    # Ensure subscription tracking is initialized
    if workflow_id not in job_orchestrator._workflow_subscriptions:
        job_orchestrator._workflow_subscriptions[workflow_id] = set()

    # Clear any existing staged jobs and stage new ones
    if workflow_id in job_orchestrator._workflows:
        job_orchestrator.clear_staged_configs(workflow_id)
        for source_name in workflow_spec.source_names:
            job_orchestrator.stage_config(
                workflow_id, source_name=source_name, params={}, aux_source_names={}
            )

    return job_orchestrator.commit_workflow(workflow_id)


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

        # Verify internal state unchanged
        internal_grid = plot_orchestrator.get_grid(grid_id)
        assert 'new_param' not in internal_grid.cells[cell_id].config.params
        # Geometry is frozen and cannot be modified
        assert internal_grid.cells[cell_id].geometry.row == 0

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

        # Verify internal state unchanged
        internal_grid = plot_orchestrator.get_grid(grid_id)
        assert 'new_param' not in internal_grid.cells[cell_id].config.params
        # Geometry is frozen and cannot be modified
        assert internal_grid.cells[cell_id].geometry.row == 0


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
            geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='out1',
                source_names=['src1'],
                plot_name='plot1',
                params={},
            ),
        )
        cell_2 = PlotCell(
            geometry=CellGeometry(row=1, col=1, row_span=1, col_span=1),
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

    def test_add_cell_subscribes_to_workflow(self, plot_orchestrator, plot_cell):
        """Each add should subscribe appropriately."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        plot_orchestrator.add_plot(grid_id, plot_cell)

    def test_remove_cell_unsubscribes_from_workflow(self, plot_orchestrator, plot_cell):
        """Remove cell should unsubscribe."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        plot_orchestrator.remove_plot(cell_id)

    def test_update_config_resubscribes(
        self, plot_orchestrator, plot_cell, workflow_id
    ):
        """Update config should unsubscribe and resubscribe."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        new_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='new_output',
            source_names=['new_source'],
            plot_name='new_plot',
            params={},
        )
        plot_orchestrator.update_plot_config(cell_id, new_config)


class TestWorkflowIntegrationAndPlotCreationTiming:
    """Tests for workflow integration and plot creation timing."""

    def test_workflow_commit_after_cell_added_creates_plot(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """Workflow commit AFTER cell added should create plot when data arrives."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        _ = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Commit workflow (PlotOrchestrator subscribes, waiting for data)
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Plot not created yet (no data)
        assert fake_plotting_controller.call_count() == 0

        # Simulate data arrival by populating JobService
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        # Add data for ALL sources (plot requires both source1 and source2)
        for source_name in plot_cell.config.source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name=plot_cell.config.output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Now plot should be created
        assert fake_plotting_controller.call_count() == 1
        calls = fake_plotting_controller.get_calls()
        # With two-phase creation, create_plot_from_pipeline is called (not create_plot)
        assert calls[0]['_from_pipeline'] is True
        assert calls[0]['plot_name'] == plot_cell.config.plot_name

    def test_workflow_commit_before_cell_added_creates_plot_when_cell_added(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """Adding cell subscribes to workflow, then workflow commit creates plot."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        # Add cell (subscribes to workflow)
        _ = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for ALL sources
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for source_name in plot_cell.config.source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name=plot_cell.config.output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Verify plot was created
        assert fake_plotting_controller.call_count() == 1

    def test_multiple_cells_subscribed_to_same_workflow_all_receive_plots(
        self,
        plot_orchestrator,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """Multiple cells subscribed to same workflow should all receive plots."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        # Add multiple cells with same workflow_id
        cell_1 = PlotCell(
            geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='out1',
                source_names=['src1'],
                plot_name='plot1',
                params={},
            ),
        )
        cell_2 = PlotCell(
            geometry=CellGeometry(row=1, col=1, row_span=1, col_span=1),
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

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for both cells
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for cell in [cell_1, cell_2]:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(
                    source_name=cell.config.source_names[0], job_number=job_number
                ),
                output_name=cell.config.output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Both plots should be created
        assert fake_plotting_controller.call_count() == 2
        calls = fake_plotting_controller.get_calls()
        assert {c['plot_name'] for c in calls} == {'plot1', 'plot2'}

    def test_cell_removed_before_workflow_commit_no_plot_created(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
    ):
        """Cell removed before workflow commit should not create plot."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Remove cell before workflow commits
        plot_orchestrator.remove_plot(cell_id)

        # Commit workflow - should not create plot
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)

        # Verify no plot was created
        assert fake_plotting_controller.call_count() == 0

    def test_update_config_resubscribes_and_new_workflow_commit_creates_plot(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_id_2,
        workflow_spec_2,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
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

        # Commit new workflow
        job_ids = commit_workflow_for_test(
            job_orchestrator, workflow_id_2, workflow_spec_2
        )
        job_number = job_ids[0].job_number

        # Simulate data arrival for new workflow
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        result_key = ResultKey(
            workflow_id=workflow_id_2,
            job_id=JobId(source_name='new_source', job_number=job_number),
            output_name='new_output',
        )
        fake_data_service[result_key] = sc.scalar(1.0)

        # Verify plot was created for new workflow
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
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        callback.assert_called_once()
        call_kwargs = callback.call_args[1]
        assert call_kwargs['grid_id'] == grid_id
        assert call_kwargs['cell_id'] == cell_id
        assert call_kwargs['cell'] == plot_cell
        assert call_kwargs['plot'] is None  # No plot yet
        assert set(call_kwargs.keys()) == {
            'grid_id',
            'cell_id',
            'cell',
            'plot',
            'error',
        }
        # error should be None
        assert call_kwargs['error'] is None

    def test_on_cell_updated_called_when_plot_created_with_plot_object(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """on_cell_updated called when plot created (with plot object)."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Should have been called once when cell was added
        assert callback.call_count == 1

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for ALL sources
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for source_name in plot_cell.config.source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name=plot_cell.config.output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Called 3x: add cell, commit (waiting), data arrival (plot created)
        assert callback.call_count == 3
        call_kwargs = callback.call_args[1]
        assert call_kwargs['grid_id'] == grid_id
        assert call_kwargs['cell_id'] == cell_id
        assert call_kwargs['cell'] == plot_cell
        assert call_kwargs['plot'] == fake_plotting_controller._plot_object
        assert call_kwargs['error'] is None  # No error

    def test_on_cell_updated_called_when_plot_fails_with_error_message(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """on_cell_updated called when plot fails (with error message)."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Configure controller to raise exception
        fake_plotting_controller.configure_to_raise(ValueError('Test error'))

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for ALL sources (will trigger plot creation that fails)
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for source_name in plot_cell.config.source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name=plot_cell.config.output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Called 3x: add cell, commit (waiting), data arrival (error)
        assert callback.call_count == 3
        call_kwargs = callback.call_args[1]
        assert call_kwargs['grid_id'] == grid_id
        assert call_kwargs['cell_id'] == cell_id
        assert call_kwargs['cell'] == plot_cell
        assert call_kwargs['plot'] is None  # No plot
        assert 'Test error' in call_kwargs['error']  # Error message

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
        call_kwargs = callback.call_args[1]
        assert call_kwargs['plot'] is None  # No plot yet after update

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
        assert call_args[1] == plot_cell.geometry

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
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """PlottingController exception calls on_cell_updated with error."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        plot_orchestrator.add_plot(grid_id, plot_cell)

        fake_plotting_controller.configure_to_raise(
            RuntimeError('Plot creation failed')
        )
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for ALL sources (will trigger plot creation that fails)
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for source_name in plot_cell.config.source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name=plot_cell.config.output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Called 3x: add cell, commit (waiting), data arrival (error)
        assert callback.call_count == 3
        call_kwargs = callback.call_args[1]
        assert 'Plot creation failed' in call_kwargs['error']

    def test_plotting_controller_raises_exception_orchestrator_remains_usable(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
    ):
        """PlottingController raises exception but orchestrator remains usable."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        fake_plotting_controller.configure_to_raise(
            RuntimeError('Plot creation failed')
        )
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)

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
        self, plot_orchestrator, workflow_id
    ):
        """Remove grid with multiple cells unsubscribes all from JobOrchestrator."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        # Add multiple cells
        for i in range(3):
            cell = PlotCell(
                geometry=CellGeometry(row=i, col=0, row_span=1, col_span=1),
                config=PlotConfig(
                    workflow_id=workflow_id,
                    output_name=f'out{i}',
                    source_names=[f'src{i}'],
                    plot_name=f'plot{i}',
                    params={},
                ),
            )
            plot_orchestrator.add_plot(grid_id, cell)

        # Remove grid
        plot_orchestrator.remove_grid(grid_id)

    def test_shutdown_with_multiple_grids_all_unsubscribed(
        self, plot_orchestrator, workflow_id
    ):
        """Shutdown with multiple grids should unsubscribe all."""
        # Add multiple grids with cells
        for i in range(2):
            grid_id = plot_orchestrator.add_grid(title=f'Grid {i}', nrows=3, ncols=3)
            cell = PlotCell(
                geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
                config=PlotConfig(
                    workflow_id=workflow_id,
                    output_name=f'out{i}',
                    source_names=[f'src{i}'],
                    plot_name=f'plot{i}',
                    params={},
                ),
            )
            plot_orchestrator.add_plot(grid_id, cell)

        # Shutdown
        plot_orchestrator.shutdown()

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
    ):
        """Update config to different workflow_id unsubscribes old, subscribes new."""
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

    def test_multiple_updates_to_same_cell_config_subscription_management_correct(
        self, plot_orchestrator, plot_cell, workflow_id
    ):
        """
        Multiple updates to same cell config maintain correct subscriptions.

        Updating the same cell multiple times should not leak subscriptions.
        """
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

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

    def test_remove_grid_with_cells_that_have_pending_plot_creation(
        self, plot_orchestrator, plot_cell, workflow_id, workflow_spec, job_orchestrator
    ):
        """Remove grid with cells that have pending plot creation."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        plot_orchestrator.add_plot(grid_id, plot_cell)

        # Remove grid before workflow commits
        plot_orchestrator.remove_grid(grid_id)

        # Now commit workflow - should not crash
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)

    def test_add_cell_immediately_remove_it_then_workflow_commits_no_error(
        self, plot_orchestrator, plot_cell, workflow_id, workflow_spec, job_orchestrator
    ):
        """Add cell, immediately remove it, then workflow commits - no error."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Immediately remove
        plot_orchestrator.remove_plot(cell_id)

        # Workflow commits
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)


class TestLateSubscriberPlotRetrieval:
    """Test that late subscribers can retrieve existing plots via get_cell_state."""

    def test_get_cell_state_returns_none_for_cell_without_plot(
        self, plot_orchestrator, plot_cell
    ):
        """get_cell_state should return (None, None) for cell without plot."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Cell exists but workflow hasn't committed yet
        plot, error = plot_orchestrator.get_cell_state(cell_id)
        assert plot is None
        assert error is None

    def test_get_cell_state_returns_plot_after_workflow_commits(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_data_service,
    ):
        """get_cell_state should return plot after workflow commits."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for ALL sources
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for source_name in plot_cell.config.source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name=plot_cell.config.output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Now get_cell_state should return the plot
        plot, error = plot_orchestrator.get_cell_state(cell_id)
        assert plot is not None
        assert isinstance(plot, FakePlot)
        assert error is None

    def test_get_cell_state_returns_error_when_plot_creation_fails(
        self,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_data_service,
        fake_job_service,
    ):
        """get_cell_state should return error when plot creation fails."""

        # Create plotting controller that raises exception
        class FailingPlottingController:
            def create_plot(self, **kwargs):
                raise ValueError("Plot creation failed")

            def setup_data_pipeline(self, **kwargs):
                from ess.livedata.config.workflow_spec import JobId, ResultKey
                from ess.livedata.dashboard.extractors import LatestValueExtractor

                workflow_id = kwargs['workflow_id']
                job_number = kwargs['job_number']
                source_names = kwargs['source_names']
                output_name = kwargs['output_name']

                keys = [
                    ResultKey(
                        workflow_id=workflow_id,
                        job_id=JobId(job_number=job_number, source_name=source_name),
                        output_name=output_name,
                    )
                    for source_name in source_names
                ]

                class FakeSubscriber:
                    def __init__(self, keys_to_monitor):
                        self._keys = keys_to_monitor
                        self._extractors = {
                            key: LatestValueExtractor() for key in keys_to_monitor
                        }

                    @property
                    def keys(self):
                        return set(self._keys)

                    @property
                    def extractors(self):
                        return self._extractors

                    def trigger(self, store):
                        pass

                class FakePipe:
                    @property
                    def data(self):
                        return {}

                return FakeSubscriber(keys), FakePipe(), keys

            def create_plot_from_pipeline(self, **kwargs):
                raise ValueError("Plot creation failed")

        plot_orchestrator = PlotOrchestrator(
            plotting_controller=FailingPlottingController(),
            job_orchestrator=job_orchestrator,
            data_service=fake_data_service,
            config_store=None,
        )

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        plot_cell = PlotCell(
            geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='test_output',
                source_names=['source1'],
                plot_name='test_plot',
                params={},
            ),
        )
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival (will fail to create plot)
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        result_key = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='source1', job_number=job_number),
            output_name='test_output',
        )
        fake_data_service[result_key] = sc.scalar(1.0)

        # get_cell_state should return error (full traceback)
        plot, error = plot_orchestrator.get_cell_state(cell_id)
        assert plot is None
        assert "Plot creation failed" in error

    def test_late_subscriber_can_retrieve_existing_plots_from_all_cells(
        self,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
        fake_job_service,
    ):
        """
        Simulate late subscriber scenario: plots exist, new UI component
        initializes and retrieves them via get_cell_state.
        """
        plot_orchestrator = PlotOrchestrator(
            plotting_controller=fake_plotting_controller,
            job_orchestrator=job_orchestrator,
            data_service=fake_data_service,
            config_store=None,
        )

        # Create grid with multiple cells
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        cell_ids = []
        for i in range(3):
            plot_cell = PlotCell(
                geometry=CellGeometry(row=i, col=0, row_span=1, col_span=1),
                config=PlotConfig(
                    workflow_id=workflow_id,
                    output_name=f'output_{i}',
                    source_names=[f'source_{i}'],
                    plot_name=f'plot_{i}',
                    params={},
                ),
            )
            cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)
            cell_ids.append(cell_id)

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for all cells
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for i in range(3):
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=f'source_{i}', job_number=job_number),
                output_name=f'output_{i}',
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Simulate late subscriber (new session, page reload, etc.)
        # Late subscriber retrieves grid config
        grids = plot_orchestrator.get_all_grids()
        assert grid_id in grids

        grid_config = grids[grid_id]
        assert len(grid_config.cells) == 3

        # Late subscriber retrieves plots for all cells via get_cell_state
        for cell_id in cell_ids:
            plot, error = plot_orchestrator.get_cell_state(cell_id)
            assert plot is not None, f"Plot should exist for cell {cell_id}"
            assert isinstance(plot, FakePlot)
            assert error is None

    def test_get_cell_state_updated_when_cell_config_updated_with_running_workflow(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_data_service,
    ):
        """
        When config is updated and workflow is running, plot is immediately recreated.
        """
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number1 = job_ids[0].job_number

        # Simulate data arrival for ALL sources
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for source_name in plot_cell.config.source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number1),
                output_name=plot_cell.config.output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Verify plot exists
        plot1, error = plot_orchestrator.get_cell_state(cell_id)
        assert plot1 is not None
        assert error is None

        # Update config while workflow is still running
        new_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='new_output',
            source_names=['new_source'],
            plot_name='new_plot',
            params={},
        )
        plot_orchestrator.update_plot_config(cell_id, new_config)

        # Simulate data arrival for new config
        result_key2 = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='new_source', job_number=job_number1),
            output_name='new_output',
        )
        fake_data_service[result_key2] = sc.scalar(2.0)

        # Since workflow is still running, plot should be immediately recreated
        plot2, error = plot_orchestrator.get_cell_state(cell_id)
        assert plot2 is not None
        assert error is None
        # Note: We can't reliably test for a different instance since
        # FakePlottingController may return the same object

    def test_get_cell_state_cleared_when_cell_removed(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_data_service,
    ):
        """get_cell_state should be cleaned up when cell is removed."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = plot_orchestrator.add_plot(grid_id, plot_cell)

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for ALL sources
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for source_name in plot_cell.config.source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name=plot_cell.config.output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Verify plot exists
        plot, error = plot_orchestrator.get_cell_state(cell_id)
        assert plot is not None

        # Remove cell
        plot_orchestrator.remove_plot(cell_id)

        # get_cell_state should now return None (cell doesn't exist)
        plot, error = plot_orchestrator.get_cell_state(cell_id)
        assert plot is None
        assert error is None


class TestSourceNameFiltering:
    """Test that plots only create when their specific source_names have data."""

    def test_plot_not_created_when_job_has_different_source_names(
        self,
        plot_orchestrator,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """
        Plot should NOT be created when job exists for different source_names.

        This test demonstrates the bug where a plot wanting source_A tries
        to create when only source_B has data (same workflow, same job_number).
        """
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        # Create a plot that wants data from source_A
        plot_cell = PlotCell(
            geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='test_output',
                source_names=['source_A'],  # Plot wants source_A
                plot_name='test_plot',
                params={},
            ),
        )
        plot_orchestrator.add_plot(grid_id, plot_cell)

        # Commit workflow (notifies the plot that workflow is running)
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for source_B (NOT source_A)
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        result_key_B = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='source_B', job_number=job_number),
            output_name='test_output',
        )
        fake_data_service[result_key_B] = sc.scalar(1.0)

        # Plot should NOT be created (data is for wrong source)
        assert (
            fake_plotting_controller.call_count() == 0
        ), "Plot should not be created for different source_name"

        # Now simulate data arrival for source_A (the one the plot wants)
        result_key_A = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='source_A', job_number=job_number),
            output_name='test_output',
        )
        fake_data_service[result_key_A] = sc.scalar(2.0)

        # NOW the plot should be created
        assert (
            fake_plotting_controller.call_count() == 1
        ), "Plot should be created when correct source_name has data"
        calls = fake_plotting_controller.get_calls()
        # With two-phase creation, create_plot_from_pipeline is called
        assert calls[0]['_from_pipeline'] is True

    def test_plot_created_progressively_as_sources_arrive(
        self,
        plot_orchestrator,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """
        Plot with multiple source_names created as soon as first source has data.

        A plot wanting sources [A, B] should create when only A has data,
        then update automatically when B arrives (progressive plotting).
        """
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        # Create a plot that wants data from BOTH source_A and source_B
        plot_cell = PlotCell(
            geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='test_output',
                source_names=['source_A', 'source_B'],  # Plot wants BOTH
                plot_name='test_plot',
                params={},
            ),
        )
        plot_orchestrator.add_plot(grid_id, plot_cell)

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for source_A only
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        result_key_A = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='source_A', job_number=job_number),
            output_name='test_output',
        )
        fake_data_service[result_key_A] = sc.scalar(1.0)

        # Plot SHOULD be created with partial data (progressive plotting)
        assert (
            fake_plotting_controller.call_count() == 1
        ), "Plot should be created as soon as first source has data"
        calls = fake_plotting_controller.get_calls()
        # With two-phase creation, create_plot_from_pipeline is called
        assert calls[0]['_from_pipeline'] is True

        # When source_B arrives, the DynamicMap will automatically update
        # (no new create_plot call needed - the existing plot subscribes to updates)
        result_key_B = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='source_B', job_number=job_number),
            output_name='test_output',
        )
        fake_data_service[result_key_B] = sc.scalar(2.0)

        # Still only 1 create_plot call (plot updates via streaming, not recreation)
        assert fake_plotting_controller.call_count() == 1

    def test_immediate_creation_when_subscribing_to_running_job_with_correct_sources(
        self,
        plot_orchestrator,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """
        When subscribing to already-running job, plot creates if sources exist.

        This tests the "late subscription" path where workflow is already
        running and has data for the plot's source_names.
        """
        # Commit workflow FIRST
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Add data for source_A
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        result_key_A = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='source_A', job_number=job_number),
            output_name='test_output',
        )
        fake_data_service[result_key_A] = sc.scalar(1.0)

        # NOW add the plot (late subscription)
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        plot_cell = PlotCell(
            geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='test_output',
                source_names=['source_A'],
                plot_name='test_plot',
                params={},
            ),
        )
        plot_orchestrator.add_plot(grid_id, plot_cell)

        # Plot should be created immediately (data already exists)
        assert (
            fake_plotting_controller.call_count() == 1
        ), "Plot should be created immediately when subscribing to running job"

    def test_no_creation_when_subscribing_to_running_job_without_correct_sources(
        self,
        plot_orchestrator,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """
        When subscribing to already-running job, plot waits if sources missing.

        This tests the "late subscription" path where workflow is running
        but doesn't have data for the plot's specific source_names.
        """
        # Commit workflow FIRST
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Add data for source_B (not what the plot will want)
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        result_key_B = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='source_B', job_number=job_number),
            output_name='test_output',
        )
        fake_data_service[result_key_B] = sc.scalar(1.0)

        # NOW add the plot wanting source_A (late subscription)
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        plot_cell = PlotCell(
            geometry=CellGeometry(row=0, col=0, row_span=1, col_span=1),
            config=PlotConfig(
                workflow_id=workflow_id,
                output_name='test_output',
                source_names=['source_A'],  # Plot wants A, but only B exists
                plot_name='test_plot',
                params={},
            ),
        )
        plot_orchestrator.add_plot(grid_id, plot_cell)

        # Plot should NOT be created (wrong source)
        assert (
            fake_plotting_controller.call_count() == 0
        ), "Plot should not be created when only wrong sources exist"

        # Add data for source_A (what the plot wants)
        result_key_A = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='source_A', job_number=job_number),
            output_name='test_output',
        )
        fake_data_service[result_key_A] = sc.scalar(2.0)

        # NOW plot should be created
        assert (
            fake_plotting_controller.call_count() == 1
        ), "Plot should be created when correct source arrives"
