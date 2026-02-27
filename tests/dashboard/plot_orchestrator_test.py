# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pydantic
import pytest

from ess.livedata.dashboard.plot_data_service import PlotDataService
from ess.livedata.dashboard.plot_orchestrator import (
    CellGeometry,
    DataSourceConfig,
    GridId,
    PlotConfig,
    PlotGridConfig,
    PlotOrchestrator,
)
from ess.livedata.dashboard.plots import PresenterBase


class FakePlotParams(pydantic.BaseModel):
    """Flexible fake params model that accepts any fields."""

    model_config = pydantic.ConfigDict(extra='allow')


# Default geometry used in tests
DEFAULT_GEOMETRY = CellGeometry(row=0, col=0, row_span=1, col_span=1)


class FakePlotterSpec:
    """Fake PlotterSpec for testing."""

    def __init__(self, params_class=FakePlotParams):
        self.params = params_class


class FakePlot:
    """Dummy plot object for testing."""

    pass


class FakePlotter:
    """Fake Plotter for testing."""

    def __init__(self):
        self.kdims = None
        self._initialized_data = None
        self._cached_state = None

    def initialize_from_data(self, data):
        self._initialized_data = data

    def create_presenter(self):
        """Return a fake presenter."""
        return FakePresenter(self)

    def compute(self, data, **kwargs):
        result = FakePlot()
        self._cached_state = result
        return result

    def mark_presenters_dirty(self):
        pass

    def get_cached_state(self):
        return self._cached_state

    def has_cached_state(self):
        return self._cached_state is not None

    def __call__(self, data):
        return self.compute(data)


class FakePresenter(PresenterBase):
    """Fake presenter for testing."""

    def present(self, pipe):
        import holoviews as hv

        return hv.DynamicMap(self._plotter, streams=[pipe], cache_size=1)


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


class FakePipe:
    """Fake HoloViews Pipe for testing."""

    def __init__(self, data):
        self.data = data

    def send(self, data):
        self.data = data


class FakePlottingController:
    """
    Fake PlottingController for testing plot orchestration.

    Uses real StreamManager for data pipeline setup (avoiding duplication of
    subscription logic), but returns fake plots for easy assertion.
    """

    def __init__(self, stream_manager):
        self._stream_manager = stream_manager
        self._should_raise = False
        self._exception_to_raise = None
        self._plot_object = FakePlot()
        self._calls: list[dict] = []
        self._pipeline_setups: list[dict] = []
        self._fake_spec = FakePlotterSpec()

    def get_spec(self, plot_name: str) -> FakePlotterSpec:
        """Return a fake plotter spec that accepts any params."""
        return self._fake_spec

    def configure_to_raise(self, exception: Exception) -> None:
        """Configure the controller to raise an exception on create_plot."""
        self._should_raise = True
        self._exception_to_raise = exception

    def reset(self) -> None:
        """Reset the controller to normal behavior."""
        self._should_raise = False
        self._exception_to_raise = None
        self._calls.clear()
        self._pipeline_setups.clear()

    def call_count(self) -> int:
        """Return the number of create_plotter calls."""
        return len(self._calls)

    def get_calls(self) -> list[dict]:
        """Return all recorded calls."""
        return self._calls.copy()

    def get_pipeline_setups(self) -> list[dict]:
        """Return all recorded pipeline setup calls."""
        return self._pipeline_setups.copy()

    def setup_pipeline(
        self,
        keys_by_role,
        plot_name: str,
        params,
        on_data,
    ):
        """Set up data pipeline using real StreamManager (unified interface)."""
        from ess.livedata.dashboard.data_roles import PRIMARY

        # Extract source_names and output_name from primary keys for assertions
        primary_keys = keys_by_role.get(PRIMARY, [])
        source_names = [key.job_id.source_name for key in primary_keys]
        output_name = primary_keys[0].output_name if primary_keys else None

        # Record the call for assertions
        self._pipeline_setups.append(
            {
                'source_names': source_names,
                'output_name': output_name,
                'plot_name': plot_name,
            }
        )

        # Use real StreamManager for subscription (avoids duplicating logic)
        return self._stream_manager.make_stream(
            keys_by_role,
            on_data=on_data,
        )

    def create_plotter(
        self,
        plot_name: str,
        params: dict,
    ):
        """Create a fake plotter, recording the call for assertions."""
        self._calls.append(
            {
                'plot_name': plot_name,
                'params': params,
                '_from_create_plotter': True,
            }
        )
        if self._should_raise:
            raise self._exception_to_raise
        return FakePlotter()


@pytest.fixture
def job_number():
    """Create a test job number."""
    return uuid.uuid4()


def make_plot_config(
    workflow_id,
    source_names=None,
    output_name='test_output',
    plot_name='test_plot',
    params=None,
):
    """Helper to create a PlotConfig with a single data source."""
    from ess.livedata.dashboard.data_roles import PRIMARY

    if source_names is None:
        source_names = ['source1', 'source2']
    if params is None:
        params = FakePlotParams(param1='value1')
    data_source = DataSourceConfig(
        workflow_id=workflow_id,
        source_names=source_names,
        output_name=output_name,
    )
    return PlotConfig(
        data_sources={PRIMARY: data_source},
        plot_name=plot_name,
        params=params,
    )


@pytest.fixture
def plot_config(workflow_id):
    """Create a basic PlotConfig."""
    return make_plot_config(workflow_id)


@pytest.fixture
def plot_cell(plot_config):
    """Create a basic (geometry, config) tuple for tests."""
    return (DEFAULT_GEOMETRY, plot_config)


def add_cell_with_layer(orchestrator, grid_id, geometry, config):
    """Helper to add a cell with a single layer (replaces old add_plot)."""
    cell_id = orchestrator.add_cell(grid_id, geometry)
    orchestrator.add_layer(cell_id, config)
    return cell_id


@pytest.fixture
def fake_data_service():
    """Create a real DataService for testing."""
    from ess.livedata.dashboard.data_service import DataService

    return DataService()


@pytest.fixture
def fake_stream_manager(fake_data_service):
    """Create a real StreamManager with FakePipe factory for testing."""
    from ess.livedata.dashboard.stream_manager import StreamManager

    return StreamManager(data_service=fake_data_service)


@pytest.fixture
def fake_plotting_controller(fake_stream_manager):
    """Create a FakePlottingController with StreamManager."""
    return FakePlottingController(stream_manager=fake_stream_manager)


@pytest.fixture
def fake_job_service():
    """Create a real JobService for testing."""
    from ess.livedata.dashboard.job_service import JobService

    return JobService()


@pytest.fixture
def plot_data_service():
    """Create a PlotDataService for testing."""
    return PlotDataService()


@pytest.fixture
def plot_orchestrator(
    job_orchestrator, fake_plotting_controller, fake_data_service, plot_data_service
):
    """Create a PlotOrchestrator with real JobOrchestrator."""
    return PlotOrchestrator(
        plotting_controller=fake_plotting_controller,
        job_orchestrator=job_orchestrator,
        data_service=fake_data_service,
        instrument='dummy',
        plot_data_service=plot_data_service,
    )


def commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec):
    """
    Stage and commit a workflow for testing using public API only.

    Requires the workflow_spec to have params so that JobOrchestrator
    initializes the workflow with staged configs during construction.
    """
    # With params in workflow_spec, the workflow is already initialized
    # with staged configs. Just commit it.
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
        add_cell_with_layer(plot_orchestrator, grid_id, plot_cell[0], plot_cell[1])

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
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        # Get grid and modify a cell's layer config (params is a Pydantic model)
        grid = plot_orchestrator.get_grid(grid_id)
        grid.cells[cell_id].layers[0].config.params.new_param = 'new_value'

        # Verify internal state unchanged
        internal_grid = plot_orchestrator.get_grid(grid_id)
        assert not hasattr(
            internal_grid.cells[cell_id].layers[0].config.params, 'new_param'
        )
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
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        # Get all grids and modify a cell's layer (params is a Pydantic model)
        all_grids = plot_orchestrator.get_all_grids()
        all_grids[grid_id].cells[cell_id].layers[
            0
        ].config.params.new_param = 'new_value'

        # Verify internal state unchanged
        internal_grid = plot_orchestrator.get_grid(grid_id)
        assert not hasattr(
            internal_grid.cells[cell_id].layers[0].config.params, 'new_param'
        )
        # Geometry is frozen and cannot be modified
        assert internal_grid.cells[cell_id].geometry.row == 0


class TestCellManagement:
    """Tests for cell operations without triggering plot creation."""

    def test_add_cell_to_grid_makes_it_retrievable_in_grid_config(
        self, plot_orchestrator, plot_cell
    ):
        """Add cell to grid makes it retrievable in grid config."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        grid = plot_orchestrator.get_grid(grid_id)
        assert cell_id in grid.cells
        # Verify cell has correct geometry and layer config
        cell = grid.cells[cell_id]
        assert cell.geometry == plot_cell[0]
        assert len(cell.layers) == 1
        assert cell.layers[0].config == plot_cell[1]

    def test_remove_cell_from_grid_removes_it_from_grid_config(
        self, plot_orchestrator, plot_cell
    ):
        """Remove cell from grid removes it from grid config."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        plot_orchestrator.remove_cell(cell_id)

        grid = plot_orchestrator.get_grid(grid_id)
        assert cell_id not in grid.cells

    def test_get_layer_config_returns_correct_config(
        self, plot_orchestrator, plot_cell, plot_config
    ):
        """Get layer config returns correct config."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        # Get the layer_id from the cell
        grid = plot_orchestrator.get_grid(grid_id)
        layer_id = grid.cells[cell_id].layers[0].layer_id

        retrieved_config = plot_orchestrator.get_layer_config(layer_id)

        assert retrieved_config == plot_config

    def test_update_layer_config_changes_the_config(
        self, plot_orchestrator, plot_cell, plot_config, workflow_id
    ):
        """Update layer config replaces the layer with a new layer_id."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        # Get the original layer_id from the cell
        grid = plot_orchestrator.get_grid(grid_id)
        old_layer_id = grid.cells[cell_id].layers[0].layer_id

        new_config = make_plot_config(
            workflow_id,
            source_names=['new_source'],
            output_name='new_output',
            plot_name='new_plot',
            params=FakePlotParams(new_param='new_value'),
        )
        plot_orchestrator.update_layer_config(old_layer_id, new_config)

        # After update, the layer has a NEW layer_id (this invalidates session caches)
        grid = plot_orchestrator.get_grid(grid_id)
        new_layer_id = grid.cells[cell_id].layers[0].layer_id
        assert new_layer_id != old_layer_id

        retrieved_config = plot_orchestrator.get_layer_config(new_layer_id)
        assert retrieved_config == new_config
        assert retrieved_config.output_name == 'new_output'

    def test_add_multiple_cells_to_same_grid(self, plot_orchestrator, workflow_id):
        """Add multiple cells to same grid."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        geometry_1 = CellGeometry(row=0, col=0, row_span=1, col_span=1)
        config_1 = make_plot_config(
            workflow_id,
            source_names=['src1'],
            output_name='out1',
            plot_name='plot1',
            params=FakePlotParams(),
        )
        geometry_2 = CellGeometry(row=1, col=1, row_span=1, col_span=1)
        config_2 = make_plot_config(
            workflow_id,
            source_names=['src2'],
            output_name='out2',
            plot_name='plot2',
            params=FakePlotParams(),
        )

        cell_id_1 = add_cell_with_layer(
            plot_orchestrator, grid_id, geometry_1, config_1
        )
        cell_id_2 = add_cell_with_layer(
            plot_orchestrator, grid_id, geometry_2, config_2
        )

        grid = plot_orchestrator.get_grid(grid_id)
        assert len(grid.cells) == 2
        assert cell_id_1 in grid.cells
        assert cell_id_2 in grid.cells

    def test_add_cells_to_multiple_grids(self, plot_orchestrator, plot_cell):
        """Add cells to multiple grids."""
        grid_id_1 = plot_orchestrator.add_grid(title='Grid 1', nrows=3, ncols=3)
        grid_id_2 = plot_orchestrator.add_grid(title='Grid 2', nrows=3, ncols=3)

        cell_id_1 = add_cell_with_layer(
            plot_orchestrator, grid_id_1, plot_cell[0], plot_cell[1]
        )
        cell_id_2 = add_cell_with_layer(
            plot_orchestrator, grid_id_2, plot_cell[0], plot_cell[1]
        )

        grid_1 = plot_orchestrator.get_grid(grid_id_1)
        grid_2 = plot_orchestrator.get_grid(grid_id_2)
        assert cell_id_1 in grid_1.cells
        assert cell_id_2 in grid_2.cells


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
        """Workflow commit AFTER cell added should create plotter eagerly."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        _ = add_cell_with_layer(plot_orchestrator, grid_id, plot_cell[0], plot_cell[1])

        # Commit workflow - plotter is created eagerly (before data arrives)
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Plotter should be created eagerly when job is ready
        assert fake_plotting_controller.call_count() == 1
        calls = fake_plotting_controller.get_calls()
        assert calls[0]['_from_create_plotter'] is True
        assert calls[0]['plot_name'] == plot_cell[1].plot_name

        # Simulate data arrival by populating DataService
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        # Add data for ALL sources (plot requires both source1 and source2)
        for source_name in plot_cell[1].source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name=plot_cell[1].output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Plotter count unchanged (no new plotter created on data arrival)
        assert fake_plotting_controller.call_count() == 1

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
        _ = add_cell_with_layer(plot_orchestrator, grid_id, plot_cell[0], plot_cell[1])

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for ALL sources
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for source_name in plot_cell[1].source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name=plot_cell[1].output_name,
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
        geometry_1 = CellGeometry(row=0, col=0, row_span=1, col_span=1)
        config_1 = make_plot_config(
            workflow_id,
            source_names=['src1'],
            output_name='out1',
            plot_name='plot1',
            params=FakePlotParams(),
        )
        geometry_2 = CellGeometry(row=1, col=1, row_span=1, col_span=1)
        config_2 = make_plot_config(
            workflow_id,
            source_names=['src2'],
            output_name='out2',
            plot_name='plot2',
            params=FakePlotParams(),
        )

        add_cell_with_layer(plot_orchestrator, grid_id, geometry_1, config_1)
        add_cell_with_layer(plot_orchestrator, grid_id, geometry_2, config_2)

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for both cells
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for config in [config_1, config_2]:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(
                    source_name=config.source_names[0],
                    job_number=job_number,
                ),
                output_name=config.output_name,
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
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        # Remove cell before workflow commits
        plot_orchestrator.remove_cell(cell_id)

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
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        # Get layer_id for the cell
        grid = plot_orchestrator.get_grid(grid_id)
        layer_id = grid.cells[cell_id].layers[0].layer_id

        # Update to different workflow_id
        new_config = make_plot_config(
            workflow_id_2,
            source_names=['new_source'],
            output_name='new_output',
            plot_name='new_plot',
            params=FakePlotParams(),
        )
        plot_orchestrator.update_layer_config(layer_id, new_config)

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
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        callback.assert_called_once()
        call_kwargs = callback.call_args[1]
        assert call_kwargs['grid_id'] == grid_id
        assert call_kwargs['cell_id'] == cell_id
        # Verify cell has correct geometry and layer config
        cell = call_kwargs['cell']
        assert cell.geometry == plot_cell[0]
        assert len(cell.layers) == 1
        assert cell.layers[0].config == plot_cell[1]
        # Callback now has simplified signature (just config, no state)
        assert set(call_kwargs.keys()) == {'grid_id', 'cell_id', 'cell'}

    def test_on_cell_updated_called_when_cell_added_not_on_data_arrival(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """on_cell_updated called only when cell added, not on data arrival."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        # Should have been called once when cell was added
        assert callback.call_count == 1
        call_kwargs = callback.call_args[1]
        assert call_kwargs['grid_id'] == grid_id
        assert call_kwargs['cell_id'] == cell_id

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for ALL sources
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for source_name in plot_cell[1].source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name=plot_cell[1].output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Still only called once (no notification on workflow commit or data arrival)
        # Sessions poll PlotDataService directly instead
        assert callback.call_count == 1

    def test_on_cell_updated_called_when_plot_fails_with_error_message(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
        plot_data_service,
    ):
        """on_cell_updated called when plot creation fails (with error message)."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        add_cell_with_layer(plot_orchestrator, grid_id, plot_cell[0], plot_cell[1])

        # Should have been called once when cell was added
        assert callback.call_count == 1
        call_kwargs = callback.call_args[1]
        cell = call_kwargs['cell']
        layer_id = cell.layers[0].layer_id

        # Configure controller to raise exception on create_plotter
        fake_plotting_controller.configure_to_raise(ValueError('Test error'))

        # Commit workflow - this triggers eager plotter creation which will fail
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)

        # Only 1 call - initial cell creation
        # Error state changes are now detected via polling, not callbacks
        assert callback.call_count == 1

        # Error is stored in PlotDataService for polling-based detection
        state = plot_data_service.get(layer_id)
        assert state is not None
        assert state.error_message is not None
        assert 'Test error' in state.error_message

    def test_on_cell_updated_called_when_layer_config_updated_no_plot_yet(
        self, plot_orchestrator, plot_cell, workflow_id
    ):
        """on_cell_updated called when layer config updated (no plot yet)."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        # Get layer_id for the cell
        grid = plot_orchestrator.get_grid(grid_id)
        layer_id = grid.cells[cell_id].layers[0].layer_id

        # Update config
        new_config = make_plot_config(
            workflow_id,
            source_names=['new_source'],
            output_name='new_output',
            plot_name='new_plot',
            params=FakePlotParams(),
        )
        plot_orchestrator.update_layer_config(layer_id, new_config)

        # Should have been called twice (add + update)
        assert callback.call_count == 2
        # Callback now has simplified signature (just config, no state)
        call_kwargs = callback.call_args[1]
        assert set(call_kwargs.keys()) == {'grid_id', 'cell_id', 'cell'}

    def test_on_cell_removed_called_when_cell_removed(
        self, plot_orchestrator, plot_cell
    ):
        """on_cell_removed called when cell removed."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_removed=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )
        plot_orchestrator.remove_cell(cell_id)

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == grid_id
        # Verify the removed cell's geometry matches
        assert call_args[1] == plot_cell[0]

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
        plot_data_service,
    ):
        """PlottingController exception stores error in PlotDataService."""
        callback = CallbackCapture()
        plot_orchestrator.subscribe_to_lifecycle(on_cell_updated=callback)

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        # Should have been called once when cell was added
        assert callback.call_count == 1
        cell = plot_orchestrator.get_cell(cell_id)
        layer_id = cell.layers[0].layer_id

        fake_plotting_controller.configure_to_raise(
            RuntimeError('Plot creation failed')
        )
        # Commit workflow - this triggers eager plotter creation which will fail
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)

        # Only 1 call - initial cell creation
        # Error state changes are now detected via polling, not callbacks
        assert callback.call_count == 1

        # Error is stored in PlotDataService for polling-based detection
        state = plot_data_service.get(layer_id)
        assert state is not None
        assert state.error_message is not None
        assert 'Plot creation failed' in state.error_message

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
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        fake_plotting_controller.configure_to_raise(
            RuntimeError('Plot creation failed')
        )
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)

        # Orchestrator should still be usable
        fake_plotting_controller.reset()
        plot_orchestrator.remove_cell(cell_id)
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
        self,
        plot_orchestrator,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
    ):
        """Remove grid with cells unsubscribes all, so commit creates no plots."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        for i in range(3):
            geometry = CellGeometry(row=i, col=0, row_span=1, col_span=1)
            config = make_plot_config(
                workflow_id,
                source_names=[f'src{i}'],
                output_name=f'out{i}',
                plot_name=f'plot{i}',
                params=FakePlotParams(),
            )
            add_cell_with_layer(plot_orchestrator, grid_id, geometry, config)

        plot_orchestrator.remove_grid(grid_id)
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)

        assert plot_orchestrator.get_grid(grid_id) is None
        assert fake_plotting_controller.call_count() == 0

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
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        plot_orchestrator.remove_cell(cell_id)

        # Grid should still exist
        grid = plot_orchestrator.get_grid(grid_id)
        assert grid is not None
        assert len(grid.cells) == 0


class TestEdgeCasesAndComplexScenarios:
    """Tests for edge cases and complex scenarios."""

    def test_remove_grid_with_cells_that_have_pending_plot_creation(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
    ):
        """Remove grid with pending plots unsubscribes, so commit creates no plots."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        add_cell_with_layer(plot_orchestrator, grid_id, plot_cell[0], plot_cell[1])

        plot_orchestrator.remove_grid(grid_id)
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)

        assert plot_orchestrator.get_grid(grid_id) is None
        assert fake_plotting_controller.call_count() == 0


class TestCellRetrieval:
    """Test that cells can be retrieved via get_cell and PlotDataService."""

    def test_get_cell_returns_cell_config(self, plot_orchestrator, plot_cell):
        """get_cell should return the cell config."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        cell = plot_orchestrator.get_cell(cell_id)
        assert cell is not None
        assert cell.geometry == plot_cell[0]
        assert len(cell.layers) == 1
        assert cell.layers[0].config == plot_cell[1]

    def test_plot_data_service_has_data_after_workflow_commits(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_data_service,
        plot_data_service,
    ):
        """PlotDataService should have data after workflow commits and data arrives."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        cell = plot_orchestrator.get_cell(cell_id)
        layer_id = cell.layers[0].layer_id

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for ALL sources
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for source_name in plot_cell[1].source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name=plot_cell[1].output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # PlotDataService should have data for the layer
        state = plot_data_service.get(layer_id)
        assert state is not None
        assert state.plotter.has_cached_state()  # Has computed data
        assert state.error_message is None

    def test_plot_data_service_has_error_when_plot_creation_fails(
        self,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_data_service,
        fake_stream_manager,
    ):
        """PlotDataService should have error when plot creation fails."""
        # Create plotting controller configured to fail
        failing_controller = FakePlottingController(stream_manager=fake_stream_manager)
        failing_controller.configure_to_raise(ValueError("Plot creation failed"))

        plot_data_service = PlotDataService()
        plot_orchestrator = PlotOrchestrator(
            plotting_controller=failing_controller,
            job_orchestrator=job_orchestrator,
            data_service=fake_data_service,
            instrument='dummy',
            config_store=None,
            plot_data_service=plot_data_service,
        )

        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)
        config = make_plot_config(
            workflow_id,
            source_names=['source1'],
            output_name='test_output',
            plot_name='test_plot',
            params=FakePlotParams(),
        )
        cell_id = add_cell_with_layer(plot_orchestrator, grid_id, geometry, config)

        cell = plot_orchestrator.get_cell(cell_id)
        layer_id = cell.layers[0].layer_id

        # Commit workflow - this triggers eager plotter creation which will fail
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)

        # PlotDataService should have error for the layer
        state = plot_data_service.get(layer_id)
        assert state is not None
        assert state.error_message is not None
        assert "Plot creation failed" in state.error_message

    def test_late_subscriber_can_retrieve_existing_grids_and_plot_state(
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
        initializes and retrieves grid config and plot state from PlotDataService.
        """
        plot_data_service = PlotDataService()
        plot_orchestrator = PlotOrchestrator(
            plotting_controller=fake_plotting_controller,
            job_orchestrator=job_orchestrator,
            data_service=fake_data_service,
            instrument='dummy',
            config_store=None,
            plot_data_service=plot_data_service,
        )

        # Create grid with multiple cells
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        cell_ids = []
        for i in range(3):
            geometry = CellGeometry(row=i, col=0, row_span=1, col_span=1)
            config = make_plot_config(
                workflow_id,
                source_names=[f'source_{i}'],
                output_name=f'output_{i}',
                plot_name=f'plot_{i}',
                params=FakePlotParams(),
            )
            cell_id = add_cell_with_layer(plot_orchestrator, grid_id, geometry, config)
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

        # Late subscriber retrieves plot state from PlotDataService
        for cell_id in cell_ids:
            cell = plot_orchestrator.get_cell(cell_id)
            for layer in cell.layers:
                state = plot_data_service.get(layer.layer_id)
                assert state is not None
                assert state.error_message is None
                assert state.plotter.has_cached_state()  # Has computed data

    def test_layer_config_update_recreates_plotter_with_running_workflow(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_data_service,
        fake_plotting_controller,
        plot_data_service,
    ):
        """
        When layer config is updated and workflow is running, plotter is recreated.
        """
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        # Get layer_id and config
        cell = plot_orchestrator.get_cell(cell_id)
        layer_id = cell.layers[0].layer_id
        config = cell.layers[0].config

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number1 = job_ids[0].job_number

        # Simulate data arrival for ALL sources
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for source_name in config.source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number1),
                output_name=config.output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Verify layer has data in PlotDataService
        state1 = plot_data_service.get(layer_id)
        assert state1 is not None
        assert state1.plotter.has_cached_state()

        # Update layer config while workflow is still running
        new_config = make_plot_config(
            workflow_id,
            source_names=['new_source'],
            output_name='new_output',
            plot_name='new_plot',
            params=FakePlotParams(),
        )
        plot_orchestrator.update_layer_config(layer_id, new_config)

        # Simulate data arrival for new config
        result_key2 = ResultKey(
            workflow_id=workflow_id,
            job_id=JobId(source_name='new_source', job_number=job_number1),
            output_name='new_output',
        )
        fake_data_service[result_key2] = sc.scalar(2.0)

        # Verify plotter was recreated with new config
        assert fake_plotting_controller.call_count() == 2
        setups = fake_plotting_controller.get_pipeline_setups()
        assert setups[1]['source_names'] == ['new_source']
        assert setups[1]['output_name'] == 'new_output'

    def test_plot_data_service_cleared_when_cell_removed(
        self,
        plot_orchestrator,
        plot_cell,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_data_service,
        plot_data_service,
    ):
        """PlotDataService should be cleaned up when cell is removed."""
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        cell = plot_orchestrator.get_cell(cell_id)
        layer_id = cell.layers[0].layer_id

        # Commit workflow
        job_ids = commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)
        job_number = job_ids[0].job_number

        # Simulate data arrival for ALL sources
        import scipp as sc

        from ess.livedata.config.workflow_spec import JobId, ResultKey

        for source_name in plot_cell[1].source_names:
            result_key = ResultKey(
                workflow_id=workflow_id,
                job_id=JobId(source_name=source_name, job_number=job_number),
                output_name=plot_cell[1].output_name,
            )
            fake_data_service[result_key] = sc.scalar(1.0)

        # Verify layer has data in PlotDataService
        state = plot_data_service.get(layer_id)
        assert state is not None
        assert state.plotter.has_cached_state()

        # Remove cell
        plot_orchestrator.remove_cell(cell_id)

        # PlotDataService should no longer have the layer
        assert plot_data_service.get(layer_id) is None

        # get_cell should raise KeyError since cell doesn't exist
        import pytest

        with pytest.raises(KeyError):
            plot_orchestrator.get_cell(cell_id)


class TestSourceNameFiltering:
    """Test that plots only create when their specific source_names have data."""

    def test_plotter_created_when_workflow_commits(
        self,
        plot_orchestrator,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """
        Plotter is created eagerly when workflow commits.

        With the simplified architecture, plotter creation happens when the
        workflow starts (job committed), not when data arrives. The data
        pipeline handles filtering by source_name.
        """
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)

        # Create a plot that wants data from source_A
        geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)
        config = make_plot_config(
            workflow_id,
            source_names=['source_A'],  # Plot wants source_A
            output_name='test_output',
            plot_name='test_plot',
            params=FakePlotParams(),
        )
        add_cell_with_layer(plot_orchestrator, grid_id, geometry, config)

        # Commit workflow - plotter is created eagerly
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)

        # Plotter created when workflow commits (regardless of data)
        assert fake_plotting_controller.call_count() == 1, (
            "Plotter should be created when workflow commits"
        )
        calls = fake_plotting_controller.get_calls()
        assert calls[0]['_from_create_plotter'] is True

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
        geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)
        config = make_plot_config(
            workflow_id,
            source_names=['source_A', 'source_B'],  # Plot wants BOTH
            output_name='test_output',
            plot_name='test_plot',
            params=FakePlotParams(),
        )
        add_cell_with_layer(plot_orchestrator, grid_id, geometry, config)

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

        # Plotter SHOULD be created with partial data (progressive plotting)
        assert fake_plotting_controller.call_count() == 1, (
            "Plotter should be created as soon as first source has data"
        )
        calls = fake_plotting_controller.get_calls()
        # With multi-session architecture, create_plotter is called
        assert calls[0]['_from_create_plotter'] is True

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
        geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)
        config = make_plot_config(
            workflow_id,
            source_names=['source_A'],
            output_name='test_output',
            plot_name='test_plot',
            params=FakePlotParams(),
        )
        add_cell_with_layer(plot_orchestrator, grid_id, geometry, config)

        # Plot should be created immediately (data already exists)
        assert fake_plotting_controller.call_count() == 1, (
            "Plot should be created immediately when subscribing to running job"
        )

    def test_late_subscription_creates_plotter_immediately(
        self,
        plot_orchestrator,
        workflow_id,
        workflow_spec,
        job_orchestrator,
        fake_plotting_controller,
        fake_data_service,
    ):
        """
        Late subscription creates plotter immediately when workflow running.

        With eager plotter creation, subscribing to an already-running workflow
        creates the plotter immediately. The data pipeline handles filtering
        by source_name.
        """
        # Commit workflow FIRST
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)

        # NOW add the plot wanting source_A (late subscription)
        grid_id = plot_orchestrator.add_grid(title='Test Grid', nrows=3, ncols=3)
        geometry = CellGeometry(row=0, col=0, row_span=1, col_span=1)
        config = make_plot_config(
            workflow_id,
            source_names=['source_A'],
            output_name='test_output',
            plot_name='test_plot',
            params=FakePlotParams(),
        )
        add_cell_with_layer(plot_orchestrator, grid_id, geometry, config)

        # Plotter is created immediately (workflow already running)
        assert fake_plotting_controller.call_count() == 1, (
            "Plotter should be created when subscribing to running workflow"
        )


class TestPlotConfigIsStatic:
    """Tests for PlotConfig.is_static() method."""

    def test_is_static_returns_true_for_single_data_source_with_empty_source_names(
        self, workflow_id
    ):
        """Static overlay marker: single data source with empty source_names."""
        from ess.livedata.dashboard.data_roles import PRIMARY

        data_source = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=[],  # Empty source_names = static
            output_name='My Custom Overlay',
        )
        config = PlotConfig(
            data_sources={PRIMARY: data_source},
            plot_name='rectangles',
            params=FakePlotParams(),
        )
        assert config.is_static() is True

    def test_is_static_returns_false_for_single_data_source_with_source_names(
        self, workflow_id
    ):
        """Normal plot: single data source with actual source_names."""
        from ess.livedata.dashboard.data_roles import PRIMARY

        data_source = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=['source1'],
            output_name='result',
        )
        config = PlotConfig(
            data_sources={PRIMARY: data_source},
            plot_name='test_plot',
            params=FakePlotParams(),
        )
        assert config.is_static() is False

    def test_is_static_returns_false_for_empty_data_sources(self, workflow_id):
        """Edge case: empty data_sources dict is not static (invalid config)."""
        config = PlotConfig(
            data_sources={},
            plot_name='test_plot',
            params=FakePlotParams(),
        )
        assert config.is_static() is False

    def test_is_static_returns_false_for_multiple_data_sources(self, workflow_id):
        """Future correlation histograms: multiple data sources are not static."""
        from ess.livedata.dashboard.data_roles import PRIMARY, X_AXIS

        data_source1 = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=[],
            output_name='output1',
        )
        data_source2 = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=[],
            output_name='output2',
        )
        config = PlotConfig(
            data_sources={PRIMARY: data_source1, X_AXIS: data_source2},
            plot_name='test_plot',
            params=FakePlotParams(),
        )
        assert config.is_static() is False

    def test_static_config_can_access_output_name(self, workflow_id):
        """Static overlay's output_name holds the user's custom name."""
        from ess.livedata.dashboard.data_roles import PRIMARY

        data_source = DataSourceConfig(
            workflow_id=workflow_id,
            source_names=[],
            output_name='My Rectangles',
        )
        config = PlotConfig(
            data_sources={PRIMARY: data_source},
            plot_name='rectangles',
            params=FakePlotParams(),
        )
        assert config.output_name == 'My Rectangles'


class TestDataSubscriptionCleanup:
    """Tests for data subscription cleanup on workflow restart."""

    def test_workflow_restart_cleans_up_old_data_subscription(
        self,
        plot_orchestrator,
        plot_cell,
        job_orchestrator,
        workflow_id,
        workflow_spec,
        fake_data_service,
    ):
        """When workflow restarts, old data subscription is unregistered."""
        # Add cell with layer
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=1, ncols=1)
        add_cell_with_layer(plot_orchestrator, grid_id, plot_cell[0], plot_cell[1])

        # Commit workflow to create first data subscription
        job_ids_1 = commit_workflow_for_test(
            job_orchestrator, workflow_id, workflow_spec
        )

        # Check subscriber count after first commit
        initial_subscriber_count = len(fake_data_service._subscribers)
        assert initial_subscriber_count >= 1

        # Restart workflow (stop + start)
        job_orchestrator.stop_workflow(workflow_id)
        job_ids_2 = commit_workflow_for_test(
            job_orchestrator, workflow_id, workflow_spec
        )

        # Subscriber count should stay the same (old cleaned up, new added)
        assert len(fake_data_service._subscribers) == initial_subscriber_count
        # Job numbers should be different
        assert job_ids_1[0].job_number != job_ids_2[0].job_number

    def test_layer_removal_cleans_up_data_subscription(
        self,
        plot_orchestrator,
        plot_cell,
        job_orchestrator,
        workflow_id,
        workflow_spec,
        fake_data_service,
    ):
        """When layer is removed, its data subscription is unregistered."""
        # Add cell with layer
        grid_id = plot_orchestrator.add_grid(title='Test', nrows=1, ncols=1)
        cell_id = add_cell_with_layer(
            plot_orchestrator, grid_id, plot_cell[0], plot_cell[1]
        )

        # Commit workflow to create data subscription
        commit_workflow_for_test(job_orchestrator, workflow_id, workflow_spec)

        # Check subscriber count
        subscriber_count_with_layer = len(fake_data_service._subscribers)
        assert subscriber_count_with_layer >= 1

        # Remove the cell (which removes the layer)
        plot_orchestrator.remove_cell(cell_id)

        # Subscriber should be removed
        assert len(fake_data_service._subscribers) == subscriber_count_with_layer - 1
