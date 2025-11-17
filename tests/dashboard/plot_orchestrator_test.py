# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import holoviews as hv
import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import (
    JobId,
    JobNumber,
    ResultKey,
    WorkflowId,
    WorkflowSpec,
)
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plot_orchestrator import (
    PlotCell,
    PlotConfig,
    PlotOrchestrator,
)
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.stream_manager import StreamManager

hv.extension('bokeh')


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
def workflow_spec(workflow_id):
    """Create a test WorkflowSpec."""
    return WorkflowSpec(
        instrument=workflow_id.instrument,
        namespace=workflow_id.namespace,
        name=workflow_id.name,
        version=workflow_id.version,
        title="Test Workflow",
        description="A test workflow",
        source_names=['detector_1', 'detector_2'],
        params=None,
    )


@pytest.fixture
def workflow_registry(workflow_id, workflow_spec):
    """Create a test workflow registry."""
    return {workflow_id: workflow_spec}


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
def plot_orchestrator(job_service, plotting_controller, workflow_registry):
    """Create a PlotOrchestrator for testing."""
    return PlotOrchestrator(
        job_service=job_service,
        plotting_controller=plotting_controller,
        workflow_registry=workflow_registry,
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
        assert grid.cells[0].plot is None
        assert grid.cells[0].job_number is None
        assert grid.cells[0].error is None

    def test_add_plot_with_matching_job(
        self,
        plot_orchestrator,
        job_service,
        data_service,
        workflow_id,
        job_number,
        detector_data,
    ):
        """Test adding a plot configuration when a matching job already exists."""
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
        assert grid.cells[0].job_number == job_number
        assert grid.cells[0].plot is not None
        assert isinstance(grid.cells[0].plot, hv.DynamicMap)
        assert grid.cells[0].error is None

    def test_refresh_plots(
        self,
        plot_orchestrator,
        job_service,
        data_service,
        workflow_id,
        job_number,
        detector_data,
    ):
        """Test refreshing plots after job becomes available."""
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

        plot_orchestrator.add_plot(grid_id, cell)

        grid = plot_orchestrator.get_grid(grid_id)
        assert grid.cells[0].plot is None

        job_id = JobId(source_name='detector_1', job_number=job_number)
        result_key = ResultKey(
            workflow_id=workflow_id,
            job_id=job_id,
            output_name='intensity',
        )

        # Adding data to data_service automatically registers the job
        # via data_updated callback
        data_service[result_key] = detector_data

        plot_orchestrator.refresh_plots()

        assert grid.cells[0].job_number == job_number
        assert grid.cells[0].plot is not None
        assert isinstance(grid.cells[0].plot, hv.DynamicMap)
        assert grid.cells[0].error is None

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
        job_service,
        data_service,
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

        # Verify initial plot was created
        grid = plot_orchestrator.get_grid(grid_id)
        assert grid.cells[0].plot is not None

        # Update configuration with same workflow but different output
        new_config = PlotConfig(
            workflow_id=workflow_id,
            output_name='spectrum',  # Changed output
            source_names=['detector_1'],
            plot_name='lines',
            params={},
        )

        plot_orchestrator.update_plot_config(cell_id, new_config)

        # Verify config was updated
        updated_config = plot_orchestrator.get_plot_config(cell_id)
        assert updated_config.output_name == 'spectrum'
        assert updated_config.source_names == ['detector_1']

        # The config change was successful - we don't test whether
        # the plot was created since that depends on data availability
