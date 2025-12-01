# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import holoviews as hv
import panel as pn
import pydantic
import pytest

from ess.livedata.config.grid_templates import GridSpec
from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plot_orchestrator import (
    CellGeometry,
    PlotCell,
    PlotConfig,
    PlotOrchestrator,
)
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.stream_manager import StreamManager
from ess.livedata.dashboard.widgets.plot_grid_manager import PlotGridManager

hv.extension('bokeh')


class EmptyParams(pydantic.BaseModel):
    """Empty params model for testing."""

    pass


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
    return StreamManager(data_service=data_service, pipe_factory=hv.streams.Pipe)


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
def plot_orchestrator(plotting_controller, job_orchestrator, fake_data_service):
    """Create a PlotOrchestrator for testing."""
    return PlotOrchestrator(
        plotting_controller=plotting_controller,
        job_orchestrator=job_orchestrator,
        data_service=fake_data_service,
    )


@pytest.fixture
def grid_manager(plot_orchestrator, workflow_registry):
    """Create a PlotGridManager for testing."""
    return PlotGridManager(
        orchestrator=plot_orchestrator, workflow_registry=workflow_registry
    )


class TestPlotGridManagerInitialization:
    """Tests for PlotGridManager initialization."""

    def test_creates_panel_widget(self, grid_manager):
        """Test that manager creates a Panel widget."""
        assert isinstance(grid_manager.panel, pn.Column)


class TestMultipleManagers:
    """Tests for multiple manager instances (multi-user scenario)."""

    def test_multiple_managers_stay_synchronized(
        self, plot_orchestrator, workflow_registry
    ):
        """Test that multiple managers sharing same orchestrator stay in sync."""
        # Create managers which register as callbacks with orchestrator
        _manager1 = PlotGridManager(
            orchestrator=plot_orchestrator, workflow_registry=workflow_registry
        )
        _manager2 = PlotGridManager(
            orchestrator=plot_orchestrator, workflow_registry=workflow_registry
        )

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


@pytest.fixture
def sample_template():
    """Create a sample grid template for testing."""
    cell = PlotCell(
        geometry=CellGeometry(row=0, col=0, row_span=2, col_span=2),
        config=PlotConfig(
            workflow_id=WorkflowId.from_string('test/ns/wf/1'),
            output_name='output',
            source_names=['source1'],
            plot_name='lines',
            params=EmptyParams(),
        ),
    )
    return GridSpec(
        name='Test Template',
        title='Template Grid',
        description='',
        nrows=3,
        ncols=4,
        cells=(cell,),
    )


class TestTemplateSupport:
    """Tests for grid template support in PlotGridManager."""

    def test_template_selector_hidden_when_no_templates(
        self, plot_orchestrator, workflow_registry
    ):
        """Test that template selector is hidden when no templates provided."""
        manager = PlotGridManager(
            orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            templates=[],
        )

        assert manager._template_selector.visible is False

    def test_template_selector_visible_when_templates_provided(
        self, plot_orchestrator, workflow_registry, sample_template
    ):
        """Test that template selector is visible when templates are provided."""
        manager = PlotGridManager(
            orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            templates=[sample_template],
        )

        assert manager._template_selector.visible is True
        # Should have "-- No template --" plus the template
        assert len(manager._template_selector.options) == 2

    def test_selecting_template_populates_fields(
        self, plot_orchestrator, workflow_registry, sample_template
    ):
        """Test that selecting a template populates the form fields."""
        manager = PlotGridManager(
            orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            templates=[sample_template],
        )

        # Simulate template selection
        manager._template_selector.value = 'Test Template'

        assert manager._title_input.value == 'Template Grid'
        assert manager._nrows_input.value == 3
        assert manager._ncols_input.value == 4

    def test_selecting_template_sets_minimum_rows_cols(
        self, plot_orchestrator, workflow_registry, sample_template
    ):
        """Test that selecting template sets min rows/cols based on cell geometry."""
        manager = PlotGridManager(
            orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            templates=[sample_template],
        )

        # Before selection, min should be 2
        assert manager._nrows_input.start == 2
        assert manager._ncols_input.start == 2

        # After selection, min should be computed from cells
        manager._template_selector.value = 'Test Template'

        # Cell at (0,0) with span (2,2) requires min 2 rows and 2 cols
        assert manager._nrows_input.start == 2
        assert manager._ncols_input.start == 2

    def test_deselecting_template_resets_fields(
        self, plot_orchestrator, workflow_registry, sample_template
    ):
        """Test that deselecting template resets to defaults."""
        manager = PlotGridManager(
            orchestrator=plot_orchestrator,
            workflow_registry=workflow_registry,
            templates=[sample_template],
        )

        # Select then deselect
        manager._template_selector.value = 'Test Template'
        manager._template_selector.value = '-- No template --'

        assert manager._title_input.value == 'New Grid'
        assert manager._nrows_input.value == 3
        assert manager._ncols_input.value == 3
        assert manager._nrows_input.start == 2
        assert manager._ncols_input.start == 2
