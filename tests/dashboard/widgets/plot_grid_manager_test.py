# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import holoviews as hv
import panel as pn
import pytest
import yaml

from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plot_orchestrator import PlotOrchestrator
from ess.livedata.dashboard.plotting_controller import PlottingController
from ess.livedata.dashboard.stream_manager import StreamManager
from ess.livedata.dashboard.widgets.plot_grid_manager import (
    GridRow,
    PlotGridManager,
    _sanitize_filename,
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
def plot_orchestrator(plotting_controller, job_orchestrator, fake_data_service):
    """Create a PlotOrchestrator for testing."""
    return PlotOrchestrator(
        plotting_controller=plotting_controller,
        job_orchestrator=job_orchestrator,
        data_service=fake_data_service,
        instrument='dummy',
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


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_replaces_spaces_with_underscores(self):
        assert _sanitize_filename('My Grid Name') == 'my_grid_name'

    def test_removes_invalid_characters(self):
        assert _sanitize_filename('Grid<>:"/\\|?*Test') == 'gridtest'

    def test_lowercases_result(self):
        assert _sanitize_filename('UPPERCASE') == 'uppercase'

    def test_handles_empty_string(self):
        assert _sanitize_filename('') == ''


class TestGridRow:
    """Tests for GridRow widget."""

    def test_creates_panel_row(self, plot_orchestrator, workflow_registry):
        """Test that GridRow creates a Panel Row widget."""
        from io import StringIO

        from ess.livedata.dashboard.plot_orchestrator import PlotGridConfig

        config = PlotGridConfig(title='Test Grid', nrows=3, ncols=4)

        # Dummy callbacks
        def on_remove():
            pass

        def get_yaml():
            return StringIO('test: value')

        row = GridRow(
            grid_id=None,  # type: ignore[arg-type]
            grid_config=config,
            instrument='dummy',
            on_remove=on_remove,
            get_yaml_content=get_yaml,
        )

        assert isinstance(row.panel, pn.Row)

    def test_download_button_has_correct_filename(
        self, plot_orchestrator, workflow_registry
    ):
        """Test that download button filename includes instrument and grid name."""
        from io import StringIO

        from ess.livedata.dashboard.plot_orchestrator import PlotGridConfig

        config = PlotGridConfig(title='My Test Grid', nrows=2, ncols=3)

        row = GridRow(
            grid_id=None,  # type: ignore[arg-type]
            grid_config=config,
            instrument='bifrost',
            on_remove=lambda: None,
            get_yaml_content=lambda: StringIO(''),
        )

        # The download button is the second element in the row (index 1)
        download_button = row.panel[1]
        assert download_button.filename == 'esslivedata_bifrost_my_test_grid.yaml'

    def test_displays_grid_info(self, plot_orchestrator, workflow_registry):
        """Test that GridRow displays grid title and dimensions."""
        from io import StringIO

        from ess.livedata.dashboard.plot_orchestrator import PlotGridConfig

        config = PlotGridConfig(title='My Test Grid', nrows=2, ncols=5)

        row = GridRow(
            grid_id=None,  # type: ignore[arg-type]
            grid_config=config,
            instrument='dummy',
            on_remove=lambda: None,
            get_yaml_content=lambda: StringIO(''),
        )

        # Find the label pane
        label_pane = row.panel[0]
        assert 'My Test Grid' in label_pane.object
        assert '2x5' in label_pane.object


class TestGridDownload:
    """Tests for grid config download functionality."""

    def test_serialize_grid_returns_valid_dict(
        self, plot_orchestrator, workflow_registry
    ):
        """Test that serialize_grid returns a dict with expected structure."""
        grid_id = plot_orchestrator.add_grid(title='Download Test', nrows=4, ncols=5)

        result = plot_orchestrator.serialize_grid(grid_id)

        assert result['title'] == 'Download Test'
        assert result['nrows'] == 4
        assert result['ncols'] == 5
        assert 'cells' in result
        assert isinstance(result['cells'], list)

    def test_serialize_grid_produces_valid_yaml(
        self, plot_orchestrator, workflow_registry
    ):
        """Test that serialized grid can be dumped to YAML."""
        grid_id = plot_orchestrator.add_grid(title='YAML Test', nrows=3, ncols=3)

        result = plot_orchestrator.serialize_grid(grid_id)
        yaml_str = yaml.dump(result, default_flow_style=False)

        # Should be valid YAML that can be loaded back
        loaded = yaml.safe_load(yaml_str)
        assert loaded['title'] == 'YAML Test'

    def test_yaml_callback_generates_file_content(
        self, grid_manager, plot_orchestrator, workflow_registry
    ):
        """Test that the YAML callback generates downloadable content."""
        grid_id = plot_orchestrator.add_grid(title='Callback Test', nrows=2, ncols=2)

        # Get the callback from the manager
        callback = grid_manager._make_yaml_callback(grid_id)
        sio = callback()

        # Should return a StringIO with YAML content
        content = sio.read()
        assert 'title: Callback Test' in content
        assert 'nrows: 2' in content
        assert 'ncols: 2' in content

    def test_serialize_grid_raises_for_unknown_grid(self, plot_orchestrator):
        """Test that serialize_grid raises KeyError for unknown grid ID."""
        from uuid import uuid4

        from ess.livedata.dashboard.plot_orchestrator import GridId

        fake_grid_id = GridId(uuid4())

        with pytest.raises(KeyError):
            plot_orchestrator.serialize_grid(fake_grid_id)
