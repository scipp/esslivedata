# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import holoviews as hv
import panel as pn
import pytest
import yaml

from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.plot_data_service import PlotDataService
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
def plot_data_service():
    """Create a PlotDataService for testing."""
    return PlotDataService()


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


class TestGridUpload:
    """Tests for grid config upload functionality.

    Upload follows a preview-based flow:
    1. User uploads file -> form fields populated, preview shown
    2. User clicks "Add Grid" -> grid is created
    """

    def test_upload_populates_form_fields(self, grid_manager):
        """Test that uploading a file populates form fields."""
        yaml_content = b"title: Uploaded Grid\nnrows: 4\nncols: 5\ncells: []"

        class FakeEvent:
            new = yaml_content

        grid_manager._on_file_uploaded(FakeEvent())

        # Form fields should be populated
        assert grid_manager._title_input.value == 'Uploaded Grid'
        assert grid_manager._nrows_input.value == 4
        assert grid_manager._ncols_input.value == 5

    def test_upload_sets_pending_cells(self, grid_manager):
        """Test that upload stores parsed cells for later creation."""
        yaml_content = b"title: Test\nnrows: 2\nncols: 2\ncells: []"

        class FakeEvent:
            new = yaml_content

        grid_manager._on_file_uploaded(FakeEvent())

        # Pending cells should be set (empty list for this config)
        assert grid_manager._pending_upload_cells is not None
        assert grid_manager._pending_upload_cells == []

    def test_upload_then_add_grid_creates_grid(self, grid_manager, plot_orchestrator):
        """Test that uploading then clicking Add Grid creates the grid."""
        yaml_content = b"title: Uploaded Grid\nnrows: 4\nncols: 5\ncells: []"

        class FakeEvent:
            new = yaml_content

        grid_manager._on_file_uploaded(FakeEvent())

        # No grid created yet - just preview
        assert len(plot_orchestrator.get_all_grids()) == 0

        # Simulate clicking Add Grid
        grid_manager._on_add_grid(None)

        # Now grid should be created
        grids = plot_orchestrator.get_all_grids()
        assert len(grids) == 1
        grid = next(iter(grids.values()))
        assert grid.title == 'Uploaded Grid'
        assert grid.nrows == 4
        assert grid.ncols == 5

    def test_upload_invalid_yaml_does_not_update_form(self, grid_manager):
        """Test that invalid YAML does not update form fields."""
        invalid_yaml = b"title: [unbalanced bracket"
        original_title = grid_manager._title_input.value

        class FakeEvent:
            new = invalid_yaml

        # Should not raise, error handled gracefully
        grid_manager._on_file_uploaded(FakeEvent())

        # Form should remain unchanged
        assert grid_manager._title_input.value == original_title
        assert grid_manager._pending_upload_cells is None

    def test_upload_non_dict_does_not_update_form(self, grid_manager):
        """Test that non-dict YAML does not update form fields."""
        yaml_content = b"- item1\n- item2"
        original_title = grid_manager._title_input.value

        class FakeEvent:
            new = yaml_content

        # Should not raise, error handled gracefully
        grid_manager._on_file_uploaded(FakeEvent())

        # Form should remain unchanged
        assert grid_manager._title_input.value == original_title
        assert grid_manager._pending_upload_cells is None

    def test_upload_invalid_cell_does_not_update_form(self, grid_manager):
        """Test that cell with missing geometry does not update form fields."""
        yaml_content = b"""
title: Test Grid
cells:
  - config:
      workflow_id: test
"""  # Missing 'geometry' field
        original_title = grid_manager._title_input.value

        class FakeEvent:
            new = yaml_content

        # Should not raise, error handled gracefully
        grid_manager._on_file_uploaded(FakeEvent())

        # Form should remain unchanged (error occurred before form update)
        assert grid_manager._title_input.value == original_title
        assert grid_manager._pending_upload_cells is None

    def test_upload_uses_defaults_for_missing_fields(self, grid_manager):
        """Test that missing fields use sensible defaults."""
        from ess.livedata.dashboard.widgets.plot_grid_manager import _MODE_UPLOAD

        # Switch to Upload mode first
        grid_manager._mode_selector.value = _MODE_UPLOAD

        yaml_content = b"cells: []"  # Missing title, nrows, ncols

        class FakeEvent:
            new = yaml_content

        grid_manager._on_file_uploaded(FakeEvent())

        # Defaults should be applied
        assert grid_manager._title_input.value == 'Uploaded Grid'
        assert grid_manager._nrows_input.value == 3
        assert grid_manager._ncols_input.value == 3

    def test_upload_triggers_lifecycle_on_add(
        self, plot_orchestrator, workflow_registry
    ):
        """Test that Add Grid after upload triggers lifecycle callback."""
        from ess.livedata.dashboard.widgets.plot_grid_manager import _MODE_UPLOAD

        created_grids = []

        # Subscribe to lifecycle events
        plot_orchestrator.subscribe_to_lifecycle(
            on_grid_created=lambda grid_id, config: created_grids.append(
                (grid_id, config)
            ),
        )

        manager = PlotGridManager(
            orchestrator=plot_orchestrator, workflow_registry=workflow_registry
        )

        # Switch to Upload mode
        manager._mode_selector.value = _MODE_UPLOAD

        yaml_content = b"title: Sync Test\nnrows: 2\nncols: 2\ncells: []"

        class FakeEvent:
            new = yaml_content

        manager._on_file_uploaded(FakeEvent())

        # No callback yet - grid not created
        assert len(created_grids) == 0

        # Now click Add Grid
        manager._on_add_grid(None)

        # Lifecycle callback should have been called
        assert len(created_grids) == 1
        assert created_grids[0][1].title == 'Sync Test'

    def test_upload_none_value_is_ignored(self, grid_manager, plot_orchestrator):
        """Test that None value (cleared file input) is ignored."""

        class FakeEvent:
            new = None

        grid_manager._on_file_uploaded(FakeEvent())

        # Nothing should change
        assert grid_manager._pending_upload_cells is None
        assert len(plot_orchestrator.get_all_grids()) == 0

    def test_add_grid_clears_pending_upload(self, grid_manager, plot_orchestrator):
        """Test that Add Grid clears the pending upload state."""
        from ess.livedata.dashboard.widgets.plot_grid_manager import _MODE_UPLOAD

        # Switch to Upload mode
        grid_manager._mode_selector.value = _MODE_UPLOAD

        yaml_content = b"title: Test\nnrows: 2\nncols: 2\ncells: []"

        class FakeEvent:
            new = yaml_content

        grid_manager._on_file_uploaded(FakeEvent())
        assert grid_manager._pending_upload_cells is not None

        # Click Add Grid
        grid_manager._on_add_grid(None)

        # Pending upload should be cleared
        assert grid_manager._pending_upload_cells is None

    def test_source_indicator_shows_uploaded_configuration(self, grid_manager):
        """Test that source indicator updates when file is uploaded in Upload mode."""
        from ess.livedata.dashboard.widgets.plot_grid_manager import _MODE_UPLOAD

        # Initially should show empty grid (in Template mode)
        assert 'Empty grid' in grid_manager._source_indicator.object

        # Switch to Upload mode
        grid_manager._mode_selector.value = _MODE_UPLOAD
        # Before upload, shows "No file uploaded"
        assert 'No file uploaded' in grid_manager._source_indicator.object

        # Upload a file
        yaml_content = b"title: Test\nnrows: 2\nncols: 2\ncells: []"

        class FakeEvent:
            new = yaml_content

        grid_manager._file_input.filename = 'test_config.yaml'
        grid_manager._on_file_uploaded(FakeEvent())

        # Should now show uploaded filename
        assert 'Uploaded: test_config.yaml' in grid_manager._source_indicator.object


class TestModeSwitch:
    """Tests for the mode switch (Template/Upload) functionality."""

    def test_mode_switch_shows_template_selector_in_template_mode(self, grid_manager):
        """Test that template selector is visible in Template mode."""
        from ess.livedata.dashboard.widgets.plot_grid_manager import _MODE_TEMPLATE

        # Start in Template mode (default)
        assert grid_manager._mode_selector.value == _MODE_TEMPLATE
        assert grid_manager._template_selector.visible is True
        assert grid_manager._file_input.visible is False

    def test_mode_switch_shows_upload_in_upload_mode(self, grid_manager):
        """Test that file input is visible in Upload mode."""
        from ess.livedata.dashboard.widgets.plot_grid_manager import _MODE_UPLOAD

        # Switch to Upload mode
        grid_manager._mode_selector.value = _MODE_UPLOAD

        assert grid_manager._template_selector.visible is False
        assert grid_manager._file_input.visible is True

    def test_mode_switch_preserves_template_selection(self, grid_manager):
        """Test that switching modes preserves the template selection."""
        from ess.livedata.dashboard.widgets.plot_grid_manager import (
            _MODE_TEMPLATE,
            _MODE_UPLOAD,
        )

        # Record initial state
        initial_template_name = grid_manager._template_selector.value
        initial_template_object = grid_manager._selected_template

        # Switch to Upload mode and back
        grid_manager._mode_selector.value = _MODE_UPLOAD
        grid_manager._mode_selector.value = _MODE_TEMPLATE

        # Template selection should be preserved
        assert grid_manager._template_selector.value == initial_template_name
        assert grid_manager._selected_template == initial_template_object

    def test_switching_to_template_mode_clears_upload_from_preview(self, grid_manager):
        """Test that switching to Template mode shows template preview, not upload."""
        from ess.livedata.dashboard.widgets.plot_grid_manager import (
            _MODE_TEMPLATE,
            _MODE_UPLOAD,
        )

        # Switch to Upload mode and upload a file
        grid_manager._mode_selector.value = _MODE_UPLOAD
        yaml_content = b"title: Test\nnrows: 2\nncols: 2\ncells: []"

        class FakeEvent:
            new = yaml_content

        grid_manager._file_input.filename = 'test.yaml'
        grid_manager._on_file_uploaded(FakeEvent())
        assert 'Uploaded: test.yaml' in grid_manager._source_indicator.object

        # Switch back to Template mode
        grid_manager._mode_selector.value = _MODE_TEMPLATE

        # Should show empty grid (no template selected)
        assert 'Empty grid' in grid_manager._source_indicator.object

    def test_add_grid_resets_to_template_mode(self, grid_manager, plot_orchestrator):
        """Test that Add Grid resets the UI to Template mode."""
        from ess.livedata.dashboard.widgets.plot_grid_manager import (
            _MODE_TEMPLATE,
            _MODE_UPLOAD,
        )

        # Switch to Upload mode and upload
        grid_manager._mode_selector.value = _MODE_UPLOAD
        yaml_content = b"title: Test\nnrows: 2\nncols: 2\ncells: []"

        class FakeEvent:
            new = yaml_content

        grid_manager._on_file_uploaded(FakeEvent())

        # Click Add Grid
        grid_manager._on_add_grid(None)

        # Should be back in Template mode with defaults
        assert grid_manager._mode_selector.value == _MODE_TEMPLATE
        assert grid_manager._pending_upload_cells is None
        assert grid_manager._title_input.value == 'New Grid'
