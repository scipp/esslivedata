# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for WorkflowStatusWidget and WorkflowStatusListWidget."""

import panel as pn
import pytest

from ess.livedata.config.workflow_spec import JobId, WorkflowId
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.dashboard.icons import get_icon
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.widgets.workflow_status_widget import (
    WorkflowStatusListWidget,
    WorkflowStatusWidget,
    _get_unconfigured_sources,
    _group_configs_by_equality,
)


def _get_button_icons(buttons) -> list[str | None]:
    """Extract icons from a collection of buttons for testing."""
    return [obj.icon for obj in buttons if isinstance(obj, pn.widgets.Button)]


def _has_icon(icons: list[str | None], icon_name: str) -> bool:
    """Check if any icon matches the given icon name."""
    expected_icon = get_icon(icon_name)
    return expected_icon in icons


@pytest.fixture
def job_service():
    """Create a JobService for testing."""
    return JobService()


@pytest.fixture
def workflow_status_widget(
    workflow_id,
    workflow_spec,
    job_orchestrator,
    job_service,
):
    """Create a WorkflowStatusWidget for testing."""
    return WorkflowStatusWidget(
        workflow_id=workflow_id,
        workflow_spec=workflow_spec,
        orchestrator=job_orchestrator,
        job_service=job_service,
    )


class TestConfigGrouping:
    """Tests for configuration grouping logic."""

    def test_empty_staged_returns_empty_list(self):
        """Test that empty staged config returns empty groups."""
        result = _group_configs_by_equality({}, {})
        assert result == []

    def test_single_source_single_group(self):
        """Test that a single source creates a single group."""
        from ess.livedata.dashboard.job_orchestrator import JobConfig

        staged = {
            'source1': JobConfig(params={'threshold': 100.0}, aux_source_names={}),
        }
        result = _group_configs_by_equality(staged, {})

        assert len(result) == 1
        assert result[0].source_names == ('source1',)
        assert result[0].params == {'threshold': 100.0}

    def test_identical_configs_grouped_together(self):
        """Test that sources with identical configs are grouped."""
        from ess.livedata.dashboard.job_orchestrator import JobConfig

        staged = {
            'source1': JobConfig(params={'threshold': 100.0}, aux_source_names={}),
            'source2': JobConfig(params={'threshold': 100.0}, aux_source_names={}),
        }
        result = _group_configs_by_equality(staged, {})

        assert len(result) == 1
        assert set(result[0].source_names) == {'source1', 'source2'}

    def test_different_configs_separate_groups(self):
        """Test that sources with different configs are in separate groups."""
        from ess.livedata.dashboard.job_orchestrator import JobConfig

        staged = {
            'source1': JobConfig(params={'threshold': 100.0}, aux_source_names={}),
            'source2': JobConfig(params={'threshold': 200.0}, aux_source_names={}),
        }
        result = _group_configs_by_equality(staged, {})

        assert len(result) == 2
        source_names = {g.source_names[0] for g in result}
        assert source_names == {'source1', 'source2'}

    def test_modified_flag_set_when_differs_from_active(self):
        """Test that is_modified is True when staged differs from active."""
        from ess.livedata.dashboard.job_orchestrator import JobConfig

        staged = {
            'source1': JobConfig(params={'threshold': 200.0}, aux_source_names={}),
        }
        active = {
            'source1': JobConfig(params={'threshold': 100.0}, aux_source_names={}),
        }
        result = _group_configs_by_equality(staged, active)

        assert len(result) == 1
        assert result[0].is_modified is True

    def test_modified_flag_false_when_matches_active(self):
        """Test that is_modified is False when staged matches active."""
        from ess.livedata.dashboard.job_orchestrator import JobConfig

        staged = {
            'source1': JobConfig(params={'threshold': 100.0}, aux_source_names={}),
        }
        active = {
            'source1': JobConfig(params={'threshold': 100.0}, aux_source_names={}),
        }
        result = _group_configs_by_equality(staged, active)

        assert len(result) == 1
        assert result[0].is_modified is False

    def test_modified_when_source_not_in_active(self):
        """Test that is_modified is True when source is not in active config."""
        from ess.livedata.dashboard.job_orchestrator import JobConfig

        staged = {
            'source1': JobConfig(params={'threshold': 100.0}, aux_source_names={}),
        }
        active = {}
        result = _group_configs_by_equality(staged, active)

        assert len(result) == 1
        assert result[0].is_modified is True

    def test_nested_dict_params_handled_correctly(self):
        """Test that nested dictionaries in params are handled correctly."""
        from ess.livedata.dashboard.job_orchestrator import JobConfig

        # Nested dict in params (e.g., ROI configuration)
        staged = {
            'source1': JobConfig(
                params={'roi': {'x': [0, 10], 'y': [0, 20]}}, aux_source_names={}
            ),
            'source2': JobConfig(
                params={'roi': {'x': [0, 10], 'y': [0, 20]}}, aux_source_names={}
            ),
            'source3': JobConfig(
                params={'roi': {'x': [5, 15], 'y': [0, 20]}}, aux_source_names={}
            ),
        }
        result = _group_configs_by_equality(staged, {})

        # source1 and source2 should be grouped, source3 separate
        assert len(result) == 2
        # Find the group with 2 sources
        group_sizes = [len(g.source_names) for g in result]
        assert sorted(group_sizes) == [1, 2]


class TestUnconfiguredSources:
    """Tests for unconfigured sources detection."""

    def test_all_sources_unconfigured(self):
        """Test when no sources are configured."""
        result = _get_unconfigured_sources(['source1', 'source2'], {})
        assert result == ['source1', 'source2']

    def test_no_unconfigured_sources(self):
        """Test when all sources are configured."""
        from ess.livedata.dashboard.job_orchestrator import JobConfig

        staged = {
            'source1': JobConfig(params={}, aux_source_names={}),
            'source2': JobConfig(params={}, aux_source_names={}),
        }
        result = _get_unconfigured_sources(['source1', 'source2'], staged)
        assert result == []

    def test_partial_configuration(self):
        """Test when some sources are configured."""
        from ess.livedata.dashboard.job_orchestrator import JobConfig

        staged = {
            'source1': JobConfig(params={}, aux_source_names={}),
        }
        result = _get_unconfigured_sources(['source1', 'source2'], staged)
        assert result == ['source2']


class TestWorkflowStatusWidget:
    """Tests for WorkflowStatusWidget."""

    def test_creates_panel_widget(self, workflow_status_widget):
        """Test that widget creates a Panel widget."""
        panel = workflow_status_widget.panel()
        assert isinstance(panel, pn.Column)

    def test_workflow_id_property(self, workflow_status_widget, workflow_id):
        """Test that workflow_id property returns correct ID."""
        assert workflow_status_widget.workflow_id == workflow_id

    def test_initial_status_is_stopped(self, workflow_status_widget):
        """Test that initial status is STOPPED when no active jobs."""
        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'STOPPED'

    def test_gear_click_calls_orchestrator_create_adapter(
        self, workflow_status_widget, job_orchestrator, workflow_id
    ):
        """Test that gear click creates an adapter via orchestrator."""
        # Verify the widget has a reference to the orchestrator
        assert workflow_status_widget._orchestrator is job_orchestrator

        # The modal container exists and is initially empty
        assert len(workflow_status_widget._modal_container) == 0

        # Note: Full modal creation test requires a workflow spec with valid params
        # that ConfigurationModal can handle. The test fixtures use simplified specs
        # that don't work with the full ConfigurationModal widget hierarchy.

    def test_remove_click_removes_sources_from_staged(
        self, workflow_status_widget, job_orchestrator, workflow_id
    ):
        """Test that remove click removes sources from staged config."""
        # First stage some configs
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source2',
            params={'threshold': 100.0},
            aux_source_names={},
        )

        # Remove source1
        workflow_status_widget._on_remove_click(['source1'])

        # Check that only source2 remains
        staged = job_orchestrator.get_staged_config(workflow_id)
        assert 'source1' not in staged
        assert 'source2' in staged

    def test_expand_collapse_toggle(self, workflow_status_widget):
        """Test that header click toggles expansion."""
        assert workflow_status_widget._expanded is True

        # Simulate header click
        workflow_status_widget._on_header_click(None)
        assert workflow_status_widget._expanded is False

        workflow_status_widget._on_header_click(None)
        assert workflow_status_widget._expanded is True

    def test_refresh_rebuilds_widget_on_version_change(
        self, workflow_status_widget, job_orchestrator, workflow_id
    ):
        """Test that refresh() triggers a full rebuild when state version changes."""
        rebuild_count = 0
        original_build = workflow_status_widget._build_widget

        def counting_build():
            nonlocal rebuild_count
            rebuild_count += 1
            original_build()

        workflow_status_widget._build_widget = counting_build

        # Stage a config to bump version
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 50.0},
            aux_source_names={},
        )

        # Call refresh - should detect version change and rebuild
        workflow_status_widget.refresh()
        assert rebuild_count == 1

        # Call refresh again without changes - should NOT rebuild
        workflow_status_widget.refresh()
        assert rebuild_count == 1

    def test_header_shows_play_button_when_stopped_with_staged_configs(
        self, workflow_status_widget, job_orchestrator, workflow_id
    ):
        """Test that play button is shown when stopped and has staged configs."""
        # Stage a config but don't commit (workflow is stopped)
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )

        # Rebuild widget to pick up staged config
        workflow_status_widget._build_widget()

        # Check header buttons
        header_buttons = workflow_status_widget._create_header_buttons()
        button_icons = _get_button_icons(header_buttons)
        assert _has_icon(button_icons, 'player-play')  # play button

    def test_header_does_not_show_play_button_when_no_staged_configs(
        self, workflow_status_widget, job_orchestrator, workflow_id
    ):
        """Test that play button is not shown when there are no staged configs."""
        # Clear any staged configs (fixture may have default configs)
        job_orchestrator.clear_staged_configs(workflow_id)
        workflow_status_widget._build_widget()

        header_buttons = workflow_status_widget._create_header_buttons()
        button_icons = _get_button_icons(header_buttons)
        assert not _has_icon(button_icons, 'player-play')  # No play button

    def test_header_no_play_button_when_running_without_changes(
        self, workflow_status_widget, job_orchestrator, workflow_id
    ):
        """Test that play button is not shown when running and staged matches active."""
        # Stage and commit to start the workflow
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )
        job_orchestrator.commit_workflow(workflow_id)

        # Rebuild widget
        workflow_status_widget._build_widget()

        # Check header buttons - no play because staged == active
        header_buttons = workflow_status_widget._create_header_buttons()
        button_icons = _get_button_icons(header_buttons)
        assert not _has_icon(button_icons, 'player-play')  # No play button
        assert _has_icon(button_icons, 'player-stop')  # Stop button

    def test_header_shows_play_button_when_running_with_modified_staged(
        self, workflow_status_widget, job_orchestrator, workflow_id
    ):
        """Test that play button is shown when running with modified staged config."""
        # Stage and commit to start the workflow
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )
        job_orchestrator.commit_workflow(workflow_id)

        # Modify staged config (different from active)
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 200.0},  # Different value
            aux_source_names={},
        )

        # Rebuild widget
        workflow_status_widget._build_widget()

        # Check header buttons - play button should appear because staged != active
        header_buttons = workflow_status_widget._create_header_buttons()
        button_icons = _get_button_icons(header_buttons)
        assert _has_icon(button_icons, 'player-play')  # Play button (commit & restart)
        assert _has_icon(button_icons, 'player-stop')  # Stop button still there


class TestWorkflowStatusWidgetWithJobs:
    """Tests for WorkflowStatusWidget with active jobs."""

    def test_status_shows_active_when_job_running(
        self,
        workflow_status_widget,
        job_service,
        workflow_id,
        job_orchestrator,
    ):
        """Test that status shows ACTIVE when jobs are running."""
        # Stage and commit to create active job
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )
        job_ids = job_orchestrator.commit_workflow(workflow_id)
        job_number = job_ids[0].job_number

        # Create job status
        job_id = JobId(source_name='source1', job_number=job_number)
        job_status = JobStatus(
            job_id=job_id,
            workflow_id=workflow_id,
            state=JobState.active,
            start_time=1000000000000,
        )
        job_service.status_updated(job_status)

        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'ACTIVE'

    def test_status_shows_error_when_job_has_error(
        self,
        workflow_status_widget,
        job_service,
        workflow_id,
        job_orchestrator,
    ):
        """Test that status shows ERROR when any job has error."""
        # Stage and commit to create active job
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )
        job_ids = job_orchestrator.commit_workflow(workflow_id)
        job_number = job_ids[0].job_number

        # Create job status with error
        job_id = JobId(source_name='source1', job_number=job_number)
        job_status = JobStatus(
            job_id=job_id,
            workflow_id=workflow_id,
            state=JobState.error,
            error_message='Something went wrong',
            start_time=1000000000000,
        )
        job_service.status_updated(job_status)

        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'ERROR'

    def test_status_shows_pending_when_no_backend_status(
        self,
        workflow_status_widget,
        workflow_id,
        job_orchestrator,
    ):
        """Test that status shows PENDING when job committed but no backend status."""
        # Stage and commit to create active job
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )
        job_orchestrator.commit_workflow(workflow_id)

        # Don't add any job status (simulating backend not running)
        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'PENDING'

    def test_timing_shows_waiting_for_pending_job(
        self,
        workflow_status_widget,
        workflow_id,
        job_orchestrator,
    ):
        """Test that timing text shows 'Waiting for backend...' for pending job."""
        # Stage and commit to create active job
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )
        job_orchestrator.commit_workflow(workflow_id)

        # Don't add any job status (simulating backend not running)
        timing = workflow_status_widget._get_timing_text()
        assert timing == 'Waiting for backend...'

    def test_status_transitions_from_pending_to_active(
        self,
        workflow_status_widget,
        job_service,
        workflow_id,
        job_orchestrator,
    ):
        """Test status transitions from PENDING to ACTIVE when backend responds."""
        # Stage and commit to create active job
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )
        job_ids = job_orchestrator.commit_workflow(workflow_id)

        # Initially no backend status - should be PENDING
        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'PENDING'

        # Backend starts and sends status - should become ACTIVE
        job_id = JobId(source_name='source1', job_number=job_ids[0].job_number)
        job_status = JobStatus(
            job_id=job_id,
            workflow_id=workflow_id,
            state=JobState.active,
            start_time=1000000000000,
        )
        job_service.status_updated(job_status)

        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'ACTIVE'

    def test_stop_clears_to_stopped_not_pending(
        self,
        workflow_status_widget,
        workflow_id,
        job_orchestrator,
    ):
        """Test that stop button clears to STOPPED, not PENDING."""
        # Stage and commit to create pending job
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )
        job_orchestrator.commit_workflow(workflow_id)

        # Should be PENDING
        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'PENDING'

        # Stop the workflow
        workflow_status_widget._on_stop_click()

        # Should be STOPPED, not PENDING
        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'STOPPED'

    def test_status_becomes_pending_when_heartbeat_stale(
        self,
        workflow_status_widget,
        job_service,
        workflow_id,
        job_orchestrator,
    ):
        """Test status transitions from ACTIVE to PENDING when heartbeat stales."""
        import time

        # Use short timeout for testing (1 second)
        job_service._heartbeat_timeout_ns = 1_000_000_000

        # Stage and commit to create active job
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )
        job_ids = job_orchestrator.commit_workflow(workflow_id)

        # Backend sends initial status - should be ACTIVE
        job_id = JobId(source_name='source1', job_number=job_ids[0].job_number)
        job_status = JobStatus(
            job_id=job_id,
            workflow_id=workflow_id,
            state=JobState.active,
            start_time=1000000000000,
        )
        job_service.status_updated(job_status)

        # Should be ACTIVE initially
        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'ACTIVE'

        # Wait for heartbeat to become stale (> 1 second)
        time.sleep(1.1)

        # Status should now be PENDING (stale heartbeat)
        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'PENDING'

    def test_is_status_stale_returns_true_for_old_status(self, job_service):
        """Test that is_status_stale returns True for old status."""
        import time

        # Use short timeout for testing
        job_service._heartbeat_timeout_ns = 100_000_000  # 100ms

        job_id = JobId(source_name='test', job_number='test-uuid')
        job_status = JobStatus(
            job_id=job_id,
            workflow_id=WorkflowId(
                instrument='test', namespace='test', name='test', version=1
            ),
            state=JobState.active,
        )

        # Add status
        job_service.status_updated(job_status)

        # Should not be stale immediately
        assert not job_service.is_status_stale(job_id)

        # Wait for status to become stale
        time.sleep(0.15)

        # Should be stale now
        assert job_service.is_status_stale(job_id)

    def test_is_status_stale_returns_true_for_missing_job(self, job_service):
        """Test that is_status_stale returns True for job with no status."""
        job_id = JobId(source_name='nonexistent', job_number='fake-uuid')
        assert job_service.is_status_stale(job_id)


class TestWorkflowStatusListWidget:
    """Tests for WorkflowStatusListWidget."""

    def test_creates_panel_widget(
        self,
        job_orchestrator,
        job_service,
    ):
        """Test that list widget creates a Panel widget."""
        list_widget = WorkflowStatusListWidget(
            orchestrator=job_orchestrator,
            job_service=job_service,
        )
        panel = list_widget.panel()
        assert isinstance(panel, pn.Column)

    def test_creates_widget_per_workflow(
        self,
        job_orchestrator,
        job_service,
    ):
        """Test that list widget creates one widget per workflow."""
        list_widget = WorkflowStatusListWidget(
            orchestrator=job_orchestrator,
            job_service=job_service,
        )

        workflow_registry = job_orchestrator.get_workflow_registry()
        assert len(list_widget._widgets) == len(workflow_registry)
        for workflow_id in workflow_registry:
            assert workflow_id in list_widget._widgets

    def test_expand_all_expands_all_widgets(
        self,
        job_orchestrator,
        job_service,
    ):
        """Test that expand_all expands all workflow widgets."""
        list_widget = WorkflowStatusListWidget(
            orchestrator=job_orchestrator,
            job_service=job_service,
        )

        # First collapse all widgets
        for widget in list_widget._widgets.values():
            widget.set_expanded(False)
            assert not widget._expanded

        # Expand all
        list_widget._expand_all()

        # All widgets should be expanded
        for widget in list_widget._widgets.values():
            assert widget._expanded

    def test_collapse_all_collapses_all_widgets(
        self,
        job_orchestrator,
        job_service,
    ):
        """Test that collapse_all collapses all workflow widgets."""
        list_widget = WorkflowStatusListWidget(
            orchestrator=job_orchestrator,
            job_service=job_service,
        )

        # Widgets start collapsed by default, so first expand them
        for widget in list_widget._widgets.values():
            widget.set_expanded(True)
            assert widget._expanded

        # Collapse all
        list_widget._collapse_all()

        # All widgets should be collapsed
        for widget in list_widget._widgets.values():
            assert not widget._expanded

    def test_header_row_contains_expand_collapse_buttons(
        self,
        job_orchestrator,
        job_service,
    ):
        """Test that the panel contains expand/collapse all buttons."""
        list_widget = WorkflowStatusListWidget(
            orchestrator=job_orchestrator,
            job_service=job_service,
        )

        panel = list_widget.panel()
        # First element should be the header row
        header_row = panel[0]
        assert isinstance(header_row, pn.Row)

        # Header row should contain the expand/collapse buttons
        button_names = [
            obj.name for obj in header_row if isinstance(obj, pn.widgets.Button)
        ]
        assert 'Expand all' in button_names
        assert 'Collapse all' in button_names
