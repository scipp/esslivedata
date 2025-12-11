# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for WorkflowStatusWidget and WorkflowStatusListWidget."""

import panel as pn
import pytest

from ess.livedata.config.workflow_spec import JobId, WorkflowId
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.widgets.workflow_status_widget import (
    WorkflowStatusListWidget,
    WorkflowStatusWidget,
    _get_unconfigured_sources,
    _group_configs_by_equality,
)


@pytest.fixture
def job_service():
    """Create a JobService for testing."""
    return JobService()


@pytest.fixture
def configure_callback():
    """Create a mock configure callback that records calls."""
    calls = []

    def callback(workflow_id: WorkflowId, source_names: list[str]):
        calls.append((workflow_id, source_names))

    callback.calls = calls
    return callback


class FakeWorkflowController:
    """Fake workflow controller for testing."""

    def __init__(self, orchestrator):
        self._orchestrator = orchestrator

    def stop_workflow(self, workflow_id):
        """Stop a workflow."""
        return self._orchestrator.stop_workflow(workflow_id)


@pytest.fixture
def workflow_controller(job_orchestrator):
    """Create a fake workflow controller."""
    return FakeWorkflowController(job_orchestrator)


@pytest.fixture
def workflow_status_widget(
    workflow_id,
    workflow_spec,
    job_orchestrator,
    job_service,
    workflow_controller,
    configure_callback,
):
    """Create a WorkflowStatusWidget for testing."""
    return WorkflowStatusWidget(
        workflow_id=workflow_id,
        workflow_spec=workflow_spec,
        orchestrator=job_orchestrator,
        job_service=job_service,
        workflow_controller=workflow_controller,
        on_configure=configure_callback,
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

    def test_gear_click_invokes_callback(
        self, workflow_status_widget, workflow_id, configure_callback
    ):
        """Test that gear click invokes configure callback."""
        workflow_status_widget._on_gear_click(['source1'])

        assert len(configure_callback.calls) == 1
        assert configure_callback.calls[0] == (workflow_id, ['source1'])

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

    def test_status_shows_scheduled_when_no_backend_status(
        self,
        workflow_status_widget,
        workflow_id,
        job_orchestrator,
    ):
        """Test that status shows SCHEDULED when job committed but no backend status."""
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
        assert status == 'SCHEDULED'

    def test_timing_shows_waiting_for_scheduled_job(
        self,
        workflow_status_widget,
        workflow_id,
        job_orchestrator,
    ):
        """Test that timing text shows 'Waiting for backend...' for scheduled job."""
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

    def test_status_transitions_from_scheduled_to_active(
        self,
        workflow_status_widget,
        job_service,
        workflow_id,
        job_orchestrator,
    ):
        """Test status transitions from SCHEDULED to ACTIVE when backend responds."""
        # Stage and commit to create active job
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )
        job_ids = job_orchestrator.commit_workflow(workflow_id)

        # Initially no backend status - should be SCHEDULED
        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'SCHEDULED'

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

    def test_stop_clears_to_stopped_not_scheduled(
        self,
        workflow_status_widget,
        workflow_controller,
        workflow_id,
        job_orchestrator,
    ):
        """Test that stop button clears to STOPPED, not SCHEDULED."""
        # Stage and commit to create scheduled job
        job_orchestrator.stage_config(
            workflow_id,
            source_name='source1',
            params={'threshold': 100.0},
            aux_source_names={},
        )
        job_orchestrator.commit_workflow(workflow_id)

        # Should be SCHEDULED
        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'SCHEDULED'

        # Stop the workflow
        workflow_status_widget._on_stop_click()

        # Should be STOPPED, not SCHEDULED
        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'STOPPED'

    def test_status_becomes_scheduled_when_heartbeat_stale(
        self,
        workflow_status_widget,
        job_service,
        workflow_id,
        job_orchestrator,
    ):
        """Test status transitions from ACTIVE to SCHEDULED when heartbeat stales."""
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

        # Status should now be SCHEDULED (stale heartbeat)
        status, _ = workflow_status_widget._get_workflow_status()
        assert status == 'SCHEDULED'

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
        workflow_controller,
        configure_callback,
    ):
        """Test that list widget creates a Panel widget."""
        list_widget = WorkflowStatusListWidget(
            orchestrator=job_orchestrator,
            job_service=job_service,
            workflow_controller=workflow_controller,
            on_configure=configure_callback,
        )
        panel = list_widget.panel()
        assert isinstance(panel, pn.Column)

    def test_creates_widget_per_workflow(
        self,
        job_orchestrator,
        job_service,
        workflow_controller,
        configure_callback,
    ):
        """Test that list widget creates one widget per workflow."""
        list_widget = WorkflowStatusListWidget(
            orchestrator=job_orchestrator,
            job_service=job_service,
            workflow_controller=workflow_controller,
            on_configure=configure_callback,
        )

        workflow_registry = job_orchestrator.get_workflow_registry()
        assert len(list_widget._widgets) == len(workflow_registry)
        for workflow_id in workflow_registry:
            assert workflow_id in list_widget._widgets
