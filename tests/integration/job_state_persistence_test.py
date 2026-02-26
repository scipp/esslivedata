# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration tests for job state persistence in JobOrchestrator."""

from ess.livedata.config.workflow_spec import JobNumber, WorkflowId
from ess.livedata.dashboard.config_store import ConfigStoreManager
from ess.livedata.handlers.monitor_workflow_specs import MonitorDataParams
from ess.livedata.parameter_models import Scale, TimeUnit, TOAEdges
from tests.integration.backend import DashboardBackend


def test_active_job_persisted_and_restored(tmp_path) -> None:
    """
    Test that active job state is persisted and restored across sessions.

    This test verifies:
    1. Starting a workflow creates an active job with a job_number
    2. The active job state (job_number, jobs) is persisted to config store
    3. Creating a new backend with the same config store restores the active job
    4. Subscribers are notified of the restored active job
    """
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )
    source_names = ['monitor1', 'monitor2']
    custom_params = MonitorDataParams(
        toa_edges=TOAEdges(
            start=5.0,
            stop=15.0,
            num_bins=150,
            scale=Scale.LINEAR,
            unit=TimeUnit.MS,
        )
    )

    # Start workflow in first backend session
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend1:
        job_ids = backend1.workflow_controller.start_workflow(
            workflow_id=workflow_id,
            source_names=source_names,
            config=custom_params,
        )
        assert len(job_ids) == 2

        # Get the job_number from the first backend
        original_job_number = job_ids[0].job_number

        # Verify active config is available
        active_config = backend1.job_orchestrator.get_active_config(workflow_id)
        assert len(active_config) == 2
        assert 'monitor1' in active_config
        assert 'monitor2' in active_config

    # Create second backend with same config directory
    # This simulates a dashboard restart
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend2:
        # Verify active job was restored
        restored_active = backend2.job_orchestrator.get_active_config(workflow_id)
        assert len(restored_active) == 2, "Active jobs should be restored"
        assert 'monitor1' in restored_active
        assert 'monitor2' in restored_active

        # Verify job_number is the same
        restored_job_number = backend2.job_orchestrator.get_active_job_number(
            workflow_id
        )
        assert restored_job_number is not None, "Current job should be restored"
        assert restored_job_number == original_job_number, (
            "Job number should match original"
        )

        # Verify params were also restored correctly
        assert restored_active['monitor1'].params == custom_params.model_dump()


def test_subscriber_notified_on_job_restoration(tmp_path) -> None:
    """
    Test that subscribers are notified when an active job is restored.

    This verifies Option A behavior: passive restoration notifies subscribers
    so they can display restored job state.
    """
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )
    source_names = ['monitor1']

    # Start workflow and persist state
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend1:
        job_ids = backend1.workflow_controller.start_workflow(
            workflow_id=workflow_id,
            source_names=source_names,
            config=MonitorDataParams(),
        )
        original_job_number = job_ids[0].job_number

    # Create second backend and subscribe before restoration
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend2:
        from ess.livedata.dashboard.job_orchestrator import WorkflowCallbacks

        # Track subscriber notifications
        notified_job_numbers = []

        def track_notification(job_number: JobNumber) -> None:
            notified_job_numbers.append(job_number)

        # Subscribe to workflow - should be immediately notified of restored job
        backend2.job_orchestrator.subscribe_to_workflow(
            workflow_id,
            WorkflowCallbacks(on_started=track_notification),
        )

        # Subscriber should have been notified immediately with restored job_number
        assert len(notified_job_numbers) == 1
        assert notified_job_numbers[0] == original_job_number


def test_job_transition_persists_previous_job(tmp_path) -> None:
    """
    Test that stopping old jobs and starting new ones persists both states.

    When committing a workflow that already has an active job:
    1. Old job moves to 'previous'
    2. New job becomes 'current'
    3. Both current and previous are persisted
    4. After restoration, both current and previous are restored
    """
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )
    source_names = ['monitor1']

    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend1:
        # Start first job
        job_ids_1 = backend1.workflow_controller.start_workflow(
            workflow_id=workflow_id,
            source_names=source_names,
            config=MonitorDataParams(),
        )
        first_job_number = job_ids_1[0].job_number

        # Start second job (stops first)
        job_ids_2 = backend1.workflow_controller.start_workflow(
            workflow_id=workflow_id,
            source_names=source_names,
            config=MonitorDataParams(
                toa_edges=TOAEdges(
                    start=10.0, stop=20.0, num_bins=100, scale=Scale.LINEAR
                )
            ),
        )
        second_job_number = job_ids_2[0].job_number

        # Verify state in first backend using public API
        current_job_number = backend1.job_orchestrator.get_active_job_number(
            workflow_id
        )
        previous_job_number = backend1.job_orchestrator.get_previous_job_number(
            workflow_id
        )
        assert current_job_number == second_job_number
        assert previous_job_number == first_job_number

    # Restore in second backend
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend2:
        # Both current and previous should be restored
        restored_current = backend2.job_orchestrator.get_active_job_number(workflow_id)
        restored_previous = backend2.job_orchestrator.get_previous_job_number(
            workflow_id
        )

        assert restored_current is not None
        assert restored_current == second_job_number

        assert restored_previous is not None
        assert restored_previous == first_job_number


def test_corrupted_job_state_does_not_break_restoration(tmp_path) -> None:
    """
    Test that corrupted job state doesn't prevent config restoration.

    If the persisted job state is corrupted (e.g., invalid UUID), the
    orchestrator should log a warning but still restore the config.
    """
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )

    # Manually create a config with corrupted job state
    config_manager = ConfigStoreManager(instrument='dummy', config_dir=tmp_path)
    config_store = config_manager.get_store('workflow_configs')

    config_store[str(workflow_id)] = {
        'source_names': ['monitor1'],
        'params': MonitorDataParams().model_dump(),
        'aux_source_names': {},
        'current_job': {
            'job_number': 'not-a-valid-uuid',  # Corrupted!
            'jobs': {
                'monitor1': {
                    'params': MonitorDataParams().model_dump(),
                    'aux_source_names': {},
                }
            },
        },
    }

    # Backend should start successfully despite corrupted job state
    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend:
        # Config should be restored
        staged = backend.job_orchestrator.get_staged_config(workflow_id)
        assert 'monitor1' in staged

        # Active job should not be restored (corruption)
        active = backend.job_orchestrator.get_active_config(workflow_id)
        assert len(active) == 0, "Corrupted job state should not be restored"


def test_no_active_job_persists_empty_state(tmp_path) -> None:
    """
    Test that workflows without active jobs persist correctly.

    A workflow with only staged config (no committed job) should persist
    just the config, not job state.
    """
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )

    with DashboardBackend(
        instrument='dummy', dev=True, transport='none', config_dir=tmp_path
    ) as backend1:
        # Just check that the workflow exists with default config
        # Don't start it
        staged = backend1.job_orchestrator.get_staged_config(workflow_id)
        assert len(staged) > 0  # Default sources from spec

    # Verify stored config doesn't have job state
    config_manager = ConfigStoreManager(instrument='dummy', config_dir=tmp_path)
    config_store = config_manager.get_store('workflow_configs')

    # The workflow won't be persisted until it's committed, so there should be nothing
    # in the store yet (since we only initialized but never committed)
    stored = config_store.get(str(workflow_id))
    # If anything is stored, it should not have current_job
    if stored:
        assert 'current_job' not in stored
