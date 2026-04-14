# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for automatic workflow deactivation on backend shutdown.

When a backend worker shuts down, it sends a final heartbeat marking all jobs
as stopped. The dashboard's JobService forwards these to JobOrchestrator via
its on_status_updated callback, which deactivates the workflow so the widget
transitions to the normal STOPPED state.
"""

from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.core.timestamp import Timestamp


class TestReconcileStoppedJobs:
    def test_deactivates_workflow_when_all_jobs_stopped(
        self, job_orchestrator, job_service, workflow_id
    ):
        """Workflow with all jobs reported stopped is deactivated."""
        job_ids = job_orchestrator.commit_workflow(workflow_id)
        assert job_orchestrator.get_active_job_number(workflow_id) is not None

        _report_all_stopped(job_service, job_ids, workflow_id)

        assert job_orchestrator.get_active_job_number(workflow_id) is None

    def test_does_not_deactivate_when_some_jobs_still_active(
        self, job_orchestrator, job_service, workflow_id
    ):
        """Workflow stays active if not all jobs have reported stopped."""
        job_ids = job_orchestrator.commit_workflow(workflow_id)

        # Only report one job as stopped
        _report_status(job_service, job_ids[0], workflow_id, JobState.stopped)

        assert job_orchestrator.get_active_job_number(workflow_id) is not None

    def test_does_not_deactivate_when_no_statuses_received(
        self, job_orchestrator, workflow_id
    ):
        """Workflow stays active if no heartbeats have been received yet."""
        job_orchestrator.commit_workflow(workflow_id)

        assert job_orchestrator.get_active_job_number(workflow_id) is not None

    def test_does_not_send_stop_commands_to_backend(
        self, job_orchestrator, job_service, workflow_id, fake_message_sink
    ):
        """Reconciliation does not send commands — the backend is already dead."""
        job_ids = job_orchestrator.commit_workflow(workflow_id)
        messages_before = len(fake_message_sink.messages)

        _report_all_stopped(job_service, job_ids, workflow_id)

        assert len(fake_message_sink.messages) == messages_before

    def test_sets_stopped_reason_to_backend_shutdown(
        self, job_orchestrator, job_service, workflow_id
    ):
        """Stopped reason distinguishes backend shutdown from user stop."""
        from ess.livedata.dashboard.job_orchestrator import StoppedReason

        job_ids = job_orchestrator.commit_workflow(workflow_id)
        _report_all_stopped(job_service, job_ids, workflow_id)

        assert (
            job_orchestrator.get_stopped_reason(workflow_id)
            == StoppedReason.backend_shutdown
        )

    def test_user_stop_sets_stopped_reason_to_user(self, job_orchestrator, workflow_id):
        """User-initiated stop sets reason to user."""
        from ess.livedata.dashboard.job_orchestrator import StoppedReason

        job_orchestrator.commit_workflow(workflow_id)
        job_orchestrator.stop_workflow(workflow_id)

        assert job_orchestrator.get_stopped_reason(workflow_id) == StoppedReason.user

    def test_bumps_version_on_deactivation(
        self, job_orchestrator, job_service, workflow_id
    ):
        """Version is incremented so widgets detect the state change."""
        job_ids = job_orchestrator.commit_workflow(workflow_id)
        version_before = job_orchestrator.get_workflow_state_version(workflow_id)
        _report_all_stopped(job_service, job_ids, workflow_id)

        assert job_orchestrator.get_workflow_state_version(workflow_id) > version_before

    def test_idempotent_when_already_stopped(
        self, job_orchestrator, job_service, workflow_id
    ):
        """Reporting stopped on an already-stopped workflow is a no-op."""
        job_ids = job_orchestrator.commit_workflow(workflow_id)
        _report_all_stopped(job_service, job_ids, workflow_id)
        _report_all_stopped(job_service, job_ids, workflow_id)  # second round is safe

        assert job_orchestrator.get_active_job_number(workflow_id) is None

    def test_cleans_up_job_states_on_deactivation(
        self, job_orchestrator, job_service, workflow_id
    ):
        """Job state tracking entries are removed when a workflow is deactivated."""
        job_ids = job_orchestrator.commit_workflow(workflow_id)
        _report_all_stopped(job_service, job_ids, workflow_id)

        assert not any(jid in job_orchestrator._job_states for jid in job_ids)

    def test_cleans_up_job_states_on_user_stop(
        self, job_orchestrator, job_service, workflow_id
    ):
        """Job state entries are cleaned up on user-initiated stop too."""
        job_ids = job_orchestrator.commit_workflow(workflow_id)

        for job_id in job_ids:
            _report_status(job_service, job_id, workflow_id, JobState.active)

        job_orchestrator.stop_workflow(workflow_id)

        assert not any(jid in job_orchestrator._job_states for jid in job_ids)

    def test_reconciles_independently_per_workflow(
        self, job_orchestrator, job_service, workflow_id, workflow_id_2
    ):
        """Only workflows with all jobs stopped are deactivated."""
        job_ids_1 = job_orchestrator.commit_workflow(workflow_id)
        job_orchestrator.commit_workflow(workflow_id_2)

        # Only stop jobs for workflow 1
        _report_all_stopped(job_service, job_ids_1, workflow_id)

        assert job_orchestrator.get_active_job_number(workflow_id) is None
        assert job_orchestrator.get_active_job_number(workflow_id_2) is not None


def _report_status(job_service, job_id: JobId, workflow_id, state: JobState) -> None:
    job_service.status_updated(
        JobStatus(
            job_id=job_id,
            workflow_id=workflow_id,
            state=state,
            start_time=Timestamp.from_ns(1_000_000_000),
        )
    )


def _report_all_stopped(job_service, job_ids: list[JobId], workflow_id) -> None:
    for job_id in job_ids:
        _report_status(job_service, job_id, workflow_id, JobState.stopped)
