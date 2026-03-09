# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Integration tests for the interaction between JobOrchestrator, Orchestrator,
and DataService during workflow restart.

These tests verify observable behavior when a workflow is recommitted:
data from the previous job should be cleaned up, and late-arriving data
for the old job should not accumulate.

The "race condition" tests simulate or trigger the threading hazards
between the background update thread (Orchestrator.update) and the UI
thread (JobOrchestrator.commit_workflow).
"""

import sys
import threading

import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.core.message import STATUS_STREAM_ID, StreamId, StreamKind
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_orchestrator import JobOrchestrator
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.orchestrator import Orchestrator
from ess.livedata.dashboard.service_registry import ServiceRegistry
from ess.livedata.fakes import FakeMessageSink


def _data_stream_id(key: ResultKey) -> StreamId:
    return StreamId(kind=StreamKind.LIVEDATA_DATA, name=key.model_dump_json())


def _make_result_key(
    workflow_id: WorkflowId,
    source_name: str,
    job_number,
    output_name: str = 'result',
) -> ResultKey:
    return ResultKey(
        workflow_id=workflow_id,
        job_id=JobId(source_name=source_name, job_number=job_number),
        output_name=output_name,
    )


def _make_system(*workflow_specs):
    """Wire up real JobOrchestrator, Orchestrator, and DataService."""
    data_service = DataService()
    job_service = JobService()
    fake_sink = FakeMessageSink()
    registry = {spec.get_id(): spec for spec in workflow_specs}

    job_orchestrator = JobOrchestrator(
        command_service=CommandService(sink=fake_sink),
        workflow_registry=registry,
        data_service=data_service,
        job_service=job_service,
    )

    # Orchestrator uses the same JobOrchestrator for active-job filtering
    orchestrator = Orchestrator(
        message_source=fake_sink,  # unused; we call forward() directly
        data_service=data_service,
        job_service=job_service,
        service_registry=ServiceRegistry(),
        job_orchestrator=job_orchestrator,
    )

    return orchestrator, job_orchestrator, data_service, job_service


class TestRestartCleanup:
    """
    Integration tests: after a workflow restart, old data is cleaned up
    and late-arriving old data is rejected by the ingest filter.
    """

    def test_old_data_is_removed_after_recommit(self, workflow_spec):
        """Data from the previous job is not present after recommit."""
        orchestrator, job_orchestrator, data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        # First commit
        job_ids = job_orchestrator.commit_workflow(workflow_id)
        old_number = job_ids[0].job_number

        # Simulate data arriving for old job via Orchestrator.forward()
        for source_name in workflow_spec.source_names:
            key = _make_result_key(workflow_id, source_name, old_number)
            orchestrator.forward(_data_stream_id(key), sc.scalar(1.0))

        assert len(data_service) == len(workflow_spec.source_names)

        # Recommit
        job_orchestrator.commit_workflow(workflow_id)

        # Old data should be gone
        assert len(data_service) == 0

    def test_late_data_for_old_job_is_rejected(self, workflow_spec):
        """Data arriving for the old job after recommit does not enter DataService."""
        orchestrator, job_orchestrator, data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        # First commit, capture old job number
        job_ids = job_orchestrator.commit_workflow(workflow_id)
        old_number = job_ids[0].job_number

        # Recommit — old job is no longer active
        job_orchestrator.commit_workflow(workflow_id)

        # Late-arriving data for old job
        key = _make_result_key(workflow_id, workflow_spec.source_names[0], old_number)
        orchestrator.forward(_data_stream_id(key), sc.scalar(42.0))

        # Should not create a new entry
        assert len(data_service) == 0

    def test_new_data_accepted_after_recommit(self, workflow_spec):
        """Data for the new job is accepted after recommit."""
        orchestrator, job_orchestrator, data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        # First commit + data
        job_ids_1 = job_orchestrator.commit_workflow(workflow_id)
        old_number = job_ids_1[0].job_number
        for sn in workflow_spec.source_names:
            key = _make_result_key(workflow_id, sn, old_number)
            orchestrator.forward(_data_stream_id(key), sc.scalar(1.0))

        # Recommit
        job_ids_2 = job_orchestrator.commit_workflow(workflow_id)
        new_number = job_ids_2[0].job_number

        # New data arrives
        data = sc.scalar(99.0)
        for sn in workflow_spec.source_names:
            key = _make_result_key(workflow_id, sn, new_number)
            orchestrator.forward(_data_stream_id(key), data)

        assert len(data_service) == len(workflow_spec.source_names)
        # Verify it's the new data
        for sn in workflow_spec.source_names:
            key = _make_result_key(workflow_id, sn, new_number)
            assert key in data_service

    def test_late_old_status_is_rejected(self, workflow_spec):
        """Job status for the old job after recommit does not enter JobService."""
        orchestrator, job_orchestrator, _, job_service = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        job_ids = job_orchestrator.commit_workflow(workflow_id)
        old_job_id = job_ids[0]

        # Recommit
        job_orchestrator.commit_workflow(workflow_id)

        # Late-arriving status for old job
        status = JobStatus(
            job_id=old_job_id,
            workflow_id=workflow_id,
            state=JobState.active,
        )
        orchestrator.forward(STATUS_STREAM_ID, status)

        assert len(job_service.job_statuses) == 0

    def test_other_workflow_data_survives_recommit(
        self, workflow_spec, workflow_spec_2
    ):
        """Recommitting one workflow does not affect data from another workflow."""
        orchestrator, job_orchestrator, data_service, _ = _make_system(
            workflow_spec, workflow_spec_2
        )

        wf_id_1 = workflow_spec.get_id()
        wf_id_2 = workflow_spec_2.get_id()

        # Commit both workflows
        jobs_1 = job_orchestrator.commit_workflow(wf_id_1)
        jobs_2 = job_orchestrator.commit_workflow(wf_id_2)

        # Data for both workflows
        key_1 = _make_result_key(
            wf_id_1, workflow_spec.source_names[0], jobs_1[0].job_number
        )
        key_2 = _make_result_key(
            wf_id_2, workflow_spec_2.source_names[0], jobs_2[0].job_number
        )
        orchestrator.forward(_data_stream_id(key_1), sc.scalar(1.0))
        orchestrator.forward(_data_stream_id(key_2), sc.scalar(2.0))

        assert len(data_service) == 2

        # Recommit only workflow 1
        job_orchestrator.commit_workflow(wf_id_1)

        # Workflow 2's data survives
        assert key_2 in data_service
        assert key_1 not in data_service

    def test_multiple_restarts_do_not_leak(self, workflow_spec):
        """Repeated restarts do not leave orphaned data in DataService."""
        orchestrator, job_orchestrator, data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        for _ in range(5):
            job_ids = job_orchestrator.commit_workflow(workflow_id)
            job_number = job_ids[0].job_number

            # Simulate data arriving
            for sn in workflow_spec.source_names:
                key = _make_result_key(workflow_id, sn, job_number)
                orchestrator.forward(_data_stream_id(key), sc.scalar(1.0))

        # After 5 restarts, only the last job's data should remain
        assert len(data_service) == len(workflow_spec.source_names)


class TestConcurrentForwardAndCommit:
    """
    Thread-safety tests for concurrent Orchestrator.forward() (background
    thread) and JobOrchestrator.commit_workflow() (UI thread).

    These tests exercise the data-flow lock that serializes active-set
    mutations and DataService cleanup against message processing.
    """

    @pytest.mark.slow
    def test_concurrent_forward_and_commit_does_not_crash(self, workflow_spec):
        """Concurrent data forwarding and workflow restart must not crash."""
        orchestrator, job_orchestrator, _data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        # Initial commit
        job_ids = job_orchestrator.commit_workflow(workflow_id)

        errors: list[Exception] = []
        stop = threading.Event()

        def forward_loop():
            """Continuously forward data, as the background update thread does."""
            i = 0
            while not stop.is_set():
                # Forward for the current active job (may change mid-loop)
                for sn in workflow_spec.source_names:
                    key = _make_result_key(workflow_id, sn, job_ids[0].job_number)
                    try:
                        with job_orchestrator.data_flow_lock:
                            orchestrator.forward(
                                _data_stream_id(key), sc.scalar(float(i))
                            )
                    except Exception as exc:
                        errors.append(exc)
                        return
                i += 1

        thread = threading.Thread(target=forward_loop)
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            thread.start()
            # Restart the workflow many times while forward_loop runs
            for _ in range(20):
                job_ids = job_orchestrator.commit_workflow(workflow_id)
            stop.set()
            thread.join(timeout=5.0)
        finally:
            sys.setswitchinterval(old_interval)

        assert not errors, f"Forward loop crashed: {errors[0]}"

    @pytest.mark.slow
    def test_concurrent_forward_and_commit_does_not_leak(self, workflow_spec):
        """Repeated concurrent restarts must not leave orphaned data."""
        orchestrator, job_orchestrator, data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        errors: list[Exception] = []
        stop = threading.Event()
        # Shared mutable ref so forward_loop always uses the latest job_ids
        current_job_ids = [None]

        def forward_loop():
            """Continuously forward data for the current job."""
            i = 0
            while not stop.is_set():
                jids = current_job_ids[0]
                if jids is None:
                    continue
                for sn in workflow_spec.source_names:
                    key = _make_result_key(workflow_id, sn, jids[0].job_number)
                    try:
                        with job_orchestrator.data_flow_lock:
                            orchestrator.forward(
                                _data_stream_id(key), sc.scalar(float(i))
                            )
                    except Exception as exc:
                        errors.append(exc)
                        return
                i += 1

        thread = threading.Thread(target=forward_loop)
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            # Restart many times
            for _ in range(20):
                current_job_ids[0] = job_orchestrator.commit_workflow(workflow_id)

            thread.start()
            # Let the forward loop run for a bit with the last job
            stop.wait(timeout=0.2)
            stop.set()
            thread.join(timeout=5.0)
        finally:
            sys.setswitchinterval(old_interval)

        assert not errors, f"Forward loop crashed: {errors[0]}"

        # Only the last job's data should be present
        last_number = current_job_ids[0][0].job_number
        for key in data_service:
            assert key.job_id.job_number == last_number, (
                f"Orphaned key from old job: {key}"
            )
