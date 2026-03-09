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
import uuid

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


class TestRaceOrphanedBuffer:
    """
    Proves that a message which passes the active-job filter before
    commit_workflow discards the old job number can re-create a buffer
    that cleanup already deleted. The orphaned buffer is never cleaned
    up by subsequent restarts, leaking memory indefinitely.

    The test simulates the race deterministically: after commit_workflow
    cleans up, we write directly to DataService (representing a message
    that already passed the filter on the background thread).
    """

    @pytest.mark.xfail(
        reason="Orphaned buffer from race window is never cleaned up",
        strict=True,
    )
    def test_late_write_after_cleanup_does_not_persist(self, workflow_spec):
        """An in-flight write after cleanup must not survive the next restart."""
        _, job_orchestrator, data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        # Commit 1: workflow running, data arrives
        job_ids_1 = job_orchestrator.commit_workflow(workflow_id)
        old_number = job_ids_1[0].job_number
        old_key = _make_result_key(
            workflow_id, workflow_spec.source_names[0], old_number
        )
        data_service[old_key] = sc.scalar(1.0)

        # Commit 2: cleanup deletes old data
        job_orchestrator.commit_workflow(workflow_id)
        assert old_key not in data_service

        # Simulate the race: background thread was in-flight (already past
        # the is_active_job_number check) and writes after cleanup ran.
        data_service[old_key] = sc.scalar(42.0)

        # Commit 3: this restart cleans up commit 2's job_number,
        # NOT the orphaned key from commit 1.
        job_orchestrator.commit_workflow(workflow_id)

        # The orphaned buffer should not persist.
        assert old_key not in data_service


class TestRaceDictIteration:
    """
    Proves that concurrent iteration over DataService (as done by
    _cleanup_previous_job_data's list comprehension) and insertion
    (as done by Orchestrator.forward) can crash with RuntimeError.

    The list comprehension ``[k for k in data_service if cond]`` iterates
    the internal dict via Python bytecode. The GIL can switch between
    iterations, allowing another thread to insert a key and trigger
    ``RuntimeError: dictionary changed size during iteration``.
    """

    @pytest.mark.xfail(
        reason="DataService dict iteration is not thread-safe",
        strict=True,
    )
    @pytest.mark.slow
    def test_concurrent_iteration_and_insertion_does_not_crash(self, workflow_spec):
        """Iterating DataService while another thread inserts must not crash."""
        data_service = DataService()
        workflow_id = workflow_spec.get_id()
        job_number = uuid.uuid4()

        # Populate with enough keys to make iteration span multiple GIL switches
        for i in range(200):
            key = _make_result_key(workflow_id, f'source_{i}', job_number)
            data_service[key] = sc.scalar(float(i))

        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)  # Maximize GIL switching
        try:
            errors: list[Exception] = []
            stop = threading.Event()

            def iterate_like_cleanup():
                """Iterate with a filter, same as _cleanup_previous_job_data."""
                while not stop.is_set():
                    try:
                        _ = [
                            k for k in data_service if k.job_id.job_number == job_number
                        ]
                    except RuntimeError as exc:
                        errors.append(exc)
                        return

            def insert_new_keys():
                """Insert keys with a different job_number, simulating
                Orchestrator.forward for another workflow."""
                other_number = uuid.uuid4()
                i = 0
                while not stop.is_set():
                    key = _make_result_key(workflow_id, f'other_{i}', other_number)
                    data_service[key] = sc.scalar(1.0)
                    i += 1

            t_iter = threading.Thread(target=iterate_like_cleanup)
            t_insert = threading.Thread(target=insert_new_keys)
            t_iter.start()
            t_insert.start()

            # 1 second is enough to trigger the race reliably
            stop.wait(timeout=1.0)
            stop.set()
            t_iter.join(timeout=5.0)
            t_insert.join(timeout=5.0)

            assert not errors, f"Dict iteration crashed: {errors[0]}"
        finally:
            sys.setswitchinterval(old_interval)
