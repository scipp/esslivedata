# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Integration tests for the interaction between JobOrchestrator, Orchestrator,
and DataService during workflow restart.

These tests verify observable behavior around the generation flip at commit:
buffers are cleared so the new generation starts blank, late data from the
replaced generation is rejected, and a stopped workflow's data is retained
under its stable keys until the next commit.

The "race condition" tests exercise the threading hazards between the
background update thread (Orchestrator.update) and the UI thread
(JobOrchestrator.commit_workflow). The single-generation invariant test is
the merge gate for the stable-key model: it must fail on an implementation
whose generation flip is not atomic with the buffer clear.
"""

import sys
import threading

import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import DataKey, JobId, ResultKey, WorkflowId
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.core.message import STATUS_STREAM_ID, StreamId, StreamKind
from ess.livedata.dashboard.active_job_registry import ActiveJobRegistry
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.data_service import DataService, DataServiceSubscriber
from ess.livedata.dashboard.extractors import LatestValueExtractor
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


def _data_keys(workflow_spec) -> list[DataKey]:
    return [
        DataKey(
            workflow_id=workflow_spec.get_id(), source_name=sn, output_name='result'
        )
        for sn in workflow_spec.source_names
    ]


class _KeysSubscriber(DataServiceSubscriber):
    """Subscriber over a fixed set of keys, for pulling data with stamps."""

    def __init__(self, keys: list[DataKey]) -> None:
        self._extractors = {key: LatestValueExtractor() for key in keys}
        super().__init__()

    @property
    def extractors(self):
        return self._extractors

    def on_updated(self, updated_keys) -> None:
        pass


def _make_system(*workflow_specs):
    """Wire up real JobOrchestrator, Orchestrator, DataService, and
    ActiveJobRegistry."""
    data_service = DataService()
    job_service = JobService()
    fake_sink = FakeMessageSink()
    registry = {spec.get_id(): spec for spec in workflow_specs}

    active_job_registry = ActiveJobRegistry(
        data_service=data_service, job_service=job_service
    )

    job_orchestrator = JobOrchestrator(
        command_service=CommandService(sink=fake_sink),
        workflow_registry=registry,
        active_job_registry=active_job_registry,
        job_service=job_service,
    )

    orchestrator = Orchestrator(
        message_source=fake_sink,  # unused; we call forward() directly
        data_service=data_service,
        job_service=job_service,
        service_registry=ServiceRegistry(),
        job_orchestrator=job_orchestrator,
        active_job_registry=active_job_registry,
    )

    return orchestrator, job_orchestrator, data_service, job_service


class TestRestartCleanup:
    """
    Integration tests: a recommit clears the workflow's buffers (blank slate
    for the new generation) and late-arriving old data is rejected by the
    ingest filter; a stop retains data.
    """

    def test_data_cleared_on_recommit(self, workflow_spec):
        """The generation flip clears the workflow's buffers: the plot goes
        blank until the new generation's first data arrives."""
        orchestrator, job_orchestrator, data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        job_ids = job_orchestrator.commit_workflow(workflow_id)
        for source_name in workflow_spec.source_names:
            key = _make_result_key(workflow_id, source_name, job_ids[0].job_number)
            orchestrator.forward(_data_stream_id(key), sc.scalar(1.0))

        subscriber = _KeysSubscriber(_data_keys(workflow_spec))
        data_service.register_subscriber(subscriber)
        assert len(data_service.snapshot(subscriber)) == len(workflow_spec.source_names)

        job_orchestrator.commit_workflow(workflow_id)

        assert data_service.snapshot(subscriber) == {}

    def test_identical_config_recommit_also_clears(self, workflow_spec):
        """A pure restart (staged config unchanged) clears too — uniform
        semantics, matching the fresh buffers that per-commit keys used to
        provide."""
        orchestrator, job_orchestrator, data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        job_ids = job_orchestrator.commit_workflow(workflow_id)
        key = _make_result_key(
            workflow_id, workflow_spec.source_names[0], job_ids[0].job_number
        )
        orchestrator.forward(_data_stream_id(key), sc.scalar(1.0))
        assert key.data_key in data_service

        # No staging between commits: byte-identical config.
        job_orchestrator.commit_workflow(workflow_id)

        subscriber = _KeysSubscriber(_data_keys(workflow_spec))
        data_service.register_subscriber(subscriber)
        assert data_service.snapshot(subscriber) == {}

    def test_data_retained_after_stop(self, workflow_spec):
        """Stopping does not evict: the stopped generation's data stays
        displayed under its stable keys until the next commit."""
        orchestrator, job_orchestrator, data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        job_ids = job_orchestrator.commit_workflow(workflow_id)
        for sn in workflow_spec.source_names:
            key = _make_result_key(workflow_id, sn, job_ids[0].job_number)
            orchestrator.forward(_data_stream_id(key), sc.scalar(1.0))

        job_orchestrator.stop_workflow(workflow_id)

        for data_key in _data_keys(workflow_spec):
            assert data_key in data_service

    def test_late_data_for_old_job_is_rejected(self, workflow_spec):
        """Data arriving for the old job after recommit does not enter
        DataService — old-parameter data can never sit under a stable key
        whose active config is new."""
        orchestrator, job_orchestrator, data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        job_ids = job_orchestrator.commit_workflow(workflow_id)
        old_number = job_ids[0].job_number

        job_orchestrator.commit_workflow(workflow_id)

        key = _make_result_key(workflow_id, workflow_spec.source_names[0], old_number)
        orchestrator.forward(_data_stream_id(key), sc.scalar(42.0))

        assert len(data_service) == 0

    def test_new_data_accepted_after_recommit(self, workflow_spec):
        """Data for the new generation is accepted after recommit."""
        orchestrator, job_orchestrator, data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        job_ids_1 = job_orchestrator.commit_workflow(workflow_id)
        for sn in workflow_spec.source_names:
            key = _make_result_key(workflow_id, sn, job_ids_1[0].job_number)
            orchestrator.forward(_data_stream_id(key), sc.scalar(1.0))

        job_ids_2 = job_orchestrator.commit_workflow(workflow_id)
        new_number = job_ids_2[0].job_number

        for sn in workflow_spec.source_names:
            key = _make_result_key(workflow_id, sn, new_number)
            orchestrator.forward(_data_stream_id(key), sc.scalar(99.0))

        for data_key in _data_keys(workflow_spec):
            assert data_key in data_service
            assert data_service[data_key].value == 99.0

    def test_status_for_replaced_job_is_still_accepted(self, workflow_spec):
        """JobStatus keeps job-number semantics: the just-replaced generation
        may still report its final states (it is the registry's ``last``)."""
        orchestrator, job_orchestrator, _, job_service = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        job_ids = job_orchestrator.commit_workflow(workflow_id)
        old_job_id = job_ids[0]

        job_orchestrator.commit_workflow(workflow_id)

        status = JobStatus(
            job_id=old_job_id, workflow_id=workflow_id, state=JobState.active
        )
        orchestrator.forward(STATUS_STREAM_ID, status)

        assert len(job_service.job_statuses) == 1

    def test_status_beyond_window_is_observed_but_not_adopted(self, workflow_spec):
        """Two recommits push the first generation out of the (current, last)
        window. Its heartbeats are still observed (ADR 0008) — the job is an
        orphan for reconciliation to stop — but they do not displace the
        committed current generation."""
        orchestrator, job_orchestrator, _, job_service = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()

        job_ids = job_orchestrator.commit_workflow(workflow_id)
        oldest_job_id = job_ids[0]

        job_orchestrator.commit_workflow(workflow_id)
        job_ids_3 = job_orchestrator.commit_workflow(workflow_id)

        status = JobStatus(
            job_id=oldest_job_id, workflow_id=workflow_id, state=JobState.active
        )
        orchestrator.forward(STATUS_STREAM_ID, status)

        assert len(job_service.job_statuses) == 1
        assert (
            job_orchestrator.get_active_job_number(workflow_id)
            == job_ids_3[0].job_number
        )

    def test_other_workflow_data_survives_recommit(
        self, workflow_spec, workflow_spec_2
    ):
        """Recommitting one workflow does not affect data from another workflow."""
        orchestrator, job_orchestrator, data_service, _ = _make_system(
            workflow_spec, workflow_spec_2
        )

        wf_id_1 = workflow_spec.get_id()
        wf_id_2 = workflow_spec_2.get_id()

        jobs_1 = job_orchestrator.commit_workflow(wf_id_1)
        jobs_2 = job_orchestrator.commit_workflow(wf_id_2)

        key_1 = _make_result_key(
            wf_id_1, workflow_spec.source_names[0], jobs_1[0].job_number
        )
        key_2 = _make_result_key(
            wf_id_2, workflow_spec_2.source_names[0], jobs_2[0].job_number
        )
        orchestrator.forward(_data_stream_id(key_1), sc.scalar(1.0))
        orchestrator.forward(_data_stream_id(key_2), sc.scalar(2.0))

        job_orchestrator.commit_workflow(wf_id_1)
        job_orchestrator.commit_workflow(wf_id_1)

        assert key_2.data_key in data_service
        assert key_1.data_key not in data_service


class TestConcurrentForwardAndCommit:
    """
    Thread-safety tests for concurrent Orchestrator.forward() (background
    thread) and JobOrchestrator.commit_workflow() (UI thread).

    The generation flip (begin_generation + buffer clear) must be atomic
    with respect to ingest: post-flip, buffers may hold only
    current-generation data.
    """

    @pytest.mark.slow
    def test_concurrent_forward_and_commit_does_not_crash(self, workflow_spec):
        """Concurrent data forwarding and workflow restart must not crash."""
        orchestrator, job_orchestrator, _data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()
        registry = job_orchestrator.active_job_registry

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
                        with registry.ingestion_guard():
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
    def test_commit_ingest_race_preserves_single_generation_invariant(
        self, workflow_spec
    ):
        """Old-generation data must never survive the flip.

        Each round, an ingest batch for the pre-commit generation races the
        commit that replaces it, barrier-synchronized to maximize
        interleavings. Whichever order the race resolves, the invariant is
        the same: after the round's post-commit data lands, every stamp in
        the buffers is the new generation's. A stale stamp means an
        old-generation write slipped past the flip — the failure mode of a
        stable-key implementation whose flip is not guarded.

        A free-running checker additionally asserts that no single pull ever
        observes a mix of generations (the clear is atomic to pulls).
        """
        orchestrator, job_orchestrator, data_service, _ = _make_system(workflow_spec)
        workflow_id = workflow_spec.get_id()
        registry = job_orchestrator.active_job_registry
        subscriber = _KeysSubscriber(_data_keys(workflow_spec))
        data_service.register_subscriber(subscriber)

        rounds = 50
        batches_per_round = 10
        start_barrier = threading.Barrier(3)
        end_barrier = threading.Barrier(3)
        errors: list[Exception] = []
        violations: list[str] = []
        stop_checker = threading.Event()
        current_job_ids = [job_orchestrator.commit_workflow(workflow_id)]

        def _forward_batch(job_ids, value: float) -> None:
            with registry.ingestion_guard():
                for sn in workflow_spec.source_names:
                    key = _make_result_key(workflow_id, sn, job_ids[0].job_number)
                    orchestrator.forward(_data_stream_id(key), sc.scalar(value))

        def ingest_loop():
            try:
                for i in range(rounds):
                    job_ids = current_job_ids[0]  # pre-commit generation
                    start_barrier.wait()
                    # Several guard-held batches widen the exposure: each one
                    # is a separate check-then-write window racing the flip.
                    for _ in range(batches_per_round):
                        _forward_batch(job_ids, float(i))
                    end_barrier.wait()
            except threading.BrokenBarrierError:
                return
            except Exception as exc:
                errors.append(exc)
                start_barrier.abort()

        def commit_loop():
            try:
                for _ in range(rounds):
                    start_barrier.wait()
                    new_ids = job_orchestrator.commit_workflow(workflow_id)
                    end_barrier.wait()
                    # Safe: ingest reads current_job_ids before start_barrier
                    # of the next round.
                    current_job_ids[0] = new_ids
            except threading.BrokenBarrierError:
                return
            except Exception as exc:
                errors.append(exc)
                start_barrier.abort()

        def checker_loop():
            while not stop_checker.is_set():
                data, stamps = data_service.snapshot_with_stamps(subscriber)
                observed = {stamps[key] for key in data}
                if len(observed) > 1:
                    violations.append(f"mixed generations in one pull: {observed}")
                    return

        threads = [
            threading.Thread(target=ingest_loop),
            threading.Thread(target=commit_loop),
            threading.Thread(target=checker_loop),
        ]
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            for thread in threads:
                thread.start()
            for i in range(rounds):
                start_barrier.wait()
                end_barrier.wait()
                # Race resolved. Whichever order it took, buffers must be
                # empty now: batch-then-flip means the flip cleared the batch;
                # flip-then-batch means the filter dropped it. Any surviving
                # entry is an old-generation write that slipped past the flip.
                data, stamps = data_service.snapshot_with_stamps(subscriber)
                if data:
                    violations.append(
                        f"round {i}: stale generation survived the flip: "
                        f"{set(stamps.values())}"
                    )
                    start_barrier.abort()
                    break
                # Land data for the new generation so the next round's flip
                # has non-empty buffers to clear.
                new_number = job_orchestrator.get_active_job_number(workflow_id)
                new_ids = [JobId(source_name='unused', job_number=new_number)]
                _forward_batch(new_ids, value=-1.0)
        except threading.BrokenBarrierError:
            pass
        finally:
            stop_checker.set()
            sys.setswitchinterval(old_interval)
            for thread in threads:
                thread.join(timeout=10.0)

        assert not errors, f"Race harness crashed: {errors[0]}"
        assert not violations, violations[0]
