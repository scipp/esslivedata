# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for job adoption from heartbeat observation and desired/observed
reconciliation (ADR 0008).

The persisted current_job record restores desired state and labels observed
jobs, but only a heartbeat observation establishes the generation that admits
data. Reconciliation re-issues stops while observed heartbeats contradict the
desired state.
"""

import uuid

import pytest

from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.core.job_manager import JobCommand
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.dashboard.active_job_registry import ActiveJobRegistry
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.job_orchestrator import (
    STOP_REISSUE_INTERVAL_SECONDS,
    JobOrchestrator,
)
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.fakes import FakeMessageSink


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store() -> dict:
    """Config store shared between orchestrator incarnations (dashboard
    restarts)."""
    return {}


@pytest.fixture
def make_system(workflow_registry, store, clock):
    """Build a fresh orchestrator incarnation sharing the persisted store."""

    def _make(*, heartbeat_timeout_ns: int | None = None):
        data_service = DataService()
        kwargs = (
            {}
            if heartbeat_timeout_ns is None
            else {'heartbeat_timeout_ns': heartbeat_timeout_ns}
        )
        job_service = JobService(**kwargs)
        registry = ActiveJobRegistry(data_service=data_service, job_service=job_service)
        sink = FakeMessageSink()
        orchestrator = JobOrchestrator(
            command_service=CommandService(sink=sink),
            workflow_registry=workflow_registry,
            active_job_registry=registry,
            job_service=job_service,
            config_store=store,
            clock=clock,
        )
        return orchestrator, registry, job_service, sink

    return _make


def _heartbeat(
    job_service: JobService,
    job_id: JobId,
    workflow_id,
    state: JobState = JobState.active,
    start_time_ns: int = 1_000_000_000,
) -> None:
    job_service.status_updated(
        JobStatus(
            job_id=job_id,
            workflow_id=workflow_id,
            state=state,
            start_time=Timestamp.from_ns(start_time_ns),
        )
    )


def _sent_stop_job_ids(sink: FakeMessageSink) -> list[JobId]:
    return [
        msg.value.job_id
        for msg in sink.messages
        if isinstance(msg.value, JobCommand) and msg.value.action.value == 'stop'
    ]


class TestAdoptionAfterRestart:
    def test_record_restores_desired_state_but_admits_no_data(
        self, make_system, workflow_id
    ):
        """After a restart the record says what should be running, but only
        an observation establishes the generation that admits data."""
        orchestrator1, _, _, _ = make_system()
        job_ids = orchestrator1.commit_workflow(workflow_id)
        job_number = job_ids[0].job_number

        orchestrator2, registry2, _, _ = make_system()

        assert orchestrator2.get_active_job_number(workflow_id) == job_number
        assert not registry2.is_current(workflow_id, job_number)

    def test_heartbeat_adopts_recorded_job(self, make_system, workflow_id):
        orchestrator1, _, _, _ = make_system()
        job_ids = orchestrator1.commit_workflow(workflow_id)
        job_number = job_ids[0].job_number
        committed_config = orchestrator1.get_active_config(workflow_id)

        _, registry2, job_service2, _ = make_system()
        _heartbeat(job_service2, job_ids[0], workflow_id)

        assert registry2.is_current(workflow_id, job_number)
        resolved = registry2.resolve_config(workflow_id, job_number)
        assert resolved is not None
        assert resolved.keys() == committed_config.keys()

    def test_stopped_heartbeat_does_not_adopt(self, make_system, workflow_id):
        orchestrator1, _, _, _ = make_system()
        job_ids = orchestrator1.commit_workflow(workflow_id)

        _, registry2, job_service2, _ = make_system()
        _heartbeat(job_service2, job_ids[0], workflow_id, state=JobState.stopped)

        assert not registry2.is_current(workflow_id, job_ids[0].job_number)

    def test_all_stopped_heartbeats_after_restart_deactivate(
        self, make_system, workflow_id
    ):
        """A backend that stopped the jobs while the dashboard was down
        reports stopped states; the workflow transitions to stopped instead
        of waiting forever."""
        orchestrator1, _, _, _ = make_system()
        job_ids = orchestrator1.commit_workflow(workflow_id)

        orchestrator2, _, job_service2, _ = make_system()
        for job_id in job_ids:
            _heartbeat(job_service2, job_id, workflow_id, state=JobState.stopped)

        assert orchestrator2.get_active_job_number(workflow_id) is None

    def test_predecessor_heartbeat_does_not_displace_record(
        self, make_system, workflow_id
    ):
        """When several job_numbers heartbeat for one workflow, the record
        resolves which is current (ADR 0008)."""
        orchestrator1, _, _, _ = make_system()
        job_ids = orchestrator1.commit_workflow(workflow_id)
        recorded_number = job_ids[0].job_number

        orchestrator2, registry2, job_service2, _ = make_system()
        predecessor = JobId(source_name='source1', job_number=uuid.uuid4())
        _heartbeat(job_service2, predecessor, workflow_id)

        assert orchestrator2.get_active_job_number(workflow_id) == recorded_number
        assert not registry2.is_current(workflow_id, predecessor.job_number)


class TestOrphanAdoption:
    def test_orphan_adopted_with_unknown_config(self, make_system, workflow_id):
        """A record miss degrades, it does not lie: the job is visible and
        admits data, without params provenance."""
        orchestrator, registry, job_service, _ = make_system()
        orphan = JobId(source_name='source1', job_number=uuid.uuid4())

        _heartbeat(job_service, orphan, workflow_id)

        assert orchestrator.get_active_job_number(workflow_id) == orphan.job_number
        assert registry.is_current(workflow_id, orphan.job_number)
        assert registry.resolve_config(workflow_id, orphan.job_number) is None
        active = orchestrator.get_active_config(workflow_id)
        assert active['source1'].params == {}

    def test_second_source_extends_adopted_job(self, make_system, workflow_id):
        orchestrator, _, job_service, _ = make_system()
        job_number = uuid.uuid4()

        for source in ('source1', 'source2'):
            _heartbeat(
                job_service,
                JobId(source_name=source, job_number=job_number),
                workflow_id,
            )

        assert set(orchestrator.get_active_config(workflow_id)) == {
            'source1',
            'source2',
        }

    def test_latest_start_time_wins_among_recordless_jobs(
        self, make_system, workflow_id
    ):
        orchestrator, _, job_service, _ = make_system()
        older = JobId(source_name='source1', job_number=uuid.uuid4())
        newer = JobId(source_name='source1', job_number=uuid.uuid4())

        _heartbeat(job_service, older, workflow_id, start_time_ns=1_000)
        _heartbeat(job_service, newer, workflow_id, start_time_ns=2_000)
        _heartbeat(job_service, older, workflow_id, start_time_ns=1_000)

        assert orchestrator.get_active_job_number(workflow_id) == newer.job_number

    def test_earlier_start_does_not_displace_later(self, make_system, workflow_id):
        orchestrator, _, job_service, _ = make_system()
        newer = JobId(source_name='source1', job_number=uuid.uuid4())
        older = JobId(source_name='source1', job_number=uuid.uuid4())

        _heartbeat(job_service, newer, workflow_id, start_time_ns=2_000)
        _heartbeat(job_service, older, workflow_id, start_time_ns=1_000)

        assert orchestrator.get_active_job_number(workflow_id) == newer.job_number

    def test_adopted_job_is_stoppable(self, make_system, workflow_id):
        orchestrator, _, job_service, sink = make_system()
        orphan = JobId(source_name='source1', job_number=uuid.uuid4())
        _heartbeat(job_service, orphan, workflow_id)

        assert orchestrator.stop_workflow(workflow_id) is True
        assert orphan in _sent_stop_job_ids(sink)
        assert orchestrator.get_active_job_number(workflow_id) is None

    def test_adopted_job_is_not_persisted_as_record(
        self, make_system, store, workflow_id
    ):
        """The record must not lie: an adopted job's params are unknown, so
        no current_job record is written for it."""
        _, _, job_service, _ = make_system()
        orphan = JobId(source_name='source1', job_number=uuid.uuid4())

        _heartbeat(job_service, orphan, workflow_id)

        stored = store.get(str(workflow_id))
        assert stored is None or 'current_job' not in stored

    def test_user_stopped_job_is_not_readopted(self, make_system, workflow_id):
        """A stop-lagged job's continued heartbeats must not resurrect it;
        reconciliation re-issues the stop instead."""
        orchestrator, _, job_service, _ = make_system()
        job_ids = orchestrator.commit_workflow(workflow_id)
        _heartbeat(job_service, job_ids[0], workflow_id)
        orchestrator.stop_workflow(workflow_id)

        _heartbeat(job_service, job_ids[0], workflow_id)

        assert orchestrator.get_active_job_number(workflow_id) is None

    def test_bumps_version_on_adoption(self, make_system, workflow_id):
        orchestrator, _, job_service, _ = make_system()
        version_before = orchestrator.get_workflow_state_version(workflow_id)

        _heartbeat(
            job_service,
            JobId(source_name='source1', job_number=uuid.uuid4()),
            workflow_id,
        )

        assert orchestrator.get_workflow_state_version(workflow_id) > version_before


class TestReconciliation:
    def test_reissues_stop_after_interval_while_heartbeats_continue(
        self, make_system, clock, workflow_id
    ):
        """A swallowed stop recovers: while the job keeps heartbeating
        against desired stopped, the stop is re-issued, rate-bounded."""
        orchestrator, _, job_service, sink = make_system()
        job_ids = orchestrator.commit_workflow(workflow_id)
        _heartbeat(job_service, job_ids[0], workflow_id)
        orchestrator.stop_workflow(workflow_id)
        stops_after_user_stop = len(_sent_stop_job_ids(sink))

        _heartbeat(job_service, job_ids[0], workflow_id)
        orchestrator.reconcile_observed_jobs()
        assert len(_sent_stop_job_ids(sink)) == stops_after_user_stop

        clock.advance(STOP_REISSUE_INTERVAL_SECONDS + 1)
        _heartbeat(job_service, job_ids[0], workflow_id)
        orchestrator.reconcile_observed_jobs()
        assert len(_sent_stop_job_ids(sink)) == stops_after_user_stop + 1

        orchestrator.reconcile_observed_jobs()
        assert len(_sent_stop_job_ids(sink)) == stops_after_user_stop + 1

    def test_no_reissue_for_desired_running_job(self, make_system, clock, workflow_id):
        orchestrator, _, job_service, sink = make_system()
        job_ids = orchestrator.commit_workflow(workflow_id)
        _heartbeat(job_service, job_ids[0], workflow_id)

        clock.advance(STOP_REISSUE_INTERVAL_SECONDS + 1)
        orchestrator.reconcile_observed_jobs()

        assert _sent_stop_job_ids(sink) == []

    def test_stops_superseded_unknown_job(self, make_system, workflow_id):
        """An observed job that is neither the desired generation nor a
        known predecessor is stopped promptly."""
        orchestrator, _, job_service, sink = make_system()
        orchestrator.commit_workflow(workflow_id)
        unknown = JobId(source_name='source1', job_number=uuid.uuid4())
        _heartbeat(job_service, unknown, workflow_id)

        orchestrator.reconcile_observed_jobs()

        assert unknown in _sent_stop_job_ids(sink)

    def test_ignores_stale_heartbeats(self, make_system, workflow_id):
        orchestrator, _, job_service, sink = make_system(heartbeat_timeout_ns=-1)
        job_ids = orchestrator.commit_workflow(workflow_id)
        _heartbeat(job_service, job_ids[0], workflow_id)
        orchestrator.stop_workflow(workflow_id)
        stops_after_user_stop = len(_sent_stop_job_ids(sink))

        _heartbeat(job_service, job_ids[0], workflow_id)
        orchestrator.reconcile_observed_jobs()

        assert len(_sent_stop_job_ids(sink)) == stops_after_user_stop

    def test_prunes_stale_statuses(self, make_system, workflow_id):
        orchestrator, _, job_service, _ = make_system(heartbeat_timeout_ns=-1)
        _heartbeat(
            job_service,
            JobId(source_name='source1', job_number=uuid.uuid4()),
            workflow_id,
            state=JobState.stopped,
        )

        orchestrator.reconcile_observed_jobs()

        assert job_service.job_statuses == {}


class TestCommitSupersedesObserved:
    def test_recommit_stops_job_whose_stop_was_lost(self, make_system, workflow_id):
        """After a user stop whose send was swallowed, the job keeps
        heartbeating with no local owner; the next commit supersedes it by
        stopping it alongside starting the new generation."""
        orchestrator, _, job_service, sink = make_system()
        job_ids = orchestrator.commit_workflow(workflow_id)
        _heartbeat(job_service, job_ids[0], workflow_id)
        orchestrator.stop_workflow(workflow_id)
        sink.messages.clear()

        _heartbeat(job_service, job_ids[0], workflow_id)
        orchestrator.commit_workflow(workflow_id)

        assert job_ids[0] in _sent_stop_job_ids(sink)
