# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""JobOrchestrator against the desired-state table.

A commit or stop writes one record and takes effect only once the writer
confirms it is stored. Records read back from the table (the dashboard's own,
another writer's, or those of unknown workflows) keep the orchestrator in sync.
"""

import uuid

import pytest
import scipp as sc
import yaml

from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.core.desired_state import (
    DesiredJob,
    DesiredStateWriteError,
    DesiredWorkflow,
    InMemoryDesiredStateLog,
)
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.config_store import FileBackedConfigStore
from ess.livedata.dashboard.job_orchestrator import JobOrchestrator
from ess.livedata.dashboard.notification_queue import (
    NotificationQueue,
    NotificationType,
)
from ess.livedata.fakes import FakeMessageSink


class FlakyWriter:
    """In-memory desired-state log whose writes fail while ``failing`` is set."""

    def __init__(self, *, failing: bool = False) -> None:
        self.log = InMemoryDesiredStateLog()
        self.failing = failing

    def write(self, workflow_id: WorkflowId, desired: DesiredWorkflow | None) -> None:
        if self.failing:
            raise DesiredStateWriteError("NOT_ENOUGH_REPLICAS")
        self.log.write(workflow_id, desired)


class ObservingWriter:
    """Writer that records what the dashboard shows while a write is in flight."""

    def __init__(self, observe) -> None:
        self._log = InMemoryDesiredStateLog()
        self._observe = observe
        self.observed: list = []

    def write(self, workflow_id: WorkflowId, desired: DesiredWorkflow | None) -> None:
        self.observed.append(self._observe())
        self._log.write(workflow_id, desired)


def _record(workflow_id: WorkflowId, *sources: str) -> DesiredWorkflow:
    return DesiredWorkflow(
        workflow_id=workflow_id,
        job_number=uuid.uuid4(),
        jobs={
            s: DesiredJob(params={'threshold': 1.0}, aux_source_names={})
            for s in sources
        },
    )


@pytest.fixture
def notification_queue() -> NotificationQueue:
    return NotificationQueue()


@pytest.fixture
def make_orchestrator(workflow_registry, active_job_registry, notification_queue):
    def make(writer, config_store=None) -> JobOrchestrator:
        return JobOrchestrator(
            command_service=CommandService(sink=FakeMessageSink()),
            desired_state=writer,
            workflow_registry=workflow_registry,
            active_job_registry=active_job_registry,
            config_store=config_store,
            notification_queue=notification_queue,
        )

    return make


@pytest.fixture
def data_key(workflow_id) -> DataKey:
    return DataKey(workflow_id=workflow_id, source_name='source1', output_name='result')


def _errors(queue: NotificationQueue) -> list[str]:
    return [
        e.message
        for e in queue.get_all_events()
        if e.notification_type is NotificationType.ERROR
    ]


class TestCommit:
    def test_generation_flips_only_after_the_write_is_stored(
        self, make_orchestrator, workflow_id, data_service, data_key
    ):
        orchestrator = None
        writer = ObservingWriter(
            lambda: (
                orchestrator.get_active_job_number(workflow_id),
                data_key in data_service,
            )
        )
        orchestrator = make_orchestrator(writer)
        data_service.set_item(data_key, sc.DataArray(sc.scalar(1.0)))

        (job_id, *_) = orchestrator.commit_workflow(workflow_id)

        assert writer.observed == [(None, True)]
        assert orchestrator.get_active_job_number(workflow_id) == job_id.job_number
        assert orchestrator.active_job_registry.is_current(
            workflow_id, job_id.job_number
        )
        assert data_key not in data_service

    def test_failed_write_changes_nothing_and_is_reported(
        self,
        make_orchestrator,
        workflow_id,
        data_service,
        data_key,
        notification_queue,
    ):
        writer = FlakyWriter()
        orchestrator = make_orchestrator(writer)
        (old, *_) = orchestrator.commit_workflow(workflow_id)
        data_service.set_item(data_key, sc.DataArray(sc.scalar(1.0)))
        writer.failing = True
        version = orchestrator.get_workflow_state_version(workflow_id)

        assert orchestrator.commit_workflow(workflow_id) == []

        assert orchestrator.get_active_job_number(workflow_id) == old.job_number
        assert orchestrator.active_job_registry.is_current(workflow_id, old.job_number)
        assert data_key in data_service
        assert orchestrator.get_workflow_state_version(workflow_id) == version
        (error,) = _errors(notification_queue)
        assert 'Test Workflow' in error
        assert 'NOT_ENOUGH_REPLICAS' in error

    def test_failed_write_is_not_persisted(self, make_orchestrator, workflow_id):
        store: dict[str, dict] = {}
        orchestrator = make_orchestrator(FlakyWriter(failing=True), store)

        orchestrator.commit_workflow(workflow_id)

        assert store == {}


class TestStop:
    def test_stop_clears_current_and_retains_data(
        self, make_orchestrator, workflow_id, data_service, data_key
    ):
        log = InMemoryDesiredStateLog()
        orchestrator = make_orchestrator(log)
        (job_id, *_) = orchestrator.commit_workflow(workflow_id)
        data_service.set_item(data_key, sc.DataArray(sc.scalar(1.0)))

        assert orchestrator.stop_workflow(workflow_id)

        assert log.read(1) == [(workflow_id, None)]
        assert orchestrator.get_active_job_number(workflow_id) is None
        assert not orchestrator.active_job_registry.is_current(
            workflow_id, job_id.job_number
        )
        assert data_key in data_service

    def test_failed_stop_changes_nothing_and_is_reported(
        self, make_orchestrator, workflow_id, notification_queue
    ):
        writer = FlakyWriter()
        orchestrator = make_orchestrator(writer)
        (job_id, *_) = orchestrator.commit_workflow(workflow_id)
        writer.failing = True

        assert not orchestrator.stop_workflow(workflow_id)

        assert orchestrator.get_active_job_number(workflow_id) == job_id.job_number
        assert len(_errors(notification_queue)) == 1


class TestReadBack:
    def test_startup_table_becomes_current(
        self, make_orchestrator, workflow_id, workflow_id_2
    ):
        log = InMemoryDesiredStateLog()
        record = _record(workflow_id, 'source1')
        log.write(workflow_id, record)
        log.write(workflow_id_2, _record(workflow_id_2, 'source_a'))
        log.write(workflow_id_2, None)

        orchestrator = make_orchestrator(log)
        orchestrator.apply_desired_updates(log.reader().bootstrap(timeout_s=1.0))

        assert orchestrator.get_active_job_number(workflow_id) == record.job_number
        assert orchestrator.get_active_config(workflow_id) == record.jobs
        assert orchestrator.active_job_registry.is_current(
            workflow_id, record.job_number
        )
        assert orchestrator.get_active_job_number(workflow_id_2) is None

    def test_own_write_read_back_is_a_noop(
        self, make_orchestrator, workflow_id, data_service, data_key
    ):
        log = InMemoryDesiredStateLog()
        reader = log.reader()
        orchestrator = make_orchestrator(log)
        orchestrator.commit_workflow(workflow_id)
        data_service.set_item(data_key, sc.DataArray(sc.scalar(1.0)))
        version = orchestrator.get_workflow_state_version(workflow_id)

        orchestrator.apply_desired_updates(reader.poll())

        assert orchestrator.get_workflow_state_version(workflow_id) == version
        assert data_key in data_service

    def test_record_from_another_writer_becomes_current(
        self, make_orchestrator, workflow_id, data_service, data_key
    ):
        orchestrator = make_orchestrator(InMemoryDesiredStateLog())
        orchestrator.commit_workflow(workflow_id)
        data_service.set_item(data_key, sc.DataArray(sc.scalar(1.0)))
        version = orchestrator.get_workflow_state_version(workflow_id)
        foreign = _record(workflow_id, 'source2')

        orchestrator.apply_desired_updates([(workflow_id, foreign)])

        assert orchestrator.get_active_job_number(workflow_id) == foreign.job_number
        assert set(orchestrator.get_active_config(workflow_id)) == {'source2'}
        assert orchestrator.active_job_registry.is_current(
            workflow_id, foreign.job_number
        )
        assert data_key not in data_service
        assert orchestrator.get_workflow_state_version(workflow_id) == version + 1

    def test_tombstone_from_another_writer_clears_current(
        self, make_orchestrator, workflow_id, data_service, data_key
    ):
        orchestrator = make_orchestrator(InMemoryDesiredStateLog())
        orchestrator.commit_workflow(workflow_id)
        data_service.set_item(data_key, sc.DataArray(sc.scalar(1.0)))

        orchestrator.apply_desired_updates([(workflow_id, None)])

        assert orchestrator.get_active_job_number(workflow_id) is None
        assert data_key in data_service

    def test_nothing_is_written_back(self, make_orchestrator, workflow_id):
        log = InMemoryDesiredStateLog()
        orchestrator = make_orchestrator(log)

        orchestrator.apply_desired_updates(
            [(workflow_id, _record(workflow_id, 'source1')), (workflow_id, None)]
        )

        assert log.read(0) == []


class TestUnknownEntries:
    @pytest.fixture
    def unknown_id(self) -> WorkflowId:
        return WorkflowId(instrument='test_instrument', name='removed', version=3)

    def test_are_listed_and_never_removed_by_the_dashboard(
        self, make_orchestrator, unknown_id
    ):
        log = InMemoryDesiredStateLog()
        record = _record(unknown_id, 'source1')
        log.write(unknown_id, record)
        orchestrator = make_orchestrator(log)

        orchestrator.apply_desired_updates(log.reader().bootstrap(timeout_s=1.0))

        assert orchestrator.get_unknown_workflows() == [record]
        assert len(log.read(0)) == 1
        assert orchestrator.get_desired_job_ids() == set()

    def test_removal_writes_tombstone(self, make_orchestrator, unknown_id):
        log = InMemoryDesiredStateLog()
        orchestrator = make_orchestrator(log)
        orchestrator.apply_desired_updates([(unknown_id, _record(unknown_id, 's'))])
        version = orchestrator.get_unknown_workflows_version()

        assert orchestrator.remove_unknown_workflow(unknown_id)

        assert log.read(0) == [(unknown_id, None)]
        assert orchestrator.get_unknown_workflows() == []
        assert orchestrator.get_unknown_workflows_version() == version + 1

    def test_failed_removal_keeps_entry(
        self, make_orchestrator, unknown_id, notification_queue
    ):
        orchestrator = make_orchestrator(FlakyWriter(failing=True))
        orchestrator.apply_desired_updates([(unknown_id, _record(unknown_id, 's'))])

        assert not orchestrator.remove_unknown_workflow(unknown_id)

        assert len(orchestrator.get_unknown_workflows()) == 1
        assert len(_errors(notification_queue)) == 1

    def test_tombstone_from_another_writer_removes_entry(
        self, make_orchestrator, unknown_id
    ):
        orchestrator = make_orchestrator(InMemoryDesiredStateLog())
        orchestrator.apply_desired_updates([(unknown_id, _record(unknown_id, 's'))])

        orchestrator.apply_desired_updates([(unknown_id, None)])

        assert orchestrator.get_unknown_workflows() == []


class TestConfigStore:
    def test_commit_persists_staged_configs_only(self, make_orchestrator, workflow_id):
        store: dict[str, dict] = {}
        orchestrator = make_orchestrator(InMemoryDesiredStateLog(), store)

        orchestrator.commit_workflow(workflow_id)

        assert list(store[str(workflow_id)]) == ['jobs']

    def test_legacy_store_file_with_current_job_loads(
        self, tmp_path, make_orchestrator, workflow_id
    ):
        path = tmp_path / 'workflow_configs.yaml'
        legacy = {
            str(workflow_id): {
                'jobs': {
                    'source1': {'params': {'threshold': 5.0}, 'aux_source_names': {}}
                },
                'current_job': {
                    'job_number': str(uuid.uuid4()),
                    'jobs': {
                        'source1': {
                            'params': {'threshold': 5.0},
                            'aux_source_names': {},
                        }
                    },
                },
            }
        }
        path.write_text(yaml.safe_dump(legacy))

        orchestrator = make_orchestrator(
            InMemoryDesiredStateLog(), FileBackedConfigStore(path)
        )

        staged = orchestrator.get_staged_config(workflow_id)
        assert staged['source1'].params == {'threshold': 5.0}
        assert orchestrator.get_active_job_number(workflow_id) is None
