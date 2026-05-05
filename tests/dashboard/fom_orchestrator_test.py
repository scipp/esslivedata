# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for FOMOrchestrator."""

from __future__ import annotations

from typing import Any

import pydantic
import pytest
import scipp as sc

from ess.livedata.config.models import ConfigKey
from ess.livedata.config.workflow_spec import (
    REDUCTION,
    JobId,
    WorkflowConfig,
    WorkflowOutputsBase,
    WorkflowSpec,
)
from ess.livedata.core.job import JobState, JobStatus
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.core.message import COMMANDS_STREAM_ID
from ess.livedata.core.stream_alias import BindStreamAlias, UnbindStreamAlias
from ess.livedata.dashboard.active_job_registry import ActiveJobRegistry
from ess.livedata.dashboard.command_service import CommandService
from ess.livedata.dashboard.config_store import ConfigStore, InMemoryConfigStore
from ess.livedata.dashboard.data_service import DataService
from ess.livedata.dashboard.fom_orchestrator import (
    FOMOrchestrator,
    FOMSlot,
)
from ess.livedata.dashboard.job_service import JobService
from ess.livedata.dashboard.notification_queue import NotificationQueue
from ess.livedata.fakes import FakeMessageSink
from ess.livedata.handlers.config_handler import ConfigUpdate


class _Params(pydantic.BaseModel):
    threshold: float = 1.0


class _Outputs(WorkflowOutputsBase):
    result: sc.DataArray = pydantic.Field(title='Result')


def _spec(name: str, sources: list[str]) -> WorkflowSpec:
    return WorkflowSpec(
        instrument='test',
        group=REDUCTION,
        name=name,
        version=1,
        title=name,
        description='',
        source_names=sources,
        params=_Params,
        outputs=_Outputs,
    )


@pytest.fixture
def workflow_a() -> WorkflowSpec:
    return _spec('workflow_a', ['det_1', 'det_2'])


@pytest.fixture
def workflow_b() -> WorkflowSpec:
    return _spec('workflow_b', ['det_3'])


@pytest.fixture
def sink() -> FakeMessageSink:
    return FakeMessageSink()


@pytest.fixture
def data_service() -> DataService:
    return DataService()


@pytest.fixture
def job_service() -> JobService:
    return JobService()


@pytest.fixture
def active_job_registry(
    data_service: DataService, job_service: JobService
) -> ActiveJobRegistry:
    return ActiveJobRegistry(data_service=data_service, job_service=job_service)


@pytest.fixture
def notification_queue() -> NotificationQueue:
    return NotificationQueue()


def _make_orchestrator(
    *specs: WorkflowSpec,
    sink: FakeMessageSink,
    active_job_registry: ActiveJobRegistry,
    job_service: JobService,
    notification_queue: NotificationQueue | None = None,
    config_store: ConfigStore | None = None,
    n_slots: int = 2,
) -> FOMOrchestrator:
    registry = {spec.get_id(): spec for spec in specs}
    return FOMOrchestrator(
        command_service=CommandService(sink=sink),
        workflow_registry=registry,
        active_job_registry=active_job_registry,
        job_service=job_service,
        notification_queue=notification_queue,
        config_store=config_store,
        n_slots=n_slots,
    )


def _last_batch(sink: FakeMessageSink) -> list[tuple[ConfigKey, Any]]:
    """Return the most recent batch as (config_key, value) tuples."""
    last = sink.published_messages[-1]
    return [
        (msg.value.config_key, msg.value.value)
        for msg in last
        if msg.stream == COMMANDS_STREAM_ID and isinstance(msg.value, ConfigUpdate)
    ]


class TestSlotCount:
    def test_default_n_slots_is_two(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        assert orch.slot_names == [FOMSlot('fom-0'), FOMSlot('fom-1')]

    def test_custom_n_slots(self, workflow_a, sink, active_job_registry, job_service):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            n_slots=4,
        )
        assert len(orch.slot_names) == 4
        assert orch.slot_names[-1] == FOMSlot('fom-3')

    def test_zero_slots_rejected(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        with pytest.raises(ValueError, match="n_slots"):
            _make_orchestrator(
                workflow_a,
                sink=sink,
                active_job_registry=active_job_registry,
                job_service=job_service,
                n_slots=0,
            )

    def test_unknown_slot_raises(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        with pytest.raises(KeyError):
            orch.commit_slot(
                FOMSlot('fom-99'),
                workflow_id=workflow_a.get_id(),
                source_names=['det_1'],
                output_name='result',
                params={},
            )


class TestCommitEmptySlot:
    def test_sends_workflow_config_and_bind_only(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        orch.commit_slot(
            FOMSlot('fom-0'),
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={'threshold': 5.0},
        )
        batch = _last_batch(sink)
        assert len(batch) == 2
        kinds = [v.__class__ for _, v in batch]
        assert WorkflowConfig in kinds
        assert BindStreamAlias in kinds
        assert JobCommand not in kinds
        assert UnbindStreamAlias not in kinds

    def test_state_recorded(self, workflow_a, sink, active_job_registry, job_service):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        (job_id,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={'threshold': 5.0},
        )
        state = orch.get_slot_state(slot)
        assert state is not None
        assert state.workflow_id == workflow_a.get_id()
        assert state.source_names == ('det_1',)
        assert state.output_name == 'result'
        assert state.params == {'threshold': 5.0}
        assert state.job_ids == (job_id,)

    def test_bind_command_carries_alias_and_job(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-1')
        (job_id,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_2'],
            output_name='result',
            params={},
        )
        binds = [v for _, v in _last_batch(sink) if isinstance(v, BindStreamAlias)]
        assert len(binds) == 1
        assert binds[0].alias == slot
        assert binds[0].job_id == job_id
        assert binds[0].output_name == 'result'

    def test_workflow_config_carries_job_number(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        (job_id,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={'threshold': 7.0},
        )
        wcs = [v for _, v in _last_batch(sink) if isinstance(v, WorkflowConfig)]
        assert len(wcs) == 1
        assert wcs[0].job_number == job_id.job_number
        assert wcs[0].params == {'threshold': 7.0}

    def test_all_messages_share_message_id(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        orch.commit_slot(
            FOMSlot('fom-0'),
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        ids = {v.message_id for _, v in _last_batch(sink)}
        assert len(ids) == 1
        assert next(iter(ids)) is not None

    def test_active_job_registry_activated(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        (job_id,) = orch.commit_slot(
            FOMSlot('fom-0'),
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        assert active_job_registry.is_active(job_id.job_number)

    def test_version_bumped(self, workflow_a, sink, active_job_registry, job_service):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        v0 = orch.get_slot_state_version(slot)
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        assert orch.get_slot_state_version(slot) > v0


class TestCommitBoundSlot:
    def test_sends_full_four_command_batch(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_2'],
            output_name='result',
            params={'threshold': 9.0},
        )
        batch = _last_batch(sink)
        kinds = [v.__class__ for _, v in batch]
        assert kinds.count(JobCommand) == 1
        assert kinds.count(UnbindStreamAlias) == 1
        assert kinds.count(WorkflowConfig) == 1
        assert kinds.count(BindStreamAlias) == 1

    def test_stop_targets_previous_job(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        (first,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_2'],
            output_name='result',
            params={},
        )
        stops = [
            v
            for _, v in _last_batch(sink)
            if isinstance(v, JobCommand) and v.action == JobAction.stop
        ]
        assert len(stops) == 1
        assert stops[0].job_id == first

    def test_unbind_targets_alias(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_2'],
            output_name='result',
            params={},
        )
        unbinds = [v for _, v in _last_batch(sink) if isinstance(v, UnbindStreamAlias)]
        assert unbinds[0].alias == slot

    def test_previous_job_deactivated(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        (first,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        (second,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_2'],
            output_name='result',
            params={},
        )
        assert not active_job_registry.is_active(first.job_number)
        assert active_job_registry.is_active(second.job_number)

    def test_state_replaced(
        self, workflow_a, workflow_b, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            workflow_b,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={'threshold': 1.0},
        )
        orch.commit_slot(
            slot,
            workflow_id=workflow_b.get_id(),
            source_names=['det_3'],
            output_name='result',
            params={'threshold': 2.0},
        )
        state = orch.get_slot_state(slot)
        assert state is not None
        assert state.workflow_id == workflow_b.get_id()
        assert state.source_names == ('det_3',)
        assert state.params == {'threshold': 2.0}


class TestReleaseSlot:
    def test_release_empty_returns_false(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        assert orch.release_slot(FOMSlot('fom-0')) is False
        assert sink.published_messages == []

    def test_release_bound_sends_stop_and_unbind(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        (job_id,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        assert orch.release_slot(slot) is True

        batch = _last_batch(sink)
        stops = [
            v
            for _, v in batch
            if isinstance(v, JobCommand) and v.action == JobAction.stop
        ]
        unbinds = [v for _, v in batch if isinstance(v, UnbindStreamAlias)]
        assert len(stops) == 1
        assert stops[0].job_id == job_id
        assert len(unbinds) == 1
        assert unbinds[0].alias == slot

    def test_release_retains_config_with_no_job_number(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        (job_id,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={'threshold': 3.0},
        )
        orch.release_slot(slot)
        state = orch.get_slot_state(slot)
        assert state is not None
        assert state.workflow_id == workflow_a.get_id()
        assert state.source_names == ('det_1',)
        assert state.output_name == 'result'
        assert state.params == {'threshold': 3.0}
        assert state.job_number is None
        assert state.is_running is False
        assert not active_job_registry.is_active(job_id.job_number)

    def test_release_already_stopped_returns_false(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        orch.release_slot(slot)
        n_batches = len(sink.published_messages)
        assert orch.release_slot(slot) is False
        assert len(sink.published_messages) == n_batches


class TestStartSlot:
    def test_start_empty_returns_none(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        assert orch.start_slot(FOMSlot('fom-0')) is None

    def test_start_running_returns_none(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        n_batches = len(sink.published_messages)
        assert orch.start_slot(slot) is None
        assert len(sink.published_messages) == n_batches

    def test_start_stopped_relaunches_with_same_config(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        (first,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={'threshold': 5.0},
        )
        orch.release_slot(slot)
        new = orch.start_slot(slot)
        assert new is not None
        (new_id,) = new
        assert new_id.source_name == 'det_1'
        assert new_id != first
        state = orch.get_slot_state(slot)
        assert state is not None
        assert state.is_running is True
        assert state.params == {'threshold': 5.0}
        assert active_job_registry.is_active(new_id.job_number)
        # No Stop / Unbind since previous slot was already stopped.
        kinds = [v.__class__ for _, v in _last_batch(sink)]
        assert JobCommand not in kinds
        assert UnbindStreamAlias not in kinds
        assert WorkflowConfig in kinds
        assert BindStreamAlias in kinds


class TestClearSlot:
    def test_clear_empty_returns_false(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        assert orch.clear_slot(FOMSlot('fom-0')) is False

    def test_clear_running_returns_false(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        assert orch.clear_slot(slot) is False
        assert orch.get_slot_state(slot) is not None

    def test_clear_stopped_drops_state(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        orch.release_slot(slot)
        v0 = orch.get_slot_state_version(slot)
        assert orch.clear_slot(slot) is True
        assert orch.get_slot_state(slot) is None
        assert orch.get_slot_state_version(slot) > v0


class TestCommitStoppedSlot:
    def test_commit_after_release_skips_stop_and_unbind(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        orch.release_slot(slot)
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_2'],
            output_name='result',
            params={},
        )
        kinds = [v.__class__ for _, v in _last_batch(sink)]
        assert JobCommand not in kinds
        assert UnbindStreamAlias not in kinds
        assert kinds.count(WorkflowConfig) == 1
        assert kinds.count(BindStreamAlias) == 1


class TestResetSlot:
    def test_reset_empty_returns_false(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        assert orch.reset_slot(FOMSlot('fom-0')) is False

    def test_reset_stopped_returns_false(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        orch.release_slot(slot)
        assert orch.reset_slot(slot) is False

    def test_reset_bound_sends_reset(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        (job_id,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        assert orch.reset_slot(slot) is True
        cmds = [v for _, v in _last_batch(sink) if isinstance(v, JobCommand)]
        assert len(cmds) == 1
        assert cmds[0].action == JobAction.reset
        assert cmds[0].job_id == job_id

    def test_reset_does_not_clear_slot(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        orch.reset_slot(slot)
        assert orch.get_slot_state(slot) is not None


class TestMultiSource:
    def test_commit_empty_slot_with_two_sources_emits_2_workflow_configs_and_2_binds(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        job_ids = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1', 'det_2'],
            output_name='result',
            params={'threshold': 4.0},
        )
        assert len(job_ids) == 2
        # All source-jobs share one job_number.
        assert {jid.job_number for jid in job_ids} == {job_ids[0].job_number}
        assert {jid.source_name for jid in job_ids} == {'det_1', 'det_2'}

        batch = _last_batch(sink)
        kinds = [v.__class__ for _, v in batch]
        assert kinds.count(WorkflowConfig) == 2
        assert kinds.count(BindStreamAlias) == 2
        assert JobCommand not in kinds
        assert UnbindStreamAlias not in kinds
        # All four messages share one message_id.
        assert len({v.message_id for _, v in batch}) == 1
        # Binds carry the slot alias and per-source job_ids.
        binds = [v for _, v in batch if isinstance(v, BindStreamAlias)]
        assert {b.alias for b in binds} == {slot}
        assert {b.job_id for b in binds} == set(job_ids)

    def test_commit_replaces_with_n_stops_one_unbind_n_configs_n_binds(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        prev_ids = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1', 'det_2'],
            output_name='result',
            params={},
        )
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_2'],
            output_name='result',
            params={'threshold': 9.0},
        )
        batch = _last_batch(sink)
        kinds = [v.__class__ for _, v in batch]
        assert kinds.count(JobCommand) == 2  # one Stop per prev source-job
        assert kinds.count(UnbindStreamAlias) == 1  # single Unbind for the alias
        assert kinds.count(WorkflowConfig) == 1
        assert kinds.count(BindStreamAlias) == 1
        stops = [
            v
            for _, v in batch
            if isinstance(v, JobCommand) and v.action == JobAction.stop
        ]
        assert {s.job_id for s in stops} == set(prev_ids)

    def test_release_multi_source_sends_n_stops_one_unbind(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        prev_ids = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1', 'det_2'],
            output_name='result',
            params={},
        )
        orch.release_slot(slot)
        batch = _last_batch(sink)
        stops = [
            v
            for _, v in batch
            if isinstance(v, JobCommand) and v.action == JobAction.stop
        ]
        unbinds = [v for _, v in batch if isinstance(v, UnbindStreamAlias)]
        assert {s.job_id for s in stops} == set(prev_ids)
        assert len(unbinds) == 1

    def test_reset_multi_source_sends_n_resets(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        prev_ids = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1', 'det_2'],
            output_name='result',
            params={},
        )
        assert orch.reset_slot(slot) is True
        resets = [
            v
            for _, v in _last_batch(sink)
            if isinstance(v, JobCommand) and v.action == JobAction.reset
        ]
        assert {r.job_id for r in resets} == set(prev_ids)

    def test_one_source_job_stopped_transitions_slot_to_stopped(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        prev_ids = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1', 'det_2'],
            output_name='result',
            params={},
        )
        orch.on_job_status_updated(
            JobStatus(
                job_id=prev_ids[0],
                workflow_id=workflow_a.get_id(),
                state=JobState.stopped,
            )
        )
        state = orch.get_slot_state(slot)
        assert state is not None
        assert state.is_running is False
        # Shared job_number deactivated.
        assert not active_job_registry.is_active(prev_ids[0].job_number)

    def test_empty_sources_rejected(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        with pytest.raises(ValueError, match="source_names"):
            orch.commit_slot(
                FOMSlot('fom-0'),
                workflow_id=workflow_a.get_id(),
                source_names=[],
                output_name='result',
                params={},
            )

    def test_persisted_multi_source_round_trip(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        store = InMemoryConfigStore()
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            config_store=store,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1', 'det_2'],
            output_name='result',
            params={'threshold': 5.0},
        )
        entry = store[slot]
        assert entry['source_names'] == ['det_1', 'det_2']

        # New orchestrator restores both sources.
        orch2 = _make_orchestrator(
            workflow_a,
            sink=FakeMessageSink(),
            active_job_registry=ActiveJobRegistry(
                data_service=DataService(),
                job_service=JobService(),
            ),
            job_service=JobService(),
            config_store=store,
        )
        restored = orch2.get_slot_state(slot)
        assert restored is not None
        assert restored.source_names == ('det_1', 'det_2')


class TestMultipleSlots:
    def test_independent_slots(
        self, workflow_a, workflow_b, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            workflow_b,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        s0 = FOMSlot('fom-0')
        s1 = FOMSlot('fom-1')
        (j0,) = orch.commit_slot(
            s0,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        (j1,) = orch.commit_slot(
            s1,
            workflow_id=workflow_b.get_id(),
            source_names=['det_3'],
            output_name='result',
            params={},
        )
        assert j0 != j1
        assert orch.get_slot_state(s0).source_names == ('det_1',)
        assert orch.get_slot_state(s1).source_names == ('det_3',)
        assert active_job_registry.is_active(j0.job_number)
        assert active_job_registry.is_active(j1.job_number)

    def test_release_one_does_not_affect_other(
        self, workflow_a, workflow_b, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            workflow_b,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        s0 = FOMSlot('fom-0')
        s1 = FOMSlot('fom-1')
        orch.commit_slot(
            s0,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        orch.commit_slot(
            s1,
            workflow_id=workflow_b.get_id(),
            source_names=['det_3'],
            output_name='result',
            params={},
        )
        orch.release_slot(s0)
        assert orch.get_slot_state(s0) is not None
        assert orch.get_slot_state(s0).is_running is False
        assert orch.get_slot_state(s1) is not None
        assert orch.get_slot_state(s1).is_running is True


class TestAcknowledgement:
    def test_success_pushes_notification(
        self,
        workflow_a,
        sink,
        active_job_registry,
        job_service,
        notification_queue,
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            notification_queue=notification_queue,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        msg_id = next(iter({v.message_id for _, v in _last_batch(sink)}))
        # Two ACKs (matches expected_count for empty-slot commit)
        orch.process_acknowledgement(msg_id, "ACK")
        orch.process_acknowledgement(msg_id, "ACK")
        events = notification_queue.get_all_events()
        assert len(events) == 1
        assert 'fom-0' in events[0].message

    def test_partial_failure_pushes_error(
        self,
        workflow_a,
        sink,
        active_job_registry,
        job_service,
        notification_queue,
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            notification_queue=notification_queue,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        msg_id = next(iter({v.message_id for _, v in _last_batch(sink)}))
        orch.process_acknowledgement(msg_id, "ACK")
        orch.process_acknowledgement(msg_id, "ERR", error_message="alias busy")
        events = notification_queue.get_all_events()
        assert any('failed' in e.message for e in events)

    def test_unknown_message_id_ignored(
        self,
        workflow_a,
        sink,
        active_job_registry,
        job_service,
        notification_queue,
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            notification_queue=notification_queue,
        )
        orch.process_acknowledgement("does-not-exist", "ACK")
        assert notification_queue.get_all_events() == []


class TestOnJobStatusUpdated:
    def test_marks_slot_stopped_when_backend_reports_stopped(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        (job_id,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={'threshold': 4.0},
        )
        orch.on_job_status_updated(
            JobStatus(
                job_id=job_id,
                workflow_id=workflow_a.get_id(),
                state=JobState.stopped,
            )
        )
        state = orch.get_slot_state(slot)
        assert state is not None
        assert state.is_running is False
        assert state.params == {'threshold': 4.0}
        assert not active_job_registry.is_active(job_id.job_number)

    def test_active_status_does_not_clear_slot(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        (job_id,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        orch.on_job_status_updated(
            JobStatus(
                job_id=job_id,
                workflow_id=workflow_a.get_id(),
                state=JobState.active,
            )
        )
        assert orch.get_slot_state(slot) is not None

    def test_status_for_unrelated_job_ignored(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        import uuid as _uuid

        unrelated = JobId(source_name='other', job_number=_uuid.uuid4())
        orch.on_job_status_updated(
            JobStatus(
                job_id=unrelated,
                workflow_id=workflow_a.get_id(),
                state=JobState.stopped,
            )
        )
        assert orch.get_slot_state(slot) is not None


class TestPersistence:
    def test_commit_writes_to_store(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        store = InMemoryConfigStore()
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            config_store=store,
        )
        slot = FOMSlot('fom-0')
        (job_id,) = orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={'threshold': 5.0},
        )
        entry = store[slot]
        assert entry['version'] == 2
        assert entry['workflow_id'] == str(workflow_a.get_id())
        assert entry['source_names'] == ['det_1']
        assert entry['output_name'] == 'result'
        assert entry['params'] == {'threshold': 5.0}
        assert entry['job_number'] == str(job_id.job_number)

    def test_release_persists_with_null_job_number(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        store = InMemoryConfigStore()
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            config_store=store,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={'threshold': 5.0},
        )
        orch.release_slot(slot)
        entry = store[slot]
        assert entry['job_number'] is None
        assert entry['params'] == {'threshold': 5.0}

    def test_clear_removes_from_store(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        store = InMemoryConfigStore()
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            config_store=store,
        )
        slot = FOMSlot('fom-0')
        orch.commit_slot(
            slot,
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        orch.release_slot(slot)
        orch.clear_slot(slot)
        assert slot not in store

    def test_restore_loads_running_slot_as_probing(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        import uuid as _uuid

        store = InMemoryConfigStore()
        slot = FOMSlot('fom-0')
        job_number = _uuid.uuid4()
        store[slot] = {
            'version': 2,
            'workflow_id': str(workflow_a.get_id()),
            'source_names': ['det_1'],
            'output_name': 'result',
            'params': {'threshold': 7.0},
            'aux_source_names': {},
            'job_number': str(job_number),
        }
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            config_store=store,
        )
        state = orch.get_slot_state(slot)
        assert state is not None
        assert state.workflow_id == workflow_a.get_id()
        assert state.params == {'threshold': 7.0}
        assert state.job_number == job_number
        assert state.is_running is True
        assert active_job_registry.is_active(job_number)

    def test_restore_loads_stopped_slot(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        store = InMemoryConfigStore()
        slot = FOMSlot('fom-0')
        store[slot] = {
            'version': 2,
            'workflow_id': str(workflow_a.get_id()),
            'source_names': ['det_1'],
            'output_name': 'result',
            'params': {'threshold': 7.0},
            'aux_source_names': {},
            'job_number': None,
        }
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            config_store=store,
        )
        state = orch.get_slot_state(slot)
        assert state is not None
        assert state.is_running is False
        assert state.job_number is None

    def test_restore_drops_unknown_workflow(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        store = InMemoryConfigStore()
        store[FOMSlot('fom-0')] = {
            'version': 2,
            'workflow_id': 'other/ns/missing/1',
            'source_names': ['det_1'],
            'output_name': 'result',
            'params': {},
            'aux_source_names': {},
            'job_number': None,
        }
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            config_store=store,
        )
        assert orch.get_slot_state(FOMSlot('fom-0')) is None

    def test_restore_drops_unknown_source(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        store = InMemoryConfigStore()
        store[FOMSlot('fom-0')] = {
            'version': 2,
            'workflow_id': str(workflow_a.get_id()),
            'source_names': ['det_does_not_exist'],
            'output_name': 'result',
            'params': {},
            'aux_source_names': {},
            'job_number': None,
        }
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            config_store=store,
        )
        assert orch.get_slot_state(FOMSlot('fom-0')) is None

    def test_restore_drops_unknown_output(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        store = InMemoryConfigStore()
        store[FOMSlot('fom-0')] = {
            'version': 2,
            'workflow_id': str(workflow_a.get_id()),
            'source_names': ['det_1'],
            'output_name': 'no_such_output',
            'params': {},
            'aux_source_names': {},
            'job_number': None,
        }
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            config_store=store,
        )
        assert orch.get_slot_state(FOMSlot('fom-0')) is None

    def test_restore_drops_invalid_params(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        store = InMemoryConfigStore()
        store[FOMSlot('fom-0')] = {
            'version': 2,
            'workflow_id': str(workflow_a.get_id()),
            'source_names': ['det_1'],
            'output_name': 'result',
            'params': {'threshold': 'not-a-float'},
            'aux_source_names': {},
            'job_number': None,
        }
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            config_store=store,
        )
        assert orch.get_slot_state(FOMSlot('fom-0')) is None

    def test_restore_drops_unknown_schema_version(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        store = InMemoryConfigStore()
        store[FOMSlot('fom-0')] = {
            'version': 999,
            'workflow_id': str(workflow_a.get_id()),
            'source_names': ['det_1'],
            'output_name': 'result',
            'params': {},
            'aux_source_names': {},
            'job_number': None,
        }
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
            config_store=store,
        )
        assert orch.get_slot_state(FOMSlot('fom-0')) is None

    def test_no_store_works(self, workflow_a, sink, active_job_registry, job_service):
        orch = _make_orchestrator(
            workflow_a,
            sink=sink,
            active_job_registry=active_job_registry,
            job_service=job_service,
        )
        # Without a store, mutations must not raise.
        orch.commit_slot(
            FOMSlot('fom-0'),
            workflow_id=workflow_a.get_id(),
            source_names=['det_1'],
            output_name='result',
            params={},
        )
        orch.release_slot(FOMSlot('fom-0'))
        orch.clear_slot(FOMSlot('fom-0'))


def _persisted_running(workflow_a: WorkflowSpec, job_number) -> dict:
    return {
        'version': 2,
        'workflow_id': str(workflow_a.get_id()),
        'source_names': ['det_1'],
        'output_name': 'result',
        'params': {'threshold': 1.0},
        'aux_source_names': {},
        'job_number': str(job_number) if job_number is not None else None,
    }


def _orch_with_probe(
    workflow_a, sink, active_job_registry, job_service, store, *, timeout=0.0
) -> FOMOrchestrator:
    return FOMOrchestrator(
        command_service=CommandService(sink=sink),
        workflow_registry={workflow_a.get_id(): workflow_a},
        active_job_registry=active_job_registry,
        job_service=job_service,
        config_store=store,
        n_slots=2,
        probe_timeout_seconds=timeout,
    )


class TestProbe:
    def test_tick_before_deadline_is_noop(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        import uuid as _uuid

        store = InMemoryConfigStore()
        job_number = _uuid.uuid4()
        store[FOMSlot('fom-0')] = _persisted_running(workflow_a, job_number)
        orch = _orch_with_probe(
            workflow_a,
            sink,
            active_job_registry,
            job_service,
            store,
            timeout=60.0,
        )
        orch.tick()
        state = orch.get_slot_state(FOMSlot('fom-0'))
        assert state is not None
        assert state.is_running is True
        assert active_job_registry.is_active(job_number)

    def test_tick_after_deadline_without_status_clears_job_number(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        import uuid as _uuid

        store = InMemoryConfigStore()
        job_number = _uuid.uuid4()
        store[FOMSlot('fom-0')] = _persisted_running(workflow_a, job_number)
        orch = _orch_with_probe(
            workflow_a, sink, active_job_registry, job_service, store
        )
        orch.tick()
        state = orch.get_slot_state(FOMSlot('fom-0'))
        assert state is not None
        assert state.is_running is False
        assert state.params == {'threshold': 1.0}  # config retained
        assert not active_job_registry.is_active(job_number)
        assert store[FOMSlot('fom-0')]['job_number'] is None

    def test_fresh_status_resolves_probe(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        import uuid as _uuid

        store = InMemoryConfigStore()
        job_number = _uuid.uuid4()
        store[FOMSlot('fom-0')] = _persisted_running(workflow_a, job_number)
        orch = _orch_with_probe(
            workflow_a, sink, active_job_registry, job_service, store
        )
        job_id = JobId(source_name='det_1', job_number=job_number)
        job_service.status_updated(
            JobStatus(
                job_id=job_id,
                workflow_id=workflow_a.get_id(),
                state=JobState.active,
            )
        )
        orch.on_job_status_updated(job_service.job_statuses[job_id])
        orch.tick()
        state = orch.get_slot_state(FOMSlot('fom-0'))
        assert state is not None
        assert state.is_running is True
        assert state.job_number == job_number

    def test_stopped_status_during_probe_transitions_to_stopped(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        import uuid as _uuid

        store = InMemoryConfigStore()
        job_number = _uuid.uuid4()
        store[FOMSlot('fom-0')] = _persisted_running(workflow_a, job_number)
        orch = _orch_with_probe(
            workflow_a, sink, active_job_registry, job_service, store
        )
        job_id = JobId(source_name='det_1', job_number=job_number)
        orch.on_job_status_updated(
            JobStatus(
                job_id=job_id,
                workflow_id=workflow_a.get_id(),
                state=JobState.stopped,
            )
        )
        state = orch.get_slot_state(FOMSlot('fom-0'))
        assert state is not None
        assert state.is_running is False
        assert state.params == {'threshold': 1.0}

    def test_user_commit_during_probe_supersedes_probe(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        import uuid as _uuid

        store = InMemoryConfigStore()
        old_job_number = _uuid.uuid4()
        store[FOMSlot('fom-0')] = _persisted_running(workflow_a, old_job_number)
        orch = _orch_with_probe(
            workflow_a, sink, active_job_registry, job_service, store
        )
        new_jobs = orch.commit_slot(
            FOMSlot('fom-0'),
            workflow_id=workflow_a.get_id(),
            source_names=['det_2'],
            output_name='result',
            params={'threshold': 9.0},
        )
        # tick() must not undo the just-committed job.
        orch.tick()
        state = orch.get_slot_state(FOMSlot('fom-0'))
        assert state is not None
        assert state.is_running is True
        assert state.job_number == new_jobs[0].job_number
        assert active_job_registry.is_active(new_jobs[0].job_number)
        assert not active_job_registry.is_active(old_job_number)

    def test_no_probe_for_stopped_persisted_slot(
        self, workflow_a, sink, active_job_registry, job_service
    ):
        store = InMemoryConfigStore()
        slot = FOMSlot('fom-0')
        store[slot] = _persisted_running(workflow_a, None)
        orch = _orch_with_probe(
            workflow_a, sink, active_job_registry, job_service, store
        )
        orch.tick()  # nothing pending; no-op
        state = orch.get_slot_state(slot)
        assert state is not None
        assert state.is_running is False
