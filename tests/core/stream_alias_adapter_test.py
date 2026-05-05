# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pytest

from ess.livedata.config.acknowledgement import AcknowledgementResponse
from ess.livedata.config.workflow_spec import JobId, WorkflowConfig, WorkflowId
from ess.livedata.core.job_manager import JobManager
from ess.livedata.core.stream_alias import (
    BindStreamAlias,
    StreamAliasRegistry,
    UnbindStreamAlias,
)
from ess.livedata.core.stream_alias_adapter import StreamAliasAdapter

from .job_manager_test import FakeJobFactory


@pytest.fixture
def factory() -> FakeJobFactory:
    return FakeJobFactory()


@pytest.fixture
def manager(factory: FakeJobFactory) -> JobManager:
    return JobManager(factory)


@pytest.fixture
def workflow_config() -> WorkflowConfig:
    return WorkflowConfig(
        identifier=WorkflowId(
            instrument="test",
            name="test_workflow",
            version=1,
        )
    )


@pytest.fixture
def hosted_job_id(manager: JobManager, workflow_config: WorkflowConfig) -> JobId:
    """A job_id known to the manager (scheduled, not yet active)."""
    return manager.schedule_job("source-x", workflow_config)


@pytest.fixture
def registry() -> StreamAliasRegistry:
    return StreamAliasRegistry()


@pytest.fixture
def adapter(registry: StreamAliasRegistry, manager: JobManager) -> StreamAliasAdapter:
    return StreamAliasAdapter(registry=registry, job_manager=manager)


def _bind_value(
    *,
    alias: str,
    job_id: JobId,
    output_name: str = 'result',
    message_id: str | None = None,
) -> dict:
    return BindStreamAlias(
        alias=alias,
        job_id=job_id,
        output_name=output_name,
        message_id=message_id,
    ).model_dump(mode='json')


def _unbind_value(*, alias: str, message_id: str | None = None) -> dict:
    return UnbindStreamAlias(alias=alias, message_id=message_id).model_dump(mode='json')


class TestBind:
    def test_actor_acks_success(
        self,
        adapter: StreamAliasAdapter,
        registry: StreamAliasRegistry,
        hosted_job_id: JobId,
    ) -> None:
        ack = adapter.bind(
            'ignored',
            _bind_value(alias='fom-0', job_id=hosted_job_id, message_id='msg-1'),
        )
        assert ack is not None
        assert ack.message_id == 'msg-1'
        assert ack.device == 'fom-0'
        assert ack.response == AcknowledgementResponse.ACK
        assert registry.lookup(hosted_job_id, 'result') == 'fom-0'

    def test_non_actor_silent(
        self, adapter: StreamAliasAdapter, registry: StreamAliasRegistry
    ) -> None:
        unknown = JobId(source_name='other', job_number=uuid.uuid4())
        ack = adapter.bind(
            'ignored',
            _bind_value(alias='fom-0', job_id=unknown, message_id='msg-1'),
        )
        assert ack is None
        assert not registry.has('fom-0')

    def test_conflicting_alias_acks_error(
        self,
        adapter: StreamAliasAdapter,
        registry: StreamAliasRegistry,
        hosted_job_id: JobId,
    ) -> None:
        # Pre-bind (job, 'result') under fom-0, then try fom-1 for the same pair.
        registry.bind('fom-0', hosted_job_id, 'result')
        ack = adapter.bind(
            'ignored',
            _bind_value(
                alias='fom-1',
                job_id=hosted_job_id,
                output_name='result',
                message_id='msg-2',
            ),
        )
        assert ack is not None
        assert ack.response == AcknowledgementResponse.ERR
        assert 'already bound' in (ack.message or '')
        # Original binding intact.
        assert registry.lookup(hosted_job_id, 'result') == 'fom-0'

    def test_multi_bind_under_same_alias(
        self,
        adapter: StreamAliasAdapter,
        registry: StreamAliasRegistry,
        manager: JobManager,
        workflow_config: WorkflowConfig,
    ) -> None:
        """Same alias can host multiple (job, output) bindings."""
        first = manager.schedule_job('det_1', workflow_config)
        second = manager.schedule_job('det_2', workflow_config)
        ack1 = adapter.bind(
            'ignored',
            _bind_value(alias='fom-0', job_id=first, message_id='msg-a'),
        )
        ack2 = adapter.bind(
            'ignored',
            _bind_value(alias='fom-0', job_id=second, message_id='msg-b'),
        )
        assert ack1 is not None
        assert ack1.response == AcknowledgementResponse.ACK
        assert ack2 is not None
        assert ack2.response == AcknowledgementResponse.ACK
        assert registry.lookup(first, 'result') == 'fom-0'
        assert registry.lookup(second, 'result') == 'fom-0'

    def test_actor_without_message_id_returns_none(
        self,
        adapter: StreamAliasAdapter,
        registry: StreamAliasRegistry,
        hosted_job_id: JobId,
    ) -> None:
        ack = adapter.bind('ignored', _bind_value(alias='fom-0', job_id=hosted_job_id))
        assert ack is None
        # Side effect still applied.
        assert registry.has('fom-0')


class TestUnbind:
    def test_actor_acks_success(
        self,
        adapter: StreamAliasAdapter,
        registry: StreamAliasRegistry,
        hosted_job_id: JobId,
    ) -> None:
        registry.bind('fom-0', hosted_job_id, 'result')
        ack = adapter.unbind(
            'ignored', _unbind_value(alias='fom-0', message_id='msg-3')
        )
        assert ack is not None
        assert ack.message_id == 'msg-3'
        assert ack.response == AcknowledgementResponse.ACK
        assert not registry.has('fom-0')

    def test_unknown_alias_silent(self, adapter: StreamAliasAdapter) -> None:
        ack = adapter.unbind(
            'ignored',
            _unbind_value(alias='not-bound-anywhere', message_id='msg-4'),
        )
        assert ack is None

    def test_actor_without_message_id_returns_none(
        self,
        adapter: StreamAliasAdapter,
        registry: StreamAliasRegistry,
        hosted_job_id: JobId,
    ) -> None:
        registry.bind('fom-0', hosted_job_id, 'result')
        ack = adapter.unbind('ignored', _unbind_value(alias='fom-0'))
        assert ack is None
        assert not registry.has('fom-0')
