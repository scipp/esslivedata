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
def hosted_job_id(manager: JobManager) -> JobId:
    """A job_id known to the manager (scheduled, not yet active)."""
    job_id = JobId(source_name="source-x", job_number=uuid.uuid4())
    workflow_config = WorkflowConfig(
        identifier=WorkflowId(
            instrument="test",
            namespace="data_reduction",
            name="test_workflow",
            version=1,
        ),
        job_id=job_id,
    )
    return manager.schedule_job(workflow_config)


@pytest.fixture
def registry() -> StreamAliasRegistry:
    return StreamAliasRegistry()


@pytest.fixture
def adapter(registry: StreamAliasRegistry, manager: JobManager) -> StreamAliasAdapter:
    return StreamAliasAdapter(registry=registry, job_manager=manager)


class TestBind:
    def test_actor_acks_success(
        self,
        adapter: StreamAliasAdapter,
        registry: StreamAliasRegistry,
        hosted_job_id: JobId,
    ) -> None:
        ack = adapter.bind(
            BindStreamAlias(
                alias='fom-0',
                job_id=hosted_job_id,
                output_name='result',
                message_id='msg-1',
            )
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
            BindStreamAlias(
                alias='fom-0',
                job_id=unknown,
                output_name='result',
                message_id='msg-1',
            )
        )
        assert ack is None
        assert not registry.has('fom-0')

    def test_no_replace_acks_error(
        self,
        adapter: StreamAliasAdapter,
        registry: StreamAliasRegistry,
        hosted_job_id: JobId,
    ) -> None:
        registry.bind('fom-0', hosted_job_id, 'result')
        ack = adapter.bind(
            BindStreamAlias(
                alias='fom-0',
                job_id=hosted_job_id,
                output_name='other',
                message_id='msg-2',
            )
        )
        assert ack is not None
        assert ack.response == AcknowledgementResponse.ERR
        assert 'already bound' in (ack.message or '')
        # Original binding intact.
        assert registry.lookup(hosted_job_id, 'result') == 'fom-0'

    def test_actor_without_message_id_returns_none(
        self,
        adapter: StreamAliasAdapter,
        registry: StreamAliasRegistry,
        hosted_job_id: JobId,
    ) -> None:
        ack = adapter.bind(
            BindStreamAlias(alias='fom-0', job_id=hosted_job_id, output_name='result')
        )
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
        ack = adapter.unbind(UnbindStreamAlias(alias='fom-0', message_id='msg-3'))
        assert ack is not None
        assert ack.message_id == 'msg-3'
        assert ack.response == AcknowledgementResponse.ACK
        assert not registry.has('fom-0')

    def test_unknown_alias_silent(self, adapter: StreamAliasAdapter) -> None:
        ack = adapter.unbind(
            UnbindStreamAlias(alias='not-bound-anywhere', message_id='msg-4')
        )
        assert ack is None

    def test_actor_without_message_id_returns_none(
        self,
        adapter: StreamAliasAdapter,
        registry: StreamAliasRegistry,
        hosted_job_id: JobId,
    ) -> None:
        registry.bind('fom-0', hosted_job_id, 'result')
        ack = adapter.unbind(UnbindStreamAlias(alias='fom-0'))
        assert ack is None
        assert not registry.has('fom-0')
