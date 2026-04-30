# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for :class:`UnrollingSinkAdapter`, including stream-alias mirror emission."""

from __future__ import annotations

import uuid

import pytest
import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.core.stream_alias import AliasedResult, StreamAliasRegistry
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.fakes import FakeMessageSink
from ess.livedata.kafka.sink import UnrollingSinkAdapter

WORKFLOW_ID = WorkflowId(instrument='test', namespace='reduction', name='wf', version=1)


def _make_job_id() -> JobId:
    return JobId(source_name='detector_1', job_number=uuid.uuid4())


def _make_result_message(
    *, job_id: JobId, outputs: dict[str, sc.DataArray]
) -> Message[sc.DataGroup]:
    """Build a message of the same shape OrchestratingProcessor emits."""
    result_key = ResultKey(workflow_id=WORKFLOW_ID, job_id=job_id)
    return Message(
        timestamp=Timestamp.from_ns(0),
        stream=StreamId(
            kind=StreamKind.LIVEDATA_DATA, name=result_key.model_dump_json()
        ),
        value=sc.DataGroup(outputs),
    )


@pytest.fixture
def downstream() -> FakeMessageSink:
    return FakeMessageSink()


def _scalar(value: float) -> sc.DataArray:
    return sc.DataArray(sc.scalar(value, unit='counts'))


class TestUnrollingWithoutRegistry:
    def test_passthrough_for_non_datagroup(self, downstream: FakeMessageSink) -> None:
        adapter = UnrollingSinkAdapter(downstream)
        msg = Message(
            timestamp=Timestamp.from_ns(0),
            stream=StreamId(kind=StreamKind.LIVEDATA_DATA, name='raw'),
            value=_scalar(1.0),
        )
        adapter.publish_messages([msg])
        assert downstream.messages == [msg]

    def test_unrolls_datagroup_into_per_output_messages(
        self, downstream: FakeMessageSink
    ) -> None:
        adapter = UnrollingSinkAdapter(downstream)
        job_id = _make_job_id()
        adapter.publish_messages(
            [
                _make_result_message(
                    job_id=job_id,
                    outputs={'a': _scalar(1.0), 'b': _scalar(2.0)},
                )
            ]
        )
        assert len(downstream.messages) == 2
        names = {
            ResultKey.model_validate_json(m.stream.name).output_name
            for m in downstream.messages
        }
        assert names == {'a', 'b'}
        kinds = {m.stream.kind for m in downstream.messages}
        assert kinds == {StreamKind.LIVEDATA_DATA}


class TestUnrollingWithRegistry:
    def test_no_binding_no_mirror(self, downstream: FakeMessageSink) -> None:
        registry = StreamAliasRegistry()
        adapter = UnrollingSinkAdapter(downstream, alias_registry=registry)
        job_id = _make_job_id()
        adapter.publish_messages(
            [_make_result_message(job_id=job_id, outputs={'a': _scalar(1.0)})]
        )
        assert len(downstream.messages) == 1
        assert downstream.messages[0].stream.kind == StreamKind.LIVEDATA_DATA

    def test_bound_output_emits_mirror_alongside_data(
        self, downstream: FakeMessageSink
    ) -> None:
        registry = StreamAliasRegistry()
        job_id = _make_job_id()
        registry.bind('fom-0', job_id, 'a')
        adapter = UnrollingSinkAdapter(downstream, alias_registry=registry)
        adapter.publish_messages(
            [
                _make_result_message(
                    job_id=job_id,
                    outputs={'a': _scalar(1.0), 'b': _scalar(2.0)},
                )
            ]
        )
        # Two data messages plus one mirror.
        assert len(downstream.messages) == 3
        data_msgs = [
            m for m in downstream.messages if m.stream.kind == StreamKind.LIVEDATA_DATA
        ]
        fom_msgs = [
            m for m in downstream.messages if m.stream.kind == StreamKind.LIVEDATA_FOM
        ]
        assert len(data_msgs) == 2
        assert len(fom_msgs) == 1

        mirror = fom_msgs[0]
        assert isinstance(mirror.value, AliasedResult)
        assert mirror.value.alias == 'fom-0'
        assert sc.identical(mirror.value.data, _scalar(1.0))
        # Mirror keeps the ResultKey JSON in stream.name.
        rk = ResultKey.model_validate_json(mirror.stream.name)
        assert rk.job_id == job_id
        assert rk.output_name == 'a'

    def test_mirror_independent_per_alias(self, downstream: FakeMessageSink) -> None:
        registry = StreamAliasRegistry()
        job_id = _make_job_id()
        registry.bind('fom-0', job_id, 'a')
        registry.bind('fom-1', job_id, 'b')
        adapter = UnrollingSinkAdapter(downstream, alias_registry=registry)
        adapter.publish_messages(
            [
                _make_result_message(
                    job_id=job_id,
                    outputs={'a': _scalar(1.0), 'b': _scalar(2.0)},
                )
            ]
        )
        fom_msgs = [
            m for m in downstream.messages if m.stream.kind == StreamKind.LIVEDATA_FOM
        ]
        aliases = {m.value.alias for m in fom_msgs}
        assert aliases == {'fom-0', 'fom-1'}

    def test_other_jobs_unaffected(self, downstream: FakeMessageSink) -> None:
        registry = StreamAliasRegistry()
        bound_job = _make_job_id()
        other_job = _make_job_id()
        registry.bind('fom-0', bound_job, 'a')
        adapter = UnrollingSinkAdapter(downstream, alias_registry=registry)
        adapter.publish_messages(
            [_make_result_message(job_id=other_job, outputs={'a': _scalar(99.0)})]
        )
        kinds = {m.stream.kind for m in downstream.messages}
        assert kinds == {StreamKind.LIVEDATA_DATA}
