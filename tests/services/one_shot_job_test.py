# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""End-to-end tests for the one-shot job mechanism.

A "one-shot" job is one whose primary source resolves to zero physical streams.
JobManager finalizes such jobs synchronously inside ``schedule_job`` and the
result reaches the sink via the config-handler return path on the same loop
tick as the ack — with no dependency on data traffic to drive activation or
emission.

The mechanism lives in the orchestrator core, not in any particular service.
We exercise it via the data_reduction service builder simply because it is
the most readily available wiring; the same path applies to every service
using ``OrchestratingProcessor``.
"""

from __future__ import annotations

from typing import Any

import pytest
import scipp as sc

from ess.livedata.config import instrument_registry, workflow_spec
from ess.livedata.config.models import ConfigKey
from ess.livedata.config.workflow_spec import DefaultOutputs, WorkflowId
from ess.livedata.core.job import JobState
from ess.livedata.core.message import StreamKind
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.services.data_reduction import make_reduction_service_builder
from tests.helpers.livedata_app import LivedataApp

ONE_SHOT_WORKFLOW_NAME = 'one_shot_test_workflow'

# A workflow source name is treated as one-shot iff
# ``resolve_stream_names({source}, instrument, stream_mapping)`` yields the
# empty set. That happens precisely when the name is absent from all three of:
#   - ``StreamMapping.all_stream_names`` (no physical stream subscribed),
#   - ``instrument.detector_names`` (no logical→physical detector expansion),
#   - ``instrument.monitors`` (no logical→physical monitor expansion).
# We pick a name that satisfies all three to drive the chopperless code path
# for these tests. No Kafka adapter, no message merging, no routing wiring
# beyond the standard service builder is involved — one-shot detection is
# purely a property of the routing tables.
#
# A real consumer (the planned chopper workflow) shares this exact mechanism
# for its chopperless case: the workflow declares a single logical source
# name and resolves to the empty set on instruments without chopper PVs. The
# chopper-equipped case lives in the routing layer (logical→physical
# expansion plus a merging adapter that rewrites incoming chopper messages
# to the same logical name); that wiring is orthogonal to the mechanism
# under test here.
ONE_SHOT_SOURCE = 'no_physical_stream'

ONE_SHOT_OUTPUT_VALUE = 42.0


class _OneShotWorkflow:
    """Test workflow that emits a fixed result on finalize.

    Matches the ``Workflow`` protocol; ``accumulate``/``clear`` are unreachable
    for a one-shot job (no data is ever pushed) but must exist for protocol
    compliance.
    """

    def accumulate(
        self, data: dict[str, Any], *, start_time: Timestamp, end_time: Timestamp
    ) -> None:
        pass

    def finalize(self) -> dict[str, Any]:
        return {'result': sc.DataArray(sc.scalar(ONE_SHOT_OUTPUT_VALUE))}

    def clear(self) -> None:
        pass


def _ensure_one_shot_spec_registered() -> WorkflowId:
    """Register the one-shot workflow spec on the dummy instrument.

    Idempotent: spec registration is a global side effect on the shared
    registry; subsequent calls within the same process are no-ops. Returns the
    workflow id either way so callers can address the workflow uniformly.
    """
    instrument = instrument_registry['dummy']
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='data_reduction',
        name=ONE_SHOT_WORKFLOW_NAME,
        version=1,
    )
    if workflow_id in instrument.workflow_factory:
        return workflow_id
    handle = instrument.register_spec(
        namespace='data_reduction',
        name=ONE_SHOT_WORKFLOW_NAME,
        version=1,
        title='One-shot test',
        description='Test workflow with no physical streams.',
        source_names=[ONE_SHOT_SOURCE],
        outputs=DefaultOutputs,
    )

    @handle.attach_factory()
    def _one_shot_factory() -> _OneShotWorkflow:
        return _OneShotWorkflow()

    return workflow_id


@pytest.fixture
def app() -> LivedataApp:
    # Building the service builder triggers ``load_factories`` on the dummy
    # instrument, which is what populates ``instrument_registry['dummy']``
    # via the dummy specs module import. Register the test workflow afterwards
    # so the dummy instrument is ready to host it.
    builder = make_reduction_service_builder(instrument='dummy')
    workflow_id = _ensure_one_shot_spec_registered()
    livedata_app = LivedataApp.from_service_builder(builder)
    livedata_app.workflow_id = workflow_id  # type: ignore[attr-defined]
    return livedata_app


def _publish_start_command(
    app: LivedataApp, workflow_id: WorkflowId, message_id: str | None = None
) -> None:
    config_key = ConfigKey(
        source_name=ONE_SHOT_SOURCE,
        service_name='data_reduction',
        key='workflow_config',
    )
    config = workflow_spec.WorkflowConfig(identifier=workflow_id, message_id=message_id)
    app.publish_config_message(key=config_key, value=config.model_dump(mode='json'))


def test_start_command_emits_result_with_no_data_traffic(app: LivedataApp) -> None:
    """Single config message → single step → ack + result, no data ever flows."""
    _publish_start_command(app, app.workflow_id, message_id='msg-1')

    app.service.step()

    streams_by_kind = {m.stream.kind for m in app.sink.messages}
    assert StreamKind.LIVEDATA_RESPONSES in streams_by_kind
    assert StreamKind.LIVEDATA_DATA in streams_by_kind

    data_messages = [
        m for m in app.sink.messages if m.stream.kind == StreamKind.LIVEDATA_DATA
    ]
    assert len(data_messages) == 1
    assert data_messages[0].value.values == ONE_SHOT_OUTPUT_VALUE


def test_result_emitted_on_first_step_after_command(app: LivedataApp) -> None:
    """Result is bound to the command-bearing tick, not deferred to a later one."""
    # First step with no commands and no data: no result must be emitted.
    app.service.step()
    assert all(m.stream.kind != StreamKind.LIVEDATA_DATA for m in app.sink.messages)

    _publish_start_command(app, app.workflow_id)
    app.service.step()

    data_messages = [
        m for m in app.sink.messages if m.stream.kind == StreamKind.LIVEDATA_DATA
    ]
    assert len(data_messages) == 1


def test_no_result_emitted_when_no_command_sent(app: LivedataApp) -> None:
    """Sanity check: idle service produces no data messages on its own."""
    for _ in range(3):
        app.service.step()

    assert all(m.stream.kind != StreamKind.LIVEDATA_DATA for m in app.sink.messages)


def test_ack_and_result_carry_matching_workflow_identity(app: LivedataApp) -> None:
    """The ack and the data result both refer to the same scheduled job."""
    _publish_start_command(app, app.workflow_id, message_id='msg-corr')

    app.service.step()

    ack_messages = [
        m for m in app.sink.messages if m.stream.kind == StreamKind.LIVEDATA_RESPONSES
    ]
    data_messages = [
        m for m in app.sink.messages if m.stream.kind == StreamKind.LIVEDATA_DATA
    ]
    assert len(ack_messages) == 1
    assert ack_messages[0].value.message_id == 'msg-corr'
    assert len(data_messages) == 1
    # Result stream name encodes workflow_id; verify it matches the started one.
    assert app.workflow_id.name in data_messages[0].stream.name


def test_one_shot_job_emits_stopped_status(app: LivedataApp) -> None:
    """Without a stopped status, the dashboard would keep displaying the job
    as running because the one-shot job never enters the active/scheduled
    state that ``_report_status`` walks. Verify the status reaches the sink.
    """
    _publish_start_command(app, app.workflow_id)

    app.service.step()

    # Status messages land on a dedicated bucket of FakeMessageSink. Filter
    # to the JobStatus-shaped messages for the workflow under test (the bucket
    # also receives ServiceStatus heartbeats).
    one_shot_statuses = [
        m
        for m in app.sink.status_messages
        if hasattr(m.value, 'workflow_id') and m.value.workflow_id == app.workflow_id
    ]
    assert len(one_shot_statuses) == 1
    status = one_shot_statuses[0].value
    assert status.state == JobState.stopped
    assert status.error_message is None
