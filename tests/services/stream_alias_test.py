# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""End-to-end tests for stream-alias binding via :class:`LivedataApp`.

Drives a real backend service (no Kafka) through a configure → bind → push →
unbind cycle and asserts that mirror messages appear on
:class:`StreamKind.LIVEDATA_FOM` only while the binding is active.
"""

from __future__ import annotations

import uuid

import pytest

from ess.livedata.config import instrument_registry
from ess.livedata.config.acknowledgement import (
    AcknowledgementResponse,
    CommandAcknowledgement,
)
from ess.livedata.config.models import ConfigKey
from ess.livedata.config.workflow_spec import (
    JobId,
    WorkflowConfig,
    WorkflowId,
)
from ess.livedata.core.message import StreamKind
from ess.livedata.core.stream_alias import (
    AliasedResult,
    BindStreamAlias,
    UnbindStreamAlias,
)
from ess.livedata.services.detector_data import make_detector_service_builder
from tests.helpers.livedata_app import LivedataApp

INSTRUMENT = 'dummy'
SOURCE_NAME = 'panel_0'
SERVICE_NAME = 'detector_data'
# A workflow output known to be a scalar — see detector_data_test.py for the
# full list.
ALIAS_TARGET_OUTPUT = 'counts_total'


def _get_workflow_id() -> WorkflowId:
    instrument_config = instrument_registry[INSTRUMENT]
    for wid, spec in instrument_config.workflow_factory.items():
        if spec.namespace == SERVICE_NAME:
            return wid
    raise ValueError("No data_reduction workflow registered for dummy.")


@pytest.fixture
def app() -> LivedataApp:
    builder = make_detector_service_builder(instrument=INSTRUMENT)
    return LivedataApp.from_service_builder(builder)


@pytest.fixture
def hosted_job(app: LivedataApp) -> JobId:
    """Configure a workflow with a known job_number and step the service."""
    job_number = uuid.uuid4()
    workflow_id = _get_workflow_id()
    workflow_config_key = ConfigKey(
        source_name=SOURCE_NAME,
        service_name=SERVICE_NAME,
        key='workflow_config',
    )
    config = WorkflowConfig(identifier=workflow_id, job_number=job_number)
    app.publish_config_message(
        key=workflow_config_key, value=config.model_dump(mode='json')
    )
    app.step()
    app.sink.messages.clear()
    app.sink.published_messages.clear()
    return JobId(source_name=SOURCE_NAME, job_number=job_number)


def _bind_message(
    *, alias: str, job_id: JobId, output_name: str, message_id: str | None = None
) -> dict:
    return BindStreamAlias(
        alias=alias,
        job_id=job_id,
        output_name=output_name,
        message_id=message_id,
    ).model_dump(mode='json')


def _unbind_message(*, alias: str, message_id: str | None = None) -> dict:
    return UnbindStreamAlias(alias=alias, message_id=message_id).model_dump(mode='json')


def _ack_messages(app: LivedataApp) -> list[CommandAcknowledgement]:
    """Return CommandAcknowledgement values seen by the sink."""
    return [
        msg.value
        for batch in app.sink.published_messages
        for msg in batch
        if isinstance(msg.value, CommandAcknowledgement)
    ]


def test_bind_emits_fom_mirror(app: LivedataApp, hosted_job: JobId) -> None:
    bind_key = ConfigKey(key=BindStreamAlias.key)
    app.publish_config_message(
        key=bind_key,
        value=_bind_message(
            alias='fom-0',
            job_id=hosted_job,
            output_name=ALIAS_TARGET_OUTPUT,
            message_id='bind-1',
        ),
    )
    # Push events so the workflow produces a result.
    app.publish_events(size=2000, time=2)
    app.step()

    fom_msgs = [
        m for m in app.sink.messages if m.stream.kind == StreamKind.LIVEDATA_FOM
    ]
    data_msgs = [
        m for m in app.sink.messages if m.stream.kind == StreamKind.LIVEDATA_DATA
    ]
    assert len(fom_msgs) == 1, (
        f"Expected one mirror, got {len(fom_msgs)}: "
        f"{[m.stream for m in app.sink.messages]}"
    )
    assert isinstance(fom_msgs[0].value, AliasedResult)
    assert fom_msgs[0].value.alias == 'fom-0'
    # LIVEDATA_DATA stream is unaffected.
    assert any(m for m in data_msgs)

    acks = _ack_messages(app)
    bind_ack = next(a for a in acks if a.message_id == 'bind-1')
    assert bind_ack.response == AcknowledgementResponse.ACK


def test_unbind_stops_mirror(app: LivedataApp, hosted_job: JobId) -> None:
    bind_key = ConfigKey(key=BindStreamAlias.key)
    unbind_key = ConfigKey(key=UnbindStreamAlias.key)
    app.publish_config_message(
        key=bind_key,
        value=_bind_message(
            alias='fom-0',
            job_id=hosted_job,
            output_name=ALIAS_TARGET_OUTPUT,
        ),
    )
    app.publish_events(size=2000, time=2)
    app.step()
    assert any(m for m in app.sink.messages if m.stream.kind == StreamKind.LIVEDATA_FOM)

    app.sink.messages.clear()
    app.sink.published_messages.clear()

    app.publish_config_message(
        key=unbind_key,
        value=_unbind_message(alias='fom-0', message_id='unbind-1'),
    )
    app.publish_events(size=1000, time=4)
    app.step()

    fom_after = [
        m for m in app.sink.messages if m.stream.kind == StreamKind.LIVEDATA_FOM
    ]
    data_after = [
        m for m in app.sink.messages if m.stream.kind == StreamKind.LIVEDATA_DATA
    ]
    assert fom_after == []
    assert data_after  # Original data stream continues.

    acks = _ack_messages(app)
    unbind_ack = next(a for a in acks if a.message_id == 'unbind-1')
    assert unbind_ack.response == AcknowledgementResponse.ACK


def test_bind_unknown_job_is_silent(app: LivedataApp) -> None:
    unknown_job = JobId(source_name=SOURCE_NAME, job_number=uuid.uuid4())
    bind_key = ConfigKey(key=BindStreamAlias.key)
    app.publish_config_message(
        key=bind_key,
        value=_bind_message(
            alias='fom-0',
            job_id=unknown_job,
            output_name=ALIAS_TARGET_OUTPUT,
            message_id='bind-unknown',
        ),
    )
    app.step()

    acks = _ack_messages(app)
    assert all(a.message_id != 'bind-unknown' for a in acks)
