# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""A worker runs the valid jobs of a desired-state topic with malformed records.

A malformed record stays on the compacted topic until its key is overwritten,
so a worker reads it at every start. The monitor service here reads the topic
through the real :class:`KafkaDesiredStateReader` on a fake consumer, with the
records of ``hostile_wire.malformed_desired_records`` around one valid record.
"""

from __future__ import annotations

import json
import uuid

import pytest

from ess.livedata.config import instrument_registry
from ess.livedata.config.streams import stream_kind_to_topic
from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.desired_state import DesiredJob, DesiredWorkflow
from ess.livedata.core.message import StreamKind
from ess.livedata.core.message_batcher import NaiveMessageBatcher
from ess.livedata.fakes import FakeMessageSink
from ess.livedata.kafka.message_adapter import FakeKafkaMessage
from ess.livedata.kafka.sink import UnrollingSinkAdapter
from ess.livedata.services.monitor_data import make_monitor_service_builder
from tests.helpers import hostile_wire
from tests.helpers.desired_state_consumer import (
    FakeRecord,
    eof,
    make_reader,
    record,
)
from tests.helpers.livedata_app import FakeConsumer

INSTRUMENT = 'dummy'
SOURCE = 'monitor1'
SECOND_NS = 1_000_000_000


def _monitor_workflow_id() -> WorkflowId:
    for workflow_id, spec in instrument_registry[INSTRUMENT].workflow_factory.items():
        if spec.group.name == 'monitor_data':
            return workflow_id
    raise ValueError(f'No monitor_data workflow for {INSTRUMENT}')


class Worker:
    """Monitor service reading the desired state through the Kafka reader."""

    def __init__(self) -> None:
        builder = make_monitor_service_builder(instrument=INSTRUMENT)
        builder.message_batcher = NaiveMessageBatcher()
        reader, self.desired = make_reader(
            stream_kind_to_topic(
                instrument=INSTRUMENT, kind=StreamKind.LIVEDATA_DESIRED
            )
        )
        self._data = FakeConsumer()
        self.sink = FakeMessageSink()
        self.service = builder.from_consumer(
            consumer=self._data,
            sink=UnrollingSinkAdapter(self.sink),
            desired_state=reader,
            use_background_source=False,
        )
        self._monitor_topic = stream_kind_to_topic(
            instrument=INSTRUMENT, kind=StreamKind.MONITOR_EVENTS
        )
        self._time_ns = hostile_wire.REALISTIC_EPOCH_NS

    def step_with_events(self) -> None:
        """Publish one well-formed monitor event message and run one step."""
        self._time_ns += SECOND_NS
        self._data.add_message(
            FakeKafkaMessage(
                value=hostile_wire.ev44_events(SOURCE, reference_time_ns=self._time_ns),
                topic=self._monitor_topic,
                timestamp=self._time_ns // 1_000_000,
            )
        )
        self.service.step()

    def result_job_numbers(self) -> set[str]:
        return {
            json.loads(msg.stream.name)['job_id']['job_number']
            for msg in self.sink.messages
            if msg.stream.kind == StreamKind.LIVEDATA_DATA
        }


def _interleaved(valid: FakeRecord) -> list[FakeRecord]:
    """The malformed records with ``valid`` in the middle."""
    malformed = [
        FakeRecord(key, value)
        for key, value in hostile_wire.malformed_desired_records(INSTRUMENT).values()
    ]
    middle = len(malformed) // 2
    return [*malformed[:middle], valid, *malformed[middle:]]


@pytest.fixture
def worker() -> Worker:
    return Worker()


@pytest.fixture
def desired() -> DesiredWorkflow:
    workflow_id = _monitor_workflow_id()
    return DesiredWorkflow(
        workflow_id=workflow_id,
        job_number=uuid.uuid4(),
        jobs={SOURCE: DesiredJob(params={}, aux_source_names={})},
    )


def _assert_job_runs(worker: Worker, desired: DesiredWorkflow) -> None:
    for _ in range(3):
        worker.step_with_events()
    assert worker.result_job_numbers() == {str(desired.job_number)}


def test_worker_runs_valid_job_read_at_startup_among_malformed_records(
    worker: Worker, desired: DesiredWorkflow
) -> None:
    worker.desired.batches = [
        [*_interleaved(record(desired.workflow_id, desired)), eof()]
    ]
    _assert_job_runs(worker, desired)


def test_worker_runs_valid_job_read_after_startup_among_malformed_records(
    worker: Worker, desired: DesiredWorkflow
) -> None:
    worker.desired.batches = [[eof()]]
    worker.service.step()
    worker.desired.batches = [_interleaved(record(desired.workflow_id, desired))]
    _assert_job_runs(worker, desired)
