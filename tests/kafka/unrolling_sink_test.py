# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unrolling of job results into one message per output, and per subject."""

import uuid

import scipp as sc

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.core.message import Message, MessageSink, StreamId, StreamKind
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.kafka.sink import UnrollingSinkAdapter


class CollectingSink(MessageSink[sc.DataArray]):
    def __init__(self) -> None:
        self.messages: list[Message[sc.DataArray]] = []

    def publish_messages(self, messages: list[Message[sc.DataArray]]) -> None:
        self.messages.extend(messages)


def _publish(value: sc.DataGroup) -> list[tuple[ResultKey, sc.DataArray]]:
    job_id = JobId(source_name='chopper_cascade', job_number=uuid.uuid4())
    key = ResultKey(
        workflow_id=WorkflowId(instrument='dream', name='wavelength_lut', version=1),
        job_id=job_id,
    )
    sink = CollectingSink()
    UnrollingSinkAdapter(sink).publish_messages(
        [
            Message(
                timestamp=Timestamp.from_ns(0),
                stream=StreamId(
                    kind=StreamKind.LIVEDATA_DATA, name=key.model_dump_json()
                ),
                value=value,
            )
        ]
    )
    return [
        (ResultKey.model_validate_json(msg.stream.name), msg.value)
        for msg in sink.messages
    ]


def _table(value: float) -> sc.DataArray:
    return sc.DataArray(sc.scalar(value))


def test_plain_output_is_attributed_to_the_jobs_own_source() -> None:
    ((key, value),) = _publish(sc.DataGroup({'bands': _table(1.0)}))

    assert key.subject is None
    assert key.data_key.source_name == 'chopper_cascade'
    assert key.data_key.output_name == 'bands'
    assert sc.identical(value, _table(1.0))


def test_subject_keyed_output_yields_one_message_per_subject() -> None:
    results = _publish(
        sc.DataGroup(
            {
                'lookup_table': sc.DataGroup(
                    {'mantle_detector': _table(1.0), 'monitor_cave': _table(2.0)}
                ),
                'bands': _table(3.0),
            }
        )
    )

    by_data_key = {
        (key.data_key.source_name, key.data_key.output_name): value
        for key, value in results
    }
    assert set(by_data_key) == {
        ('mantle_detector', 'lookup_table'),
        ('monitor_cave', 'lookup_table'),
        ('chopper_cascade', 'bands'),
    }
    assert sc.identical(by_data_key['mantle_detector', 'lookup_table'], _table(1.0))
    assert sc.identical(by_data_key['monitor_cave', 'lookup_table'], _table(2.0))


def test_subject_keyed_results_keep_the_jobs_identity() -> None:
    """The subject names the entity described, not a second job."""
    results = _publish(
        sc.DataGroup({'lookup_table': sc.DataGroup({'mantle_detector': _table(1.0)})})
    )

    ((key, _),) = results
    assert key.job_id.source_name == 'chopper_cascade'
    assert key.subject == 'mantle_detector'
