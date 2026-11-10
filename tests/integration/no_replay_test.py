# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for the no-replay contract of the commands topic.

``tests/kafka/consumer_test.py`` pins ``assign_all_partitions`` against a fake
consumer: the assignment carries each partition's current high watermark. Only
a broker can show the consequence that follows from it, because it is a fact
about durable records the service never reads -- a worker coming up finds the
commands topic's history behind its assigned offsets and acts on none of it,
however long those commands have been sitting there. Commands are imperative
(a replayed reset would wipe data), unlike the desired state, which a worker
reads from the beginning (see ``desired_state_test.py``).

The proof is ordering, not timing. Both resets go to the same single-partition
topic, the stale one first. The acknowledgement of the later reset establishes
that the worker consumed past the earlier one's offset, so the missing
acknowledgement of the stale reset cannot be read as "not reached yet" -- the
reading that would make a timing-based version of this test vacuous. The job
both resets select runs throughout, restored from the desired state, so a
replayed stale reset would be acknowledged.
"""

import uuid

import pytest
from confluent_kafka import Producer

from ess.livedata.config import config_names
from ess.livedata.config.acknowledgement import CommandAcknowledgement
from ess.livedata.config.config_loader import load_config
from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.job import JobState
from ess.livedata.core.job_manager import JobAction, JobCommand
from ess.livedata.kafka.desired_state import KafkaDesiredStateWriter
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.integration.helpers import (
    Status,
    TopicTail,
    commit_workflow,
    job_in_state,
    topic_high_watermark,
    wait_for_watermark_advance,
)
from tests.integration.service_process import ServiceGroup

WORKFLOW_ID = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)
COMMANDS_TOPIC = 'dummy_livedata_commands'
RESPONSES_TOPIC = 'dummy_livedata_responses'


def send_reset(message_id: str) -> None:
    producer = Producer(load_config(namespace=config_names.kafka, env='dev'))
    command = JobCommand(
        action=JobAction.reset, workflow_id=WORKFLOW_ID, message_id=message_id
    )
    producer.produce(COMMANDS_TOPIC, value=command.model_dump_json().encode())
    assert producer.flush(timeout=10.0) == 0


@pytest.mark.integration
def test_restarted_service_ignores_commands_issued_while_down(
    desired_state: KafkaDesiredStateWriter,
    monitor_services: ServiceGroup,
    heartbeats: TopicTail[Status],
) -> None:
    """A restarted worker acts on commands issued after it came back, only.

    ``fake_monitors`` keeps producing throughout, so the raw monitor topic
    accumulates a backlog across the downtime as well; the restarted worker
    skips past that for the same reason.
    """
    service = monitor_services['monitor_data']
    responses = TopicTail(RESPONSES_TOPIC, CommandAcknowledgement.model_validate_json)
    try:
        job_id = commit_workflow(
            desired_state, WORKFLOW_ID, 'monitor1', MonitorDataParams()
        )
        heartbeats.wait_for(job_in_state(job_id, JobState.active))

        service.stop()
        heartbeats.wait_for(job_in_state(job_id, JobState.stopped))

        stale_id = f'stale-{uuid.uuid4()}'
        commands_before = topic_high_watermark(COMMANDS_TOPIC)
        send_reset(stale_id)
        # The command is a durable record on the topic before the worker
        # returns, which is what makes its later absence a statement about
        # offsets rather than about a lost send.
        wait_for_watermark_advance(COMMANDS_TOPIC, since=commands_before)

        service.start()
        heartbeats.wait_for(job_in_state(job_id, JobState.active), timeout=60.0)

        fresh_id = f'fresh-{uuid.uuid4()}'
        send_reset(fresh_id)
        responses.wait_for(lambda ack: ack.message_id == fresh_id, timeout=30.0)

        assert all(ack.message_id != stale_id for ack in responses.received)
    finally:
        responses.close()
