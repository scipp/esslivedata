# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for service liveness after malformed data on a real topic.

``tests/services/hostile_input_liveness_test.py`` establishes the same
invariant in-process, handing payloads straight to the service loop. What it
cannot cover is everything between broker and loop: a real consumer, schema
routing off a topic shared by many sources, and a service that takes its whole
process down if containment fails.

Liveness here is proven by ordering, not by timing. The fake producer is
restricted to ``monitor1``, which leaves ``monitor2`` -- a configured monitor
source of the dummy instrument -- with no producer other than this test. Poison
and probe messages are published to the same single-partition topic in that
order, so a workflow result for ``monitor2`` can only exist if the service
consumed past the poison. Observing continued output for ``monitor1`` instead
would prove nothing: it cannot tell "survived the poison" from "has not reached
the poison yet".

Only the malformed family of the corpus is injected. The well-formed-but-insane
timestamps are a batcher concern, covered in-process where the clock can be
controlled; over real Kafka they would only add wall-clock-dependent silence
windows to wait out.
"""

from __future__ import annotations

import time
from collections.abc import Generator

import pytest
from confluent_kafka import Producer

from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.config.streams import stream_kind_to_topic
from ess.livedata.config.workflow_spec import DataKey, WorkflowId
from ess.livedata.core.message import StreamKind
from ess.livedata.workflows.monitor_workflow_specs import MonitorDataParams
from tests.helpers import hostile_wire
from tests.integration.conftest import IntegrationEnv, create_service_group
from tests.integration.helpers import wait_for_backend_condition, wait_for_job_data
from tests.integration.service_process import ServiceGroup

WORKFLOW_ID = WorkflowId(instrument='dummy', name='monitor_histogram', version=1)
MONITOR_TOPIC = stream_kind_to_topic('dummy', StreamKind.MONITOR_EVENTS)

#: Source the fake producer keeps feeding, so the poison arrives mid-stream.
LIVE_SOURCE = 'monitor1'
#: Source only this test produces for, carrying the proof of progress.
PROBE_SOURCE = 'monitor2'


@pytest.fixture
def probe_monitor_services(request) -> Generator[ServiceGroup, None, None]:
    """Monitor services whose fake producer is restricted to ``monitor1``.

    ``monitor2`` remains a configured source of the workflow, but nothing
    produces for it, which reserves it as this test's private channel.
    """
    yield from create_service_group(
        request,
        {
            'fake_monitors': (
                'ess.livedata.services.fake_monitors',
                {'mode': 'ev44', 'num_monitors': 1},
            ),
            'monitor_data': (
                'ess.livedata.services.monitor_data',
                {
                    'dev': True,
                    'readiness_messages': ['Service started', 'kafka_consumer_ready'],
                },
            ),
        },
    )


def _probe_outputs(backend) -> list[DataKey]:
    """Data keys the dashboard holds for the probe source."""
    return [
        key
        for key in backend.data_service
        if key.workflow_id == WORKFLOW_ID and key.source_name == PROBE_SOURCE
    ]


def _publish_probe(producer: Producer) -> None:
    """Publish one well-formed event message for the probe source."""
    producer.produce(
        MONITOR_TOPIC,
        value=hostile_wire.ev44_events(PROBE_SOURCE, reference_time_ns=time.time_ns()),
    )
    producer.flush(timeout=10.0)


@pytest.mark.integration
@pytest.mark.services('probe_monitor')
def test_malformed_payloads_do_not_stall_service(
    integration_env: IntegrationEnv,
) -> None:
    """Malformed payloads on the input topic do not stop a running pipeline."""
    backend = integration_env.backend
    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=WORKFLOW_ID,
        source_names=[LIVE_SOURCE, PROBE_SOURCE],
        config=MonitorDataParams(),
    )
    jobs = {job_id.source_name: job_id for job_id in job_ids}
    wait_for_job_data(backend, WORKFLOW_ID, [jobs[LIVE_SOURCE]], timeout=30.0)
    # Precondition of the proof below: the probe source is silent until we
    # produce for it ourselves.
    assert not _probe_outputs(backend)

    producer = Producer(load_config(namespace=config_names.kafka, env='dev'))
    for payload in hostile_wire.malformed_corpus(LIVE_SOURCE).values():
        producer.produce(MONITOR_TOPIC, value=payload)
    # Flush before the first probe: only delivered messages have an offset, and
    # the proof rests on every probe sitting behind the poison in the partition.
    producer.flush(timeout=30.0)

    def probe_output_arrived() -> bool:
        """Keep the probe source fed until its first result comes back."""
        _publish_probe(producer)
        return bool(_probe_outputs(backend))

    wait_for_backend_condition(
        backend, probe_output_arrived, timeout=60.0, poll_interval=1.0
    )
    assert integration_env.services['monitor_data'].is_running()
