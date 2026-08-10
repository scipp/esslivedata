# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for the f144 log-data path into the timeseries service.

``tests/services/timeseries_test.py`` hands decoded log data straight to the
service loop. What only the wire can show is everything ahead of it: an f144
flatbuffer on the instrument's log topic, resolved through the stream mapping
(topic plus in-payload source name) to a configured source, given its unit
from the instrument's stream record, and published back as a result the
dashboard admits under the job it created.

The f144 payloads are produced by the test rather than by ``fake_logdata``:
that service hardcodes the source name ``detector_tank_angle_r0``, which is
not among the dummy instrument's configured log sources (``motion1``,
``motion2``), so its messages are dropped as unmapped and no timeseries job
would ever see them.
"""

import time
from collections.abc import Generator

import pytest
from confluent_kafka import Producer

from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.config.streams import stream_kind_to_topic
from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.core.message import StreamKind
from tests.helpers import hostile_wire
from tests.integration.conftest import IntegrationEnv, create_service_group
from tests.integration.helpers import NoParams, wait_for_backend_condition
from tests.integration.service_process import ServiceGroup

WORKFLOW_ID = WorkflowId(instrument='dummy', name='timeseries_data', version=1)
SOURCE_NAME = 'motion1'
LOG_TOPIC = stream_kind_to_topic('dummy', StreamKind.LOG)


@pytest.fixture
def timeseries_services(request) -> Generator[ServiceGroup, None, None]:
    """The timeseries service alone; the test is its own log-data producer."""
    yield from create_service_group(
        request,
        {
            'timeseries': (
                'ess.livedata.services.timeseries',
                {
                    'dev': True,
                    'readiness_messages': ['Service started', 'kafka_consumer_ready'],
                },
            ),
        },
    )


@pytest.mark.integration
@pytest.mark.services('timeseries')
def test_f144_log_data_reaches_dashboard_as_timeseries(
    integration_env: IntegrationEnv,
) -> None:
    """f144 payloads on the log topic become results for the timeseries job."""
    backend = integration_env.backend
    job_ids = backend.workflow_controller.start_workflow(
        workflow_id=WORKFLOW_ID,
        source_names=[SOURCE_NAME],
        config=NoParams(),
    )
    assert len(job_ids) == 1
    assert job_ids[0].source_name == SOURCE_NAME

    producer = Producer(load_config(namespace=config_names.kafka, env='dev'))

    def publish_reading_and_check_for_result() -> bool:
        """Publish one log reading, then report whether a result came back.

        Readings are fed on every poll rather than in one burst up front: log
        data published before the job is committed is preprocessed but has no
        job to reach, so a burst could be spent entirely on the race with the
        commit and leave nothing to observe.
        """
        producer.produce(
            LOG_TOPIC,
            value=hostile_wire.f144_log(SOURCE_NAME, timestamp_ns=time.time_ns()),
        )
        producer.flush(timeout=10.0)
        return any(
            key.workflow_id == WORKFLOW_ID and key.source_name == SOURCE_NAME
            for key in backend.data_service
        )

    wait_for_backend_condition(
        backend,
        publish_reading_and_check_for_result,
        timeout=60.0,
        poll_interval=1.0,
    )
