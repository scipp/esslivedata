# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Pytest fixtures for integration tests."""

import logging
from collections.abc import Generator
from dataclasses import dataclass

import pytest

from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.config.streams import stream_kind_to_topic
from ess.livedata.core.desired_state import fold_updates
from ess.livedata.core.job import JobStatus, ServiceStatus
from ess.livedata.core.message import StreamKind
from ess.livedata.kafka.desired_state import (
    KafkaDesiredStateReader,
    KafkaDesiredStateWriter,
)
from ess.livedata.kafka.x5f2_compat import x5f2_to_status
from ess.livedata.scripts.dev import ensure_topics_exist
from tests.integration.backend import DashboardBackend
from tests.integration.helpers import TopicTail
from tests.integration.service_process import ServiceGroup, ServiceProcess

logger = logging.getLogger(__name__)


@dataclass
class IntegrationEnv:
    """
    Integration test environment containing all components.

    Attributes
    ----------
    backend:
        Dashboard backend instance
    services:
        Service group containing all running services
    instrument:
        Instrument name being tested
    """

    backend: DashboardBackend
    services: ServiceGroup
    instrument: str


def _get_instrument_and_log_level(request):
    """Extract instrument and log_level from pytest markers."""
    instrument_marker = request.node.get_closest_marker('instrument')
    instrument = instrument_marker.args[0] if instrument_marker else 'dummy'

    log_level_marker = request.node.get_closest_marker('log_level')
    log_level_name = log_level_marker.args[0] if log_level_marker else 'INFO'

    return instrument, log_level_name


def create_service_group(
    request, services_config: dict[str, tuple[str, dict]]
) -> Generator[ServiceGroup, None, None]:
    """Common service group creation and lifecycle management.

    Public so that a test module can define its own ``<name>_services`` fixture
    for a service combination the shared fixtures do not offer.

    Parameters
    ----------
    request:
        Pytest request object for accessing markers
    services_config:
        Mapping of service names to (module_path, extra_kwargs) tuples

    Yields
    ------
    :
        ServiceGroup instance
    """
    instrument, log_level = _get_instrument_and_log_level(request)
    ensure_topics_exist(instrument)

    services_dict = {}
    for name, (module_path, extra_kwargs) in services_config.items():
        # Services with Kafka consumers should wait for both process start
        # and Kafka consumer readiness to ensure functional availability
        readiness_messages = extra_kwargs.pop('readiness_messages', None)
        services_dict[name] = ServiceProcess(
            module_path,
            log_level=log_level,
            instrument=instrument,
            readiness_messages=readiness_messages,
            **extra_kwargs,
        )

    services = ServiceGroup(services_dict)
    logger.info("Starting services for instrument: %s", instrument)

    try:
        services.start_all()
        yield services
    finally:
        services.stop_all()


@pytest.fixture
def dashboard_backend(request) -> Generator[DashboardBackend, None, None]:
    """
    Pytest fixture providing a DashboardBackend instance.

    The backend is automatically started and stopped, using the 'dummy'
    instrument by default. Tests can override the instrument and log level:

    @pytest.mark.instrument('bifrost')
    @pytest.mark.log_level('DEBUG')
    def test_something(dashboard_backend):
        ...

    Yields
    ------
    :
        DashboardBackend instance
    """
    instrument, log_level_name = _get_instrument_and_log_level(request)

    ensure_topics_exist(instrument)

    logger.info("Creating dashboard backend for instrument: %s", instrument)
    with DashboardBackend(
        instrument=instrument, dev=True, log_level=log_level_name
    ) as backend:
        yield backend


@pytest.fixture
def integration_env(dashboard_backend: DashboardBackend, request) -> IntegrationEnv:
    """
    Pytest fixture providing complete integration test environment.

    Combines dashboard backend and services into a single environment object.
    Tests must specify which services to use with a marker:

    @pytest.mark.services('monitor')  # Uses monitor_services fixture
    @pytest.mark.services('detector')  # Uses detector_services fixture
    @pytest.mark.services('reduction')  # Uses reduction_services fixture

    Design note: Each test uses exactly ONE service group to focus on a single
    service pipeline. Multiple service groups per test are not supported - tests
    should be scoped to one workflow type (monitor, detector, or reduction).

    Parameters
    ----------
    dashboard_backend:
        Dashboard backend fixture
    request:
        Pytest request object for accessing markers

    Returns
    -------
    :
        IntegrationEnv containing backend, services, and instrument name
    """
    # Get services type from marker (required)
    # Design: Single service group per test to focus on one service pipeline
    services_marker = request.node.get_closest_marker('services')
    if not services_marker:
        pytest.fail(
            "integration_env fixture requires "
            "@pytest.mark.services('monitor'|'detector'|'reduction')"
        )

    # Validate single service group (not multiple)
    if len(services_marker.args) != 1:
        pytest.fail(
            f"@pytest.mark.services() expects exactly one argument, "
            f"got {len(services_marker.args)}. Each test should focus on a "
            f"single service group: 'monitor', 'detector', or 'reduction'."
        )

    services_type = services_marker.args[0]

    # Map service type to fixture name
    fixture_name = f'{services_type}_services'

    # Dynamically request the appropriate services fixture
    try:
        services = request.getfixturevalue(fixture_name)
    except Exception as e:
        pytest.fail(
            f"Failed to load services fixture '{fixture_name}' "
            f"for services type '{services_type}': {e}"
        )

    # Get instrument
    marker = request.node.get_closest_marker('instrument')
    instrument = marker.args[0] if marker else 'dummy'

    # Service readiness (including Kafka consumer readiness) is now handled
    # by ServiceProcess via configurable readiness_messages. No need to wait
    # here - services are already confirmed ready when start_all() completes.

    return IntegrationEnv(
        backend=dashboard_backend, services=services, instrument=instrument
    )


@pytest.fixture
def monitor_services(request) -> Generator[ServiceGroup, None, None]:
    """
    Pytest fixture providing monitor-related services.

    Starts fake_monitors and monitor_data services for testing.
    Typically used through the integration_env fixture.

    Yields
    ------
    :
        ServiceGroup containing fake_monitors and monitor_data
    """
    yield from create_service_group(
        request,
        {
            'fake_monitors': (
                'ess.livedata.services.fake_monitors',
                {'mode': 'ev44'},
            ),
            'monitor_data': (
                'ess.livedata.services.monitor_data',
                {
                    'dev': True,
                    'readiness_messages': [
                        'Service started',
                        'kafka_consumer_ready',
                    ],
                },
            ),
        },
    )


@pytest.fixture
def detector_services(request) -> Generator[ServiceGroup, None, None]:
    """
    Pytest fixture providing detector-related services.

    Starts fake_detectors and detector_data services for testing.
    Typically used through the integration_env fixture.

    Yields
    ------
    :
        ServiceGroup containing fake_detectors and detector_data
    """
    yield from create_service_group(
        request,
        {
            'fake_detectors': ('ess.livedata.services.fake_detectors', {}),
            'detector_data': (
                'ess.livedata.services.detector_data',
                {
                    'dev': True,
                    'readiness_messages': [
                        'Service started',
                        'kafka_consumer_ready',
                    ],
                },
            ),
        },
    )


@pytest.fixture
def reduction_services(request) -> Generator[ServiceGroup, None, None]:
    """
    Pytest fixture providing data reduction services.

    Starts fake_detectors, detector_data, and data_reduction services for testing.
    Typically used through the integration_env fixture.

    Yields
    ------
    :
        ServiceGroup containing all reduction pipeline services
    """
    yield from create_service_group(
        request,
        {
            'fake_detectors': ('ess.livedata.services.fake_detectors', {}),
            'detector_data': (
                'ess.livedata.services.detector_data',
                {
                    'dev': True,
                    'readiness_messages': [
                        'Service started',
                        'kafka_consumer_ready',
                    ],
                },
            ),
            'data_reduction': (
                'ess.livedata.services.data_reduction',
                {
                    'dev': True,
                    'readiness_messages': [
                        'Service started',
                        'kafka_consumer_ready',
                    ],
                },
            ),
        },
    )


def _clear_desired_state(writer: KafkaDesiredStateWriter, topic: str) -> None:
    """Tombstone every workflow in the desired table."""
    config = load_config(namespace=config_names.kafka, env='dev')
    with KafkaDesiredStateReader(kafka_config=config, topic=topic) as reader:
        for workflow_id in fold_updates({}, reader.bootstrap(timeout_s=30.0)):
            writer.write(workflow_id, None)


@pytest.fixture
def desired_state(request) -> Generator[KafkaDesiredStateWriter, None, None]:
    """
    Pytest fixture providing a writer of the instrument's desired-state topic.

    The topic is compacted and outlives the test, and every worker that starts
    reads it from the beginning. The table is therefore emptied before and after
    the test. Request this fixture before the services fixture, so that services
    start with an empty table.
    """
    instrument, _ = _get_instrument_and_log_level(request)
    ensure_topics_exist(instrument)
    topic = stream_kind_to_topic(
        instrument=instrument, kind=StreamKind.LIVEDATA_DESIRED
    )
    writer = KafkaDesiredStateWriter(
        kafka_config=load_config(namespace=config_names.kafka, env='dev'),
        topic=topic,
    )
    _clear_desired_state(writer, topic)
    yield writer
    _clear_desired_state(writer, topic)


@pytest.fixture
def heartbeats(
    request,
) -> Generator[TopicTail[JobStatus | ServiceStatus], None, None]:
    """Pytest fixture providing the job and service statuses published from now on."""
    instrument, _ = _get_instrument_and_log_level(request)
    ensure_topics_exist(instrument)
    tail = TopicTail(
        stream_kind_to_topic(instrument=instrument, kind=StreamKind.LIVEDATA_STATUS),
        x5f2_to_status,
    )
    yield tail
    tail.close()
