# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Pytest fixtures for integration tests."""

import logging
from collections.abc import Generator
from dataclasses import dataclass

import pytest
from confluent_kafka import KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.config.streams import get_stream_mapping
from tests.integration.backend import DashboardBackend
from tests.integration.service_process import ServiceGroup, ServiceProcess

logger = logging.getLogger(__name__)


def _ensure_topics_exist(instrument: str) -> None:
    """Create the topics consumers validate at startup.

    Consumers fail fast on missing topics (``validate_topics_exist``), and
    broker-side auto-creation only triggers on produce, so a pristine broker
    (CI, fresh local setup) needs the topics created up front. Idempotent.
    """
    mapping = get_stream_mapping(instrument=instrument, dev=True)
    topics = (
        mapping.detector_topics
        | mapping.area_detector_topics
        | mapping.monitor_topics
        | mapping.log_topics
        | {
            mapping.topics.livedata_commands,
            mapping.topics.livedata_data,
            mapping.topics.livedata_responses,
            mapping.topics.livedata_roi,
            mapping.topics.livedata_status,
            mapping.topics.filewriter,
        }
    )
    admin = AdminClient(load_config(namespace=config_names.kafka, env='dev'))
    futures = admin.create_topics(
        [NewTopic(topic, num_partitions=1) for topic in sorted(topics)]
    )
    for topic, future in futures.items():
        try:
            future.result()
        except KafkaException as e:
            if e.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise
            logger.debug("Topic already exists: %s", topic)


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


def _create_service_group(
    request, services_config: dict[str, tuple[str, dict]]
) -> Generator[ServiceGroup, None, None]:
    """Common service group creation and lifecycle management.

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
    _ensure_topics_exist(instrument)

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
        services.start_all(startup_delay=10.0)
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

    _ensure_topics_exist(instrument)

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
    yield from _create_service_group(
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
                        'Kafka consumer ready and polling',
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
    yield from _create_service_group(
        request,
        {
            'fake_detectors': ('ess.livedata.services.fake_detectors', {}),
            'detector_data': (
                'ess.livedata.services.detector_data',
                {
                    'dev': True,
                    'readiness_messages': [
                        'Service started',
                        'Kafka consumer ready and polling',
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
    yield from _create_service_group(
        request,
        {
            'fake_detectors': ('ess.livedata.services.fake_detectors', {}),
            'detector_data': (
                'ess.livedata.services.detector_data',
                {
                    'dev': True,
                    'readiness_messages': [
                        'Service started',
                        'Kafka consumer ready and polling',
                    ],
                },
            ),
            'data_reduction': (
                'ess.livedata.services.data_reduction',
                {
                    'dev': True,
                    'readiness_messages': [
                        'Service started',
                        'Kafka consumer ready and polling',
                    ],
                },
            ),
        },
    )
