# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Pytest fixtures for integration tests."""

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass

import pytest

from .backend import DashboardBackend
from .service_process import ServiceGroup, ServiceProcess

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


@pytest.fixture
def dashboard_backend(request) -> Generator[DashboardBackend, None, None]:
    """
    Pytest fixture providing a DashboardBackend instance.

    The backend is automatically started and stopped, and uses the 'dummy'
    instrument by default. Tests can override the instrument using a marker:

    @pytest.mark.instrument('bifrost')
    def test_something(dashboard_backend):
        ...

    Tests can also override log level using a marker:

    @pytest.mark.log_level('DEBUG')
    def test_something(dashboard_backend):
        ...

    Yields
    ------
    :
        DashboardBackend instance
    """
    # Get instrument from marker or use default
    marker = request.node.get_closest_marker('instrument')
    instrument = marker.args[0] if marker else 'dummy'

    # Get log level from marker or use default
    log_level_marker = request.node.get_closest_marker('log_level')
    log_level_name = log_level_marker.args[0] if log_level_marker else 'INFO'
    log_level = getattr(logging, log_level_name)

    logger.info("Creating dashboard backend for instrument: %s", instrument)
    backend = DashboardBackend(instrument=instrument, dev=True, log_level=log_level)

    try:
        backend.start()
        yield backend
    finally:
        backend.stop()


@pytest.fixture
def monitor_services(request) -> Generator[ServiceGroup, None, None]:
    """
    Pytest fixture providing monitor-related services.

    Starts fake_monitors and monitor_data services for testing.
    Uses the 'dummy' instrument by default.

    Tests can override the instrument using a marker:

    @pytest.mark.instrument('bifrost')
    def test_something(monitor_services):
        ...

    Tests can also override log level using a marker:

    @pytest.mark.log_level('DEBUG')
    def test_something(monitor_services):
        ...

    Yields
    ------
    :
        ServiceGroup containing fake_monitors and monitor_data
    """
    # Get instrument from marker or use default
    marker = request.node.get_closest_marker('instrument')
    instrument = marker.args[0] if marker else 'dummy'

    # Get log level from marker or use default
    log_level_marker = request.node.get_closest_marker('log_level')
    log_level = log_level_marker.args[0] if log_level_marker else 'INFO'

    services = ServiceGroup(
        {
            'fake_monitors': ServiceProcess(
                'ess.livedata.services.fake_monitors',
                log_level=log_level,
                instrument=instrument,
                mode='ev44',
            ),
            'monitor_data': ServiceProcess(
                'ess.livedata.services.monitor_data',
                log_level=log_level,
                instrument=instrument,
                dev=True,
            ),
        }
    )

    logger.info("Starting monitor services for instrument: %s", instrument)
    try:
        services.start_all(startup_delay=5.0)
        yield services
    finally:
        services.stop_all()


@pytest.fixture
def integration_env(dashboard_backend: DashboardBackend, request) -> IntegrationEnv:
    """
    Pytest fixture providing complete integration test environment.

    Combines dashboard backend and services into a single environment object.
    Tests must specify which services to use with a marker:

    @pytest.mark.services('monitor')  # Uses monitor_services fixture
    @pytest.mark.services('detector')  # Uses detector_services fixture
    @pytest.mark.services('reduction')  # Uses reduction_services fixture

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
    services_marker = request.node.get_closest_marker('services')
    if not services_marker:
        pytest.fail(
            "integration_env fixture requires "
            "@pytest.mark.services('monitor'|'detector'|'reduction')"
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

    # Wait for Kafka consumers to complete group coordination and be ready to poll.
    # Services log "Service started" when their threads start, but their Kafka
    # consumers need additional time to join consumer groups, complete partition
    # assignment, and begin polling. We wait for "Kafka consumer ready" log message.
    logger.info("Waiting for services to be ready for message consumption...")
    start_time = time.time()
    timeout = 5.0
    while time.time() - start_time < timeout:
        combined_output = ''.join(
            service.get_stdout() + service.get_stderr()
            for service in services.services.values()
        )
        if 'Kafka consumer ready and polling' in combined_output:
            logger.info("Services ready after %.3fs", time.time() - start_time)
            break
        time.sleep(0.05)
    else:
        logger.warning(
            "Did not see 'Kafka consumer ready' within %.1fs, proceeding anyway",
            timeout,
        )

    return IntegrationEnv(
        backend=dashboard_backend, services=services, instrument=instrument
    )


@pytest.fixture
def detector_services(request) -> Generator[ServiceGroup, None, None]:
    """
    Pytest fixture providing detector-related services.

    Starts fake_detectors and detector_data services for testing.
    Uses the 'dummy' instrument by default.

    Yields
    ------
    :
        ServiceGroup containing fake_detectors and detector_data
    """
    marker = request.node.get_closest_marker('instrument')
    instrument = marker.args[0] if marker else 'dummy'

    # Get log level from marker or use default
    log_level_marker = request.node.get_closest_marker('log_level')
    log_level = log_level_marker.args[0] if log_level_marker else 'INFO'

    services = ServiceGroup(
        {
            'fake_detectors': ServiceProcess(
                'ess.livedata.services.fake_detectors',
                log_level=log_level,
                instrument=instrument,
            ),
            'detector_data': ServiceProcess(
                'ess.livedata.services.detector_data',
                log_level=log_level,
                instrument=instrument,
                dev=True,
            ),
        }
    )

    logger.info("Starting detector services for instrument: %s", instrument)
    try:
        services.start_all(startup_delay=5.0)
        yield services
    finally:
        services.stop_all()


@pytest.fixture
def reduction_services(request) -> Generator[ServiceGroup, None, None]:
    """
    Pytest fixture providing data reduction services.

    Starts fake_monitors, monitor_data, and data_reduction services for testing.
    Uses the 'dummy' instrument by default.

    Yields
    ------
    :
        ServiceGroup containing all reduction pipeline services
    """
    marker = request.node.get_closest_marker('instrument')
    instrument = marker.args[0] if marker else 'dummy'

    # Get log level from marker or use default
    log_level_marker = request.node.get_closest_marker('log_level')
    log_level = log_level_marker.args[0] if log_level_marker else 'INFO'

    services = ServiceGroup(
        {
            'fake_monitors': ServiceProcess(
                'ess.livedata.services.fake_monitors',
                log_level=log_level,
                instrument=instrument,
                mode='ev44',
            ),
            'monitor_data': ServiceProcess(
                'ess.livedata.services.monitor_data',
                log_level=log_level,
                instrument=instrument,
                dev=True,
            ),
            'data_reduction': ServiceProcess(
                'ess.livedata.services.data_reduction',
                log_level=log_level,
                instrument=instrument,
                dev=True,
            ),
        }
    )

    logger.info("Starting reduction services for instrument: %s", instrument)
    try:
        services.start_all(startup_delay=5.0)
        yield services
    finally:
        services.stop_all()
