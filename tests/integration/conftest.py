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

    services_dict = {}
    for name, (module_path, extra_kwargs) in services_config.items():
        services_dict[name] = ServiceProcess(
            module_path,
            log_level=log_level,
            instrument=instrument,
            **extra_kwargs,
        )

    services = ServiceGroup(services_dict)
    logger.info("Starting services for instrument: %s", instrument)

    try:
        services.start_all(startup_delay=5.0)
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
    # Get instrument from marker or use default
    marker = request.node.get_closest_marker('instrument')
    instrument = marker.args[0] if marker else 'dummy'

    # Get log level from marker or use default
    log_level_marker = request.node.get_closest_marker('log_level')
    log_level_name = log_level_marker.args[0] if log_level_marker else 'INFO'

    logger.info("Creating dashboard backend for instrument: %s", instrument)
    backend = DashboardBackend(
        instrument=instrument, dev=True, log_level=log_level_name
    )

    try:
        backend.start()
        yield backend
    finally:
        backend.stop()


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
            'fake_monitors': ('ess.livedata.services.fake_monitors', {'mode': 'ev44'}),
            'monitor_data': ('ess.livedata.services.monitor_data', {'dev': True}),
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
            'detector_data': ('ess.livedata.services.detector_data', {'dev': True}),
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
            'detector_data': ('ess.livedata.services.detector_data', {'dev': True}),
            'data_reduction': ('ess.livedata.services.data_reduction', {'dev': True}),
        },
    )
