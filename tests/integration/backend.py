# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Reusable dashboard backend for integration testing without GUI components."""

import logging
from contextlib import ExitStack
from types import TracebackType

from ess.livedata.dashboard.dashboard_services import DashboardServices
from ess.livedata.dashboard.kafka_transport import DashboardKafkaTransport


class DashboardBackend:
    """
    Reusable dashboard backend for integration tests (no GUI).

    This class provides access to all dashboard services without the Panel/GUI
    components. It's designed to be used in integration tests as a context manager.

    Parameters
    ----------
    instrument:
        Instrument name (e.g., 'dummy', 'dream', 'bifrost')
    dev:
        Use dev mode with simplified topic structure
    log_level:
        Logging level
    """

    def __init__(
        self,
        *,
        instrument: str = 'dummy',
        dev: bool = True,
        log_level: int | str = logging.INFO,
    ):
        self._instrument = instrument
        self._dev = dev
        self._logger = logging.getLogger(f'{instrument}_dashboard_backend')
        self._logger.setLevel(log_level)

        self._exit_stack = ExitStack()
        self._exit_stack.__enter__()

        # Create Kafka transport for integration tests
        transport = DashboardKafkaTransport(
            instrument=instrument, dev=dev, logger=self._logger
        )

        # Setup all dashboard services (no GUI components)
        self._services = DashboardServices(
            instrument=instrument,
            dev=dev,
            exit_stack=self._exit_stack,
            logger=self._logger,
            pipe_factory=lambda data: None,  # No-op for tests
            transport=transport,
        )

        self._logger.info("DashboardBackend initialized for %s", instrument)

    # Expose services as properties for easy access in tests
    @property
    def command_service(self):
        return self._services.command_service

    @property
    def workflow_config_service(self):
        return self._services.workflow_config_service

    @property
    def data_service(self):
        return self._services.data_service

    @property
    def stream_manager(self):
        return self._services.stream_manager

    @property
    def job_service(self):
        return self._services.job_service

    @property
    def job_controller(self):
        return self._services.job_controller

    @property
    def plotting_controller(self):
        return self._services.plotting_controller

    @property
    def orchestrator(self):
        return self._services.orchestrator

    @property
    def correlation_controller(self):
        return self._services.correlation_controller

    @property
    def workflow_controller(self):
        return self._services.workflow_controller

    def start(self) -> None:
        """Start the background message source."""
        self._services.start()
        self._logger.info("DashboardBackend started")

    def update(self) -> None:
        """Process one batch of messages from Kafka."""
        self._services.orchestrator.update()

    def stop(self) -> None:
        """Stop the background message source and clean up resources."""
        self._services.stop()
        self._exit_stack.__exit__(None, None, None)
        self._logger.info("DashboardBackend stopped")

    def __enter__(self) -> 'DashboardBackend':
        """Enter context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.stop()
