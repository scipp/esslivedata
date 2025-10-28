#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Single-process launcher for ESSlivedata services.

This launcher runs all ESSlivedata services (fakes, processors, and dashboard)
in a single process using threads. Services still communicate via Kafka.

Phase 1 implementation: Thread-per-service with Kafka communication.
"""

import argparse
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, NoReturn

from ess.livedata.config.instruments import available_instruments


def setup_logging(log_level: str) -> None:
    """Configure logging for all services."""
    level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


class ServiceThread:
    """Wrapper for running a service in a thread."""

    def __init__(
        self, name: str, service_func: Callable[..., Any], kwargs: dict[str, Any]
    ):
        self.name = name
        self.service_func = service_func
        self.kwargs = kwargs
        self.thread: threading.Thread | None = None
        self.logger = logging.getLogger(f"launcher.{name}")
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the service in a background thread."""
        self.logger.info("Starting %s service...", self.name)
        self.thread = threading.Thread(
            target=self._run_wrapper, name=f"service-{self.name}", daemon=False
        )
        self.thread.start()
        self.logger.info("%s service started", self.name)

    def _run_wrapper(self) -> None:
        """Wrapper to handle service execution and errors."""
        try:
            self.service_func(**self.kwargs)
        except KeyboardInterrupt:
            self.logger.info("%s service received shutdown signal", self.name)
        except Exception as e:
            self.logger.exception("%s service failed: %s", self.name, e)
        finally:
            self._stop_event.set()

    def stop(self) -> None:
        """Signal the service to stop."""
        self.logger.info("Stopping %s service...", self.name)
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the service thread to finish."""
        if self.thread:
            self.thread.join(timeout)
            if self.thread.is_alive():
                self.logger.warning("%s service did not stop cleanly", self.name)
            else:
                self.logger.info("%s service stopped", self.name)


class Launcher:
    """Manages lifecycle of all ESSlivedata services in a single process."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = logging.getLogger("launcher")
        self.services: list[ServiceThread] = []
        self._shutdown_event = threading.Event()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers."""

        def signal_handler(signum: int, frame: Any) -> None:
            self.logger.info("Received signal %s, initiating shutdown...", signum)
            self._shutdown_event.set()
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _create_fake_detector_service(self) -> ServiceThread:
        """Create the fake detector data service."""
        from ess.livedata.services import fake_detectors

        kwargs = {
            'instrument': self.args.instrument,
            'log_level': self.args.log_level,
        }
        if self.args.nexus_file:
            kwargs['nexus_file'] = self.args.nexus_file

        return ServiceThread('fake-detectors', fake_detectors.run_service, kwargs)

    def _create_fake_monitor_service(self) -> ServiceThread:
        """Create the fake monitor data service."""
        from ess.livedata.services import fake_monitors

        kwargs = {
            'instrument': self.args.instrument,
            'log_level': self.args.log_level,
            'mode': self.args.monitor_mode,
            'num_monitors': self.args.num_monitors,
        }

        return ServiceThread('fake-monitors', fake_monitors.run_service, kwargs)

    def _create_fake_logdata_service(self) -> ServiceThread:
        """Create the fake log data service."""
        from ess.livedata.services import fake_logdata

        kwargs = {
            'instrument': self.args.instrument,
            'log_level': self.args.log_level,
        }

        return ServiceThread('fake-logdata', fake_logdata.run_service, kwargs)

    def _create_monitor_data_service(self) -> ServiceThread:
        """Create the monitor data processing service."""
        from ess.livedata.services import monitor_data

        kwargs = {
            'instrument': self.args.instrument,
            'dev': self.args.dev,
            'log_level': self.args.log_level,
        }

        return ServiceThread(
            'monitor-data',
            self._run_processing_service_wrapper(monitor_data.main),
            kwargs,
        )

    def _create_detector_data_service(self) -> ServiceThread:
        """Create the detector data processing service."""
        from ess.livedata.services import detector_data

        kwargs = {
            'instrument': self.args.instrument,
            'dev': self.args.dev,
            'log_level': self.args.log_level,
        }

        return ServiceThread(
            'detector-data',
            self._run_processing_service_wrapper(detector_data.main),
            kwargs,
        )

    def _create_data_reduction_service(self) -> ServiceThread:
        """Create the data reduction service."""
        from ess.livedata.services import data_reduction

        kwargs = {
            'instrument': self.args.instrument,
            'dev': self.args.dev,
            'log_level': self.args.log_level,
        }

        return ServiceThread(
            'data-reduction',
            self._run_processing_service_wrapper(data_reduction.main),
            kwargs,
        )

    def _create_timeseries_service(self) -> ServiceThread:
        """Create the timeseries service."""
        from ess.livedata.services import timeseries

        kwargs = {
            'instrument': self.args.instrument,
            'dev': self.args.dev,
            'log_level': self.args.log_level,
        }

        return ServiceThread(
            'timeseries', self._run_processing_service_wrapper(timeseries.main), kwargs
        )

    def _run_processing_service_wrapper(
        self, main_func: Callable[[], None]
    ) -> Callable[..., None]:
        """Wrapper to adapt processing service main functions to accept kwargs."""

        def wrapper(**kwargs: Any) -> None:
            # Set up sys.argv for the service's argument parser
            sys.argv = ['launcher']
            sys.argv.append(f'--instrument={kwargs["instrument"]}')
            if kwargs.get('dev'):
                sys.argv.append('--dev')
            sys.argv.append(f'--log-level={kwargs["log_level"]}')
            main_func()

        return wrapper

    def _run_dashboard(self) -> None:
        """Run the dashboard in the main thread (Panel requirement)."""
        from ess.livedata.dashboard import reduction

        self.logger.info("Starting dashboard service...")

        # Set up sys.argv for the dashboard's argument parser
        sys.argv = ['launcher']
        sys.argv.append(f'--instrument={self.args.instrument}')
        if self.args.dev:
            sys.argv.append('--dev')
        sys.argv.append(f'--log-level={self.args.log_level}')

        try:
            reduction.main()
        except KeyboardInterrupt:
            self.logger.info("Dashboard received shutdown signal")
        except Exception as e:
            self.logger.exception("Dashboard failed: %s", e)

    def start(self) -> None:
        """Start all services."""
        self.logger.info("=" * 60)
        self.logger.info("ESSlivedata Launcher Starting")
        self.logger.info("Instrument: %s", self.args.instrument)
        self.logger.info("Development mode: %s", self.args.dev)
        self.logger.info("=" * 60)

        # Create services based on enabled flags
        if self.args.fake_detectors:
            self.services.append(self._create_fake_detector_service())

        if self.args.fake_monitors:
            self.services.append(self._create_fake_monitor_service())

        if self.args.fake_logdata:
            self.services.append(self._create_fake_logdata_service())

        if self.args.monitor_data:
            self.services.append(self._create_monitor_data_service())

        if self.args.detector_data:
            self.services.append(self._create_detector_data_service())

        if self.args.data_reduction:
            self.services.append(self._create_data_reduction_service())

        if self.args.timeseries:
            self.services.append(self._create_timeseries_service())

        # Start all background services
        for service in self.services:
            service.start()
            # Small delay to avoid thundering herd on Kafka
            time.sleep(0.1)

        # Run dashboard in main thread if enabled
        if self.args.dashboard:
            self._run_dashboard()
        else:
            # If no dashboard, wait for shutdown signal
            self.logger.info("All services running. Press Ctrl+C to stop.")
            try:
                self._shutdown_event.wait()
            except KeyboardInterrupt:
                pass

    def stop(self) -> None:
        """Stop all services gracefully."""
        self.logger.info("Stopping all services...")

        # Stop services in reverse order
        for service in reversed(self.services):
            service.stop()

        # Wait for services to finish
        for service in reversed(self.services):
            service.join(timeout=5.0)

        self.logger.info("All services stopped")

    def run(self) -> NoReturn:
        """Main entry point for the launcher."""
        try:
            self.start()
        finally:
            self.stop()
            sys.exit(0)


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run all ESSlivedata services in a single process',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Common arguments
    parser.add_argument(
        '--instrument',
        choices=available_instruments(),
        default='dummy',
        help='Select the instrument',
    )
    parser.add_argument(
        '--dev',
        action='store_true',
        default=False,
        help='Run in development mode with simplified topic naming',
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level',
    )

    # Service selection flags
    service_group = parser.add_argument_group('Service Selection')
    service_group.add_argument(
        '--all',
        action='store_true',
        help='Enable all services',
    )
    service_group.add_argument(
        '--fake-detectors',
        action='store_true',
        help='Enable fake detector data service',
    )
    service_group.add_argument(
        '--fake-monitors',
        action='store_true',
        help='Enable fake monitor data service',
    )
    service_group.add_argument(
        '--fake-logdata',
        action='store_true',
        help='Enable fake log data service',
    )
    service_group.add_argument(
        '--monitor-data',
        action='store_true',
        help='Enable monitor data processing service',
    )
    service_group.add_argument(
        '--detector-data',
        action='store_true',
        help='Enable detector data processing service',
    )
    service_group.add_argument(
        '--data-reduction',
        action='store_true',
        help='Enable data reduction service',
    )
    service_group.add_argument(
        '--timeseries',
        action='store_true',
        help='Enable timeseries service',
    )
    service_group.add_argument(
        '--dashboard',
        action='store_true',
        help='Enable reduction dashboard',
    )

    # Service-specific arguments
    fake_group = parser.add_argument_group('Fake Service Options')
    fake_group.add_argument(
        '--nexus-file',
        type=str,
        help='NeXus file for fake detector data (optional)',
    )
    fake_group.add_argument(
        '--monitor-mode',
        choices=['ev44', 'da00'],
        default='da00',
        help='Monitor data mode for fake monitors',
    )
    fake_group.add_argument(
        '--num-monitors',
        type=int,
        default=2,
        choices=range(1, 11),
        metavar='1-10',
        help='Number of monitors to simulate (1-10)',
    )

    # Preset configurations
    preset_group = parser.add_argument_group('Preset Configurations')
    preset_group.add_argument(
        '--dream-demo',
        action='store_true',
        help='Run the DREAM demo configuration (all services for DREAM instrument)',
    )

    return parser


def apply_presets(args: argparse.Namespace) -> None:
    """Apply preset configurations based on flags."""
    if args.dream_demo:
        args.instrument = 'dream'
        args.dev = True
        args.all = True
        args.monitor_mode = 'da00'

    if args.all:
        args.fake_detectors = True
        args.fake_monitors = True
        args.fake_logdata = True
        args.monitor_data = True
        args.detector_data = True
        args.data_reduction = True
        args.timeseries = True
        args.dashboard = True


def main() -> NoReturn:
    """Main entry point for the launcher."""
    parser = create_parser()
    args = parser.parse_args()

    # Apply preset configurations
    apply_presets(args)

    # Validate that at least one service is enabled
    service_flags = [
        args.fake_detectors,
        args.fake_monitors,
        args.fake_logdata,
        args.monitor_data,
        args.detector_data,
        args.data_reduction,
        args.timeseries,
        args.dashboard,
    ]
    if not any(service_flags):
        parser.error('No services enabled. Use --all or enable specific services.')

    # Setup logging
    setup_logging(args.log_level)

    # Create and run launcher
    launcher = Launcher(args)
    launcher.run()


if __name__ == "__main__":
    main()
