# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from contextlib import ExitStack
from typing import Any, Protocol, Self

import structlog

from ..config.instruments import available_instruments
from .processor import Processor


class ServiceBase(ABC):
    def __init__(self, *, name: str | None = None, log_level: int = logging.INFO):
        self._logger = structlog.get_logger()
        self._silence_noisy_loggers()
        self._running = False
        self._setup_signal_handlers()

    @staticmethod
    def _silence_noisy_loggers() -> None:
        """Silence third-party loggers that produce excessive output."""
        # scipp.transform_coords logs info messages that are not useful and would show
        # with every workflow call
        scipp_logger = logging.getLogger('scipp')  # noqa: TID251
        scipp_logger.setLevel(logging.WARNING)

    @property
    def is_running(self) -> bool:
        return self._running

    def _setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        self._logger.info("Registered signal handlers")

    def _handle_shutdown(self, signum: int, _: Any) -> None:
        """Handle shutdown signals"""
        self._logger.info("Received signal %d, initiating shutdown...", signum)
        self._notify_processor_shutdown()
        self.stop()
        self._notify_processor_stopped()
        sys.exit(0)

    def _notify_processor_shutdown(self) -> None:  # noqa: B027
        """Notify processor of shutdown if it supports lifecycle hooks.

        Override in subclasses to provide processor-specific shutdown notification.
        """

    def _notify_processor_stopped(self) -> None:  # noqa: B027
        """Notify processor that shutdown completed if it supports lifecycle hooks.

        Override in subclasses to provide processor-specific stopped notification.
        """

    def start(self, blocking: bool = True) -> None:
        """Start the service and block until stopped"""
        self._logger.info("Starting service...")
        self._running = True
        self._start_impl()
        self._logger.info("Service started")
        if blocking:
            self.run_forever()

    def stop(self) -> None:
        """Stop the service gracefully"""
        self._logger.info("Stopping service...")
        self._running = False
        self._stop_impl()
        self._logger.info("Service stopped")

    @abstractmethod
    def _start_impl(self) -> None:
        """Start the service implementation"""

    @abstractmethod
    def run_forever(self) -> None:
        """Block forever, waiting for signals"""

    @abstractmethod
    def _stop_impl(self) -> None:
        """Stop the service implementation"""


class StartStoppable(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class Service(ServiceBase):
    """
    Complete service with proper lifecycle management.

    Calls the injected processor in a loop with a configurable poll interval.
    If resources were passed, this class should be used as a context manager.
    """

    def __init__(
        self,
        *,
        processor: Processor,
        name: str | None = None,
        log_level: int = logging.INFO,
        poll_interval: float = 0.01,
        resources: ExitStack | None = None,
    ):
        super().__init__(name=name, log_level=log_level)
        self._poll_interval = poll_interval
        self._processor = processor
        self._thread: threading.Thread | None = None
        self._resources = resources

    def __enter__(self) -> Self:
        """Enter the context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager protocol, ensuring resources are cleaned up."""
        if self.is_running:
            self.stop()
        if self._resources is not None:
            self._logger.info("Closing resources...")
            self._resources.close()
            self._logger.info("Resources closed")

    def _start_impl(self) -> None:
        """Start the service and block until stopped"""
        self._thread = threading.Thread(target=self._run_loop)
        self._thread.start()

    def run_forever(self) -> None:
        """Block forever, waiting for signals"""
        while self.is_running:
            try:
                signal.pause()
            except KeyboardInterrupt:
                self.stop()

    def step(self) -> None:
        """Run one step of the service loop for testing purposes"""
        if self.is_running:
            raise RuntimeError("Service is running, cannot step")
        self._processor.process()

    def _run_loop(self) -> None:
        """Main service loop"""
        try:
            while self.is_running:
                start_time = time.monotonic()
                self._processor.process()
                elapsed = time.monotonic() - start_time
                remaining = max(0.0, self._poll_interval - elapsed)
                if remaining > 0:
                    time.sleep(remaining)
        except Exception as e:
            self._logger.exception("Error in service loop")
            self._notify_processor_error(str(e))
            self._running = False
            # Send a signal to the main thread to unblock it
            if threading.current_thread() is not threading.main_thread():
                os.kill(os.getpid(), signal.SIGINT)
        finally:
            self._logger.info("Service loop stopped")

    def _notify_processor_shutdown(self) -> None:
        """Notify processor of shutdown if it supports lifecycle hooks."""
        if hasattr(self._processor, 'shutdown'):
            try:
                self._processor.shutdown()
            except Exception:
                self._logger.exception("Error notifying processor of shutdown")

    def _notify_processor_stopped(self) -> None:
        """Notify processor that shutdown completed if it supports lifecycle hooks."""
        if hasattr(self._processor, 'report_stopped'):
            try:
                self._processor.report_stopped()
            except Exception:
                self._logger.exception("Error notifying processor of stopped")

    def _notify_processor_error(self, error_message: str) -> None:
        """Notify processor of error if it supports lifecycle hooks."""
        if hasattr(self._processor, 'report_error'):
            try:
                self._processor.report_error(error_message)
            except Exception:
                self._logger.exception("Error notifying processor of error")

    def _stop_impl(self) -> None:
        """Stop the service gracefully"""
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join()

    @staticmethod
    def setup_arg_parser(
        description: str, *, dev_flag: bool = True
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            '--instrument',
            choices=available_instruments(),
            default='dummy',
            help='Select the instrument',
        )
        if dev_flag:
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
        parser.add_argument(
            '--log-json-file',
            default=None,
            metavar='PATH',
            help='Write JSON-formatted logs to this file',
        )
        parser.add_argument(
            '--no-stdout-log',
            action='store_true',
            default=False,
            help='Disable logging to stdout',
        )
        return parser


def get_env_defaults(
    *, parser: argparse.ArgumentParser, prefix: str = 'LIVEDATA'
) -> dict[str, Any]:
    """Get defaults from environment variables based on parser arguments."""
    env_defaults = {}
    for action in parser._actions:
        if action.dest == 'help':
            continue
        # Start with the parser's default value
        default_value = action.default

        # Convert --arg-name to LIVEDATA_ARG_NAME
        env_name = f"{prefix}_{action.dest.upper().replace('-', '_')}"
        env_val = os.getenv(env_name)

        # Override with environment variable if present
        if env_val is not None:
            if isinstance(default_value, bool):
                env_defaults[action.dest] = env_val.lower() in ('true', '1', 'yes')
            elif isinstance(default_value, int):
                env_defaults[action.dest] = int(env_val)
            else:
                env_defaults[action.dest] = env_val
        else:
            # Use parser's default value
            env_defaults[action.dest] = default_value
    return env_defaults
