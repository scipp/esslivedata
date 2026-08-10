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
        # Set by anything that asks for shutdown -- a signal, a worker loop that
        # died, an explicit stop() -- so that a blocking start() wakes up.
        self._shutdown_requested = threading.Event()
        self._shutdown_signum: int | None = None

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
        """Install handlers for graceful shutdown.

        Called from :meth:`start`, not from ``__init__``: a handler firing
        mid-construction would run against a half-built service, and a service
        that was never started has nothing to shut down anyway.
        """
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        self._logger.info("Registered signal handlers")

    def _handle_shutdown(self, signum: int, _: Any) -> None:
        """Ask the main thread to shut down, doing nothing that can block.

        A handler runs on the main thread between bytecodes, so it must not
        take a lock the interrupted frame may hold. Logging here deadlocks the
        process whenever the signal lands inside another log call -- and a
        service logs constantly, so this happens. Everything that logs, stops
        threads or finalizes runs in :meth:`_shut_down` instead; raising
        SystemExit is the async-signal-safe way to unwind the main thread out
        of whatever it is blocked in and get there.
        """
        self._shutdown_signum = signum
        self._shutdown_requested.set()
        sys.exit(self._exit_code)

    @property
    def _exit_code(self) -> int:
        """Process exit code on shutdown; nonzero signals a fault to the supervisor."""
        return 0

    def _finalize_processor(self) -> None:  # noqa: B027
        """Finalize processor after the worker thread has stopped.

        Override in subclasses to provide processor-specific finalization.
        """

    def start(self, blocking: bool = True) -> None:
        """Start the service and, unless ``blocking`` is False, run until stopped"""
        if not blocking:
            self._launch()
            return
        try:
            self._launch()
            self.run_forever()
        except SystemExit:
            # A signal handler's exit, raised wherever the main thread happened
            # to be -- possibly still in startup. The shutdown it asked for
            # runs below, off the handler.
            pass
        self._shut_down()

    def _launch(self) -> None:
        self._setup_signal_handlers()
        self._logger.info("Starting service...")
        self._running = True
        self._start_impl()
        self._logger.info("Service started")

    def _shut_down(self) -> None:
        """Stop and finalize on the main thread, then exit the process.

        Runs once :meth:`run_forever` has returned or been unwound, so the
        logging and thread joins here are outside any signal handler.
        """
        if self._shutdown_signum is not None:
            self._logger.info(
                "Received signal %d, initiating shutdown...", self._shutdown_signum
            )
        self.stop()
        self._finalize_processor()
        sys.exit(self._exit_code)

    def stop(self) -> None:
        """Stop the service gracefully"""
        self._logger.info("Stopping service...")
        self._running = False
        self._shutdown_requested.set()
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
        self._worker_error: str | None = None

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
        """Block until a signal, an explicit stop, or the worker loop's death."""
        self._shutdown_requested.wait()

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
            self._worker_error = str(e)
            self._running = False
        finally:
            self._logger.info("Service loop stopped")
            # Wake a blocking start(), which reports the error via the exit
            # code. Signalling the process instead would land the shutdown in
            # a signal handler, where it cannot log or join threads.
            self._shutdown_requested.set()

    @property
    def _exit_code(self) -> int:
        """Nonzero when the worker loop died on an error, so that
        ``restart: on-failure`` supervisors actually restart the service."""
        return 1 if self._worker_error is not None else 0

    def _finalize_processor(self) -> None:
        """Finalize processor after the worker thread has joined."""
        try:
            self._processor.finalize(error=self._worker_error)
        except Exception:
            self._logger.exception("Error finalizing processor")

    def _stop_impl(self) -> None:
        """Stop the service gracefully"""
        # is_alive() is False for a thread that was created but never started,
        # which a signal landing mid-startup makes reachable, and join() raises
        # on such a thread.
        if (
            self._thread is not None
            and self._thread.is_alive()
            and self._thread is not threading.current_thread()
        ):
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
