# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service process lifecycle management for integration tests."""

import logging
import subprocess
import sys
import threading
import time
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)


class ServiceProcess:
    """
    Manages a service as a subprocess for integration testing.

    This class wraps subprocess.Popen to launch ESSlivedata services
    (e.g., fake_monitors, monitor_data) with proper argument handling,
    lifecycle management, and cleanup.

    Parameters
    ----------
    service_module:
        Python module name to run (e.g., 'ess.livedata.services.fake_monitors')
    log_level:
        Logging level for the subprocess (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    **kwargs:
        Service-specific arguments passed as command-line flags
        (e.g., instrument='dummy', dev=True becomes --instrument dummy --dev)
    """

    def __init__(
        self,
        service_module: str,
        *,
        log_level: str = 'INFO',
        **kwargs: Any,
    ):
        self.service_module = service_module
        self.log_level = log_level
        self.kwargs = kwargs
        self.process: subprocess.Popen | None = None
        self._stdout_lines: list[str] = []
        self._stderr_lines: list[str] = []
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _stream_output(self, stream, output_list: list[str], stream_name: str) -> None:
        """Read from a stream and forward lines to stderr for pytest capture."""
        try:
            for line in iter(stream.readline, ''):
                if self._stop_event.is_set():
                    break
                if line:
                    output_list.append(line)
                    # Forward to stderr - pytest will capture this
                    # Prefix with service name for clarity
                    sys.stderr.write(f"[{self.service_module}] {line}")
                    sys.stderr.flush()
        except ValueError:
            # Stream was closed
            pass

    def start(self, startup_delay: float = 2.0) -> None:
        """
        Start the service subprocess.

        Parameters
        ----------
        startup_delay:
            Time to wait after starting the service for it to initialize (seconds)
        """
        # Build command line arguments
        args = [sys.executable, '-m', self.service_module]

        # Add log level to command line
        args.extend(['--log-level', self.log_level])

        for key, value in self.kwargs.items():
            # Convert Python naming to CLI flags (underscore to hyphen)
            flag_name = key.replace('_', '-')

            if isinstance(value, bool):
                # Boolean flags: only add if True
                if value:
                    args.append(f'--{flag_name}')
            else:
                # Regular arguments
                args.extend([f'--{flag_name}', str(value)])

        logger.info("Starting service: %s with args: %s", self.service_module, args)

        # Start subprocess with pipes for stdout/stderr
        self.process = subprocess.Popen(  # noqa: S603
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Start threads to stream output
        self._stop_event.clear()
        self._stdout_thread = threading.Thread(
            target=self._stream_output,
            args=(self.process.stdout, self._stdout_lines, 'stdout'),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stream_output,
            args=(self.process.stderr, self._stderr_lines, 'stderr'),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        # Wait for service to report it has started (or crash)
        start_time = time.time()
        service_started = False
        while time.time() - start_time < startup_delay:
            # Check if process crashed
            if self.process.poll() is not None:
                # Wait for threads to finish reading any error output
                self._stop_event.set()
                if self._stdout_thread:
                    self._stdout_thread.join(timeout=1.0)
                if self._stderr_thread:
                    self._stderr_thread.join(timeout=1.0)

                stdout = ''.join(self._stdout_lines)
                stderr = ''.join(self._stderr_lines)
                raise RuntimeError(
                    f"Service {self.service_module} failed to start.\n"
                    f"Exit code: {self.process.returncode}\n"
                    f"STDOUT: {stdout}\n"
                    f"STDERR: {stderr}"
                )

            # Check if we've seen "Service started" in output
            combined_output = ''.join(self._stdout_lines + self._stderr_lines)
            if 'Service started' in combined_output:
                service_started = True
                break

            time.sleep(0.01)  # Small sleep to avoid busy-waiting

        if not service_started:
            logger.warning(
                "Did not see 'Service started' message for %s within %.1fs, "
                "but process is still running",
                self.service_module,
                startup_delay,
            )

        logger.info(
            "Service %s started with PID %s (took %.3fs)",
            self.service_module,
            self.process.pid,
            time.time() - start_time,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the service subprocess gracefully.

        Parameters
        ----------
        timeout:
            Maximum time to wait for graceful shutdown (seconds)
        """
        if self.process is None:
            logger.warning("Service %s was not running", self.service_module)
            return

        if self.process.poll() is not None:
            logger.info("Service %s already stopped", self.service_module)
            return

        logger.info(
            "Stopping service %s (PID %s)", self.service_module, self.process.pid
        )

        # Try graceful termination first
        self.process.terminate()

        try:
            self.process.wait(timeout=timeout)
            logger.info("Service %s stopped gracefully", self.service_module)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Service %s did not stop within %s seconds, killing",
                self.service_module,
                timeout,
            )
            self.process.kill()
            self.process.wait()

        # Signal threads to stop and wait for them to finish
        self._stop_event.set()
        if self._stdout_thread and self._stdout_thread.is_alive():
            self._stdout_thread.join(timeout=0.1)
            if self._stdout_thread.is_alive():
                logger.warning("stdout thread did not stop for %s", self.service_module)
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=0.1)
            if self._stderr_thread.is_alive():
                logger.warning("stderr thread did not stop for %s", self.service_module)

        # Close pipes
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.stderr:
            self.process.stderr.close()

    def is_running(self) -> bool:
        """Check if the service process is currently running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def get_stdout(self) -> str:
        """Get accumulated stdout from the service."""
        return ''.join(self._stdout_lines)

    def get_stderr(self) -> str:
        """Get accumulated stderr from the service."""
        return ''.join(self._stderr_lines)

    def __enter__(self) -> 'ServiceProcess':
        """Enter context manager - starts the service."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - stops the service."""
        self.stop()


class ServiceGroup:
    """
    Manages multiple services as a group for integration testing.

    This is useful for starting multiple interdependent services
    (e.g., fake_monitors + monitor_data) with proper ordering and cleanup.

    Parameters
    ----------
    services:
        Dictionary mapping service names to ServiceProcess instances
    """

    def __init__(self, services: dict[str, ServiceProcess]):
        self.services = services

    def start_all(self, startup_delay: float = 2.0) -> None:
        """
        Start all services in the group.

        Parameters
        ----------
        startup_delay:
            Time to wait after starting each service (seconds)
        """
        for name, service in self.services.items():
            logger.info("Starting service group member: %s", name)
            service.start(startup_delay=startup_delay)

    def stop_all(self, timeout: float = 10.0) -> None:
        """
        Stop all services in the group (in reverse order).

        Parameters
        ----------
        timeout:
            Maximum time to wait for each service to stop (seconds)
        """
        # Stop in reverse order
        for name, service in reversed(list(self.services.items())):
            logger.info("Stopping service group member: %s", name)
            service.stop(timeout=timeout)

    def __enter__(self) -> 'ServiceGroup':
        """Enter context manager - starts all services."""
        self.start_all()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager - stops all services."""
        self.stop_all()

    def __getitem__(self, name: str) -> ServiceProcess:
        """Get a service by name."""
        return self.services[name]
