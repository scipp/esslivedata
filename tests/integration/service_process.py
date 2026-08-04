# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Service process lifecycle management for integration tests."""

import logging
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from types import TracebackType
from typing import Any, Self

logger = logging.getLogger(__name__)

#: Time allowed for a service to report readiness. Generous because a Kafka
#: consumer may spend up to ``kafka.consumer._LEADER_TIMEOUT`` (30 s) waiting
#: for partition leadership to settle before it can consume at all. Startup
#: normally takes a fraction of this; the wait ends as soon as the service is
#: ready, so a large value costs nothing in the healthy case.
READINESS_TIMEOUT = 60.0


class ServiceProcess:
    """
    Manages a service as a subprocess for integration testing.

    This class wraps subprocess.Popen to launch ESSlivedata services
    (e.g., fake_monitors, monitor_data) with proper argument handling,
    lifecycle management, and cleanup.

    Service readiness is detected by matching log output against expected
    messages. This is a pragmatic approach for testing, though production
    deployments should use proper health/liveness endpoints (future enhancement).

    Parameters
    ----------
    service_module:
        Python module name to run (e.g., 'ess.livedata.services.fake_monitors')
    log_level:
        Logging level for the subprocess (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    readiness_messages:
        Log messages to wait for during startup. Services are considered ready
        when ALL messages have appeared in the output. Defaults to waiting for
        "Service started". For services with Kafka consumers, consider also
        waiting for "kafka_consumer_ready" to ensure functional
        readiness. Pass an empty list for a process that logs no readiness
        message and whose readiness the caller establishes itself.
    **kwargs:
        Service-specific arguments passed as command-line flags
        (e.g., instrument='dummy', dev=True becomes --instrument dummy --dev)

    Notes
    -----
    Long-term improvement: Replace log message matching with HTTP health/liveness
    endpoints that can verify actual service functionality (e.g., Kafka connectivity,
    resource availability). This would provide more robust readiness detection and
    align with standard production practices.
    """

    def __init__(
        self,
        service_module: str,
        *,
        log_level: str = 'INFO',
        readiness_messages: list[str] | None = None,
        **kwargs: Any,
    ):
        self.service_module = service_module
        self.log_level = log_level
        self.readiness_messages = (
            ['Service started'] if readiness_messages is None else readiness_messages
        )
        self.kwargs = kwargs
        self.process: subprocess.Popen | None = None
        self._stdout_lines: list[str] = []
        self._stderr_lines: list[str] = []
        # Background threads are used to continuously read from stdout/stderr pipes.
        # This prevents the subprocess from blocking when pipe buffers fill up,
        # which would cause the service to hang. The threads also forward output
        # to pytest for real-time visibility in test logs.
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _stream_output(self, stream, output_list: list[str], stream_name: str) -> None:
        """Read from a stream and forward lines to stderr for pytest capture.

        This runs in a background thread to continuously drain the subprocess pipes.
        Without this, the subprocess would block when the pipe buffer fills up,
        causing the service to hang indefinitely.

        Note: stream.readline() is a blocking I/O call that efficiently waits for
        data without spinning (consuming CPU). The thread sleeps until data arrives.
        """
        try:
            # iter(stream.readline, '') blocks on readline() until data arrives or EOF
            # This is efficient - the thread sleeps while waiting, no CPU spinning
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

    def _build_command_args(self) -> list[str]:
        """Build command line arguments for the service subprocess."""
        args = [sys.executable, '-m', self.service_module]
        args.extend(['--log-level', self.log_level])

        for key, value in self.kwargs.items():
            flag_name = key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    args.append(f'--{flag_name}')
            else:
                args.extend([f'--{flag_name}', str(value)])

        return args

    def _start_output_threads(self) -> None:
        """Start background threads to capture stdout and stderr."""
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

    def _wait_for_service_ready(self, readiness_timeout: float) -> None:
        """
        Wait for the service to report readiness or crash during startup.

        Checks for all configured readiness messages. Services are considered
        ready only when ALL messages have appeared in the output.

        Raises
        ------
        RuntimeError:
            If the service exits, or does not report readiness within
            ``readiness_timeout``. A service that has not reported readiness
            cannot be assumed to consume anything, so proceeding would turn a
            startup failure into an unrelated timeout further down the test.
        """
        start_time = time.time()
        missing_messages = set(self.readiness_messages)

        while time.time() - start_time < readiness_timeout:
            if self.process.poll() is not None:
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

            combined_output = ''.join(self._stdout_lines + self._stderr_lines)

            # Check which readiness messages have appeared
            for msg in list(missing_messages):
                if msg in combined_output:
                    missing_messages.remove(msg)
                    logger.debug(
                        "Service %s: detected readiness message '%s'",
                        self.service_module,
                        msg,
                    )

            # All readiness messages found
            if not missing_messages:
                break

            time.sleep(0.01)

        if missing_messages:
            raise RuntimeError(
                f"Service {self.service_module} did not report readiness within "
                f"{readiness_timeout:.1f}s (process still running). Missing: "
                f"{sorted(missing_messages)}\n"
                f"STDOUT: {self.get_stdout()}\n"
                f"STDERR: {self.get_stderr()}"
            )

        logger.info(
            "Service %s ready with PID %s (took %.3fs)",
            self.service_module,
            self.process.pid,
            time.time() - start_time,
        )

    def start(self, readiness_timeout: float = READINESS_TIMEOUT) -> None:
        """
        Start the service subprocess and wait until it reports readiness.

        Parameters
        ----------
        readiness_timeout:
            Maximum time to wait for the readiness messages (seconds)
        """
        args = self._build_command_args()
        logger.info("Starting service: %s with args: %s", self.service_module, args)

        # Readiness is detected by matching captured output, so a restarted
        # process must not inherit the previous run's readiness messages.
        self._stdout_lines.clear()
        self._stderr_lines.clear()

        self.process = subprocess.Popen(  # noqa: S603
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._start_output_threads()
        try:
            self._wait_for_service_ready(readiness_timeout)
        except RuntimeError:
            # Nothing has taken ownership of the process yet, so a service that
            # never became ready would otherwise outlive the test run.
            self.stop()
            raise

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the service subprocess gracefully.

        Uses ExitStack to ensure all cleanup steps complete even if some raise
        exceptions, preventing resource leaks from partially-completed shutdowns.

        Parameters
        ----------
        timeout:
            Maximum time to wait for graceful shutdown (seconds)
        """
        if self.process is None:
            logger.warning("Service %s was not running", self.service_module)
            return

        # Use ExitStack to ensure all cleanup happens even if exceptions occur
        with ExitStack() as cleanup_stack:
            # Register pipe cleanup
            if self.process.stdout:
                cleanup_stack.callback(self.process.stdout.close)
            if self.process.stderr:
                cleanup_stack.callback(self.process.stderr.close)

            # Register thread cleanup
            def stop_threads():
                self._stop_event.set()
                if self._stdout_thread and self._stdout_thread.is_alive():
                    self._stdout_thread.join(timeout=0.1)
                    if self._stdout_thread.is_alive():
                        logger.warning(
                            "stdout thread did not stop for %s", self.service_module
                        )
                if self._stderr_thread and self._stderr_thread.is_alive():
                    self._stderr_thread.join(timeout=0.1)
                    if self._stderr_thread.is_alive():
                        logger.warning(
                            "stderr thread did not stop for %s", self.service_module
                        )

            cleanup_stack.callback(stop_threads)

            if self.process.poll() is not None:
                # Already dead (clean exit or crash-injection kill): still reap
                # and run the pipe/thread cleanup, else the unclosed pipe
                # FileIOs surface as ResourceWarnings that pytest escalates to
                # an error on the next test.
                self.process.wait()
                logger.info("Service %s already stopped", self.service_module)
                return

            logger.info(
                "Stopping service %s (PID %s)", self.service_module, self.process.pid
            )

            # Terminate/kill the process (done first, cleanup happens via ExitStack)
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

    def __enter__(self) -> Self:
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

    Uses ExitStack to ensure robust cleanup: if any service fails to stop,
    all other services will still be stopped.

    Parameters
    ----------
    services:
        Dictionary mapping service names to ServiceProcess instances
    """

    def __init__(self, services: dict[str, ServiceProcess]):
        self.services = services

    def start_all(self, readiness_timeout: float = READINESS_TIMEOUT) -> None:
        """
        Start all services in the group.

        If any service fails to start, all previously started services are
        stopped to prevent orphaned processes.

        Parameters
        ----------
        readiness_timeout:
            Maximum time to wait for each service to report readiness (seconds)
        """
        # Use ExitStack to track started services for automatic cleanup on failure
        with ExitStack() as stack:
            for name, service in self.services.items():
                logger.info("Starting service group member: %s", name)
                service.start(readiness_timeout=readiness_timeout)
                # Register cleanup in case a later service fails to start
                stack.callback(service.stop)
            # All services started successfully, don't clean up
            stack.pop_all()

    def stop_all(self, timeout: float = 10.0) -> Self:
        """
        Stop all services in the group (in reverse order).

        Uses ExitStack to ensure all services are stopped even if some raise
        exceptions during shutdown. Exceptions are logged but don't prevent
        other services from being stopped.

        Parameters
        ----------
        timeout:
            Maximum time to wait for each service to stop (seconds)
        """
        # Use ExitStack to ensure all services get stopped even if some raise
        with ExitStack() as stack:
            # Register all services for cleanup (in reverse order)
            for name, service in reversed(list(self.services.items())):

                def make_stop_callback(service_name: str, svc: ServiceProcess):
                    """Create a closure that stops a specific service."""

                    def stop_callback():
                        logger.info("Stopping service group member: %s", service_name)
                        svc.stop(timeout=timeout)

                    return stop_callback

                stack.callback(make_stop_callback(name, service))

    def __enter__(self) -> Self:
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
