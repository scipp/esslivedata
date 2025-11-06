# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration test for dashboard with null transport.

Note: This test is intentionally not marked with @pytest.mark.integration even
though it is technically an integration test (runs a subprocess and verifies
port binding). The integration marker is currently used to skip tests that
require Kafka/services in CI. This test explicitly does NOT require Kafka
(it verifies the null transport works in isolation) and should run in CI.

In the future, the test markers should be refined to distinguish between:
- Tests requiring Kafka/services (currently marked as "integration")
- Tests requiring subprocesses but no external services (this test)
"""

import socket
import subprocess
import sys
import time


def _wait_for_port(port: int, timeout: float = 10.0) -> bool:
    """
    Wait for a port to be listening.

    Parameters
    ----------
    port:
        Port number to check
    timeout:
        Maximum time to wait in seconds

    Returns
    -------
    :
        True if port is listening, False if timeout exceeded
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(('localhost', port), timeout=1.0):
                return True
        except (OSError, TimeoutError):
            time.sleep(0.1)
    return False


class TestDashboardNullTransport:
    """Integration tests for dashboard with null transport."""

    def test_dashboard_starts_with_null_transport(self) -> None:
        """Verify dashboard can start with --transport none without Kafka."""
        # Start the dashboard process
        process = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                '-m',
                'ess.livedata.dashboard.reduction',
                '--instrument',
                'dummy',
                '--transport',
                'none',
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait for the dashboard to start and listen on port 5009
            port_ready = _wait_for_port(5009, timeout=10.0)
            assert port_ready, "Dashboard failed to open port 5009 within timeout"

            # Verify the process is still running
            assert process.poll() is None, "Dashboard process exited unexpectedly"

        finally:
            # Clean up: terminate the process
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            finally:
                # Close pipes to avoid resource warnings
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
