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

import time
import urllib.request

from .service_process import ServiceProcess


def _wait_for_http_response(port: int, timeout: float = 10.0) -> bool:
    """
    Wait for an HTTP 200 response from the port.

    Parameters
    ----------
    port:
        Port number to connect to
    timeout:
        Maximum time to wait in seconds

    Returns
    -------
    :
        True if we got a 200 response with non-empty content, False otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(
                f'http://localhost:{port}/', timeout=1.0
            ) as response:
                if response.status == 200:
                    content = response.read().decode('utf-8')
                    return bool(content)  # Should have HTML content
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.1)
    return False


class TestDashboardNullTransport:
    """Integration tests for dashboard with null transport."""

    def test_dashboard_starts_with_null_transport(self) -> None:
        """Verify dashboard can start with --transport none without Kafka."""
        service = ServiceProcess(
            'ess.livedata.dashboard.reduction',
            instrument='dummy',
            transport='none',
            no_fetch_announcements=True,
        )

        with service:
            # Wait for HTTP response from the dashboard
            http_ready = _wait_for_http_response(5009, timeout=10.0)
            assert http_ready, "Dashboard did not respond to HTTP request"

            # Verify the process is still running
            assert service.is_running(), "Dashboard process exited unexpectedly"
