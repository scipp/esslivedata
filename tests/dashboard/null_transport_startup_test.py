# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Test that the dashboard serves pages with ``--transport none``.

Kafka-free operation is only provable by starting the real entry point, so
this test spawns the dashboard as a subprocess and probes its HTTP port.
"""

import time
import urllib.request

import pytest

from tests.integration.service_process import ServiceProcess


def _wait_for_http_response(port: int, timeout: float = 30.0) -> bool:
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


@pytest.mark.slow
def test_dashboard_starts_with_null_transport() -> None:
    """Verify dashboard can start with --transport none without Kafka."""
    # The dashboard is a Bokeh server, not a core.Service, so it logs none of
    # the readiness messages ServiceProcess greps for; the HTTP probe below is
    # its readiness signal.
    service = ServiceProcess(
        'ess.livedata.dashboard.reduction',
        instrument='dummy',
        transport='none',
        no_fetch_announcements=True,
        readiness_messages=[],
    )

    with service:
        assert _wait_for_http_response(5009), "Dashboard did not respond to HTTP"
        assert service.is_running(), "Dashboard process exited unexpectedly"
