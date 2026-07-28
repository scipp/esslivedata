# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the launch plumbing in ``scripts/drive_dashboard.py``.

No browser is driven here, but the module imports Playwright at import time, so
these carry the ``browser`` marker to run in the environment that has it
(``tox -e browser``) rather than being silently skipped everywhere.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("playwright.sync_api")
from tests.helpers.browser import drive_dashboard


@pytest.mark.browser
def test_free_port_offers_a_port_nothing_is_listening_on():
    port = drive_dashboard._free_port()
    assert 1024 < port <= 65535
    assert not drive_dashboard._port_in_use(port)


@pytest.mark.browser
def test_free_port_releases_the_port_it_offers():
    # The probe socket must not still hold the port it just reported, or every
    # launch would hand its server a port the server cannot bind.
    port = drive_dashboard._free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


@pytest.mark.browser
def test_wait_until_ready_reports_a_dead_server_instead_of_waiting_it_out(tmp_path):
    # A server whose port was taken between being chosen and being bound exits
    # at once. Waiting out the full timeout would bury the reason; the caller
    # gets the exit code and the server's own log tail instead.
    log = tmp_path / "dashboard.log"
    with log.open("w") as logf:
        proc = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-c",
                "print('OSError: [Errno 98] Address already in use'); "
                "raise SystemExit(3)",
            ],
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    proc.wait(timeout=10)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exited with code 3") as exc_info:
        drive_dashboard._wait_until_ready(
            f"http://localhost:{drive_dashboard._free_port()}", log, proc
        )
    assert time.monotonic() - started < 10, "did not fail fast on a dead server"
    assert "Address already in use" in str(exc_info.value)
