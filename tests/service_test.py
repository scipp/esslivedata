# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import signal
import subprocess
import sys
import textwrap
import time

from ess.livedata import Service


class FakeProcessor:
    def __init__(self):
        self.call_count = 0

    def process(self) -> None:
        self.call_count += 1

    def finalize(self, *, error: str | None = None) -> None:
        pass


def test_create_start_stop_service() -> None:
    processor = FakeProcessor()
    service = Service(processor=processor)
    assert processor.call_count == 0
    service.start(blocking=False)
    assert service.is_running
    time.sleep(0.2)
    assert processor.call_count > 0
    service.stop()
    assert not service.is_running


# These run a real blocking service in a subprocess: constructing a Service
# installs process-wide signal handlers, and the worker loop signals the main
# thread via SIGINT, so the exit code is only observable from outside the process.
_SERVICE_SCRIPT = textwrap.dedent(
    """
    from ess.livedata import Service

    class Processor:
        def process(self):
            {body}
        def finalize(self, *, error=None):
            pass

    Service(processor=Processor(), poll_interval=0.005).start()
    """
)


def test_worker_loop_error_exits_nonzero() -> None:
    script = _SERVICE_SCRIPT.format(body='raise RuntimeError("boom")')
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 1


def test_clean_shutdown_exits_zero() -> None:
    script = _SERVICE_SCRIPT.format(body='pass')
    proc = subprocess.Popen([sys.executable, "-c", script])  # noqa: S603
    time.sleep(1.0)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 0
