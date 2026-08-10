# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import concurrent.futures as cf
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

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


# These run a real blocking service in a subprocess: starting one installs
# process-wide signal handlers and the shutdown ends in sys.exit, so the exit
# code is only observable from outside the process.
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


def _shutdown_during_startup() -> int:
    """Start a service, SIGTERM it as it starts up, and return its exit code.

    Signalling on the handlers-registered line aims the signal at the startup
    log calls that follow it. A handler doing real work there ran while the
    main thread held the stdout lock (deadlock), before the worker thread
    existed (join of an unstarted thread), or before the service was fully
    built. Stdout is drained throughout, so a full pipe cannot masquerade as a
    hang.
    """
    with subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", _SERVICE_SCRIPT.format(body='pass')],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    ) as proc:
        registered = threading.Event()

        def drain() -> None:
            for line in proc.stdout:  # type: ignore[union-attr]
                if "Registered signal handlers" in line:
                    registered.set()
            registered.set()

        reader = threading.Thread(target=drain)
        reader.start()
        try:
            registered.wait(timeout=30)
            proc.send_signal(signal.SIGTERM)
            return proc.wait(timeout=30)
        finally:
            proc.kill()
            reader.join(timeout=5)


@pytest.mark.slow
def test_signal_during_startup_shuts_down_cleanly() -> None:
    """A signal landing in the startup window must still exit cleanly.

    The window is sub-millisecond, so this runs several services at once and
    fails if any of them exits nonzero or has to be killed on timeout.
    """
    with cf.ThreadPoolExecutor(max_workers=8) as executor:
        codes = list(executor.map(lambda _: _shutdown_during_startup(), range(8)))

    assert codes == [0] * 8
