#!/usr/bin/env python
"""Drive the live reduction dashboard with Playwright for verification/screenshots.

This is the reusable kit for browser-driving the dashboard so each session does
not re-discover the layout or re-write navigation boilerplate. See the companion
notes in ``.claude/rules/dashboard-widgets.md`` ("Driving the dashboard with
Playwright").

Two ways to use it:

* As a **library** -- :class:`Dashboard` wraps a Playwright page with readiness
  waiting, tab navigation, retry-on-detach clicks, and a runtime UI inventory::

      with Dashboard.connect() as dash:  # default localhost:5011
          dash.goto_tab("Detectors")
          dash.screenshot("plots.png")

  For multi-session scenarios (state is process-global, widgets per-session),
  :meth:`Dashboard.connect_many` opens n isolated sessions on one server.

* As a **CLI** against a running server, or self-launching a Kafka-free fake
  backend seeded from the committed dummy fixture::

      # inventory the live UI (tabs + stable lt-* hooks) -- start here, not blind
      python scripts/drive_dashboard.py --map

      # launch fake backend + fixture, screenshot the Detectors grid, tear down
      python scripts/drive_dashboard.py --launch --tab Detectors --screenshot out.png

When a block driving a :class:`Dashboard` raises, everything known about the
session is printed to stdout: the browser console tail, the page state (active
tab, dialogs, ``lt-*`` hooks), a base64 PNG screenshot, and the dashboard's own
log tail. An intermittent failure is over by the time it is known to be one, so
it has to be captured as it happens -- and on stdout, where pytest folds it into
the failure report and the run-level log carries it out of CI.

The ``lt-*`` classes are the stable automation contract (see the rule file).
Plain Playwright CSS locators pierce the dashboard's open shadow DOM, so
``page.locator(".lt-tool-settings")`` works -- but **descendant combinators do
not cross shadow boundaries**. Target a workflow's button with a *compound*
selector on one element (``.lt-wf-total_counts.lt-tool-player-stop``), never a
descendant one (``.lt-wf-total_counts .lt-tool-player-stop`` matches nothing).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError

from playwright.sync_api import (
    Browser,
    Page,
    sync_playwright,
)
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# Default off 5009 (the reduction app default) so automation never collides with
# an interactive dev dashboard.
DEFAULT_PORT = 5011
DEFAULT_URL = f"http://localhost:{DEFAULT_PORT}"
# Time for Bokeh models to settle / the first refresh tick to rebuild rows after
# load, before any click. See the "cold start" window in the rule file.
SETTLE_MS = 5000
# Override the Chromium binary when the installed playwright package does not
# match the browsers available on disk (e.g. a preprovisioned container).
_CHROMIUM_ENV = "PLAYWRIGHT_CHROMIUM_EXECUTABLE"
# Console lines kept per session. A dashboard session logs steadily, so the tail
# is what matters and the cap keeps a long run from growing without bound.
CONSOLE_LIMIT = 200
# Server log lines printed when a driven session fails.
SERVER_LOG_TAIL = 60
# Base64 wraps at this width so no single log line grows unbounded. Failure
# screenshots stay PNG: they are inlined into the log, and a dashboard is flat
# colour and text, which PNG encodes smaller than even a lossy JPEG.
BASE64_WIDTH = 120


def _launch_browser(playwright) -> Browser:
    executable = os.environ.get(_CHROMIUM_ENV)
    return playwright.chromium.launch(executable_path=executable or None)


class Dashboard:
    """Thin Playwright wrapper exposing the dashboard's stable automation hooks."""

    def __init__(self, page: Page):
        self.page = page
        # A click that reaches a widget whose JS never acted on it looks like a
        # success from Python: the element was there, the click landed, nothing
        # happened. The browser console is the only place that leaves a trace,
        # so collect it from construction rather than from first failure.
        self.console: deque[str] = deque(maxlen=CONSOLE_LIMIT)
        page.on("console", lambda msg: self.console.append(f"{msg.type}: {msg.text}"))
        page.on("pageerror", lambda err: self.console.append(f"pageerror: {err}"))

    @classmethod
    @contextmanager
    def connect(cls, url: str = DEFAULT_URL, *, settle_ms: int = SETTLE_MS):
        """Open a browser on a running dashboard, waiting for it to settle."""
        with sync_playwright() as p:
            browser = _launch_browser(p)
            try:
                with _sessions(browser, url, 1, settle_ms) as dashboards:
                    yield dashboards[0]
            finally:
                browser.close()

    @classmethod
    @contextmanager
    def connect_many(
        cls, n: int, url: str = DEFAULT_URL, *, settle_ms: int = SETTLE_MS
    ):
        """Open ``n`` independent sessions on the same running dashboard.

        Each session gets its own isolated browser context -- own websocket,
        own Bokeh session -- like ``n`` separate users. Sessions connect
        sequentially, so the returned list is ordered oldest first; use this
        to script late-joiner scenarios (dashboard state is process-global,
        widgets are per-session, so "renders in one session but not another"
        is the classic multi-session regression).
        """
        with sync_playwright() as p:
            browser = _launch_browser(p)
            try:
                with _sessions(browser, url, n, settle_ms) as dashboards:
                    yield dashboards
            finally:
                browser.close()

    def tab_names(self) -> list[str]:
        """Visible titles of all tabs, in order (static tabs then grid tabs)."""
        return [t.strip() for t in self.page.locator(".bk-tab").all_inner_texts()]

    def goto_tab(self, name: str) -> None:
        """Activate a tab by its visible title and let it render."""
        self.page.get_by_text(name, exact=True).first.click()
        self.page.wait_for_timeout(SETTLE_MS)

    def click(self, selector: str, *, retries: int = 3) -> None:
        """Click a stable ``lt-*`` selector, retrying if a rebuild detaches it.

        Staging/commit/stop rebuilds the affected workflow row, so a click can
        race the rebuild and hit a detached element. Re-locate and retry.
        """
        for attempt in range(retries):
            try:
                self.page.locator(selector).first.click(timeout=4000)
                return
            except PlaywrightTimeoutError:
                if attempt == retries - 1:
                    raise
                self.page.wait_for_timeout(1000)

    def open_modal(self, trigger_selector: str, *, retries: int = 3):
        """Click a trigger and wait for its modal (``[role=dialog]``) to show.

        Returns the dialog locator. Dismiss with ``page.keyboard.press("Escape")``
        (a ModalEscapeCloser widget makes Escape work from initial focus) or by
        clicking ``.pnx-dialog-close``.
        """
        self.click(trigger_selector, retries=retries)
        dialog = self.page.locator("[role=dialog]").first
        dialog.wait_for(state="visible", timeout=10000)
        return dialog

    def screenshot(self, path: str | Path, *, full_page: bool = True) -> None:
        self.page.screenshot(path=str(path), full_page=full_page)

    def inventory(self) -> dict:
        """Runtime UI map: tabs and counts of each stable ``lt-*`` hook present.

        The ``lt-*`` hooks live in per-widget shadow roots; this walks them so the
        report reflects what is actually targetable right now.
        """
        hooks = self.page.evaluate(
            """() => {
                const counts = {};
                const walk = (root) => root.querySelectorAll('*').forEach(el => {
                    el.classList && el.classList.forEach(c => {
                        if (c.startsWith('lt-')) counts[c] = (counts[c] || 0) + 1;
                    });
                    if (el.shadowRoot) walk(el.shadowRoot);
                });
                walk(document);
                return counts;
            }"""
        )
        return {"tabs": self.tab_names(), "lt_hooks": dict(sorted(hooks.items()))}


def _open_session(browser: Browser, url: str, settle_ms: int) -> Dashboard:
    """One dashboard session: an isolated context, loaded and settled."""
    context = browser.new_context(viewport={"width": 1600, "height": 1000})
    dash = Dashboard(context.new_page())
    dash.page.goto(url, wait_until="networkidle")
    dash.page.wait_for_timeout(settle_ms)
    return dash


@contextmanager
def _sessions(
    browser: Browser, url: str, count: int, settle_ms: int
) -> Iterator[list[Dashboard]]:
    """Open ``count`` sessions, dumping diagnostics if the caller's block raises."""
    dashboards = [_open_session(browser, url, settle_ms) for _ in range(count)]
    try:
        yield dashboards
    except BaseException:
        _dump_diagnostics(dashboards)
        raise


def _dump_diagnostics(dashboards: list[Dashboard]) -> None:
    """Print everything known about each session at the moment it failed.

    Everything goes to stdout, including the screenshot, because that is the
    one channel that reaches whoever is reading. pytest folds captured output
    into the failure report, and the run-level log zip is downloadable while
    the run is still going; CI artifacts are not reachable from every
    environment we debug from, and an upload that silently matches no files
    looks exactly like a run that captured nothing.

    Every capture is guarded: this runs while an exception propagates, and a
    diagnostic that raised would replace the failure it exists to explain.
    """
    for index, dash in enumerate(dashboards):
        for line in dash.console:
            print(f"[browser-console {index}] {line}")
        try:
            print(f"[browser-state {index}] {_page_state(dash)}")
        except Exception as exc:
            print(f"[browser-state {index}] unavailable: {exc}")
        try:
            shot = dash.page.screenshot(full_page=True)
        except Exception as exc:
            print(f"[browser-screenshot {index}] unavailable: {exc}")
            continue
        encoded = base64.b64encode(shot).decode()
        # The note carries a different tag to the payload it describes, so that
        # the command it quotes does not also match the line quoting it.
        print(
            f"[browser-diagnostics] session {index} screenshot: {len(shot)} bytes "
            f"of png as base64; decode with: grep -o "
            f"'\\[browser-screenshot {index}\\] .*' log | sed 's/.*\\] //' "
            f"| tr -d '\\n' | base64 -d > shot.png"
        )
        for start in range(0, len(encoded), BASE64_WIDTH):
            chunk = encoded[start : start + BASE64_WIDTH]
            print(f"[browser-screenshot {index}] {chunk}")


def _page_state(dash: Dashboard) -> dict:
    """What the page held when it failed: the tab shown, dialogs, and hooks.

    A driven step fails because something it addressed was absent, stale, or
    never rendered, so the answer is nearly always in one of these three: which
    tab is actually up, whether a modal exists in the DOM but never became
    visible (as opposed to never being created), and which ``lt-*`` hooks are
    present to be clicked.
    """
    dialogs = dash.page.locator("[role=dialog]")
    count = dialogs.count()
    return {
        "url": dash.page.url,
        "active_tab": dash.page.locator(".bk-tab.bk-active").first.inner_text().strip(),
        "dialogs": count,
        "dialogs_visible": sum(dialogs.nth(i).is_visible() for i in range(count)),
        **dash.inventory(),
    }


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _free_port() -> int:
    """Return a port the OS reports as free, for a server we are about to spawn.

    Hand-picked port numbers collide: two checkouts running the browser suite at
    once, or a suite running alongside an interactive dashboard, both bind the
    same literal and one of them dies. Asking the OS for an ephemeral port scopes
    the choice to the machine's actual usage instead of to a constant in this
    repo.

    The port is only reserved while the probe socket is open, so a racing process
    can still take it in the gap before our server binds. That race is not closed
    here; it surfaces as an immediate, readable startup failure in
    :func:`_wait_until_ready` rather than as a silent hang.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _log_tail(log: Path, lines: int = 15) -> str:
    return "\n".join(log.read_text().splitlines()[-lines:])


def _terminate(proc: subprocess.Popen, *, timeout_s: float = 10.0) -> None:
    """Stop a launched dashboard and reap it.

    Reaping matters on the kill path too: a child that was never waited on is
    still running when its ``Popen`` is garbage-collected, and the
    ``ResourceWarning`` that ``__del__`` then raises surfaces as an unraisable
    exception in whichever test happens to be running at the time -- a failure
    with no relation to the code under test.
    """
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _wait_until_ready(
    url: str, log: Path, proc: subprocess.Popen, *, timeout_s: float = 60.0
) -> None:
    """Block until the dashboard serves 200, or fail with its log tail.

    A server that exits before serving -- most often because its port was taken
    between being chosen and being bound -- is reported as soon as it exits, so
    the caller does not wait out the whole timeout on a process already gone.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (URLError, ConnectionError, OSError):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Dashboard at {url} exited with code {proc.returncode} before "
                    f"serving.\n{_log_tail(log)}"
                ) from None
            time.sleep(0.5)
    raise TimeoutError(
        f"Dashboard at {url} not ready within {timeout_s}s.\n{_log_tail(log)}"
    )


@dataclass
class FakeDashboard:
    """A launched fake-backend dashboard: its URL plus its server log.

    ``log`` is the path to the server's merged stdout/stderr (still being
    appended to while the dashboard runs), for tests that need to assert on
    server-side behavior a browser can't observe directly -- e.g. that an
    action didn't raise an unhandled exception. Read only the tail written
    after your action (``log.read_text()[offset:]``, with ``offset`` taken
    from ``log.stat().st_size`` beforehand) so unrelated startup log lines
    can't produce a false positive.
    """

    url: str
    log: Path


@contextmanager
def _fake_dashboard(instrument: str, port: int | None = None):
    """Launch a Kafka-free fake-backend dashboard seeded from the fixture.

    Copies the committed fixture to a writable scratch dir (the dashboard writes
    to its config dir), waits for readiness, and tears the server down on exit.
    The sidebar starts collapsed: it is static here (announcements are off), so
    an open drawer would only narrow the plots under test and their screenshots.
    Yields a :class:`FakeDashboard`.

    Parameters
    ----------
    instrument:
        Name of the committed UI config fixture to seed the dashboard from.
    port:
        Port to serve on. Omit to have one allocated (see :func:`_free_port`);
        tests should, so that concurrent runs cannot collide. Pass an explicit
        port only when the URL has to be known in advance, e.g. to open it by
        hand.
    """
    fixture = REPO_ROOT / "tests/dashboard/ui_config_fixtures" / instrument
    if not fixture.is_dir():
        raise SystemExit(f"No UI config fixture for instrument {instrument!r}")
    if port is None:
        port = _free_port()
    elif _port_in_use(port):
        raise SystemExit(
            f"Port {port} is already in use (a prior dashboard?). Stop it or pass "
            f"--port with a free port."
        )
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "cfg"
        shutil.copytree(fixture, cfg / instrument)
        log = Path(tmp) / "dashboard.log"
        with log.open("w") as logf:
            proc = subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "ess.livedata.dashboard.reduction",
                    "--instrument",
                    instrument,
                    "--transport",
                    "fake",
                    "--port",
                    str(port),
                    "--config-dir",
                    str(cfg),
                    "--auto-start",
                    "--collapsed-sidebar",
                    "--no-fetch-announcements",
                ],
                cwd=REPO_ROOT,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
            try:
                _wait_until_ready(f"http://localhost:{port}", log, proc)
                yield FakeDashboard(url=f"http://localhost:{port}", log=log)
            except BaseException:
                # A browser-side symptom usually has a server-side cause, and
                # the log dies with the temporary directory below.
                print(f"[dashboard-log tail]\n{_log_tail(log, SERVER_LOG_TAIL)}")
                raise
            finally:
                _terminate(proc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url", default=DEFAULT_URL, help="Dashboard URL (running server)."
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Start a fake-backend dashboard from the fixture, then tear it down.",
    )
    parser.add_argument("--instrument", default="dummy", help="For --launch.")
    parser.add_argument(
        "--port",
        type=int,
        help="For --launch. Defaults to a free port, so parallel runs do not collide.",
    )
    parser.add_argument("--tab", help="Activate this tab before acting.")
    parser.add_argument("--screenshot", help="Write a full-page screenshot here.")
    parser.add_argument(
        "--map",
        action="store_true",
        help="Print the runtime UI inventory (tabs + stable lt-* hooks) as JSON.",
    )
    args = parser.parse_args()
    if not (args.map or args.screenshot):
        parser.error("nothing to do: pass --map and/or --screenshot")

    @contextmanager
    def target_url():
        if args.launch:
            with _fake_dashboard(args.instrument, args.port) as fake:
                yield fake.url
        else:
            yield args.url

    with target_url() as url, Dashboard.connect(url) as dash:
        if args.tab:
            dash.goto_tab(args.tab)
        if args.map:
            print(json.dumps(dash.inventory(), indent=2))
        if args.screenshot:
            dash.screenshot(args.screenshot)
            print(f"wrote {args.screenshot}")


if __name__ == "__main__":
    main()
