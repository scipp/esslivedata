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

* As a **CLI** against a running server, or self-launching a Kafka-free fake
  backend seeded from the committed dummy fixture::

      # inventory the live UI (tabs + stable lt-* hooks) -- start here, not blind
      python scripts/drive_dashboard.py --map

      # launch fake backend + fixture, screenshot the Detectors grid, tear down
      python scripts/drive_dashboard.py --launch --tab Detectors --screenshot out.png

The ``lt-*`` classes are the stable automation contract (see the rule file).
Plain Playwright CSS locators pierce the dashboard's open shadow DOM, so
``page.locator(".lt-tool-settings")`` works -- but **descendant combinators do
not cross shadow boundaries**. Target a workflow's button with a *compound*
selector on one element (``.lt-wf-total_counts.lt-tool-player-stop``), never a
descendant one (``.lt-wf-total_counts .lt-tool-player-stop`` matches nothing).
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError

from playwright.sync_api import (
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


class Dashboard:
    """Thin Playwright wrapper exposing the dashboard's stable automation hooks."""

    def __init__(self, page: Page):
        self.page = page

    @classmethod
    @contextmanager
    def connect(cls, url: str = DEFAULT_URL, *, settle_ms: int = SETTLE_MS):
        """Open a browser on a running dashboard, waiting for it to settle."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(settle_ms)
            try:
                yield cls(page)
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


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_until_ready(url: str, log: Path, *, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (URLError, ConnectionError, OSError):
            time.sleep(0.5)
    tail = "\n".join(log.read_text().splitlines()[-15:])
    raise TimeoutError(f"Dashboard at {url} not ready within {timeout_s}s.\n{tail}")


@contextmanager
def _fake_dashboard(instrument: str, port: int):
    """Launch a Kafka-free fake-backend dashboard seeded from the fixture.

    Copies the committed fixture to a writable scratch dir (the dashboard writes
    to its config dir), waits for readiness, and tears the server down on exit.
    """
    fixture = REPO_ROOT / "tests/dashboard/ui_config_fixtures" / instrument
    if not fixture.is_dir():
        raise SystemExit(f"No UI config fixture for instrument {instrument!r}")
    if _port_in_use(port):
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
                    "--no-fetch-announcements",
                ],
                cwd=REPO_ROOT,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
            try:
                _wait_until_ready(f"http://localhost:{port}", log)
                yield f"http://localhost:{port}"
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


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
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="For --launch.")
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
            with _fake_dashboard(args.instrument, args.port) as url:
                yield url
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
