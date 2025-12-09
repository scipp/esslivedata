#!/usr/bin/env python
"""
Take screenshots of the dashboard for visual verification.

This script starts the dashboard with HTTP transport (no Kafka needed),
loads pre-configured workflows and plot grids from fixtures, injects
test data via the HTTP API, and captures screenshots using Playwright.

Requirements:
    - playwright (with chromium installed: python -m playwright install chromium)

Usage:
    python scripts/screenshot_dashboard.py --output-dir screenshots/

    # With specific viewport width
    python scripts/screenshot_dashboard.py --output-dir screenshots/ --width 1400

How it works:
    1. Sets LIVEDATA_CONFIG_DIR to point to fixture configs
    2. Starts dashboard with --transport=http (no Kafka needed)
    3. Fixture configs restore:
       - Workflows with known job_numbers (from workflow_configs.yaml)
       - Plot grids already subscribed to those workflows (from plot_grids.yaml)
    4. Injects test data with matching ResultKeys
    5. Captures screenshots via Playwright
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timing constants
PAGE_RENDER_WAIT_MS = 3000  # milliseconds
DATA_INJECTION_WAIT_MS = 2000  # wait for data to propagate to plots


def wait_for_dashboard(port: int, timeout: float = 30.0) -> bool:
    """Wait for dashboard to be ready."""
    import urllib.request

    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(
                f'http://localhost:{port}/', timeout=1.0
            ) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def start_dashboard(instrument: str, port: int, config_dir: Path) -> subprocess.Popen:
    """Start the dashboard in a subprocess with fixture config dir."""
    env = os.environ.copy()
    env['LIVEDATA_CONFIG_DIR'] = str(config_dir)

    cmd = [
        sys.executable,
        '-m',
        'ess.livedata.dashboard.reduction',
        '--instrument',
        instrument,
        '--transport',
        'http',
        '--log-level',
        'INFO',
    ]
    logger.info("Starting dashboard: %s", ' '.join(cmd))
    logger.info("LIVEDATA_CONFIG_DIR=%s", config_dir)
    return subprocess.Popen(  # noqa: S603
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )


def capture_screenshots(
    port: int,
    output_dir: Path,
    width: int = 1200,
    inject_data: callable | None = None,
) -> list[Path]:
    """Capture screenshots of the dashboard using Playwright."""
    from playwright.sync_api import sync_playwright

    output_dir.mkdir(parents=True, exist_ok=True)
    screenshots = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': width, 'height': 900})

        # Navigate to dashboard
        logger.info("Loading dashboard...")
        page.goto(f'http://localhost:{port}', wait_until='networkidle')
        # Wait for periodic callback to start (triggered by browser connection)
        page.wait_for_timeout(500)

        # Inject data NOW - after browser connects (so periodic callbacks run)
        # but before navigating to plot tabs
        if inject_data is not None:
            logger.info("Injecting test data after browser connection...")
            inject_data(port)
            # Wait for data to be consumed and plots to be created
            page.wait_for_timeout(PAGE_RENDER_WAIT_MS)
        else:
            page.wait_for_timeout(PAGE_RENDER_WAIT_MS)

        # Take screenshot of initial state (Jobs tab)
        screenshot_path = output_dir / 'dashboard_jobs_tab.png'
        page.screenshot(path=str(screenshot_path), full_page=True)
        logger.info("Saved: %s", screenshot_path)
        screenshots.append(screenshot_path)

        # Click on "Manage Plots" tab
        manage_tab = page.locator('text=Manage Plots')
        if manage_tab.count() > 0:
            manage_tab.click()
            page.wait_for_timeout(1000)

            screenshot_path = output_dir / 'dashboard_manage_tab.png'
            page.screenshot(path=str(screenshot_path), full_page=True)
            logger.info("Saved: %s", screenshot_path)
            screenshots.append(screenshot_path)

        # Look for existing grid tabs loaded from config
        # Panel tabs use a specific structure - look for the clickable tab element
        # The tabs appear as divs with the title text, not as proper tab roles
        detectors_tab = page.locator('.bk-tab:has-text("Detectors")')
        if detectors_tab.count() > 0:
            logger.info("Found 'Detectors' grid tab, clicking...")
            detectors_tab.click()
            page.wait_for_timeout(DATA_INJECTION_WAIT_MS)

            screenshot_path = output_dir / 'dashboard_detectors_grid.png'
            page.screenshot(path=str(screenshot_path), full_page=True)
            logger.info("Saved: %s", screenshot_path)
            screenshots.append(screenshot_path)
        else:
            logger.info("No 'Detectors' tab found - grid may not have been loaded")

        browser.close()

    return screenshots


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('screenshots'),
        help='Output directory for screenshots (default: screenshots/)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5009,
        help='Dashboard port (default: 5009)',
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1200,
        help='Viewport width in pixels (default: 1200)',
    )
    parser.add_argument(
        '--instrument',
        type=str,
        default='dummy',
        help='Instrument name (default: dummy)',
    )

    args = parser.parse_args()

    # Get the fixtures directory (relative to this script)
    script_dir = Path(__file__).parent
    fixtures_dir = script_dir / 'screenshot_fixtures'

    if not fixtures_dir.exists():
        logger.error("Fixtures directory not found: %s", fixtures_dir)
        return 1

    # Start dashboard with fixture config dir
    process = start_dashboard(args.instrument, args.port, fixtures_dir)

    try:
        # Wait for dashboard to be ready
        logger.info("Waiting for dashboard to start...")
        if not wait_for_dashboard(args.port, timeout=30.0):
            logger.error("Dashboard failed to start")
            # Print any stderr output
            stderr = process.stderr.read().decode('utf-8', errors='replace')
            if stderr:
                logger.error("Dashboard stderr:\n%s", stderr)
            process.terminate()
            return 1

        logger.info("Dashboard is ready")

        # Give a moment for configs to load
        time.sleep(1)

        # Import the data injection function from fixtures
        # Data is injected INSIDE capture_screenshots after the browser connects
        sys.path.insert(0, str(script_dir))
        from screenshot_fixtures import inject_screenshot_data

        # Capture screenshots (data is injected after browser connects)
        screenshots = capture_screenshots(
            port=args.port,
            output_dir=args.output_dir,
            width=args.width,
            inject_data=inject_screenshot_data,
        )

        logger.info("Screenshots saved to %s", args.output_dir)
        logger.info("Captured %d screenshot(s)", len(screenshots))
        return 0

    finally:
        # Clean up
        logger.info("Shutting down dashboard...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == '__main__':
    sys.exit(main() or 0)
