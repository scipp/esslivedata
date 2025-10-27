#!/usr/bin/env python
"""
Generate screenshots of ConfigurationWidget for workflow specifications.

This script creates PNG screenshots of Panel ConfigurationWidget instances for all
workflow specifications in a given instrument. It uses a live Panel server and
Playwright with Chromium to capture the rendered widgets.

The screenshots are useful for documentation, stakeholder presentations, and
visual verification of widget layouts.

Requirements:
    - playwright (with chromium installed: python -m playwright install chromium)
    - System dependencies for chromium (see devcontainer.json)

Examples:
    # Generate screenshots for all dummy workflows
    python scripts/generate_widget_screenshots.py dummy

    # Generate screenshots for DREAM with custom output directory
    python scripts/generate_widget_screenshots.py dream --output-dir docs/screenshots

    # Generate with custom viewport width
    python scripts/generate_widget_screenshots.py dummy --width 1200

    # Use custom port (if 5555 is already in use)
    python scripts/generate_widget_screenshots.py dummy --port 5556
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import panel as pn
from playwright.sync_api import sync_playwright

from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.dashboard.widgets.configuration_widget import ConfigurationWidget
from ess.livedata.dashboard.workflow_configuration_adapter import (
    WorkflowConfigurationAdapter,
)

# Initialize Panel
pn.extension()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to hold current widget for server
current_widget = None


def create_widget_app(
    instrument_name: str, workflow_id: str, select_all_sources: bool = True
) -> pn.Column:
    """
    Create a Panel app for the ConfigurationWidget.

    Parameters
    ----------
    instrument_name
        Name of the instrument (e.g., 'dummy', 'dream', 'bifrost')
    workflow_id
        Workflow identifier
    select_all_sources
        If True, automatically select all available source names in the widget

    Returns
    -------
    :
        Panel Column widget containing the configuration interface
    """
    # Load the instrument config to register workflows
    _ = get_config(instrument_name)
    instrument = instrument_registry[instrument_name]

    # Get the workflow spec
    spec = instrument.workflow_factory[workflow_id]

    # Create adapter (dummy start callback)
    adapter = WorkflowConfigurationAdapter(
        spec=spec,
        persistent_config=None,
        start_callback=lambda *args, **kwargs: None,
    )

    # Create ConfigurationWidget
    widget = ConfigurationWidget(config=adapter)

    # Auto-select all source names if requested
    if select_all_sources and widget._source_selector is not None:
        all_sources = widget._source_selector.options
        if all_sources:
            widget._source_selector.value = list(all_sources)

    return widget.widget


def update_widget(
    instrument_name: str, workflow_id: str, select_all_sources: bool = True
) -> pn.Column:
    """
    Update the global widget with a new configuration.

    This allows reusing the same Panel server for multiple workflows.

    Parameters
    ----------
    instrument_name
        Name of the instrument
    workflow_id
        Workflow identifier
    select_all_sources
        If True, automatically select all available source names in the widget

    Returns
    -------
    :
        Updated Panel widget
    """
    global current_widget
    current_widget = create_widget_app(instrument_name, workflow_id, select_all_sources)
    return current_widget


def get_current_widget() -> pn.Column:
    """
    Get the current widget (used by Panel server).

    Returns
    -------
    :
        Current Panel widget
    """
    return current_widget


def generate_screenshots_for_instrument(
    instrument_name: str,
    output_dir: Path,
    port: int = 5555,
    width: int = 800,
    select_all_sources: bool = True,
) -> None:
    """
    Generate screenshots for all workflows of an instrument.

    This function:
    1. Starts a Panel server with the first workflow
    2. Iterates through all workflows, updating the server content
    3. Uses Playwright to screenshot each workflow configuration
    4. Saves screenshots to {output_dir}/{instrument_name}/{workflow_id}.png

    Parameters
    ----------
    instrument_name
        Name of the instrument (e.g., 'dummy', 'dream')
    output_dir
        Base directory for saving screenshots
    port
        Port for the Panel server (default: 5555)
    width
        Viewport width in pixels (default: 800)
    select_all_sources
        If True, automatically select all available source names before screenshot
    """
    logger.info("Generating screenshots for instrument: %s", instrument_name)

    # Load the instrument config to get all workflows
    try:
        _ = get_config(instrument_name)
    except Exception as e:
        logger.error("Failed to load config for %s: %s", instrument_name, e)
        return

    instrument = instrument_registry[instrument_name]
    workflow_ids = list(instrument.workflow_factory.keys())

    if not workflow_ids:
        logger.warning("No workflows found for %s", instrument_name)
        return

    logger.info("Found %d workflows", len(workflow_ids))

    # Create initial widget
    update_widget(instrument_name, workflow_ids[0], select_all_sources)

    # Start Panel server in background thread
    logger.info("Starting Panel server on port %d...", port)
    server_thread = threading.Thread(
        target=lambda: pn.serve(
            {'/': get_current_widget},
            port=port,
            show=False,
            verbose=False,
            start=True,
            title="ConfigurationWidget Screenshot",
        ),
        daemon=True,
    )
    server_thread.start()

    # Give server time to start
    time.sleep(3)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': width, 'height': 800})

            for i, workflow_id in enumerate(workflow_ids, 1):
                logger.info(
                    "  [%d/%d] Generating screenshot for %s...",
                    i,
                    len(workflow_ids),
                    workflow_id,
                )

                # Update the widget
                try:
                    update_widget(instrument_name, workflow_id, select_all_sources)
                except Exception as e:
                    logger.error("    Failed to create widget: %s", e)
                    continue

                # Navigate to the server (or reload)
                try:
                    page.goto(f'http://localhost:{port}', wait_until='networkidle')
                    # Give extra time for Panel to render
                    page.wait_for_timeout(2000)
                except Exception as e:
                    logger.error("    Failed to load page: %s", e)
                    continue

                # Create output path
                # Use workflow_id as filename, replacing / with _
                safe_filename = str(workflow_id).replace('/', '_')
                output_path = output_dir / instrument_name / f"{safe_filename}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Take screenshot
                try:
                    page.screenshot(path=str(output_path), full_page=True)
                    logger.info("    ✓ Saved to: %s", output_path)
                except Exception as e:
                    logger.error("    Failed to save screenshot: %s", e)
                    continue

            browser.close()
        logger.info(
            "✓ Generated %d screenshots for %s",
            len(workflow_ids),
            instrument_name,
        )
    except Exception as e:
        logger.error("Error during screenshot generation: %s", e)
        raise
    finally:
        # Server will shut down when script exits (daemon thread)
        pass


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'instrument',
        choices=available_instruments(),
        help='Instrument name to generate screenshots for',
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
        default=5555,
        help='Port for Panel server (default: 5555)',
    )

    parser.add_argument(
        '--width',
        type=int,
        default=800,
        help='Viewport width in pixels (default: 800)',
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging',
    )

    parser.add_argument(
        '--no-select-all',
        action='store_true',
        help='Do not auto-select all source names (use default selection)',
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        generate_screenshots_for_instrument(
            instrument_name=args.instrument,
            output_dir=args.output_dir,
            port=args.port,
            width=args.width,
            select_all_sources=not args.no_select_all,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
