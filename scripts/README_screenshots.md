# Widget Screenshot Generator

## Overview

The `generate_widget_screenshots.py` script generates PNG screenshots of Panel ConfigurationWidget instances for all workflow specifications in a given instrument.

These screenshots are useful for:
- Documentation
- Stakeholder presentations
- Visual verification of widget layouts
- Comparing configurations across instruments

## Requirements

The script requires:
- `playwright` Python package (already in dev dependencies)
- Chromium browser for Playwright: `python -m playwright install chromium`
- System dependencies for Chromium (already in devcontainer)

## Usage

### Basic Usage

Generate screenshots for all workflows in the dummy instrument:

```bash
python scripts/generate_widget_screenshots.py dummy
```

This creates screenshots in `screenshots/dummy/` directory.

### Custom Output Directory

```bash
python scripts/generate_widget_screenshots.py dream --output-dir docs/screenshots
```

### Custom Viewport Width

```bash
# Narrow widgets (good for mobile preview)
python scripts/generate_widget_screenshots.py dummy --width 600

# Standard width (default)
python scripts/generate_widget_screenshots.py dummy --width 800

# Wide widgets (good for desktop)
python scripts/generate_widget_screenshots.py dummy --width 1200
```

### Custom Port

If port 5555 is already in use:

```bash
python scripts/generate_widget_screenshots.py dummy --port 5556
```

### Verbose Logging

For debugging:

```bash
python scripts/generate_widget_screenshots.py dummy --verbose
```

## Output

Screenshots are saved to:
```
{output_dir}/{instrument_name}/{workflow_id}.png
```

Where `workflow_id` has `/` characters replaced with `_`.

Example:
```
screenshots/
└── dummy/
    ├── dummy_data_reduction_total_counts_1.png
    ├── dummy_detector_data_panel_0_xy_1.png
    ├── dummy_monitor_data_monitor_histogram_1.png
    └── dummy_timeseries_timeseries_data_1.png
```

## How It Works

1. Starts a Panel server in the background on the specified port
2. Creates a ConfigurationWidget for the first workflow
3. Uses Playwright with Chromium to navigate to the server
4. For each workflow:
   - Updates the widget content dynamically
   - Waits for Panel to render
   - Takes a full-page screenshot
   - Saves to the output directory
5. Server shuts down when the script exits

This approach reuses a single server instance for all workflows, making it much faster than starting/stopping the server for each screenshot.

## Troubleshooting

### Port Already in Use

If you get an error about the port being in use:
```bash
python scripts/generate_widget_screenshots.py dummy --port 5556
```

### Playwright Not Installed

If you get errors about missing browsers:
```bash
python -m playwright install chromium
```

### Missing System Dependencies

If Chromium fails to launch, ensure the system dependencies are installed. These should already be in the devcontainer, but if running locally:
```bash
# Ubuntu/Debian
python -m playwright install --with-deps chromium
```
