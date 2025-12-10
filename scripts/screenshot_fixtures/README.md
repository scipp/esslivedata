# Screenshot Fixtures

This directory contains fixture configurations and data generators for screenshot testing.
The system allows testing the dashboard with pre-configured plots and synthetic data,
without requiring Kafka.

## Directory Structure

```
screenshot_fixtures/
├── __init__.py              # Core registry and injection logic
├── README.md                # This file
└── <instrument>/            # Per-instrument fixtures
    ├── __init__.py          # Data generators (register with @fixture_registry.register)
    ├── workflow_configs.yaml # Workflow configurations with fixed job_numbers
    └── plot_configs.yaml    # Plot grid configurations
```

## How It Works

1. **workflow_configs.yaml** defines workflows with fixed job_numbers (UUIDs)
2. **plot_configs.yaml** defines plot grids that subscribe to those workflows
3. **Data generators** are Python functions that create `sc.DataArray` test data
4. The **fixture registry** matches generators to workflows by workflow_id
5. **inject_fixtures()** reads configs, calls generators, and POSTs data to the dashboard

## Adding a New Screenshot Scenario

### Step 1: Define the workflow in workflow_configs.yaml

```yaml
# Use a fixed UUID so data injection matches
dummy/detector_data/my_workflow/1:
  source_names:
    - my_detector
  params: {}
  aux_source_names: {}
  current_job:
    job_number: "00000000-0000-0000-0000-000000000003"
    jobs:
      my_detector:
        params: {}
        aux_source_names: {}
```

### Step 2: Add the plot to plot_configs.yaml

```yaml
plot_grids:
  grids:
  - title: My Plot
    nrows: 1
    ncols: 1
    cells:
    - geometry:
        row: 0
        col: 0
      config:
        workflow_id: dummy/detector_data/my_workflow/1
        output_name: current
        source_names:
        - my_detector
        plot_name: image
        params:
          # ... plot parameters
```

### Step 3: Register a data generator

In `<instrument>/__init__.py`:

```python
from screenshot_fixtures import fixture_registry
import numpy as np
import scipp as sc

# Example: 2D image data (for plot_name: image)
@fixture_registry.register('dummy/detector_data/my_workflow/1')
def make_my_detector_data() -> sc.DataArray:
    """Create 2D test data for an image plot."""
    data = np.random.default_rng(42).poisson(100, size=(64, 64)).astype(float)
    return sc.DataArray(
        sc.array(dims=['y', 'x'], values=data, unit='counts'),
        coords={
            'x': sc.arange('x', 0.0, 64.0, unit=None),
            'y': sc.arange('y', 0.0, 64.0, unit=None),
        },
    )

# Example: 1D timeseries data (for plot_name: line)
@fixture_registry.register('dummy/data_reduction/my_timeseries/1')
def make_my_timeseries_data() -> sc.DataArray:
    """Create 1D test data for a line plot."""
    time = np.linspace(0, 100, 100)
    counts = 500 * np.exp(-time / 30) + 50
    return sc.DataArray(
        sc.array(dims=['time'], values=counts, unit='counts'),
        coords={
            'time': sc.array(dims=['time'], values=time, unit='s'),
        },
    )
```

### Step 4: Import the fixture module

In `screenshot_dashboard.py`, ensure the instrument's fixture module is imported:

```python
import screenshot_fixtures.<instrument>  # noqa: F401
```

## Running Screenshots

```bash
python scripts/screenshot_dashboard.py --output-dir screenshots/ --instrument dummy
```

## Key Points

- **Single source of truth**: Job numbers are defined only in workflow_configs.yaml
- **No boilerplate**: Just write a function that returns a DataArray and decorate it
- **Automatic matching**: The registry matches generators to workflows by workflow_id string
- **Extensible**: Add new instruments by creating a new subdirectory with the same structure
