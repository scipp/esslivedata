# Adding New Instruments

This guide explains how to add support for a new instrument in ESSlivedata.

## Required Steps

1. Create a new instrument package in `src/ess/livedata/config/instruments/<instrument>/`
   - The directory name will be used as the instrument identifier
2. Create and register an `Instrument` instance (imported via the package)
3. Provide `stream_mapping` accessible from the package namespace
4. Optionally provide a `setup_factories(instrument)` function to attach workflow factories
5. Optionally provide `detector_fakes` configuration for development/testing

## Package Structure

The minimal requirements are:
- **Required**: A package with `__init__.py`
- **Required**: `stream_mapping` dict accessible from the package namespace
- **Required**: An `Instrument` instance registered with `instrument_registry`
- **Optional**: `setup_factories(instrument)` function accessible from the package
- **Optional**: `detector_fakes` dict accessible from the package (for fake data generation)

The typical convention uses this structure (but you can organize differently):

```
src/ess/livedata/config/instruments/<instrument>/
├── __init__.py          # Imports and re-exports for package namespace
├── specs.py             # Instrument instance and workflow spec registration
├── views.py             # Detector view transform functions (optional)
├── streams.py           # Stream mappings (typically defines stream_mapping and detector_fakes)
└── factories.py         # Factory implementations (setup_factories function)
```

**Note**: The file organization (`specs.py`, `streams.py`, `factories.py`) is a convention, not a requirement. Large instruments may split factories into multiple files, combine specs and streams, etc. The system only requires that:
- The package imports create an `Instrument` and register it
- `stream_mapping` is accessible from the package namespace (e.g., imported in `__init__.py`)
- If factories are needed, `setup_factories` is accessible from the package namespace
- If using fake data generators, `detector_fakes` is accessible from the package namespace

## Creating the Instrument instance

In `specs.py` (or any module imported by `__init__.py`), create and register the instrument:

```python
from ess.livedata.config import Instrument, instrument_registry

instrument = Instrument(
    name='instrument_name',
    detector_names=['detector1', 'detector2'],
    monitors=['monitor1', 'monitor2'],
    f144_attribute_registry={'motion1': {'units': 'mm'}},
)

# Register with global registry
instrument_registry.register(instrument)
```

The variable name `instrument` is conventional but not required.

## Two-Phase Registration Pattern

ESSlivedata uses a **two-phase registration pattern** to separate specifications with minimal dependencies from factory implementations:

### Phase 1: Register Specs (in `specs.py`)

Register workflow specifications with explicit metadata and return a handle:

```python
# Register a detector view spec
view_handle = instrument.register_spec(
    namespace='detector_data',
    name='detector1_xy',
    version=1,
    title='Detector 1 XY View',
    description='2D view of detector 1',
    source_names=['detector1'],
    params=DetectorViewParams,
)

# Register a data reduction workflow spec
workflow_handle = instrument.register_spec(
    namespace='data_reduction',
    name='my_workflow',
    version=1,
    title='My Workflow',
    description='Reduces data to I(Q)',
    source_names=['detector1', 'detector2'],
    params=MyWorkflowParams,
    outputs=MyWorkflowOutputs,
)
```

### Phase 2: Attach Factories (in `factories.py` or elsewhere)

Use the handle to attach the actual factory implementation. This must be done in a function called `setup_factories` that's accessible from your package namespace:

```python
def setup_factories(instrument: Instrument) -> None:
    """Initialize instrument-specific factories and workflows."""
    # Lazy imports go here (only loaded when needed)
    from ess.reduce.nexus import load_detector

    # Import the handle from wherever specs were registered
    from . import specs

    specs.view_handle.attach_factory()(make_view_function)

    @specs.workflow_handle.attach_factory()
    def make_workflow():
        # Create and return workflow
        return StreamProcessorWorkflow(...)
```

The `setup_factories` function will be called automatically by `Instrument.load_factories()` if it exists in the package namespace.

### Why Two Phases?

- **Phase 1 (specs)**: Always imported - provides metadata for UI generation and configuration discovery
- **Phase 2 (factories)**: Loaded on-demand - contains actual workflow logic and implementation
- This separation ensures the dashboard and other spec consumers don't depend on instrument-specific packages

### Critical: Lazy Import Requirements

**IMPORTANT**: The instrument package must be importable without importing other packages from the `ess` namespace (such as `ess.reduce`, `ess.dream`, `ess.powder`, `ess.sans`, `ess.loki`, etc.). All such imports must be done lazily inside the `setup_factories()` function.

**Correct pattern** (instrument packages imported inside `setup_factories`):
```python
# factories.py
def setup_factories(instrument: Instrument) -> None:
    """Initialize instrument-specific factories and workflows."""
    # Instrument package imports go here
    from ess.dream import DreamPowderWorkflow
    from ess.powder import types as powder_types
    from ess.livedata.handlers.detector_data_handler import DetectorProjection
    # ... rest of factory setup
```

**Incorrect pattern** (importing at module level):
```python
# factories.py - DON'T DO THIS!
from ess.dream import DreamPowderWorkflow  # BAD: imported at module level
from ess.powder import types as powder_types  # BAD: imported at module level

def setup_factories(instrument: Instrument) -> None:
    """Initialize instrument-specific factories and workflows."""
    # ...
```

**Why this matters**:
- The dashboard and other spec consumers need to import instrument packages to read metadata (workflow parameters, output types, etc.)
- These consumers should not be forced to install or import instrument-specific packages like `essdiffraction`, `esssans`, etc.
- Specs are imported during configuration discovery and registry initialization
- Factory implementations are only loaded when actually running the workflows

**Which imports are allowed at module level**:
- Configuration and type definitions: `ess.livedata.config.*`, `ess.livedata.parameter_models.*`
- Standard library and common dependencies: `scipp`, `pydantic`, `typing`
- Imports from your own instrument package: `from . import specs`

**Which imports must be lazy (inside `setup_factories`)**:
- Any imports from other `ess.*` packages: `ess.dream`, `ess.powder`, `ess.sans`, `ess.loki`, etc.
- Workflow implementation modules: `ess.livedata.handlers.*` (detector handlers, workflow factories, etc.) as those depend on `ess.reduce`.
- Workflow framework: `sciline`
- Geometry and data loading: `scippnexus`, etc.

## Stream Configuration

In `streams.py`, define stream mappings and optionally fake detector configuration:

```python
from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping

# Fake detector configuration for development (optional)
detector_fakes = {
    'panel_a': (1, 128**2),              # (first_id, last_id)
    'panel_b': (128**2 + 1, 2 * 128**2),
}

def _make_instrument_detectors() -> StreamLUT:
    """Define detector stream mappings for production."""
    return {
        InputStreamKey(topic='instrument_detector', source_name='panel_a'): 'panel_a',
        InputStreamKey(topic='instrument_detector', source_name='panel_b'): 'panel_b',
    }

# Stream mappings for different environments
stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'instrument_name',
        detector_names=list(detector_fakes)
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='instrument_name'),
        detectors=_make_instrument_detectors(),
    ),
}
```

### Fake Detectors

To enable development without real detector data, define pixel ID ranges in the `detector_fakes` dictionary:

- The pixel IDs should match your detector configuration and must not overlap
- The fake detector service will generate random events within these ID ranges
- Use these when running services with the `--dev` flag
- This configuration is optional and only needed if you plan to use fake data generators

## Detector Configuration in Factories

In `factories.py`, configure detectors with `detector_number` arrays.
This is optional, if `configure_detector` is not called the `detector_number` array will be loaded from the NeXus geometry file.

```python
def setup_factories(instrument: Instrument) -> None:
    """Initialize instrument-specific factories and workflows."""
    import scipp as sc

    # Configure detector with explicit detector_number
    instrument.configure_detector(
        'panel_a',
        detector_number=sc.arange('yx', 1, 128**2 + 1, unit=None).fold(
            dim='yx', sizes={'y': 128, 'x': 128}
        ),
    )
```

### Detector View Registration

ESSlivedata supports different detector view projections:

#### Logical Views (2D/3D detectors with transforms)

For detectors that need custom transforms (folding, slicing, renaming dimensions), use `instrument.add_logical_view()` in `specs.py`. This registers both the spec and transform, with the factory auto-attached during `load_factories()`:

```python
# In specs.py
from .views import get_detector_view

instrument.add_logical_view(
    name='detector_xy',
    title='Detector XY View',
    description='2D view of detector counts',
    source_names=['detector1'],
    transform=get_detector_view,
    roi_support=True,  # Enable ROI selection (default: True)
)
```

The transform function goes in a separate `views.py` to keep `specs.py` lightweight:

```python
# In views.py
import scipp as sc

def get_detector_view(da: sc.DataArray) -> sc.DataArray:
    """Transform detector data to 2D view."""
    return da.fold(dim='detector_number', sizes={'y': 128, 'x': 128})
```

For downsampling views that need proper ROI index mapping, use `reduction_dim`:

```python
# In views.py
def fold_image(da: sc.DataArray) -> sc.DataArray:
    """Fold for downsampling - don't sum here, use reduction_dim instead."""
    da = da.rename_dims({'dim_0': 'x', 'dim_1': 'y'})
    da = da.fold(dim='x', sizes={'x': 512, 'x_bin': -1})
    da = da.fold(dim='y', sizes={'y': 512, 'y_bin': -1})
    return da

# In specs.py
instrument.add_logical_view(
    name='detector_downsampled',
    title='Detector (512x512)',
    description='Downsampled detector view',
    source_names=['detector1'],
    transform=fold_image,
    reduction_dim=['x_bin', 'y_bin'],  # Dimensions to sum over
)
```

#### Geometric Projections (complex geometries)

For detectors with complex 3D geometries, use helper functions to register projections:

```python
from ess.livedata.handlers.detector_view_specs import register_detector_view_spec

# Single projection for all detectors
xy_handle = register_detector_view_spec(
    instrument=instrument,
    projection='xy_plane',
    source_names=['detector_0', 'detector_1'],
)

# Mixed projections - different projection types per detector
# Creates a unified "Detector Projection" workflow
# source_names defaults to the dict keys
projection_handle = register_detector_view_spec(
    instrument=instrument,
    projection={
        'mantle_detector': 'cylinder_mantle_z',
        'endcap_backward_detector': 'xy_plane',
        'endcap_forward_detector': 'xy_plane',
    },
)
```

Available projections:
- `xy_plane`: 2D projection onto XY plane
- `cylinder_mantle_z`: Cylindrical projection (for detectors like DREAM's mantle)

When using a dict for `projection`, each detector can use a different projection type,
but they will all appear under a single "Detector Projection" workflow in the UI.

Geometric projections are towards the sample position (assumed at the origin).
For more details see [ess.reduce.live.raw](https://scipp.github.io/essreduce/generated/modules/ess.reduce.live.raw.html).

### Geometry Files

Geometry files are needed when:
- Using geometric projections (`xy_plane`, `cylinder_mantle_z`)
- Loading `detector_number` from NeXus (if not provided explicitly)

If you configure `detector_number` explicitly via `configure_detector()`, no geometry file is needed.

To provide a geometry file:

1. Create a NeXus geometry file following the naming convention: `geometry-<instrument>-<date>.nxs`
   - Use `ess-livedata-make-geometry-nexus` to create from a regular NeXus file
   - The date should be the first date the geometry file is used in production
2. Add the file's MD5 hash to the `_registry` in [detector_data_handler.py](../../src/ess/livedata/handlers/detector_data_handler.py)
3. Upload the file to https://public.esss.dk/groups/scipp/beamlime/geometry/

Multiple geometry files can exist for an instrument (for different time periods), but only one is active at a time.

## Complete Example: Dummy Instrument

Here's the complete structure for the dummy instrument as a reference:

### `__init__.py`

```python
from .factories import setup_factories
from .streams import detector_fakes, stream_mapping

__all__ = ['detector_fakes', 'setup_factories', 'stream_mapping']
```

### `specs.py`

```python
from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.handlers.detector_view_specs import DetectorViewParams

# Create instrument
instrument = Instrument(
    name='dummy',
    detector_names=['panel_0'],
    monitors=['monitor1', 'monitor2'],
    f144_attribute_registry={'motion1': {'units': 'mm'}},
)

# Register instrument
instrument_registry.register(instrument)

# Register detector view spec
panel_0_view_handle = instrument.register_spec(
    namespace='detector_data',
    name='panel_0_xy',
    version=1,
    title='Panel 0',
    source_names=['panel_0'],
    params=DetectorViewParams,
)

# Register data reduction workflow spec
total_counts_handle = instrument.register_spec(
    name='total_counts',
    version=1,
    title='Total counts',
    description='Dummy workflow that computes total counts.',
    source_names=['panel_0'],
)
```

### `streams.py`

```python
from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping
from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping

detector_fakes = {'panel_0': (1, 128**2)}

def _make_dummy_detectors() -> StreamLUT:
    return {InputStreamKey(topic='dummy_detector', source_name='panel_0'): 'panel_0'}

stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping('dummy', detector_names=['panel_0']),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='dummy'),
        detectors=_make_dummy_detectors(),
    ),
}
```

### `factories.py`

```python
# Only config and common dependencies at module level
import scipp as sc
from ess.livedata.config import Instrument
from . import specs

def setup_factories(instrument: Instrument) -> None:
    """Initialize dummy-specific factories and workflows."""
    # Instrument packages and implementation imports go here
    import sciline
    from ess.livedata.handlers.detector_data_handler import DetectorLogicalView
    from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow

    # Configure detector
    instrument.configure_detector(
        'panel_0',
        detector_number=sc.arange('yx', 1, 128**2 + 1, unit=None).fold(
            dim='yx', sizes={'y': -1, 'x': 128}
        ),
    )

    # Attach detector view factory
    logical_view = DetectorLogicalView(instrument=instrument)
    specs.panel_0_view_handle.attach_factory()(logical_view.make_view)

    # Attach workflow factory
    @specs.total_counts_handle.attach_factory()
    def make_total_counts_workflow():
        # Define workflow implementation
        def total_counts(events):
            return events.sum()

        workflow = sciline.Pipeline((total_counts,))
        return StreamProcessorWorkflow(
            base_workflow=workflow,
            dynamic_keys={'panel_0': Events},
            target_keys={'total_counts': TotalCounts},
            accumulators=(TotalCounts,),
        )
```

## Grid Templates (Optional)

Grid templates are pre-configured plot grid layouts that users can select when creating a new grid in the dashboard. They provide a convenient starting point with commonly used workflow configurations.

### Creating Grid Templates

The easiest way to create a grid template is to design it interactively in the dashboard:

1. **Create and configure a grid in the UI**: Start the dashboard for your instrument, create a new grid, and configure it with the desired layout and workflow subscriptions.

2. **Extract the grid configuration**: The dashboard persists grid configurations to `~/.config/esslivedata/<instrument>/plot_configs.yaml`. Open this file and find the grid you created under the `plot_grids.grids` key.

3. **Create the template file**: Copy the grid configuration to a new file in your instrument's `grid_templates/` directory:

```
src/ess/livedata/config/instruments/<instrument>/
├── __init__.py
├── specs.py
├── streams.py
├── factories.py
└── grid_templates/
    └── detector_overview.yaml
```

Template files in `grid_templates/` are automatically included as package data.

Templates use the same format as persisted grids. You can optionally add a `description` field that will be shown in the template selector.

Templates are loaded once at dashboard startup. When a user selects a template, they can adjust the grid size before creating the grid. The minimum grid size is determined by the cells in the template.

## Reference Implementations

For more complex examples, see existing instrument configurations in `src/ess/livedata/config/instruments/`.