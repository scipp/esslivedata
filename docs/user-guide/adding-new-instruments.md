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

ESSlivedata uses a **two-phase registration pattern** to separate lightweight specifications from heavy factory implementations:

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
    # Heavy imports go here (only loaded when needed)
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

- **Phase 1 (specs)**: Lightweight, always imported - provides metadata for UI generation
- **Phase 2 (factories)**: Heavy, loaded on-demand - contains actual workflow logic and imports
- This separation improves startup time and allows validation without loading all dependencies

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

In `factories.py`, configure detectors with `detector_number` arrays:

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

#### Logical View (2D detectors)

For regular 2D detectors or individual layers/slices of 3D detectors:

```python
from ess.livedata.handlers.detector_data_handler import DetectorLogicalView

# Create logical view
logical_view = DetectorLogicalView(instrument=instrument)

# Attach to spec handle from specs.py
from . import specs
specs.detector_view_handle.attach_factory()(logical_view.make_view)
```

#### Geometric Projections (complex geometries)

For detectors with complex 3D geometries, use helper functions to register projections:

```python
from ess.livedata.handlers.detector_view_specs import register_detector_view_spec

# XY plane projection (e.g., for endcap detectors)
xy_handle = register_detector_view_spec(
    instrument=instrument,
    projection='xy_plane',
    source_names=['endcap_backward_detector', 'endcap_forward_detector'],
)

# Cylindrical projection (e.g., for mantle detectors)
cylinder_handle = register_detector_view_spec(
    instrument=instrument,
    projection='cylinder_mantle_z',
    source_names=['mantle_detector'],
)
```

Available projections:
- `xy_plane`: 2D projection onto XY plane
- `cylinder_mantle_z`: Cylindrical projection (for detectors like DREAM's mantle)

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
import scipp as sc
from ess.livedata.config import Instrument
from . import specs

def setup_factories(instrument: Instrument) -> None:
    """Initialize dummy-specific factories and workflows."""
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

## Reference Implementations

For more complex examples, see existing instrument configurations in `src/ess/livedata/config/instruments/`:

- [dummy](../../src/ess/livedata/config/instruments/dummy/): Simple example with logical view
- [dream](../../src/ess/livedata/config/instruments/dream/): Complex setup with multiple detector types including cylindrical projection
- [loki](../../src/ess/livedata/config/instruments/loki/): Multiple detector panels with XY projections
- [bifrost](../../src/ess/livedata/config/instruments/bifrost/): Spectroscopy example with custom workflows
- [nmx](../../src/ess/livedata/config/instruments/nmx/): Crystallography with computed detector numbers
