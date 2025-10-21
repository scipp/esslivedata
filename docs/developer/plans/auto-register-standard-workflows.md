# Auto-register Standard Workflows

## Problem

Current refactor requires boilerplate in every instrument module:

```python
# specs.py - repeated everywhere
monitor_workflow_handle = register_monitor_workflow_specs(
    instrument=instrument, source_names=['monitor1', 'monitor2']
)
timeseries_workflow_handle = register_timeseries_workflow_specs(
    instrument=instrument, source_names=['motion1']
)

# factories.py - repeated everywhere
attach_monitor_workflow_factory(specs.monitor_workflow_handle)
attach_timeseries_workflow_factory(specs.timeseries_workflow_handle)
```

Additional issues:
- Detector names duplicated between specs.py and factories.py
- Monitor names not stored centrally on `Instrument`
- Timeseries names (f144_attribute_registry) exist but aren't used consistently
- **Import-time side effects**: factories.py executes expensive operations (workflow creation, detector setup) when imported
- **Unclear control flow**: Registry lookup (`instrument = instrument_registry['loki']`) couples factories.py to global state

## Solution

### 1. Centralize metadata on `Instrument`

```python
@dataclass(kw_only=True)
class Instrument:
    name: str
    detector_names: list[str] = field(default_factory=list)  # NEW - declared upfront
    monitors: list[str] = field(default_factory=list)        # NEW
    f144_attribute_registry: dict[str, dict] = field(...)    # existing - use for timeseries
    _detector_numbers: dict[str, sc.Variable] = field(...)   # existing - per-detector config
```

**Key change**: `detector_names` are declared in specs.py, not added dynamically.

### 2. Auto-register specs in `Instrument.__post_init__`

When monitors/f144_attribute_registry are set, automatically register standard workflow specs and store handles privately (`_monitor_workflow_handle`, `_timeseries_workflow_handle`).

### 3. Make `load_factories()` an Instrument method

Replace standalone function with instance method that manages factory initialization:

```python
class Instrument:
    def load_factories(self) -> None:
        """Load and initialize instrument-specific factories."""
        # Import instrument-specific factories module
        module = importlib.import_module(
            f'ess.livedata.config.instruments.{self.name}.factories'
        )

        # Auto-attach standard factories if specs were registered
        if hasattr(self, '_monitor_workflow_handle'):
            attach_monitor_workflow_factory(self._monitor_workflow_handle)
        if hasattr(self, '_timeseries_workflow_handle'):
            attach_timeseries_workflow_factory(self._timeseries_workflow_handle)

        # Call instrument-specific setup if defined
        if hasattr(module, 'setup_factories'):
            module.setup_factories(self)

        # Auto-load detector_numbers from nexus for unconfigured detectors
        for name in self.detector_names:
            if name not in self._detector_numbers:
                self._load_detector_from_nexus(name)
```

**Benefits**:
- Clean API: `instrument.load_factories()` instead of `load_factories('loki')`
- No registry lookup needed in factories.py
- Explicit control over when expensive operations happen

### 4. Rename `add_detector()` → `configure_detector()`

Current `add_detector()` is misleading - detector names are already declared. The method actually **configures** per-detector metadata:

```python
class Instrument:
    def configure_detector(
        self,
        name: str,
        detector_number: sc.Variable | None = None,
        *,
        detector_group_name: str | None = None,
    ) -> None:
        """Configure detector-specific metadata."""
        # Store explicit detector_number (NMX case: computed arrays)
        if detector_number is not None:
            self._detector_numbers[name] = detector_number
            return
        # Otherwise will be auto-loaded from nexus file later
```

**Use cases**:
- **Explicit configuration** (NMX): Provide `detector_number` as computed scipp array
- **Auto-loading** (LOKI): Omit `configure_detector()` call, let `load_factories()` load from nexus

### 5. Explicit `setup_factories()` in instrument modules

Replace import-time side effects with explicit function that receives instrument:

```python
# loki/factories.py - NO import-time side effects, just definitions

def setup_factories(instrument: Instrument) -> None:
    """Initialize LOKI-specific factories and workflows."""
    # Expensive operations happen here, not at import time
    base_workflow = loki.live._configured_Larmor_AgBeh_workflow()
    base_workflow[Filename[SampleRun]] = get_nexus_geometry_filename('loki')

    xy_projection = DetectorProjection(
        instrument=instrument,
        projection='xy_plane',
        pixel_noise='cylindrical',
        resolution={...},
    )

    # Register factories using locally created objects
    @specs.xy_projection_handles['xy_plane']['view'].attach_factory()
    def _xy_projection_view_factory(source_name: str, params: DetectorViewParams):
        return xy_projection.make_view(source_name, params=params)

    # ... other factory registrations
```

```python
# nmx/factories.py - configure detectors with explicit metadata

def setup_factories(instrument: Instrument) -> None:
    """Initialize NMX-specific factories and configure detectors."""
    # Configure detectors with computed detector_number arrays
    dim = 'detector_number'
    sizes = {'x': 1280, 'y': 1280}
    for panel in range(3):
        instrument.configure_detector(
            f'detector_panel_{panel}',
            detector_number=sc.arange(
                'detector_number', panel * 1280**2 + 1, (panel + 1) * 1280**2 + 1
            ).fold(dim=dim, sizes=sizes),
        )

    # Other expensive setup
    panels_view = DetectorLogicalView(
        instrument=instrument,
        config=LogicalViewConfig(...),
    )
    # Register factories...
```

### 6. Simplified instrument code

```python
# loki/specs.py
instrument = Instrument(
    name='loki',
    detector_names=[f'loki_detector_{i}' for i in range(9)],  # Declared upfront
    monitors=['monitor1', 'monitor2'],                         # Auto-registers specs
)
instrument_registry.register(instrument)

# loki/factories.py - NO import-time side effects
def setup_factories(instrument: Instrument) -> None:
    # Detectors auto-loaded from nexus - no configure_detector() needed

    # Expensive factory setup (only runs when load_factories() called)
    base_workflow = loki.live._configured_Larmor_AgBeh_workflow()
    xy_projection = DetectorProjection(instrument=instrument, ...)
    # Register factories...

# Usage (in service builder or tests)
instrument = instrument_registry['loki']
instrument.load_factories()  # Explicit, clean API
```

```python
# nmx/specs.py
instrument = Instrument(
    name='nmx',
    detector_names=['detector_panel_0', 'detector_panel_1', 'detector_panel_2'],
    monitors=['monitor1'],
)
instrument_registry.register(instrument)

# nmx/factories.py
def setup_factories(instrument: Instrument) -> None:
    # Explicit detector configuration with computed arrays
    for panel in range(3):
        instrument.configure_detector(
            f'detector_panel_{panel}',
            detector_number=sc.arange(...).fold(...),
        )
    # Register factories...
```

## Benefits

- **Eliminates boilerplate**: 2+ lines of standard factory attachment per instrument
- **No import-time side effects**: Expensive operations only run when explicitly triggered
- **Clean API**: `instrument.load_factories()` instead of standalone function
- **No registry lookup**: factories.py receives instrument directly as parameter
- **Explicit control flow**: Clear when and where expensive operations happen
- **Centralized metadata**: Detector/monitor/timeseries names defined once in specs.py
- **Flexible detector config**: Auto-load from nexus OR explicit configuration
- **Better Python style**: Import defines, method call executes
- **Preserves two-phase registration**: Lightweight specs, heavy factories loaded separately
- **Still explicit**: Declare monitors → get workflows automatically
