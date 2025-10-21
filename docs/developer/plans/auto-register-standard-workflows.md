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

## Solution

### 1. Centralize metadata on `Instrument`

```python
@dataclass(kw_only=True)
class Instrument:
    name: str
    detectors: list[str] = field(default_factory=list)     # NEW
    monitors: list[str] = field(default_factory=list)      # NEW
    f144_attribute_registry: dict[str, dict] = field(...)  # existing - use for timeseries
```

### 2. Auto-register specs in `Instrument.__post_init__`

When monitors/f144_attribute_registry are set, automatically register standard workflow specs and store handles privately (`_monitor_workflow_handle`, `_timeseries_workflow_handle`).

### 3. Auto-attach factories in `load_factories()`

After importing `{instrument}.factories` module, check for stored handles and auto-attach standard factories:

```python
def load_factories(instrument: str) -> None:
    importlib.import_module(f'.{instrument}.factories', __package__)

    # Auto-attach standard factories based on registered specs
    inst = instrument_registry[instrument]
    if hasattr(inst, '_monitor_workflow_handle'):
        attach_monitor_workflow_factory(inst._monitor_workflow_handle)
    if hasattr(inst, '_timeseries_workflow_handle'):
        attach_timeseries_workflow_factory(inst._timeseries_workflow_handle)
```

### 4. Simplified instrument code

```python
# loki/specs.py
instrument = Instrument(
    name='loki',
    detectors=[f'loki_detector_{i}' for i in range(9)],  # centralized
    monitors=['monitor1', 'monitor2'],                   # auto-registers specs
)
instrument_registry.register(instrument)

# loki/factories.py
instrument = instrument_registry['loki']

# Use centralized detector names
for name in instrument.detector_names:
    instrument.add_detector(name)

# Monitor/timeseries factories auto-attached - no code needed!
```

## Benefits

- Eliminates 2+ lines of boilerplate per instrument
- Detector/monitor/timeseries names defined once
- Preserves two-phase registration (lightweight specs, heavy factories)
- `load_factories()` is the perfect hook (already called by all services)
- Still explicit (declare monitors → get workflows) but automatic
