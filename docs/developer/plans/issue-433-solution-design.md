# Issue #433: Solution Design

## Core Insight

**Pydantic parameter models are already independent of heavy instrument packages.**

The problem: workflow factories (which use `ess.loki`, `ess.reduce`, etc.) and specs (which the frontend needs) are currently registered together. This forces frontend to import heavy dependencies.

The solution: **Two-phase registration with handle-based linking.**

## Solution Overview

Split registration into two phases:
1. **Spec registration** (lightweight) → returns a `SpecHandle`
2. **Factory attachment** (heavy) → uses the handle to attach implementation

### API Design

**Note**: `workflow_factory.py` stays as one module - it's already lightweight (no `ess.reduce`, `ess.loki`, etc. imports). The heavy dependencies are in:
- `loki.py`, `dream.py`, etc. (import `ess.loki`, `ess.reduce`, `ess.sans`)
- `detector_data_handler.py` (imports `ess.reduce.live.raw`)

These are the modules we split.

#### WorkflowFactory Changes

Add in `src/ess/livedata/handlers/workflow_factory.py`:

```python
@dataclass(frozen=True)
class SpecHandle:
    """Handle for attaching factories to registered specs."""
    workflow_id: WorkflowId
    _factory: WorkflowFactory  # Reference back to factory

    def attach_factory(self) -> Callable[[Callable[..., Workflow]], Callable[..., Workflow]]:
        """Decorator to attach factory implementation to this spec."""
        return self._factory.attach_factory(self.workflow_id)


class WorkflowFactory:
    def register_spec(self, spec: WorkflowSpec) -> SpecHandle:
        """
        Register workflow spec, return handle for later factory attachment.

        Validates spec_id uniqueness and stores spec.
        Returns SpecHandle for attaching factory later.
        """
        spec_id = spec.get_id()
        if spec_id in self._workflow_specs:
            raise ValueError(f"Workflow spec '{spec_id}' already registered.")
        self._workflow_specs[spec_id] = spec
        return SpecHandle(workflow_id=spec_id, _factory=self)

    def attach_factory(self, workflow_id: WorkflowId) -> decorator:
        """
        Decorator to attach factory to a previously registered spec.

        Validates:
        - Spec exists
        - Factory's params type hint matches spec.params (using `is not`)
        """
        if workflow_id not in self._workflow_specs:
            raise ValueError(f"Spec '{workflow_id}' not registered. Call register_spec() first.")

        spec = self._workflow_specs[workflow_id]

        def decorator(factory: Callable[..., Workflow]) -> Callable[..., Workflow]:
            # Validate params type hint matches spec
            type_hints = typing.get_type_hints(factory, globalns=factory.__globals__)
            inferred_params = type_hints.get('params', None)

            if spec.params is not None and inferred_params is not None:
                if spec.params is not inferred_params:  # Use `is not`
                    raise TypeError(f"Params type mismatch for {workflow_id}")
            elif spec.params is None and inferred_params is not None:
                raise TypeError(f"Factory has params but spec has none for {workflow_id}")
            elif spec.params is not None and inferred_params is None:
                raise TypeError(f"Spec has params but factory has none for {workflow_id}")

            self._factories[workflow_id] = factory
            return factory

        return decorator
```

**Immutability**: Specs are immutable after `register_spec()` - no mutation of `spec.params`.

#### Instrument Convenience Method

```python
class Instrument:
    def register_spec(self, *, namespace='data_reduction', name, version,
                     title, description='', source_names=None,
                     params=None, aux_sources=None, outputs=None) -> SpecHandle:
        """Register workflow spec. params must be explicit (not inferred)."""
        spec = WorkflowSpec(
            instrument=self.name, namespace=namespace, name=name, version=version,
            title=title, description=description, source_names=list(source_names or []),
            params=params, aux_sources=aux_sources, outputs=outputs
        )
        return self.workflow_factory.register_spec(spec)
```

## File Structure

Split each instrument config into a submodule with separate files:

```
config/instruments/
  loki/
    __init__.py       # Only imports specs (lightweight) - enables get_config() discovery
    specs.py          # Lightweight: Pydantic models + spec registration
    factories.py      # Heavy: imports ess.loki/ess.reduce/etc., factory implementations

handlers/
  detector_view_specs.py    # Lightweight: spec registration helpers
  detector_data_handler.py  # Heavy: factory implementations (imports ess.reduce.live)
```

### Why Submodules?

The existing codebase uses two key mechanisms:

1. **`available_instruments()`**: Uses `pkgutil.iter_modules()` to discover instruments by module/package name
2. **`get_config(instrument)`**: Uses `importlib.import_module(f'.{instrument}', __package__)` to load the instrument

These require **one module per instrument** where the module name matches the instrument name. A flat file structure (`loki_specs.py`, `loki_factories.py`) would:
- Break discovery (would see `'loki_specs'` and `'loki_factories'` as two instruments)
- Break `get_config('loki')` (no `loki.py` module exists)

**Submodules solve this**: `loki/` is discovered as the `'loki'` instrument, and `get_config('loki')` imports `loki/__init__.py`.

### Instrument `__init__.py` Pattern

```python
# loki/__init__.py
"""Loki instrument configuration.

This module provides lightweight spec registration for frontend use.
Backend services must explicitly import .factories to attach implementations.
"""
from . import specs

__all__ = ['specs']
```

**Critical constraint**: The `__init__.py` must **NOT** import `factories` because:
- Frontend (dashboard) calls `get_config(instrument)` to register workflows
- Frontend does not have heavy dependencies (`ess.loki`, `ess.reduce`) installed
- Importing factories would cause `ImportError` in the frontend

Backend services import factories explicitly **after** calling `get_config()`.

### Example: loki/specs.py (Lightweight)

```python
from ess.livedata import parameter_models
from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.handlers.detector_view_specs import register_detector_view_specs

# Pydantic models (no heavy dependencies)
class SansWorkflowParams(pydantic.BaseModel):
    q_edges: parameter_models.QEdges = ...
    wavelength_edges: parameter_models.WavelengthEdges = ...

class LokiAuxSources(AuxSourcesBase):
    incident_monitor: Literal['monitor1'] = ...

# Instrument setup
instrument = Instrument(name='loki')
_detector_names = [f'loki_detector_{bank}' for bank in range(9)]

# Register detector view specs (no file access, no ess.reduce)
register_detector_view_specs(instrument, projections=['xy_plane'],
                             source_names=_detector_names)

# Register workflow specs with handles
i_of_q_handle = instrument.register_spec(
    name='i_of_q', version=1, title='I(Q)',
    source_names=_detector_names,
    params=None,  # Explicit!
    aux_sources=LokiAuxSources,
    outputs=IofQOutputs,
)

i_of_q_params_handle = instrument.register_spec(
    name='i_of_q_with_params', version=1, title='I(Q) with params',
    source_names=_detector_names,
    params=SansWorkflowParams,  # Explicit! Frontend needs this for UI
    aux_sources=LokiAuxSources,
    outputs=IofQWithTransmissionOutputs,
)

instrument_registry.register(instrument)
```

### Example: loki/factories.py (Heavy)

```python
# Heavy imports OK - backend only
import ess.loki.live
from ess import loki
from ess.reduce.nexus.types import NeXusData, SampleRun
from ess.livedata.config import instrument_registry
from ess.livedata.handlers.detector_data_handler import (
    DetectorProjection, get_nexus_geometry_filename
)
from .specs import SansWorkflowParams, _detector_names, i_of_q_handle, i_of_q_params_handle

instrument = instrument_registry['loki']

# Add detectors (triggers file load via add_detector internals)
for name in _detector_names:
    instrument.add_detector(name)

# Expensive initialization
_base_workflow = loki.live._configured_Larmor_AgBeh_workflow()
_base_workflow[Filename[SampleRun]] = get_nexus_geometry_filename('loki')

# Register detector factories (DetectorProjection.__init__ handles registration)
_xy_projection = DetectorProjection(instrument=instrument, projection='xy_plane', ...)

# Attach workflow factories using handles
@i_of_q_handle.attach_factory()
def _i_of_q(source_name: str) -> StreamProcessorWorkflow:
    wf = _base_workflow.copy()
    wf[NeXusDetectorName] = source_name
    return StreamProcessorWorkflow(wf, dynamic_keys=..., target_keys={'i_of_q': IofQ[SampleRun]})

@i_of_q_params_handle.attach_factory()
def _i_of_q_with_params(source_name: str, params: SansWorkflowParams) -> StreamProcessorWorkflow:
    # Type hint validated against spec!
    wf = _base_workflow.copy()
    wf[sans_types.QBins] = params.q_edges.get_edges()
    # ... rest of implementation
```

## Why workflow_factory.py Doesn't Need Splitting

**Question**: Should `workflow_factory.py` be split into separate spec/factory modules?

**Answer**: No. It's already lightweight.

**Analysis**:
- `workflow_factory.py` imports only: `WorkflowSpec`, `WorkflowConfig`, `WorkflowId` (all lightweight)
- Defines `Workflow` protocol in-place (no external dependency)
- Frontend never imports it directly - accesses via `instrument_registry[name].workflow_factory`
- Frontend uses it ONLY as `Mapping[WorkflowId, WorkflowSpec]` (reads specs, never calls `.create()`)
- Backend calls both `.get()` (specs) and `.create()` (factories) from `JobManager`

**Conclusion**: Keep it as one cohesive module. The heavy dependencies are in instrument configs (`loki.py`) and handlers (`detector_data_handler.py`).

## Detector Data Handler Split

**Problem**: `detector_data_handler.py` imports `ess.reduce.live.raw` at module level, making it a heavy dependency that the frontend cannot import.

**Solution**: Split into two modules:

1. **`detector_view_specs.py`** (lightweight):
   - Contains `DetectorViewParams` (Pydantic model)
   - Contains `register_detector_view_specs()` helper function
   - Imports: only lightweight dependencies (Pydantic, workflow specs)
   - Used by: instrument specs modules (`{instrument}/specs.py`)

2. **`detector_data_handler.py`** (heavy):
   - Contains `DetectorProjection`, `DetectorLogicalView` (factory implementations)
   - Imports: `ess.reduce.live.raw` and other heavy dependencies
   - Used by: instrument factories modules (`{instrument}/factories.py`)

**Integration pattern:**

```python
# In loki/specs.py (lightweight)
from ess.livedata.handlers.detector_view_specs import register_detector_view_specs

register_detector_view_specs(instrument, projections=['xy_plane'], source_names=_detector_names)
# Specs registered, handles returned

# In loki/factories.py (heavy)
from ess.livedata.handlers.detector_data_handler import DetectorProjection

_xy_projection = DetectorProjection(instrument=instrument, projection='xy_plane', ...)
# Factory implementations attached using handles from specs
```

This split is **instrument-agnostic** - the same `detector_view_specs.py` and `detector_data_handler.py` modules are used by all instruments.

## Key Changes

1. **Explicit params**: `spec.params` must be provided during spec registration (no inference from factory)
2. **Handle pattern**: Specs return handles, factories use handles to attach (no string duplication)
3. **Immutable specs**: No mutation of `spec.params` after registration
4. **Type validation**: Factory's `params` type hint validated against spec using `is not`
5. **add_detector preserved**: Keep `instrument.add_detector(name)` API - only used in factories files
6. **Detector name lists**: Simple lists in specs, `add_detector()` calls in factories

## Backend Service Integration

Backend services need to load both specs and factories. The pattern is:

```python
# Example: src/ess/livedata/services/detector_data.py

from ess.livedata.config import instrument_registry
from ess.livedata.config.instruments import get_config
import importlib


def make_detector_service_builder(
    *, instrument: str, dev: bool = True, log_level: int = logging.INFO
) -> DataServiceBuilder:
    # Load specs (lightweight)
    _ = get_config(instrument)

    # Load factories (heavy) - only imported in backend
    importlib.import_module(f'ess.livedata.config.instruments.{instrument}.factories')

    # Now instrument_registry[instrument].workflow_factory has both specs and factories
    preprocessor_factory = DetectorHandlerFactory(
        instrument=instrument_registry[instrument]
    )
    # ... rest of service setup
```

**Why this works:**
- `get_config(instrument)` triggers instrument registration via specs
- Explicit factory import attaches factory implementations to registered specs
- Services can use `instrument_registry[instrument].workflow_factory.create()` which needs factories
- Frontend never imports factories, avoiding `ImportError` for missing dependencies

## Migration Path

1. Implement `SpecHandle`, `register_spec()`, `attach_factory()` in `WorkflowFactory` (same file, no split)
2. Add `register_spec()` convenience method to `Instrument`
3. Split `detector_data_handler.py`:
   - Create `detector_view_specs.py` (lightweight: Pydantic models, spec registration helpers)
   - Keep `detector_data_handler.py` (heavy: factory implementations with ess.reduce.live.raw)
4. For each instrument (loki, dream, bifrost, etc.):
   - Create `config/instruments/{instrument}/` submodule
   - Create `{instrument}/__init__.py`: imports only `specs` module
   - Create `{instrument}/specs.py`: Pydantic models, spec registration → returns handles
   - Create `{instrument}/factories.py`: imports handles from `.specs`, attaches factory implementations
   - Remove old `{instrument}.py`
5. Update backend services to explicitly import factories after calling `get_config()`
6. Dashboard continues calling `get_config()` but only loads specs (no changes needed)

## How the Handle Pattern Works

**Spec registration** (in lightweight `{instrument}/specs.py`):
```python
# Returns handle - no factory registered yet
handle = instrument.register_spec(name='workflow', version=1, params=MyParams, ...)
```

**Factory attachment** (in heavy `{instrument}/factories.py`):
```python
from .specs import handle  # Import the handle

@handle.attach_factory()  # Uses handle to attach factory
def factory(source_name: str, params: MyParams) -> Workflow:
    # Heavy imports, expensive initialization OK here
    ...
```

**Frontend usage** (dashboard):
```python
from ess.livedata.config.instruments import get_config

_ = get_config('loki')  # Imports loki/__init__.py → imports specs → specs registered
# instrument_registry['loki'].workflow_factory now has all specs
# Frontend only needs specs for UI generation - no factories imported
```

**Backend usage** (services):
```python
from ess.livedata.config.instruments import get_config
import importlib

_ = get_config('loki')  # Imports loki/__init__.py → imports specs
# Explicitly import factories to attach implementations
importlib.import_module(f'ess.livedata.config.instruments.{instrument}.factories')
# instrument_registry['loki'].workflow_factory now has specs AND factories
```

Or more directly:
```python
from ess.livedata.config.instruments import get_config

_ = get_config('loki')  # Load specs
from ess.livedata.config.instruments.loki import factories  # Attach factories
```

**Key insight**: The handle carries the `workflow_id` and reference to `WorkflowFactory`, so:
- No string duplication (name/version/namespace specified once)
- Type validation happens at factory attachment time
- Frontend calls `get_config()` → loads only specs (lightweight)
- Backend calls `get_config()` then explicitly imports factories (heavy)

## Benefits

- **Frontend**: Zero heavy dependencies or expensive initialization (only imports specs via `get_config()`)
- **Backend**: All factories available, pays initialization cost once at startup (explicitly imports factories)
- **Type safety**: Shared Pydantic models, validated factory type hints
- **Clear separation**: Specs = metadata, Factories = implementation
- **DRY principle**: Handle pattern eliminates name/version duplication
- **Compatible with existing discovery**: Submodule structure works with `available_instruments()` and `get_config()`
- **No import errors**: Frontend never attempts to import heavy dependencies that aren't installed
