# Proposal: `ess.livedata.schemas` — a dependency-light schema surface

## Intent

Consolidate the parameter/output/instrument schema surface — currently scattered across `config/workflow_spec.py`, `config/instrument.py`, `parameter_models.py`, `handlers/*_specs.py`, and `config/instruments/<instrument>/specs.py` — into a single subpackage `ess.livedata.schemas`.

The restructuring is only worth doing if it is **meaningful**, not a cosmetic relocation. Load-bearing decisions — which dependencies the layer carries, how output templates are represented — are settled here, not deferred. The deliverable is a schema surface that already stands on its own dependency-wise and could be extracted to a standalone package when a consumer needs it; that extraction is then a packaging task, not a redesign.

Concretely:

- One enforceable dependency boundary: **pydantic + stdlib only**. No scipp inside `schemas/`.
- Output templates expressed as declarative schemas (`ArraySpec`), not `sc.DataArray` instances.
- A stable discovery API backed by filesystem convention — instruments are subpackages under `schemas/instruments/`, no explicit registry.
- Schema-vs-runtime separation pushed from per-instrument `specs.py` / `factories.py` discipline into a package-level boundary.

Out of scope as *future packaging work*, not unresolved design: publishing `ess.schemas` to PyPI, JSON Schema artifact emission for NICOS, migrating parameter models to `ess.reduce` / technique packages.

## Scope

### In

- Workflow spec types (`WorkflowSpec`, `WorkflowId`, `AuxInput`, `AuxSources`, `WorkflowOutputsBase`).
- `WorkflowId` reduced to semantic identity `(instrument, name, version)`; today's `namespace` removed from identity. UI grouping moves to `WorkflowSpec.group`.
- `ArraySpec` / `CoordSpec` replacing `sc.DataArray` default-factory templates in every `*Outputs` model.
- Instrument identity: detector and monitor names, `ComponentSpec`.
- Shared parameter models (wavelength ranges, binning edges, angle ranges, unit enums).
- Generic workflow parameter/output models (`MonitorDataParams`, `DetectorViewParams`, their outputs).
- Per-instrument schemas: enums, subclassed param models, workflow param/output models, workflow spec declarations.
- The discovery API (`list_instruments`, `get_instrument`, `get_workflows`).
- CI enforcement of the dependency boundary.

### Out

- Factory functions and any sciline/technique-package imports.
- Logical-view mechanics entirely — both transform functions and any "logical view" abstraction. Logical views appear as ordinary `WorkflowSpec`s; the transform-to-spec binding is a runtime concern (see `What moves where`).
- Stream/Kafka plumbing, topic mappings, source-to-stream binding.
- `StreamProcessorWorkflow` adapter, accumulators, dynamic keys.
- Live-mode-only constraints (stay runtime-side).
- JSON Schema emission — cheap to add once the surface is clean; wait for a consumer.
- Publication as a standalone PyPI package.

## Module layout

```
src/ess/livedata/schemas/
├── __init__.py            # public API: list_instruments, get_instrument, get_workflows
├── core.py                # WorkflowId, WorkflowSpec, WorkflowOutputsBase, AuxInput, AuxSources
├── instrument.py          # InstrumentSpec, ComponentSpec
├── arrays.py              # ArraySpec, CoordSpec
├── parameters.py          # WavelengthRange, DspacingEdges, TwoTheta, unit enums
├── workflows/
│   ├── __init__.py
│   ├── monitor.py         # MonitorDataParams, MonitorDataOutputs
│   ├── detector.py        # DetectorViewParams, DetectorViewOutputs, DetectorROIAuxSources
│   └── timeseries.py      # TimeseriesParams, TimeseriesOutputs
└── instruments/
    ├── __init__.py        # (empty)
    ├── dream/
    │   ├── __init__.py    # builds + exposes InstrumentSpec; re-exports
    │   ├── detector.py    # DreamDetectorViewParams, InstrumentConfiguration (enum)
    │   ├── monitor.py     # DreamMonitorDataParams
    │   └── powder.py      # PowderWorkflowParams, PowderReductionOutputs, ...
    ├── loki/ ...
    └── ...
```

The top level is framework (cross-cutting types); everything under `instruments/` is a plugin. Adding an instrument is adding one subpackage — no registration list, no `__init__.py` edit elsewhere. Per-instrument subpackages are public. Consumers either ask the registry or import directly for type annotations:

```python
from ess.livedata.schemas import get_workflows
specs = get_workflows('dream')

# or, for factory code:
from ess.livedata.schemas.instruments.dream.powder import PowderWorkflowParams
```

## Output templates without scipp

The current `*Outputs` models declare `sc.DataArray` fields with `default_factory` lambdas producing zero-sized arrays. The *schema* information consumers need — dims, unit, coord units — is only reachable by constructing a DataArray and reading it back. That forces scipp on anything introspecting workflow outputs (dashboard plotter selection, timeseries detection, etc.).

Replacement:

```python
# schemas/arrays.py

@dataclass(frozen=True, slots=True)
class CoordSpec:
    # consider `dim` field so we can have multiple coords for a given dim?
    unit: str | None = None

@dataclass(frozen=True, slots=True)
class ArraySpec:
    dims: tuple[str, ...]
    unit: str | None = None
    coords: Mapping[str, CoordSpec] = field(default_factory=dict)
```

`*Outputs` models declare `ArraySpec` fields:

```python
class PowderReductionOutputs(WorkflowOutputsBase):
    focussed_data_dspacing: ArraySpec = Field(
        default=ArraySpec(
            dims=('dspacing',),
            unit='dimensionless',
            coords={'dspacing': CoordSpec(unit='angstrom')},
        ),
        title='I(d)',
        description='Focussed intensity as a function of d-spacing.',
    )
```

Consumer impact:

- `WorkflowSpec.get_output_template()` returns `ArraySpec`, not `sc.DataArray`.
- `find_timeseries_outputs` becomes `len(spec.dims) == 0 and 'time' in spec.coords` — simpler and faster.
- Plotter auto-selection reads dims/coord units directly; no DataArray construction.
- Sites that actually need an empty `sc.DataArray` at runtime construct it from the `ArraySpec` locally (trivial helper on the runtime side).

Dims/unit/coord-unit is deliberately the minimum set. Bin-edge vs. bin-center, dtype, and other scipp-isms are runtime concerns and stay runtime-side.

## Workflow identity vs. grouping

Today's `WorkflowId` has `(instrument, namespace, name, version)`. `namespace` does two unrelated jobs: backend service routing (runtime) and UI grouping (cross-UI essential). Neither belongs in identity.

- Routing is architectural state — the factory that registers a workflow already knows which service runs it. No schema annotation needed.
- Grouping is a display decision. Moving a workflow from one section to another should be a metadata edit, not a new identity.

Schema identity becomes semantic only:

```python
class WorkflowId(BaseModel, frozen=True):
    instrument: str
    name: str
    version: int
```

UI grouping is a free-form string on the spec:

```python
class WorkflowSpec(BaseModel):
    ...
    group: str = Field(
        description="Display grouping label. UIs group workflows by this string."
    )
```
`WorkflowSpec` will require some cleanup to avoid legacy from esslivedata. For example, we may rename `params`->`inputs` to match `outputs`. The fields referencing "source" may also need to be reconsidered since "source" is esslivedata terminology and this proposal renamed `SourceMetadata`->`ComponentSpec`.

- No enum: the schema package does not dictate the set. Different instruments may use different groupings; the union of declared values is the set.
- Workflow names must be unique within `(instrument, version)`. Today's cross-namespace collisions (e.g. `monitor_data/default` vs. `detector_data/default`) resolve by renaming during Stage 2.
- Backend service routing is fully decoupled: the runtime handler responsible for a workflow is determined by which factory registers it, not by reading a namespace field.

## Registry API

Single type, auto-discovery via filesystem convention. Instruments are the subpackages under `schemas/instruments/`; no explicit registration list.

```python
# schemas/instrument.py
class InstrumentSpec(pydantic.BaseModel):
    name: str
    title: str
    description: str
    detectors: dict[str, ComponentSpec]
    monitors: dict[str, ComponentSpec]
    workflows: dict[WorkflowId, WorkflowSpec]
```

```python
# schemas/__init__.py
import importlib, pkgutil
from . import instruments as _instruments_pkg

def list_instruments() -> list[str]:
    """Names of available instruments. Does not import them."""
    return [m.name for m in pkgutil.iter_modules(_instruments_pkg.__path__) if m.ispkg]

def get_instrument(name: str) -> InstrumentSpec:
    """Import the instrument's subpackage (first time) and return its InstrumentSpec."""
    module = importlib.import_module(f'ess.livedata.schemas.instruments.{name}')
    return module.spec  # each instrument's __init__.py exposes `spec: InstrumentSpec`

def get_workflows(name: str) -> dict[WorkflowId, WorkflowSpec]:
    return get_instrument(name).workflows
```

Each `schemas/instruments/<inst>/__init__.py` builds its `InstrumentSpec` eagerly on first import and exposes it as a module-level `spec`. Import caching means subsequent `get_instrument(name)` calls return the same object without rebuilding. No separate registry dict is needed — Python's module cache *is* the registry.

Callers that want display metadata for the full instrument list pay the (cheap) import cost:

```python
for name in list_instruments():
    inst = get_instrument(name)
    print(inst.title, '—', inst.description)
```

Under the pydantic+stdlib dependency rule, importing an instrument subpackage is sub-millisecond.

## Schema ↔ runtime binding

The runtime side holds references to schema objects. The arrow points one way: **runtime → schema**. Schema modules never import runtime code.

- Runtime `Instrument` is constructed from `schemas.get_instrument(name)` plus stream mapping and view transforms.
- `WorkflowFactory` takes the schema's `workflows: dict[WorkflowId, WorkflowSpec]` and lets `setup_factories()` attach factory functions keyed by `WorkflowId`.
- Logical views carry no special schema-layer presence; they appear as `WorkflowSpec`s, and their transforms are attached by the runtime `WorkflowFactory` keyed by `WorkflowId`, same as any other factory.
- `ArraySpec` output templates are read at plotter-selection time; runtime constructs `sc.DataArray` from them on demand.

## Dependency rule

`ess.livedata.schemas` imports only pydantic and stdlib. Not scipp, not anything else inside `ess.livedata`.

Enforced by CI. `import-linter` contract:

```ini
[importlinter:contract:schemas-deps]
name = ess.livedata.schemas depends only on pydantic/stdlib
type = forbidden
source_modules = ess.livedata.schemas
forbidden_modules = ess.livedata, scipp
```

A grep-based test in CI suffices if `import-linter` friction is too high; either check can also run as a pre-commit hook for faster local feedback.

## What moves where

- **Core spec types** — `config/workflow_spec.py` → `schemas/core.py`. `WorkflowSpec` (now with `group: str`, no `namespace`), `WorkflowId` (reduced to `(instrument, name, version)`), `WorkflowOutputsBase`, `AuxInput`, `AuxSources`, helper enums. Runtime service routing that today reads `namespace` is relocated so the registering factory provides the routing directly.
- **Instrument identity** — `config/instrument.py` is split: schema half (`name`, `title`, `description`, `detectors`+`ComponentSpec`, `monitors`+`ComponentSpec`, workflow specs) → `schemas/instrument.py` as `InstrumentSpec`. Runtime half (transform bindings, `WorkflowFactory`, stream/Kafka, `load_factories()`) stays, now holding an `InstrumentSpec` reference.
- **Shared parameter models** — `parameter_models.py` → `schemas/parameters.py`. Straight move.
- **Generic workflow schemas** — `handlers/*_specs.py` split: pydantic models, enums, `ArraySpec` output templates → `schemas/workflows/{monitor,detector,timeseries}.py`. Registration helpers stay put; they import schemas from the new paths.
- **Output templates** — every `*Outputs` model migrates from `sc.DataArray` defaults to `ArraySpec` fields. Runtime sites that consumed DataArrays construct them from `ArraySpec` locally.
- **Per-instrument schemas** — `config/instruments/<inst>/specs.py` → `schemas/instruments/<inst>/`. Moves: instrument-specific enums, subclassed param models, workflow param/output models, workflow spec declarations, detector/monitor name lists, component metadata dict. Stays: `factories.py` (imports updated), `streams.py` (topic mapping — names move, mapping stays), `views.py` (transform *functions* plus any runtime-side helpers that build logical-view `WorkflowSpec`s and attach their transforms). Enum mirrors like DREAM's `InstrumentConfigurationEnum` are deduplicated in-place; one canonical definition lives in `schemas/instruments/dream/detector.py`.
- **Logical views become ordinary workflows.** The current `instrument.add_logical_view(...)` is reframed as a runtime-side helper that builds a `WorkflowSpec` (with a reduced-view output shape) and attaches its transform function. The schemas layer sees only the resulting `WorkflowSpec` — no `LogicalViewSpec` type, no special binding mechanism. `WorkflowFactory` binds the transform to the spec's `WorkflowId` the same way it binds any other factory.

## Execution: build, prove, roll out

Not a per-file-type wave structure across all instruments. Three stages, each ending with a green repo.

### Stage 1 — Build the core

Everything generic — the backbone, no instrument touched yet.

- Create `schemas/` with `__init__.py`, `core.py`, `instrument.py`, `arrays.py`, `parameters.py`, an empty `workflows/` subpackage, and an empty `instruments/` subpackage.
- Introduce `ArraySpec` / `CoordSpec`. Migrate `WorkflowOutputsBase` and all generic `*Outputs` to `ArraySpec`.
- Move core spec types.
- Move `ComponentSpec`, shared parameter models.
- Move generic workflow schemas into `workflows/{monitor,detector,timeseries}.py`.
- Split `Instrument` into `InstrumentSpec` + runtime wrapper; introduce registry API.
- Update consumers of output templates (plotter selection, timeseries detection, dashboard templating) to work with `ArraySpec`.
- Add runtime helper for `ArraySpec → sc.DataArray` at the one or two sites that need it.
- Add the import-boundary CI check.

Leave re-export shims at old paths so existing instrument code still imports. Stage 1 ends with every load-bearing decision made and every new type in place — no instrument moved.

### Stage 2 — Prove on DREAM

Validate the design end-to-end against the most demanding instrument before replicating.

- Create `schemas/instruments/dream/`. Move DREAM's pydantic schemas, enums, workflow specs, source metadata, detector/monitor name lists. Start with reduction workflows as those are most stand-alone (not requiring bespoke detector-view setup).
- Adapt DREAM's `factories.py`, `streams.py`, `views.py` to read from `schemas.instruments.dream`.
- Resolve cross-namespace name collisions by renaming (e.g. `monitor_data/default` → `monitor_default`) and set a `group` per `WorkflowSpec`. Verify nothing in the runtime still reads the removed `namespace` field.
- Exercise `ArraySpec` by running the full DREAM test suite and dashboard against the new outputs.
- Drive any design gaps back into Stage 1 types. This is the point where the abstraction is stress-tested — fix it here before multiplying the change by six.

### Stage 3 — Roll out

Repeat Stage 2 mechanically for each remaining instrument: dummy, loki, bifrost, odin, nmx, tbl. By this point the core is frozen; the rollout is pure relocation per instrument.

### Stage 4 — Cleanup

- Remove re-export shims from Stage 1.
- Delete the old `instrument_registry` mechanism.
- Documentation pass describing `ess.livedata.schemas` as the authoritative schema surface.

## Non-goals

- No publication as a standalone PyPI package. The dependency boundary is real at the end of this work; publishing is orthogonal packaging.
- No JSON Schema emission. Trivial to add once the surface is scipp-free; add when a consumer (NICOS) needs it.
- No movement of parameter models into `ess.reduce` or technique packages. External coordination.
- No rethink of `target_keys` / `accumulators` / `dynamic_keys`. Runtime-side, untouched.
- No new two-phase `register_spec` / `attach_factory` mechanism. Current decorator pattern continues; only the location of the schema half changes.
- **No schema-layer mechanism for source-dependent parameters (#680).** Workflows whose parameter validity, defaults, or ranges genuinely depend on which source is selected are split into multiple `WorkflowSpec`s with disjoint `source_names` subsets, each carrying its own `params` model (typically subclassing a shared base). The DREAM monitor TOF case (`monitor_cave` vs. `monitor_bunker`) becomes two specs sharing a common `DreamMonitorDataParams` base. No `params_overrides`, no source-aware validator context, no spec-level hook — structural separation via splitting is the design answer.

## Open questions

- **Instrument-subclass param models under hypothetical domain-package moves.** `DreamMonitorDataParams(MonitorDataParams)` works trivially here. Flag only — not a concern for this work.
