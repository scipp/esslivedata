# Workflow spec consolidation

Unified plan for consolidating the workflow-spec surface of `ess.livedata`. Covers the long-term direction only insofar as it makes near-term decisions legible; near-term work is the concrete deliverable. Supersedes the earlier `ess-livedata-long-term-direction.md` and `ess-livedata-schemas-stage0.md` documents.

## Problem

Today's spec surface is scattered across `config/workflow_spec.py`, `config/instrument.py`, `parameter_models.py`, `handlers/*_specs.py`, and per-instrument `config/instruments/<inst>/`. The concrete consequences:

- **Parallel interface declarations.** `ess.livedata` maintains pydantic declarations of every workflow's parameters and outputs in parallel with whatever the technique packages declare via sciline providers and scipp newtypes. Keeping them in step is a manual ongoing burden — even with nobody actively changing anything, the two surfaces drift as upstream types evolve.
- **Default drift.** A livedata parameter default is not guaranteed to match the technique package's default for the same quantity. Scientists configure the same workflow differently in different tools.
- **Scipp coupling in the schema layer.** Output templates use `sc.DataArray` default factories, parameter models carry `get_edges()`-style methods returning `sc.Variable`. Anything introspecting specs transitively imports scipp.
- **No UI story beyond the livedata dashboard.** Any out-of-process consumer (NICOS, a future web dashboard, a technique-package ipywidget UI) has nothing to import and no wire protocol to read.
- **Contribution overhead.** Adding a parameter touches livedata, its factory, and the technique package separately.

## Direction

Framing decisions that justify the near-term plan; not all of them are executed now.

**Spec vs. `WorkflowImpl` split.** The spec (identity, param model, outputs model, display metadata) is pure pydantic and carries no sciline/scipp. The workflow implementation (sciline pipeline, accumulators, target keys, translation layer) lives where the reduction code lives — today in `ess.livedata`, eventually migrated to technique packages. Dependency arrow goes impl → spec.

**Translation layer next to the pipeline, never on the param model.** Sciline binding is a free-function layer — `apply_params(pipeline, params)` and `dynamic_keys(params)` per workflow — co-located with pipeline construction. No `Annotated[T, SclineBind(...)]` on fields. This keeps the schema layer importable without sciline.

**Multi-level-defaults rule.** If a field exists in the param model, the param model is authoritative and the translation layer always applies it to the pipeline. Sciline-level `pipeline[X] = value` "defaults" are permitted only for structural constants no user, ops config, or future param will touch. Anything else is a param masquerading as a default and a silent source of streaming/batch divergence.

**No graph reshapes in livedata.** Pins on inputs feeding declared outputs are fine (they are operational facts of the Kafka topic set). `wf.insert(...)` of replacement providers, scalar substitutions like `CorrectedMonitor[EmptyBeamRun, ...] = 1.0`, rebranding runs — not fine. Each is a shadow algorithm that will silently drift from the authoritative batch version. The discipline is a local rule that kills a class of bug by construction; it also produces a finite, concrete upstream-PR backlog for technique packages, which is the cheapest probe of whether cross-compute unification is organisationally feasible.

**Runtime JSON-Schema announcement is the cross-service contract.** Each running service publishes the JSON Schema of its registered (reduced) workflow specs on the existing status topic. Authority lives at runtime, not in a compile-time import. Pydantic validators run backend-side; the UI gets JSON Schema for form rendering and optimistic validation; backend is authoritative. There is no importable universal catalog of "all workflows for all instruments", and there is no planned ancestor for one.

**Consolidation-in-repo is a complete outcome.** The schema subpackage is in-repo, framework types only, with an enforced pydantic+stdlib dependency boundary so later extraction to `ess.reduce` (the eventual home of common vocabulary) is mechanical rather than a redesign. Extraction happens when the shape has stabilised, not speculatively. If no second consumer ever materialises, consolidation-in-repo is the endpoint — not a failure.

**Per-technique schemas submodules as future cross-compute sharing.** If and when a technique-package maintainer engages concretely, that package grows an `ess.<tech>.schemas` submodule (pydantic-only) for its technique-specific param/output shapes, importing framework types from `ess.reduce.schemas`. Incremental, per technique, gated on demand. Not a precondition for anything.

## Scope

### In

- `WorkflowSpec`, `WorkflowId`, `WorkflowOutputsBase`, `AuxInput`, `AuxSources`.
- `ArraySpec` / `CoordSpec` as output-template representation.
- `InstrumentSpec`, `ComponentSpec`.
- Common parameter building blocks: ranges (wavelength, TOF, two-theta, energy), `EdgesModel` variants (linear/log scales), unit enums. Scipp-free; runtime-side converter module for `sc.Variable` / `sc.scalar` construction.
- Generic workflow param/output models for monitor, detector, timeseries.
- Per-instrument schema files relocated into `schemas/instruments/<inst>/` by filesystem convention (internal organisation; no exported registry API).
- `WorkflowSpec.group` for UI grouping; `WorkflowId` reduced to `(instrument, name, version)`; `params` → `inputs`.
- Translation-layer convention: `apply_params(pipeline, params)` / `dynamic_keys(params)` free functions co-located with workflow-impl registration.
- Dependency-boundary CI check (`schemas/` imports pydantic + stdlib only).
- Reduced-schema JSON emission on the status topic.

### Out

- No `ess.schemas` extraction. The subpackage stays inside `ess.livedata` for this work; migration to `ess.reduce.schemas` is a separable follow-up once the shape is stable.
- No composition-root refactor. The source-partitioning framing (defaults / deployment pins / job-start routing / user input) informs which fields participate in the reduced schema, but the full five-concern factory decomposition is separate work.
- No cross-repo type sharing with technique packages. Per-technique schemas submodules are opt-in, per technique, when maintainers engage.
- No `StreamProcessorWorkflow` or cross-cutting-transform move to `ess.reduce`.
- No sciline-annotation shortcut on param models.
- No wire-export of pydantic validators. Validators are Python-side; JSON Schema carries structure + bounds; backend validates authoritatively.
- No publication as a PyPI package.

## Stage 0 — in-place surgery

Non-1:1 semantic changes to `WorkflowId` and `WorkflowSpec` performed *before* any files move, so Stage 1's relocation is mechanical. Five independent commits on this branch; packaging decided when the branch is ready.

### 0a — Route services without reading `WorkflowId.namespace`

`JobManager.create` currently rejects jobs where `workflow_id.namespace != instrument.active_namespace` (set by `ServiceFactory` at startup). Replace with an explicit service tag owned by the registering factory:

- `WorkflowFactory.register(...)` gains a `service: str` argument.
- Factory keeps a `service` mapping `WorkflowId → str`.
- `JobManager` consults `factory.get_service(workflow_id)` and compares against the running service name.
- `active_namespace` stays in place but becomes unused by routing; removed in 0d once nothing else reads it.

### 0b — `BackendStatus.service_name` replaces `.namespace`

`BackendStatus` publishes `namespace`; the dashboard uses it for the worker-identity key and the status-widget label. Rename the wire field to `service_name`; update `service_registry.py` key composition and `backend_status_widget.py` display. Wire break: producers and consumers land in the same commit.

### 0c — Add `WorkflowSpec.group`, rename `params` → `inputs`

Introduce `WorkflowGroup` as a frozen pydantic model providing display metadata:

```python
class WorkflowGroup(BaseModel, frozen=True):
    name: Literal["data_reduction", "monitor_data", "detector_data", "timeseries"]
    title: str
    description: str = ""
```

Canonical instances as module-level constants (`REDUCTION`, `MONITORS`, `DETECTORS`, `TIMESERIES`); specs reference them. `WorkflowSpec` gains `group: WorkflowGroup` (required, no default) and renames `params` → `inputs` (and `WorkflowConfig.params` → `WorkflowConfig.inputs` for consistency).

### 0d — Drop `namespace` from identity

- Remove `namespace` from `WorkflowId`, `WorkflowSpec`, and every spec registration site.
- `WorkflowId.__str__` → `"{instrument}/{name}/{version}"`; `from_string` parses three parts.
- Remove `Instrument.active_namespace` and the setter in `ServiceFactory`.
- No wire-format compatibility.

## Stage 1 — carve out `schemas/`

Everything generic; no instrument touched yet. Ends with every load-bearing type in place and a green repo, behind re-export shims so existing instrument code keeps importing from old paths.

### Pre-migration audit

For every type that is a candidate to move into `schemas/` — `WorkflowSpec`, param models (including `DetectorViewParams` and per-instrument subclasses), output models, `InstrumentSpec`/`ComponentSpec` — check its field types for scipp or callable leakage. Anything that fails decomposes before the file moves: move the offending field onto a runtime-side helper, keep only declarative data on the schema type.

Runtime-side config types (`LogicalViewConfig`, `SpectrumViewSpec`, and similar) hold callables and scipp types by design — they are inputs to factory/registration logic, not part of the wire surface. They stay where they are; the audit is only about what crosses into `schemas/`.

### Module layout

```
src/ess/livedata/schemas/
├── __init__.py
├── core.py           # WorkflowId, WorkflowSpec, WorkflowOutputsBase, AuxInput, AuxSources, WorkflowGroup
├── instrument.py     # InstrumentSpec, ComponentSpec
├── arrays.py         # ArraySpec, CoordSpec
├── parameters.py     # WavelengthRange, EdgesModel variants, unit enums — scipp-free
├── workflows/
│   ├── monitor.py    # MonitorDataParams, MonitorDataOutputs
│   ├── detector.py   # DetectorViewParams, DetectorViewOutputs, DetectorROIAuxSources
│   └── timeseries.py # TimeseriesParams, TimeseriesOutputs
└── instruments/      # populated in Stage 2+; filesystem-organised, no exported registry
```

Runtime-side, next to the subpackage:

```
src/ess/livedata/
├── parameter_conversions.py   # edges_to_variable(), range_to_scalars(), etc.
└── array_conversions.py       # array_spec_to_dataarray()
```

### `ArraySpec` replaces `sc.DataArray` output templates

The current `*Outputs` models declare `sc.DataArray` with `default_factory` lambdas producing zero-sized arrays; the schema information consumers need (dims, unit, coord units) is only reachable by constructing a DataArray and reading it back.

```python
@dataclass(frozen=True, slots=True)
class CoordSpec:
    unit: str | None = None

@dataclass(frozen=True, slots=True)
class ArraySpec:
    dims: tuple[str, ...]
    unit: str | None = None
    coords: Mapping[str, CoordSpec] = field(default_factory=dict)
```

`*Outputs` models declare `ArraySpec` fields. Dims/unit/coord-unit is deliberately the minimum set; bin-edge vs. bin-center, dtype, and other scipp-isms are runtime concerns.

Consumer impact: `WorkflowSpec.get_output_template()` returns `ArraySpec`; `find_timeseries_outputs` becomes `len(spec.dims) == 0 and 'time' in spec.coords`; plotter auto-selection reads dims/coord units directly; sites that need an empty `sc.DataArray` at runtime use `array_spec_to_dataarray(spec)`.

### Parameter models go scipp-free

Today's `parameter_models.py` carries `get_start()`, `get_stop()`, `get_edges()`, `range()` methods that return `sc.Variable` / `sc.scalar`. Field declarations themselves are pure (float, int, str, enum). Strip the scipp-returning methods and relocate them as free functions in `parameter_conversions.py`:

```python
# parameter_conversions.py (runtime-side, not in schemas/)
def edges_to_variable(model: EdgesModel, *, dim: str, unit: str) -> sc.Variable: ...
def range_to_scalars(model: RangeModel) -> tuple[sc.Variable, sc.Variable]: ...
```

Call sites change from `model.get_edges()` to `edges_to_variable(model, dim=..., unit=...)`. Mechanical; ~20 call sites.

### `InstrumentSpec` / `ComponentSpec` and the runtime split

Today's `Instrument` bundles schema data (name, title, description, detector/monitor names, component metadata, workflow specs) with runtime machinery (transform bindings, `WorkflowFactory`, stream/Kafka wiring, `load_factories()`). Split:

```python
class InstrumentSpec(BaseModel):
    name: str
    title: str
    description: str
    detectors: dict[str, ComponentSpec]
    monitors: dict[str, ComponentSpec]
    workflows: dict[WorkflowId, WorkflowSpec]
```

Runtime `Instrument` holds an `InstrumentSpec` reference plus the runtime pieces. No exported discovery API; the runtime `Instrument` is built by `setup_factories()` as today.

### Translation-layer convention

Per workflow impl, co-located with factory registration:

```python
def apply_params(pipeline: sciline.Pipeline, params: DreamPowderWorkflowParams) -> None:
    pipeline[dream.powder.types.DspacingBins] = edges_to_variable(
        params.dspacing_edges, dim='dspacing', unit='angstrom'
    )
    # ...

def dynamic_keys(params: DreamPowderWorkflowParams) -> dict[str, sciline.typing.Key]:
    return {params.detector_source: NeXusData[NXdetector, SampleRun], ...}
```

Registration attaches `apply_params` and `dynamic_keys` alongside the `WorkflowSpec` and the sciline pipeline. Documented convention; not yet a formal `WorkflowImpl` bundle (that is Tier A work).

### Dependency boundary + CI

```ini
[importlinter:contract:schemas-deps]
name = ess.livedata.schemas depends only on pydantic/stdlib
type = forbidden
source_modules = ess.livedata.schemas
forbidden_modules = ess.livedata, scipp, sciline
```

A grep-based test in CI suffices if import-linter friction is too high. A standalone schemas test job (tox env) installs `ess.livedata.schemas` with pydantic only (no scipp, no sciline, no ess.livedata) and runs its unit tests — this is the real proof of the boundary.

### Reduced-schema JSON emission

Each service emits, on its existing status topic:

- Its service name (the one that already replaced `namespace` in Stage 0b).
- Per registered `WorkflowId`, the *reduced* JSON Schema: the param model minus fields filled by deployment pins or job-start routing.

Reduced schema definition (informal for this plan; formal in the implementation):

- **Pinned** — fields set by ops config at service startup. Example: calibration file path, nexus geometry filename.
- **Job-start routed** — fields resolved from a job-launch selection (e.g. "which detector source is the mantle for this run").
- **User-visible** — everything else. This is what appears in the JSON Schema on the wire.

No composition-root refactor is needed to land emission — the pinned/routed subset can be enumerated per workflow-impl registration in the near term (a list of field names excluded from emission). Full composition-root generalisation is Tier A follow-up.

## Stage 2 — prove on DREAM

DREAM is the most demanding instrument; validate the design end-to-end before replicating.

- Create `schemas/instruments/dream/`. Move DREAM's pydantic schemas, enums, workflow specs, source metadata, detector/monitor name lists. Start with reduction workflows (most stand-alone).
- Adapt DREAM's `factories.py`, `streams.py`, `views.py` to read from `schemas.instruments.dream`.
- Set `group` per `WorkflowSpec`.
- Exercise `ArraySpec` by running the full DREAM test suite and dashboard.
- Drive any design gaps back into Stage 1 types. Stress-test the abstraction here before multiplying by six.

## Stage 3 — roll out

Mechanical, per instrument: dummy, loki, bifrost, odin, nmx, tbl. Core is frozen by this point; pure relocation.

## Stage 4 — cleanup

- Remove Stage 1 re-export shims.
- Delete the old `instrument_registry` mechanism.
- Documentation pass describing `ess.livedata.schemas` as the authoritative schema surface.

## Later: migration to `ess.reduce`

Framework-only types (`WorkflowSpec`, `WorkflowId`, `ArraySpec`, `InstrumentSpec`, `ComponentSpec`, common parameter building blocks) eventually migrate into `ess.reduce.schemas` with the same pydantic+stdlib submodule-level boundary. Cross-repo, but within the same maintainer group, so organisationally cheap.

Concrete trigger for the migration: *before a second instrument joins Stage 2/3 with stable schemas.* Don't let "for now" become forever — otherwise the migration never happens and the types calcify in ess.livedata.

Per-instrument specialisations stay in `ess.livedata` regardless. `ess.reduce.schemas` is framework-only.

## What could force reconsideration

Falsification conditions. The direction is worth pursuing only if it remains reversible on evidence.

- **The second consumer never materialises.** If 18 months after consolidation no technique-package ipywidget UI and no NICOS integration is pulling from the schemas, migration to `ess.reduce` is speculative. Consolidate, stay in `ess.livedata`, move on.
- **Graph-reshape discipline fails.** If bespoke factories keep growing `wf.insert(...)` because upstream maintainers won't absorb category-C PRs, the shadow-algorithm rot wins and cross-compute unification is organisationally dead. Call it by name and stop planning around it.
- **Version mismatches become routine post-migration.** Tighten backward-compat discipline or reverse the migration.
- **Technique-package maintainers reject per-technique schemas submodules.** Cross-compute unification stays at zero; vocabulary is livedata-local. Still a coherent outcome.

## Open items

- Scope of `params` → `inputs` in Stage 0c. Default: yes, for consistency; confirm at implementation time.
- Group ordering in the UI. Declaration order of the `WorkflowGroup` module-level constants is the natural answer; confirm the dashboard honors it (not alphabetical sort).
- Stage 0 PR packaging. One branch, five commits; single PR most likely, split only if review bandwidth demands it.
- Exact format of reduced-schema announcement payload. JSON Schema + `$id`? Grouped per service, or one per `WorkflowId`? Settle at implementation time.
