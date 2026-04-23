# Proposal: `ess.livedata.schemas` — a dependency-light schema surface

## Intent

Consolidate the parameter/output/instrument schema surface — currently scattered across `config/workflow_spec.py`, `config/instrument.py`, `parameter_models.py`, `handlers/*_specs.py`, and `config/instruments/<instrument>/specs.py` — into a single subpackage `ess.livedata.schemas`.

The restructuring is only worth doing if it is **meaningful**, not a cosmetic relocation. Load-bearing decisions — which dependencies the layer carries, how output templates are represented — are settled here, not deferred. The deliverable is a schema surface that already stands on its own dependency-wise and could later be extracted to a standalone `ess.schemas` in the scipp monorepo.

Concretely:

- One enforceable dependency boundary: **pydantic + stdlib only**. No scipp inside `schemas/`.
- Output templates expressed as declarative schemas (`ArraySpec`), not `sc.DataArray` instances.
- A stable discovery API backed by filesystem convention — instruments are subpackages under `schemas/instruments/`, no explicit registry.
- Schema-vs-runtime separation pushed from per-instrument `specs.py` / `factories.py` discipline into a package-level boundary.

### Why a single consolidated schemas package (not per-technique)

The schema surface must be **one** package, not `ess.powder_schemas`, `ess.sans_schemas`, etc. Three reasons worth stating up front because others will re-litigate them:

1. **Cross-cutting types are technique-agnostic.** `WorkflowSpec`, `WorkflowId`, `ArraySpec`, `ComponentSpec`, unit enums — none of these belong to a single technique. Per-technique packages either duplicate them or force a `base_schemas` package that drags in everything anyway.
2. **One version bump per UI consumer.** Dashboard, technique-package ipywidget UIs, CLI wrappers, eventually NICOS all need to agree on the `WorkflowSpec` shape. One package = one changelog, one compatibility story. ~8 per-technique packages multiply release-coordination cost.
3. **Cross-technique workflows exist.** Monitor normalization is shared between powder/SANS/reflectometry. Splitting schemas per technique while workflows span techniques gets ugly fast.

### Extraction discipline

Framing extraction to a standalone package as "just packaging" understates what it involves. Extraction forces genuine design commitments: the validator-serialization boundary (see *Schema catalog vs. runtime catalog*), versioning contracts between schemas and consumers, and wire-vs-Python-import policy. Treat extraction as future *design* work that packaging follows, not the reverse.

**Do not extract on speculation.** Today an instrument config change is one PR in livedata. Post-extraction it becomes a scipp-monorepo PR plus a livedata dep bump, and that cost hits during live commissioning when iteration speed matters. Extract only when a *second concrete consumer* (technique-package ipywidget UI with real code, NICOS with real requirements — not "a hypothetical NICOS") is demonstrably pulling from the package.

### Explicit non-scope

Not in this work, and framed honestly rather than as "just packaging": publishing `ess.schemas` to PyPI, JSON Schema artifact emission for NICOS, migrating parameter models into `ess.reduce` / technique packages. Each is its own design conversation; this plan prepares the ground without committing to any of them.

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

### What "schemas" really means here

The mental model "schemas = trivial declarations" does not survive contact with the current code. `handlers/detector_view_specs.py` (~580 lines) and `monitor_workflow_specs.py` (~260 lines) contain construction helpers like `make_detector_view_outputs(output_ndim, roi_support, spectrum_view)` that build pydantic model classes from parameters. The CI boundary (pydantic + stdlib) still holds for them, but the accurate framing is **vocabulary + construction helpers**, not pure declarations. Expect the schemas package to carry non-trivial logic where output shape depends on configuration.

**Discussion point: scipp-free audit before Stage 1.** Some currently schema-adjacent types may carry scipp-dependent fields that would leak across the boundary. In particular `SpectrumViewSpec` / `LogicalViewConfig` today carry `Callable[[sc.DataArray, str], sc.DataArray]`-typed fields. Anything in that shape cannot cross into `schemas/` — the scipp-free invariant would be violated. These call for decomposition (split the callable into a runtime-side transform, keep only declarative fields in the schema) *before* the file relocates, not after the CI check complains. This audit belongs at the start of Stage 1, not mid-Stage 2.

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

## Schema catalog vs. runtime catalog

`ess.livedata.schemas` (eventually `ess.schemas`) is a *vocabulary*: the universe of declared workflow specs across all instruments. A concrete esslivedata backend registers factories for a subset — that subset is the **runtime catalog**.

- **Schema catalog** — `get_workflows(instrument)`; everything declared under `schemas/instruments/<inst>/`.
- **Runtime catalog** — `WorkflowFactory`'s registered `WorkflowId`s; what this deployment actually serves.

The two are allowed to diverge. Once the schemas package is released independently, the gap can widen further.

Frontend access depends on process locality:

- **Dashboard (shares the esslivedata codebase).** Builds the same `WorkflowFactory` via the same `setup_factories()` the services use; the runtime catalog is available locally, with pydantic validators intact (both sides import from `ess.schemas`). No catalog-specific wire protocol needed.
- **Out-of-process frontends (e.g. NICOS).** Would need either a `WorkflowId` set published over Kafka or a JSON-Schema export. Out of scope until a concrete requirement lands. Pydantic validators do not round-trip through JSON, so any frontend using specs for *input validation* has to share the Python classes regardless of transport.

This approach assumes every workflow esslivedata serves is declared in `ess.schemas`. If that stops being true — esslivedata begins registering workflows that live outside the schemas package — the dashboard loses local discoverability and the follow-up below becomes necessary.

**Follow-up (not in this plan): per-service `WorkflowId` announcement.** Each running service publishes its registered ID set on the existing status topic. Cleaner than implicit discovery, makes multi-service deployments self-describing, and unblocks any frontend that can't share the Python codebase. Defer until either NICOS needs the catalog or esslivedata begins serving workflows outside the schemas package.

### Discussion point: validator-serialization policy

Pydantic validators (`@field_validator`, `@model_validator`) are Python callables bound to the model class. They do not round-trip through JSON. Today the dashboard relies on them for input validation; the validator logic is authoritative, not merely documentation.

This plan implicitly assumes **(b)** below — all consumers are Python, validators travel by import. That assumption holds for dashboard + backend but needs an explicit commitment before the surface can serve out-of-process consumers. The two options:

- **(a) Validators are Python-side enrichment only.** The wire surface is the declarative fields; out-of-process consumers (NICOS over Kafka, JSON-Schema exports) reimplement validation or tolerate its absence. Validators are a livedata/dashboard convenience, not part of the contract.
- **(b) All consumers are Python.** Anyone consuming a spec must share the Python classes; wire format is implementation detail, not consumer interface.

Concrete example of what falls off in (a): DREAM's `InstrumentConfiguration.check_high_resolution_not_implemented` validator silently does not exist in a JSON-Schema export.

This decision couples to the fix-latency risk (see *Risks and operational constraints*): policy (a) lets validators evolve on the livedata side at deploy speed; policy (b) means any validator that turns out too strict during beam time needs a cross-repo release to relax. Whichever policy we adopt, we should say so out loud — half-committed is where bugs live. Resolve before extraction, not at the moment of extraction.

## Long-term direction (for discussion, not in scope)

Out of scope for this work, but recording because the plan's structure should not silently foreclose it. The likely endpoint the discussion around this plan keeps converging on:

- **`ess.schemas`** — pydantic + stdlib vocabulary, eventually in the scipp monorepo alongside technique packages. Serves esslivedata's dashboard, technique-package ipywidget UIs, CLI wrappers, potentially NICOS.
- **`ess.reduce`** (or similar home) — absorbs detector-view and monitor-histogram **transforms** currently in esslivedata. These are cross-instrument data-shaping primitives; they work equally on NeXus files and live streams. `ess.reduce` is already cross-instrument shared code for technique packages, so this is scope-consistent, not scope creep.
- **`ess.livedata`** — streaming adapter + dashboard layered over the two above. Owns Kafka plumbing, accumulators, orchestration, the runtime catalog (which workflows a given deployment serves), and livedata-local parameter subclasses for streaming-only fields (accumulation window, rate limits, cadence).

### Discussion points

- **Detector-view / monitor transform migration to `ess.reduce`.** Separable from this plan. Requires auditing which "transforms" are truly pure data ops versus entangled with streaming state (ROI aux-source feedback, dynamic detector geometry, accumulator state). The pure parts move; the reactive wrappers stay. Not every feature factors cleanly — expect implementations to need refactoring before they can move.
- **Streaming-specific parameter fields belong in livedata-local subclasses.** Fields that only exist because the workflow runs live (accumulation window, rate-limit, cadence, resolution-for-incremental-binning) do not belong in `ess.schemas`, which is the shared vocabulary across batch and live consumers. Same subclassing pattern this plan already uses for instrument extensions, applied to the live-vs-batch axis. Not every subclass needs to live in `ess.schemas`.
- **Dependency-direction concern under technique-package absorption.** If generic `MonitorDataParams` later migrates into `ess.reduce` while `DreamMonitorDataParams(MonitorDataParams)` stays in `ess.schemas`, the dependency arrow inverts (`schemas` imports from `reduce`). The livedata-local streaming-overlay pattern sidesteps this for streaming-only fields, but instrument-specific overlays of generic params still want the generic params to stay in a schemas layer. Flagging as a constraint any future `ess.reduce` absorption design must respect, not a concern to resolve now.

### What esslivedata is for, long-term

The direction above implies esslivedata as a **streaming + UI adapter** layered over shared neutron-data libraries, not as a full home-grown reduction implementation. The schemas work in this plan is compatible with both readings, but the smaller downstream design choices (where to put new detector views, what tests go where, how to factor accumulators) get simpler once the long-term answer is explicit. Worth naming the direction even though implementing it is out of scope here.

## Risks and operational constraints

Most of what's below applies after extraction to a standalone `ess.schemas`, not during the in-repo consolidation this plan implements. Framed concretely so the team can weigh them before committing to extraction, not after the first incident forces the conversation.

### Fix latency — the beam-time case

Today: an instrument scientist notices a param default is wrong at 2am mid-run. One-line PR to livedata, on-call review, redeploy — ~15 minutes.

Post-extraction, the same fix is a PR to the scipp monorepo → a different reviewer pool → merge → schemas release (or pre-release tag) → bump livedata dep → PR to livedata → review → merge → deploy. Best case hours, worst case days if the scipp-monorepo release cadence doesn't bend. **One overly-strict validator is now an hour of beam time.**

Two mitigations, both with costs:

- **(a) Keep validators light.** Structural validation only (types, required fields, pydantic does this implicitly); "this instrument accepts only values X, Y, Z" lives in livedata config rather than as a schema validator. Operational flexibility at the cost of the schema surface carrying less semantic guarantee.
- **(b) Build a livedata-side override mechanism.** Schema constraints can be relaxed locally during incidents, re-synced afterwards. Operational flexibility at the cost of a second authority over the same invariants — which tends to drift.

Discussion: which failure classes can we accept *requiring* cross-repo coordination to fix, and which cannot? Answer before extraction, not after.

### Version mismatch across services

Once `ess.schemas` is an installed dependency, backend services and the dashboard have independent install environments. Concrete failure modes:

- Backend at schemas v1.2, dashboard at v1.3 — dashboard renders fields the backend does not emit, users see ghost inputs that do nothing.
- Backend at v1.3 (new required field), dashboard at v1.2 — dashboard cannot validate what arrives, workflow appears broken from the UI side.
- Instrument subclass gains a field in v1.3 during a rolling deploy — the same workflow briefly responds differently depending on which backend instance answered; the dashboard caches a stale shape.

**One version mismatch away from losing an hour of beam time.** Mitigation needs: backward-compatibility discipline on schemas releases (new fields optional with defaults for ≥1 release), a single source of truth for "this deployment uses schemas vN" that every livedata component derives from, CI that blocks releasing components with different pins.

Discussion: version-pinning strategy. A single `schemas = "1.3.0"` line in `livedata/pyproject.toml` that dashboard and backend both honor, or per-service pins with manual coordination? The first is simpler but moves the entire livedata monorepo together; the second allows faster iteration on one service but fails exactly in the mismatch scenarios above.

### Rollback asymmetry

- Rolling back livedata with schemas unchanged: usually safe — livedata falls back to using a subset of what schemas offers.
- Rolling back schemas with livedata unchanged: unsafe — livedata may rely on a field the earlier schemas version did not have.

The on-call during a beam-time incident needs to know which way to turn the crank without reading the plan. Document the asymmetry and rehearse the rollback path before extraction, not during the first incident.

### Dev-loop friction on schema changes

Iterating on a schema change post-extraction requires editable installs across two checkouts (`pip install -e ../ess-schemas` in the livedata venv). One-time setup cost per developer, plus ongoing cost that reproducing schema-touching bugs in a clean devcontainer needs both repos in sync. The standalone schemas test job (Stage 1) absorbs much of this — most schema bugs should be caught there before reaching livedata integration — but not all.

### Breaking-change blast radius

Today a DREAM-only schema change touches DREAM's files. Post-extraction, a schemas-package change affects all instruments atomically, even when semantically scoped to one. Every schemas release implicitly forces every instrument team to re-verify against the new version. Mitigation is discipline rather than enforcement: treat framework-level types (`WorkflowSpec`, `ArraySpec`, `WorkflowId`, `ComponentSpec`) as hard-frozen once extracted, let instrument subpackages churn freely. Not enforceable by CI; requires a convention that holds under pressure.

### How these risks interact with the "extract only on real demand" rule

Each of the above is an argument to delay extraction until the benefit is concrete — and to front-load the design decisions that make extraction survivable (validator policy, version-pin strategy, rollback documentation) well before the packaging work begins. These are not packaging details; they shape what the extracted surface can safely contain.

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

- **Pre-migration audit.** Grep `handlers/detector_view_specs.py`, `LogicalViewConfig`, `SpectrumViewSpec`, and similar types for scipp-dependent fields (`Callable[[sc.DataArray, ...], ...]` and the like). Anything that would leak scipp across the boundary needs decomposition — split callables into runtime-side transforms, keep only declarative fields in what will become schema types — *before* the relevant file relocates. Discovering this mid-move is worse than doing it first.
- Create `schemas/` with `__init__.py`, `core.py`, `instrument.py`, `arrays.py`, `parameters.py`, an empty `workflows/` subpackage, and an empty `instruments/` subpackage.
- Introduce `ArraySpec` / `CoordSpec`. Migrate `WorkflowOutputsBase` and all generic `*Outputs` to `ArraySpec`.
- Move core spec types.
- Move `ComponentSpec`, shared parameter models.
- Move generic workflow schemas into `workflows/{monitor,detector,timeseries}.py`.
- Split `Instrument` into `InstrumentSpec` + runtime wrapper; introduce registry API.
- Update consumers of output templates (plotter selection, timeseries detection, dashboard templating) to work with `ArraySpec`.
- Add runtime helper for `ArraySpec → sc.DataArray` at the one or two sites that need it.
- Add the import-boundary CI check.
- **Add a standalone schemas test job.** A CI env (tox or equivalent) that installs `ess.livedata.schemas` with only pydantic as a runtime dep (no scipp, no ess.livedata) and runs its unit tests. This is the real proof of the dependency boundary. `import-linter` only verifies that imports don't reference forbidden modules; it does not prove the package actually *works* standalone — a subtle difference that matters when extraction eventually happens.

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

