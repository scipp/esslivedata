# ADR 0003: Unified declaration model for workflow context inputs

- Status: accepted
- Deciders: Simon
- Date: 2026-05-21

## Context

ADR 0002 settles where the context-stream gate lives (at the JobManager) and the failure modes it prevents. It does not settle where context streams are declared. The implementation that landed alongside ADR 0002 derives the gate set from `StreamProcessorWorkflow.context_keys` and threads context-stream declarations through `AuxSources`. Three sharp edges:

1. **`AuxSources` carries two unrelated concerns.** It was built for user-facing dynamic-aux selection (the `AuxInput.choices` widget side). Context streams piggyback on it for routing, declared as plain-string aux entries that the UI happens to ignore. ROI and bifrost motion both do this. The leak is structural: a routing primitive doubles as a workflow-execution primitive.

2. **Two parallel declarations for the same stream.** For LOKI's `detector_carriage`, `LOKI_DYNAMIC_TRANSFORMS` (a per-source dict in `specs.py`) is passed both to `DetectorROIAuxSources(dynamic_transforms=...)` (for routing) and to `DetectorViewFactory(dynamic_transforms=...)` (for graph wiring). The names must agree; nothing enforces it. For ROI, the names `'roi_rectangle'` / `'roi_polygon'` appear in `DetectorROIAuxSources.inputs`, in `DetectorROIAuxSources.initial_context_messages`, and in `detector_view/factory.py`'s `context_keys`. Three sites, manual sync.

3. **`Workflow.context_keys` is a protocol leak.** Only `JobFactory.create` consults it, and only at job-creation time. Forcing every `Workflow` implementation to expose it (current implementations: SPW with the actual keys, `AreaDetectorView` and `TimeseriesStreamProcessor` with empty stubs) couples a configuration concern to the runtime protocol.

A separate piece, today's `LogContextBinding` on `Instrument`, already covers the instrument-scoped half cleanly (bifrost rotation streams). It is f144-specific in name but not in shape; the spec-scoped (ROI) half has no equivalent first-class type.

Additional constraint: the dashboard imports workflow specs to generate UI. `specs.py` files must remain free of workflow-key imports — the dashboard must not depend on instrument-specific Sciline pipelines.

### Param-dependent context — explicit non-goal

Today's detector-view and monitor-view factories wire position-context unconditionally for sources that have a motion binding, regardless of `coordinate_mode` (TOA vs wavelength). A precise design would have the gate fire only in wavelength mode (TOA mode does not consume position). We accept the over-gate: instruments wiring a stream as context must keep the producer healthy. If "scientist must plot raw detector with broken motion controller" becomes a real operational need, the answer is to split the spec into a TOA-only and a wavelength variant, not to introduce param-dependent gating. Today this need is hypothetical; YAGNI.

## Decision

Evolve the existing `LogContextBinding` into a general `ContextInput` declaration used at both instrument and spec scope. Drop the "Log" prefix (semantically already general); add two optional fields with sensible defaults so existing call sites are unaffected:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ContextInput:
    """Declaration of one context-stream input to a workflow."""
    stream_name: str                                                # internal stream name (unresolved)
    workflow_key: Any                                               # opaque; consumed by the factory
    dependent_sources: frozenset[str]
    stream_resolver: Callable[[JobId, str], str] | None = None      # default identity: wire name == stream_name
    seed_factory: Callable[[JobId], Message] | None = None          # default: no seed
```

Field semantics:

- `stream_name` is both the *workflow-facing* name (the key the factory uses in `SPW.context_keys`) and the *unresolved* internal name.
- `stream_resolver`, when set, maps `(job_id, stream_name)` to the **wire** stream name used by routing and the gate. The bifrost case leaves the resolver unset and `wire == stream_name`. The ROI case sets a resolver that returns `f"{job_id}/{stream_name}"`.
- `seed_factory`, when set, produces the cold-start message fired at `schedule_job` time.

Two declaration sites, both producing `ContextInput` records:

1. **`instrument.add_context_input(stream_name, workflow_key, dependent_sources, *, stream_resolver=None, seed_factory=None)`** — source-scoped. Rename of `add_log_context_binding` plus two optional kwargs. Stored on `Instrument.context_inputs` (rename of `log_context_bindings`). Used for instrument-property context (motion; potentially temperature).

2. **`spec_handle.add_context_input(stream_name, workflow_key, *, stream_resolver=None, seed_factory=None)`** — spec-scoped. Late-bound from `factories.py` (mirrors `attach_factory()`), keeping `specs.py` free of workflow-key imports. `dependent_sources` is defaulted by the handle to `frozenset(spec.source_names)`. Stored on a new `WorkflowSpec.context_inputs` list.

`JobFactory.create(job_id, config)` iterates the union, filters by source membership, materialises wire stream names and seeds, then propagates:

```python
matching: list[ContextInput] = [
    *(ci for ci in instrument.context_inputs if job_id.source_name in ci.dependent_sources),
    *(ci for ci in spec.context_inputs       if job_id.source_name in ci.dependent_sources),
]
context_keys = {ci.stream_name: ci.workflow_key for ci in matching}
context_stream_names = {
    ci.stream_resolver(job_id, ci.stream_name) if ci.stream_resolver else ci.stream_name
    for ci in matching
}
seed_messages = [ci.seed_factory(job_id) for ci in matching if ci.seed_factory is not None]

workflow = factory(
    source_name=job_id.source_name, params=...,
    context_keys=context_keys,
)
job = Job(..., context_stream_names=context_stream_names)
on_schedule_seed(seed_messages)
```

`Job` carries a field→wire mapping so that incoming wire-stream messages get remapped to the workflow-facing `stream_name` before reaching SPW. The mapping derives from `matching`; the gate compares against the wire-name set.

`AuxSources` is slimmed to dynamic, user-selectable aux only — no `initial_context_messages`, no context-flavoured entries.

`Workflow.context_keys` comes off the protocol. SPW takes `context_keys` as a constructor argument and that is the end of the dict's journey.

`Job.context_aux_stream_names` (introduced by ADR 0002's implementation) is renamed to `Job.context_stream_names` — the value is no longer a subset of aux.

## Alternatives considered

| Option | Notes |
|---|---|
| **Evolve `LogContextBinding` into `ContextInput`; two declaration sites, one record shape (chosen)** | Single type covers both scopes. Existing instrument-binding API surface preserved (with optional kwargs added). Late-binding for the spec side keeps `specs.py` workflow-key-free. |
| Keep `LogContextBinding` for instrument scope; add a separate `SpecContextInput` type for spec scope | Two parallel types with overlapping shapes. The differences are all in *defaults* (and which API call you make), not data; a hierarchy adds friction without paying its way. |
| One declaration on `WorkflowSpec` as a constructor field carrying `workflow_key` | Drags workflow-key imports into `specs.py` and therefore into the dashboard. Rejected. |
| Spec lists fields with stream resolvers; factory declares workflow keys separately; JobFactory cross-references by field name | Two declarations would still need to agree on field names; restores the AuxSources-vs-factory drift in a different form. |
| Spec declares `context_inputs_for(params, source) -> set[ContextInput]` (callable) | Supports param-dependent gating but pushes branching logic into specs and adds a callable to the declaration. Rejected per the param-dependence non-goal. |
| Keep `Workflow.context_keys` as the gate source (status quo from ADR 0002 implementation) | Workable. Leaves the `AuxSources`/factory duplication unaddressed and the protocol leak in place. |

## Key design choices

### Workflow keys stay out of `specs.py` via late-binding

`spec_handle.add_context_input(...)` is invoked from `factories.py`, the same module that owns `attach_factory()`. The `SpecHandle` already references the registry; it gains a second decorator-style method. Dashboards that import `specs.py` to render UI never see a workflow-key import. (Instrument bindings already live in `factories.py` today.)

### Two scopes, one record shape

Both declaration sites produce `ContextInput`. Instrument bindings filter by `dependent_sources` explicitly. Spec-level entries default `dependent_sources` to the spec's source names, so they apply uniformly across the spec. The two cover the natural axes:

- *Motion is a property of the source — but not every workflow on that source consumes it.* `loki_detector_0` feeds the absolute-position-dependent `xy_projection` view, the `i_of_q` reduction, AND the `tube_view` (a logical sum that does not use position) and `ratemeter`. The first two need carriage; the latter two would gate forever on a stream they never read. Same shape on bifrost's `unified_detector`: cuts need rotation, the detector view and ratemeter do not. The right scope is therefore *spec-level*, declared per-handle in `factories.py`, not instrument-level. Instrument-level scope remains the right choice only for streams genuinely consumed by every spec on the source (today: none).
- *ROI is a property of the workflow.* Only detector-view workflows have ROI; reduction workflows on the same source do not. Spec-scoped.

Whole-instrument streams that are needed by every workflow on every source (a hypothetical sample-temperature stream consumed everywhere) can be declared with `dependent_sources` covering all instrument source names; this gates every workflow on every source, which matches the intended semantics.

`dependent_sources` may be a *strict subset* of a spec's `source_names` — e.g. a multi-detector spec where only one source carries an f144 stream. Today no live case exercises this; the historical LOKI example was reframed as spec-scoped above.

### Instrument bindings are source-scoped, not spec-scoped

A binding scoped to `{'unified_detector'}` activates for *every* spec whose `source_names` includes `unified_detector` — detector_view, qmap, ratemeter all gate on it indiscriminately. **If a context stream applies to a source but only some workflows on that source should gate on it, declare per-spec via `spec_handle.add_context_input` rather than as an instrument binding.** Documented as the routing rule. Today nothing fits the "shared source, divergent context need" shape — if it ever does, the answer is `add_context_input` on the handle.

### Routing vs gating

Routing (Kafka subscription set) is decided per *namespace* at service startup, ahead of any specific job. It can over-subscribe — the bandwidth cost of an unused f144 stream is negligible. Gating is decided per *job* at `JobFactory.create`, using only the declarations that resolve for that specific (spec, source). Gating must be precise; routing only needs to be permissive.

Routing pickup for instrument bindings is via `config/route_derivation.py:gather_source_names`. Spec-level `add_context_input` entries contribute to the subscription set only when `stream_resolver is None` — entries with a resolver are job-scoped (today: ROI, which routes via a dedicated `StreamKind.LIVEDATA_ROI` topic registered at service startup) and `gather_source_names` cannot meaningfully resolve them at namespace-startup time.

The routing-pickup extension is a co-requirement of this ADR, not a follow-up: without it, bifrost rotation and LOKI carriage streams are never subscribed by the `detector_data` namespace and the gate stays closed indefinitely. It must land in the same PR or in an explicit pre-merge dependency.

### Seeding

Seeds live on the context-input record: `seed_factory(job_id)` returns the cold-start `Message`. The `Instrument` and `spec_handle` APIs both accept it, but instrument-property context streams (motion, temperature) have no meaningful "no message yet" steady state; if the producer is down, the gate stays closed and the workflow does not run, which is the correct behaviour per ADR 0002. In practice the seed is supplied only for spec-level ROI today.

JobManager fires seeds at `schedule_job` via the existing `preprocess_messages` path (see ADR 0002). Reset does not re-seed: context accumulators live in `MessagePreprocessor` above the job and are not cleared by `Job.reset`, so the previously-seen value persists across the reset.

`JobManager._seed_initial_context` is also responsible for marking the seeded wire-stream names in `_seen_context_streams[job_id]`, so the gate opens on the first tick without re-firing `set_context` from cached state. The seeds and the bookkeeping live together; `JobFactory.create` returns the seed list alongside the `Job` rather than carrying both responsibilities.

### LOKI `transform_name` carrier

`ContextInput` declares `stream_name` and `workflow_key` but no graph-side label. LOKI's `add_dynamic_transform` needs the NeXus path (e.g. `/entry/instrument/detector_carriage/value`) that today lives in `TransformValueStream.transform_name`. The collapse of `LOKI_DYNAMIC_TRANSFORMS` into one `spec_handle.add_context_input(...)` call therefore loses one piece of information that the factory still needs.

Resolution: keep a small per-source `dict[str, str]` (source_name → transform_name) local to `loki/factories.py`, consulted by the factory when it sees `TransformValueLog` in `context_keys`. The dict is one-line per LOKI source today, well-localised, and does not pollute the cross-instrument `ContextInput` shape. If a second instrument needs the same plumbing the right move is to extend `ContextInput` with an opaque `payload: Any` field; meanwhile YAGNI.

### Workflow protocol stays pure

`Workflow.context_keys` is removed. SPW's constructor accepts `context_keys: dict[str, Any] | None`; nothing else queries the workflow for context. `AreaDetectorView` and `TimeseriesStreamProcessor` lose their empty-stub `context_keys` property. New workflow types do not inherit a configuration concern.

## Consequences

- `LogContextBinding` renamed to `ContextInput` in `config/stream.py`. Two optional fields added (`stream_resolver`, `seed_factory`); existing instantiations unchanged.
- `Instrument.log_context_bindings` → `Instrument.context_inputs`. `Instrument.add_log_context_binding` → `Instrument.add_context_input`. Mechanical rename across declarations and call sites. (Bifrost rotation and LOKI carriage both end up spec-scoped per the section above; the instrument-level API remains for streams genuinely consumed by every spec on a source.)
- `Instrument.get_context_keys(source_name)` continues to work — its body iterates `context_inputs`, filters by `dependent_sources`, returns `{ci.stream_name: ci.workflow_key for ci in matching}`.
- `DetectorROIAuxSources` is removed entirely. ROI declarations migrate to `spec_handle.add_context_input(...)` calls in `detector_view/factory.py` (or a colocated module).
- `bifrost_aux_sources` and `aux_sources=bifrost_aux_sources` arguments in the qmap registrations go away. The existing instrument-binding calls in bifrost's `factories.py` cover both routing (via the `gather_source_names` extension) and gating.
- LOKI's `LOKI_DYNAMIC_TRANSFORMS` dict and the dual declaration in `specs.py` + `factories.py` collapses into a `spec_handle.add_context_input(stream_name='detector_carriage', workflow_key=...)` call on the position-dependent handle(s) in `factories.py`. The `transform_name` (NeXus path) lives in a small per-source dict local to `loki/factories.py` (see § "LOKI transform_name carrier").
- `WorkflowSpec` grows a `context_inputs: list[ContextInput]` field (empty by default).
- `SpecHandle` gains an `add_context_input(...)` method that appends to the underlying spec's list, defaulting `dependent_sources` to `frozenset(spec.source_names)`.
- `JobFactory.create` learns to merge instrument + spec context inputs and pass the resulting `context_keys` to the factory. Factories that today receive `aux_source_names` and reach into a `dynamic_transforms` dict lose that branch — `context_keys` arrives ready-made.
- `Job.context_aux_stream_names` renamed to `Job.context_stream_names`. Tests updated.
- `Workflow.context_keys` removed from the protocol. SPW keeps its constructor arg; other workflow implementations drop the property stub.
- Per ADR 0002, the gate mechanism inside `JobManager` is unchanged; only the source of the gate set moves.
- Routing-pickup extension (inventory item 2) is a prerequisite of this ADR for the bifrost/LOKI motion case. Spec-level `add_context_input` entries need the equivalent routing-pickup treatment so the namespace subscribes to their resolved wire stream names.
- Validation: a spec-level `add_context_input` whose materialised wire-stream name collides with an instrument binding's wire-stream name in the merged set is a registration error. Test coverage required.
- The `Workflow` protocol becomes minimal: `accumulate`, `finalize`, `clear`. Any context handling is an SPW-internal concern, opaque to `Job` and `JobManager`.

## Addendum: changes during implementation

The decision text above captures the design as proposed. The following amendments were made during implementation. They preserve the core decision (one declaration model, two scopes, JobManager-level gate) but refine several points the original draft underweighted.

### Two record shapes via subclass split

The proposal had one `ContextInput` dataclass covering both direct-bind (workflow_key) and chain-patch (transform_path) cases via mutually-exclusive optional fields. As `transform_path` and `log_key` were added (see next item), the runtime `__post_init__` discriminator and the three downstream `transform_path is not None` checks became a sum-type-pretending-to-be-product-type smell. Final shape:

```python
class ContextInput:                            # abstract base, not instantiable
    stream_name: str
    dependent_sources: frozenset[str]
    stream_resolver: Callable | None
    seed_factory: Callable | None

class DirectBindContextInput(ContextInput):
    workflow_key: Any                          # Sciline key

class ChainPatchContextInput(ContextInput):
    transform_path: str                        # NeXus depends_on chain entry
    log_key: type[ValueLog]                    # per-binding Sciline key
```

Discrimination at consumer sites is `isinstance`-based. `Instrument.add_context_input` dispatches on its kwargs to the matching subclass; `SpecHandle.add_context_input` accepts direct-bind kwargs only (chain-patch fields are absent from its signature).

### Chain-patch fields on the record, not a separate dict

The proposal's "LOKI transform_name carrier" section kept a per-source dict (`_LOKI_TRANSFORM_NAMES`) local to `loki/factories.py`. That section is superseded: the NeXus path lives on `ChainPatchContextInput.transform_path`, and `Instrument.apply_dynamic_transforms(workflow, components)` reads the registry, groups bindings by component type, and patches the workflow's `NeXusTransformationChain[T, SampleRun]` provider in place. The same seam works for the original detector-view factory and for `_i_of_q_factory` (which receives a pre-built `LokiWorkflow()`), closing the originally-deferred #922 motivation.

### Per-binding ValueLog subclasses for multi-transform chains

The proposal's `workflow_key: Any` field treated the Sciline key as opaque. For chain-patch this is too coarse: a single Sciline key (`TransformValueLog`) shared across bindings would collapse multiple dynamic transforms into one node. `ChainPatchContextInput.log_key: type[ValueLog]` is a per-binding subclass (e.g. `class DetectorCarriageLog(ValueLog): ...`), giving each binding a distinct Sciline parameter. A new validator (`_validate_context_input_log_key_uniqueness`) rejects shared keys across different streams; another (`_validate_chain_patch_stream_consistency`) rejects same-stream entries that disagree on `(transform_path, log_key)`.

### Instrument scope as default for motion, with `skip_motion` opt-out

The "Instrument bindings are source-scoped, not spec-scoped" section recommended *spec scope* for motion. Implementation flipped this:

> Motion-context declarations live at *instrument* scope; specs that consume a source but not the geometry value (e.g. LOKI `tube_view`, bifrost `detector_ratemeter`) opt out via `SpecHandle.skip_motion()`.

Rationale: the spec-scope-by-default model makes "added a new spec consuming the source, forgot to declare carriage" a silent-wrong failure (the workflow runs with stale geometry). Instrument scope plus opt-out makes the symmetric mistake noisy: a forgotten `skip_motion()` produces a gate-on-unused-stream, not silently-wrong output. See `docs/developer/plans/motion-context-wiring.md` for the full discussion.

The validator now suppresses instrument-scope bindings for specs that have opted out, and rejects context-stream names that collide with `aux_sources` field names on the same spec (a context wire-name would silently overwrite the aux entry in the merged field→wire mapping at `JobFactory.create`).

### Job carries aux and context as separate maps

The proposal's "splice context wire names into `aux_source_names`" worked but conflated semantics: `Job.aux_source_names` returned both user-selected aux and framework-injected context wire names. Final shape carries them as separate maps:

- `Job.aux_source_names` — user-selected aux only (workflow `AuxSources`).
- `Job.context_wire_names` — framework-routed context wire names.
- `Job.input_stream_names` — combined, used by routing call sites in `JobManager`.

`JobFactory.create` still passes a single merged dict to the factory (the workflow boundary doesn't distinguish), but `Job` keeps them split for accounting.

### CI validator for `transform_path`

The original "Validation" bullet covered wire-name collisions only. Implementation adds `tests/config/motion_binding_test.py`, which walks every chain-patch binding's `transform_path` against the registered NeXus geometry artifact and fails if the path doesn't appear on the declared consumer's `depends_on` chain. Closes wrong-path as a silent failure mode without putting NeXus introspection on the construction hot path.

### `synthesise_provider` parameter-name validation

`synthesise_provider` (used by `apply_dynamic_transforms` to build per-component fused providers) validates that the synthesised function name and every parameter name match `^[a-zA-Z_][a-zA-Z0-9_]*$` before splicing into the `exec` template. Closes a latent code-injection risk if a caller ever derived a parameter name from external input.

### Module placement: `ValueLog` lives in `config/`

`ValueLog` was initially placed in `handlers/value_log.py`. Because `WorkflowSpec.context_inputs: list[DirectBindContextInput]` requires pydantic forward-ref resolution to know about `ValueLog`, the placement created a `config/` → `handlers/` import. Final placement is `config/value_log.py` next to `ContextInput`. `synthesise_provider` lives in `handlers/dynamic_transforms.py` (its only caller).
