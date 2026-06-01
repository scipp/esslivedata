# ADR 0003: Unified declaration model for workflow context bindings

- Status: accepted
- Deciders: Simon
- Date: 2026-05-21

## Context

ADR 0002 settles where the context-stream gate lives (at the JobManager) and the
failure modes it prevents. It does not settle where context streams are *declared*,
nor how a context value reaches the Sciline pipeline. Before this ADR those concerns
were spread across several mechanisms with three sharp edges:

1. **`AuxSources` carried two unrelated concerns.** It was built for user-facing
   dynamic-aux selection (the `AuxInput.choices` widget side). Context streams
   piggybacked on it for routing, declared as plain-string aux entries the UI
   happened to ignore. A routing primitive doubled as a workflow-execution
   primitive.

2. **Two parallel declarations for the same stream.** LOKI's `detector_carriage`
   was declared once for routing and once for graph wiring, with nothing enforcing
   that the names agree.

3. **`Workflow.context_keys` was a protocol leak.** Only `JobFactory.create`
   consulted it, only at job-creation time, yet every `Workflow` implementation had
   to expose it (the actual keys for SPW, empty stubs for `AreaDetectorView` and
   `TimeseriesStreamProcessor`). A configuration concern leaked into the runtime
   protocol.

A standing constraint shapes the solution: the dashboard imports workflow `specs.py`
to generate UI, so `specs.py` must stay free of workflow-key imports — the dashboard
must not depend on instrument-specific Sciline pipelines.

### Param-dependent context — explicit non-goal

Detector-view and monitor-view factories wire position context unconditionally for
sources that have a motion binding, regardless of `coordinate_mode` (TOA vs
wavelength). A precise design would gate only in wavelength mode (TOA does not consume
position). We accept the over-gate: an instrument wiring a stream as context must keep
the producer healthy. If "plot raw detector with a broken motion controller" becomes a
real operational need, the answer is to split the spec into a TOA-only and a wavelength
variant, not to introduce param-dependent gating. Today this is hypothetical; YAGNI.

## Decision

One declaration record, `ContextBinding`, declarable at two scopes, feeding both the
factory's `context_keys` (wired into `set_context`) and the job's gating set (ADR
0002).

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ContextBinding:
    stream_name: str                 # internal stream name; also the workflow-facing key
    workflow_key: Any                # opaque Sciline key, consumed by the factory
    dependent_sources: frozenset[str]
```

It is a single concrete dataclass — no subclasses, no resolver/seed callables. Two
flavours share the record, discriminated by the *type* of `workflow_key`:

- **Direct parameter.** `workflow_key` is an arbitrary Sciline key (e.g.
  `InstrumentAngle[SampleRun]`, `ROIRectangleRequest`). The stream value is fed
  straight into `set_context`.
- **NXtransformations chain patch.** `workflow_key` is a `ValueLog` subclass. The
  value is patched into the NeXus `depends_on` chain rather than bound directly.
  `Instrument.apply_dynamic_transforms` discovers these by
  `issubclass(workflow_key, ValueLog)`. The chain entry to patch is *derived* from
  `stream_name` (the f144 substream's `nexus_path`) via `Instrument.chain_patch_path`
  — there is no separate `transform_path` field, keeping a single source of truth in
  the parsed stream definitions.

### Two declaration sites

1. **`Instrument.add_context_binding(*, stream_name, dependent_sources, workflow_key)`**
   — source-scoped. Stored on `Instrument.context_bindings`. Used for instrument-property
   context (motion/geometry). Chain-patch bindings live at instrument scope only.

2. **`SpecHandle.add_context_binding(stream_name, workflow_key, dependent_sources=None)`**
   — spec-scoped. Late-bound from `factories.py` (mirroring `attach_factory()`),
   keeping `specs.py` free of workflow-key imports. `dependent_sources` defaults to the
   spec's `source_names`. Stored on `WorkflowRegistration.context_bindings`. Rejects
   `ValueLog` keys: chain-patch is instrument-scope. There are no callables — the wire
   name equals `stream_name` and there is no seed, so the gate stays closed until the
   producer publishes (correct for a context with no safe default).

### Resolution at job creation

`JobFactory.create` merges instrument- and spec-scope bindings filtered by source
membership, then builds the factory input and the gate set:

```python
instrument_inputs = [] if registration.skip_instrument_contexts \
    else instrument.context_bindings
matching = [ci for ci in (*instrument_inputs, *registration.context_bindings)
            if job_id.source_name in ci.dependent_sources]
context_keys     = {ci.stream_name: ci.workflow_key for ci in matching}
context_streams  = {ci.stream_name for ci in matching}   # wire name == stream_name
```

`context_keys` goes to the factory; `context_streams` becomes `Job.gating_streams`.
`JobFactory.create` returns a bare `Job`. `AuxSources` is slimmed to dynamic,
user-selectable aux only — no context-flavoured entries, no `initial_context_messages`.

### `skip_instrument_contexts` opt-out

Motion is a property of the *source*, but not every spec on that source consumes the
geometry value (LOKI `tube_view`, bifrost `detector_ratemeter`, the bifrost detector
view). Such a spec declares `SpecHandle.skip_instrument_contexts()`, which drops
instrument-scope bindings for that spec's jobs (spec-scope bindings are unaffected).
The opt-out is called from `factories.py`, co-located with the
`Instrument.add_context_binding` it negates: the flag is meaningless without the
binding it cancels, and the rationale is implementation knowledge. It needs no
workflow-key import, so — unlike `add_context_binding` — its placement is not forced
by the `specs.py` import constraint; co-location is a legibility choice.

Instrument scope as the default — with an opt-out — is chosen over spec scope as the
default deliberately. Spec-scope-by-default makes "added a new spec on the source,
forgot to declare carriage" a *silently-wrong* failure (the workflow runs with stale
geometry). Instrument-scope-plus-opt-out makes the symmetric mistake *noisy*: a
forgotten `skip_instrument_contexts()` gates a workflow on a stream it never reads, so
the workflow visibly never runs rather than producing wrong output.

### Chain patching

`ValueLog` lives in `config/value_log.py`, next to `ContextBinding`, so `config` does
not import `handlers`. Each chain-patch binding declares its own `ValueLog` subclass,
giving it a distinct Sciline parameter so multiple dynamic transforms can coexist on
one workflow without colliding.

`Instrument.apply_dynamic_transforms(workflow, components)` selects the chain-patch
bindings matching each `(source, component_type)`, groups them by component type, and
builds one fused per-component provider (`add_dynamic_transforms` /
`synthesise_provider` in `handlers/dynamic_transforms.py`) that replaces essreduce's
`NeXusTransformationChain[T, SampleRun]` provider and writes the latest sample of each
`ValueLog` parameter into the chain. The same seam works for the detector-view factory
and for a pre-built `LokiWorkflow()`.

### Routing pickup

Routing (the Kafka subscription set) is decided per *namespace* at service startup,
ahead of any specific job, and may over-subscribe; the bandwidth cost of an unused
f144 stream is negligible. Gating is decided per *job* at `JobFactory.create` and must
be precise. `config/route_derivation.py:gather_source_names` therefore includes context
bindings in the subscription set: spec-scope entries by stream name, instrument-scope
entries when a spec in the namespace shares a source with the binding's
`dependent_sources`. This pickup is a co-requirement of this ADR, not a follow-up:
without it the namespace never subscribes to motion streams and the gate stays closed
indefinitely.

### Workflow protocol stays pure

`Workflow.context_keys` is removed from the protocol. SPW accepts `context_keys` as a
constructor argument and that is the end of the dict's journey. `AreaDetectorView` and
`TimeseriesStreamProcessor` lose their empty-stub property. The `Workflow` protocol
reduces to `accumulate`, `finalize`, `clear`; context handling is an SPW-internal
concern, opaque to `Job` and `JobManager`.

## Alternatives considered

| Option | Notes |
|---|---|
| **One `ContextBinding` record, two declaration sites (chosen)** | Single type covers both scopes and both flavours. Late-binding for the spec side keeps `specs.py` workflow-key-free. |
| Separate types per scope or per flavour (subclass split) | Overlapping shapes whose differences are in *defaults*, not data; a hierarchy adds friction without paying its way. Rejected. |
| One declaration on `WorkflowSpec` as a constructor field carrying `workflow_key` | Drags workflow-key imports into `specs.py` and therefore the dashboard. Rejected. |
| Spec lists stream fields; factory declares keys separately; cross-reference by name | Two declarations must still agree on names; restores the routing-vs-factory drift in a new form. Rejected. |
| Spec scope as default for motion | Makes a forgotten declaration silently-wrong (stale geometry). Instrument-scope-plus-opt-out makes it noisy instead. Rejected. |
| Per-source `transform_name` dict local to `loki/factories.py` | The chain path is derived from `stream_name` via `chain_patch_path`; a parallel dict would re-introduce manual sync. Rejected. |
| Keep `Workflow.context_keys` as the gate source | Leaves the `AuxSources`/factory duplication and the protocol leak in place. Rejected. |

## Key design choices

### Workflow keys stay out of `specs.py` via late-binding

`SpecHandle.add_context_binding(...)` is invoked from `factories.py`, the module that
owns `attach_factory()`. Dashboards importing `specs.py` to render UI never see a
workflow-key import.

### Two scopes, one record shape

Instrument scope is for source properties (motion), where chain-patch also lives. Spec
scope is for context that is a property of one workflow. The spec-scope path has no
current user — it is the extension point for the motivating future case, a
**sample-temperature** stream feeding one reduction workflow: gated (no safe default),
a real shared Kafka stream (wire name == `stream_name`), no seed. That drops straight
into `SpecHandle.add_context_binding` with no further machinery.

### Chain path derived, not declared

The NeXus transformation-chain entry a chain-patch binding writes is derived from its
`stream_name` via `Instrument.chain_patch_path`, so the f144 substream definition is the
single source of truth and a binding cannot disagree with the stream it patches.

## Consequences

- `ContextBinding` lives in `config/stream.py` as a single concrete dataclass
  (`stream_name`, `workflow_key`, `dependent_sources`). `ValueLog` lives in
  `config/value_log.py`.
- `Instrument` carries `context_bindings: list[ContextBinding]` and
  `add_context_binding(...)`; `apply_dynamic_transforms(workflow, components)` patches
  chain-patch bindings into the pipeline.
- `WorkflowRegistration` carries `context_bindings` and a `skip_instrument_contexts`
  flag; `SpecHandle` gains `add_context_binding(...)` and `skip_instrument_contexts()`.
- `JobFactory.create` merges instrument + spec bindings, passes `context_keys` to the
  factory, sets `Job.gating_streams`, and returns a bare `Job`.
- `AuxSources` is dynamic, user-selectable aux only. ROI remains an `AuxSources`
  (`DetectorROIAuxSources`, job-prefixed per job) — it is not a context binding, since
  ADR 0002 routes ROI as aux rather than gating it.
- `Workflow.context_keys` is removed from the protocol; SPW keeps its constructor arg,
  other implementations drop the stub.
- `gather_source_names` includes context bindings so the namespace subscribes to gated
  streams. Without it the gate never opens for motion.
- Validators enforce the record's invariants: chain-patch `(stream_name, workflow_key)`
  uniqueness; wire-name collisions across instrument/spec scope and against aux field
  names; `dependent_sources` against registered spec `source_names`. `synthesise_provider`
  validates identifier names before splicing into its `exec` template.
- `tests/config/motion_binding_test.py` walks every chain-patch binding's derived
  transform path against the registered NeXus geometry artifact and fails if the path is
  not on the declared consumer's `depends_on` chain — closing wrong-path as a silent
  failure without putting NeXus introspection on the construction hot path.
- Per ADR 0002, the gate mechanism inside `JobManager` is unchanged; this ADR settles
  only the declaration model that feeds it.
