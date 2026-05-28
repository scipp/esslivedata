# Motion-context wiring: problem statement and design space

Background document for the ongoing work on context-stream declarations
(see ADRs [0002][adr-0002] and [0003][adr-0003], and the branch
`jobmanager-context-gate`).

[adr-0002]: ../adr/0002-context-stream-gating-at-jobmanager.md
[adr-0003]: ../adr/0003-context-binding-declaration-model.md

The goal of this document is to capture what motion-context wiring is, how
it currently works in the codebase, what knowledge lives where, and which
constraints any reliable solution must respect — without prescribing a way
forward. End goal: a mechanism that minimises the risk of workflow
implementers hooking motion up to components incorrectly.

Companion document: [adr-0002-0003-review.html](./adr-0002-0003-review.html)
covers the broader review of ADRs 0002 and 0003. This document focuses
specifically on the motion-context wiring concern raised during that review.

## 1. What is "motion-context wiring"?

The Sciline workflow graphs underlying our detector reductions are
parametrised by absolute positions and orientations of components
(detectors, choppers, sample, monitors). Many of those positions are
*dynamic*: a motor moves; an f144 / NXlog stream publishes the new
readback value; the workflow graph must re-render the affected branch.

Concretely, NeXus encodes a component's position as a `depends_on` chain:
a linked list of transformations from the component, through intermediate
frames, to the laboratory root. Some transformations in the chain are
*static* (constant `value` attribute); others are *dynamic* (their `value`
is supplied by an f144 stream, identified by the transformation node's
`source` attribute).

Producing the absolute position of a component therefore requires:

1. Locating the component's `depends_on` chain in the NeXus file.
2. For every dynamic transformation in the chain, consuming the latest
   value from the corresponding f144 stream.
3. Patching that value into the Sciline graph so the downstream
   `Position[Component]` provider produces a correct result.

This is what we mean by *motion-context wiring*: the connection between
(a) a workflow that needs a component's position, (b) the NeXus
`depends_on` chain that defines how the position is computed, and (c) the
f144 streams that supply the dynamic values.

## 2. State in the current branch

The branch `jobmanager-context-gate` lands a unified declaration model
(`ContextBinding`) for context streams in general — including, but not
limited to, motion. See [the review HTML](./adr-0002-0003-review.html) §3
for the full mechanism.

### 2.1 The declaration record

From `src/ess/livedata/config/stream.py:91`:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ContextBinding:
    stream_name:        str
    workflow_key:       Any                                    # Sciline key
    dependent_sources:  frozenset[str]
    stream_resolver:    Callable[[JobId, str], str] | None = None
    seed_factory:       Callable[[JobId], Message] | None = None
```

For motion specifically: `stream_resolver` and `seed_factory` are both
`None`. The wire name equals the stream name; the gate stays closed until
the producer publishes a real value (motion has no meaningful "no message
yet" default).

### 2.2 Concrete declarations today

**LOKI rear-bank carriage** — at
`src/ess/livedata/config/instruments/loki/factories.py`:

```python
_LOKI_TRANSFORM_NAMES: dict[str, str] = {
    'loki_detector_0': '/entry/instrument/detector_carriage/value',
}

specs.xy_projection_handle.add_context_binding(
    stream_name='detector_carriage',
    workflow_key=TransformValueLog,
    dependent_sources=frozenset({'loki_detector_0'}),
)
```

The `_LOKI_TRANSFORM_NAMES` dict is passed to `DetectorViewFactory` as
`transform_names`. The factory consults it when it sees `TransformValueLog`
in `context_keys` and calls
`add_dynamic_transform(workflow, transform_name=path)`.

**Bifrost rotation** — at
`src/ess/livedata/config/instruments/bifrost/factories.py`:

```python
def _add_qmap_rotation_context(handle):
    handle.add_context_binding(
        stream_name='detector_tank_angle_r0',
        workflow_key=InstrumentAngle[SampleRun],
    )
    handle.add_context_binding(
        stream_name='rotation_stage',
        workflow_key=SampleAngle[SampleRun],
    )

for handle in (specs.qmap_handle, specs.elastic_qmap_handle,
               specs.elastic_qmap_custom_handle):
    _add_qmap_rotation_context(handle)
```

Bifrost has no `transform_name` dict equivalent because the cut workflow
consumes the f144 values directly as Sciline parameters, not by patching
a `depends_on` chain. The two cases — LOKI's chain patch and bifrost's
parameter binding — both feed the workflow via `set_context`, but the
graph wiring differs.

### 2.3 What the workflow implementer is responsible for, today

For each (workflow, component-whose-position-is-consumed) pair, the
implementer manually writes, in `factories.py`:

1. The wire stream name (`'detector_carriage'`).
2. The Sciline workflow key (`TransformValueLog` for chain patching, or
   `InstrumentAngle[SampleRun]` for a typed scalar).
3. The set of source names this binding applies to.
4. For LOKI-style chain-patch wiring: the NeXus path of the transformation
   to patch (`/entry/instrument/detector_carriage/value`), in a separate
   per-instrument `dict[str, str]` carried to `DetectorViewFactory`.

The framework, given those declarations, handles the rest:
`JobFactory.create` merges instrument- and spec-level entries, derives
the gate set, splices wire names into `aux_source_names` for routing,
and (for ROI-like cases) fires cold-start seeds. See HTML §3 for the
merge logic and §4 for the runtime gate.

### 2.4 Failure modes under manual wiring

The hand-written declarations expose several failure shapes:

- **Typos in `stream_name`** — caught at `Instrument.__post_init__` if the
  stream isn't in `instrument.streams`; raises
  `"ContextBinding references unknown stream …"`.
- **Typos in `dependent_sources`** — caught at registration by
  `_validate_binding_dependent_sources`; raises if a binding lists a
  source no spec advertises.
- **Wire-name collision** between instrument- and spec-level entries —
  caught by `_validate_context_binding_wire_name_collisions` (added in
  commit `2dd70b25`).
- **Wrong `workflow_key`** — no validation. A binding may name an
  unrelated Sciline key; the workflow factory has to be wired correctly
  for the binding to have effect. Mismatches surface as `UnsatisfiedGraphError`
  at first finalize, or as silently-wrong results if the key is valid but
  semantically wrong.
- **Missing declaration on a new spec that shares the source** — silent
  in the new model: the workflow runs without the context value (`None`
  in Sciline params) and either crashes or produces wrong output. This
  is exactly the failure mode ADR 0002's gate is designed to prevent,
  but only fires when `Job.context_stream_names` lists the stream — which
  in turn requires the declaration. Forgetting the declaration removes
  the gate's protection.
- **Wrong NeXus path** in `_LOKI_TRANSFORM_NAMES` — passed to
  `add_dynamic_transform`; produces a NeXus-side error if the path is
  invalid, or silently patches the wrong transformation if the path is
  valid but unintended.
- **Forgot to register on additional consumers** — `unified_detector`
  feeds three qmap-family specs; each needs the binding repeated (see
  the `_add_qmap_rotation_context` helper above). Adding a fourth qmap
  variant requires remembering to call the helper.

Note the asymmetry: typos and structural collisions are caught at
registration. Semantic errors (wrong key, wrong path, missing
declaration) are not — they surface at runtime, possibly silently.

## 3. What knowledge the NeXus file already encodes

The NeXus file under `src/ess/livedata/config/instruments/<inst>/`
(referenced via the `nexus_file` attribute on each `Instrument`) carries:

- **The full `depends_on` chain** for every component (detector, monitor,
  sample, choppers, etc.).
- **For each transformation in each chain**: its kind (translation,
  rotation), its vector, and its `value` source — either a constant
  attribute or a reference to an f144 group via the `source` attribute.
- **The f144 group itself**: topic, source, units. Already parsed by
  the `streams_parsed.py` codegen into `F144Stream` records.

What's *not* in the NeXus file: the mapping from f144 stream to Sciline
workflow key (`TransformValueLog` vs `InstrumentAngle[SampleRun]` vs …).
That's a workflow-side concern — it depends on what the workflow does
with the value, not what the value is.

## 4. Where each piece of knowledge currently lives

| Knowledge                                                | Source of truth                                      | Currently encoded in                            |
| -------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------- |
| Stream exists; topic; source; units                      | NeXus file                                           | `streams_parsed.py` (codegen) → `instrument.streams` |
| Which streams are dynamic transforms in which chain      | NeXus `depends_on` + `source` attributes             | Hand-written `_LOKI_TRANSFORM_NAMES`            |
| Which component a chain belongs to                       | NeXus `depends_on` head                              | Hand-written `dependent_sources` on `ContextBinding` |
| Whether a workflow needs a component's absolute position | Sciline graph (whether it consumes `Position[X]`)    | Implicit; workflow author knows it              |
| The Sciline key that should receive the dynamic value    | Workflow design                                      | Hand-written `workflow_key` on `ContextBinding`   |
| The NeXus path to patch                                  | NeXus `depends_on` chain                             | Hand-written `_LOKI_TRANSFORM_NAMES` value      |

Rows 1, 2, 3, 6 are derivable from the NeXus file. Rows 4, 5 are
workflow-internal — not derivable from NeXus.

## 5. Geometry context vs process context

The automation thesis applies cleanly to *geometry context* — values
participating in a `depends_on` chain.

It does *not* apply to *process context* — values that parametrise the
workflow but have no representation in NeXus geometry:

| Category         | Examples                                          | Encoded in NeXus geometry? |
| ---------------- | ------------------------------------------------- | -------------------------- |
| Geometry context | Detector carriage, sample rotation, chopper position (as a transformation) | Yes — `depends_on` chains  |
| Process context  | Chopper setpoint / phase, sample temperature, ROI selection, dashboard parameters | No                         |

Both today use the same `ContextBinding` carrier. The distinction matters
because the source of truth for *what should be wired* differs.

## 6. Sciline-side constraints

The framework hands the workflow factory a `context_keys: dict[str, type]`
mapping wire stream names to Sciline keys. The factory uses it in one
of two ways:

- **Direct parameter binding** (bifrost): each entry becomes a
  `set_context({key: value})` call. The key is a typed Sciline key
  (`InstrumentAngle[SampleRun]`).
- **Chain patching** (LOKI): one entry has the value `TransformValueLog`;
  the factory calls `add_dynamic_transform(workflow, transform_name=…)`
  which substitutes the value into the chain at `transform_name`. The
  `TransformValueLog` key today is singular per chain.

Implications for automation:

- A chain with more than one dynamic transform needs the Sciline key to
  be parameterisable by transform position (or one key per transform).
  Today no chain has more than one dynamic transform in use; the
  singular key is sufficient. A future case (e.g. bifrost's
  `unified_detector` if both rotation streams entered the chain rather
  than being parameter-bound) would need this.
- The dichotomy "chain patch vs parameter bind" is itself a workflow
  decision: bifrost could have been wired via chain patching but chose
  parameter binding because the cut workflow consumes the angles directly,
  not via a chain-derived `Position`. So automation derived from the
  NeXus chain alone cannot decide which side it lands on — that's a
  workflow-design choice.

## 7. Constraints any reliable solution must address

Independent of which mechanism is chosen:

- **Multi-spec sources.** `loki_detector_0` feeds `xy_projection`,
  `tube_view`, `i_of_q`, `ratemeter`; only some consume position. Per-spec
  scope is the correct granularity. (This is the conclusion already in
  the current branch — see HTML §3 "Two scopes" and the commit history
  for the late instrument→spec scope shift.)
- **Per-source variation.** `loki_detector_0` has a `detector_carriage`
  dynamic; `loki_detector_1` does not. The set of streams to wire varies
  with the source within one spec.
- **Opt-out / freeze cases.** Calibration or diagnostic modes may want
  to override or freeze a position. Any automation must allow the
  workflow to bypass it.
- **Observability.** The workflow implementer should be able to inspect
  what streams a workflow consumes. Automation that hides the wiring
  makes operational debugging harder; logging the auto-emitted records
  or an inspection API is part of the requirement.
- **Error timing.** Registration-time errors are loud and CI-catchable.
  Runtime errors (especially silent-wrong results) are dangerous. A
  reliable solution shifts as many errors as possible to registration
  time.
- **The `transform_name` problem.** Today this is a per-source dict
  local to LOKI's `factories.py` (see ADR 0003 § "LOKI transform_name
  carrier"). The path is in NeXus; a reliable solution should not
  require hand-keying it.
- **Coupling to NeXus availability at registration.** Auto-wiring
  requires the NeXus file (or its codegen output) at registration time.
  This is already true for the current codegen path
  (`streams_parsed.py`); auto-wiring extends what the codegen needs to
  expose.
- **The chopper case.** Chopper *setpoint* and *phase* streams parametrise
  the wavelength frame, which is process context, not geometry. But
  chopper *position* is geometry context (the chopper is a NeXus
  component with a `depends_on` chain). A workflow consuming both must
  be able to declare both, even though they need different wiring
  mechanisms.

## 8. The design space

Three points on a spectrum, from most manual to most automated:

1. **Status quo.** Per-handle `add_context_binding(stream_name=…,
   workflow_key=…, dependent_sources=…)`; per-instrument `transform_names`
   dict. Workflow author writes everything.

2. **Declared-need, derived-wiring.** Workflow author declares "this
   handle uses position of component X" (a single call). The framework
   walks X's `depends_on` chain, identifies dynamic transformations, and
   emits the corresponding `ContextBinding` records and
   `add_dynamic_transform` calls. The workflow key remains a workflow
   concern — either auto-supplied for chain patching, or named explicitly
   for direct parameter binding.

3. **Graph-introspected, fully derived.** Framework inspects the workflow's
   Sciline graph at registration; for every `Position[X]` (or related
   key) the graph consumes, walks X's chain, identifies dynamics, emits
   bindings. Zero per-(workflow, component) declarations.

Open questions across the design space:

- Where does the chain walk happen — extension to the `streams_parsed.py`
  codegen (one-time per NeXus file), or runtime at `Instrument.__post_init__`
  (re-derived each process)?
- For non-geometry context (chopper setpoint, temperature, ROI): does
  the same declaration record (`ContextBinding`) carry it, or does it get
  its own type? Today the same record carries both; the question is
  whether that survives once geometry becomes automated.
- What is the inspection / debugging story for auto-wired bindings? At
  minimum: logging on startup, and an API surface that returns the
  effective set of `ContextBinding` records per (spec, source).
- How are workflow-side opt-outs declared (freeze, override)? As an
  argument to the chain walker, as a per-handle override, or via a
  separate Sciline-side mechanism that takes priority over auto-wiring?

## 9. Resolution on this branch

The branch lands a pragmatic refinement of option 1 that closes the
silent-omission failure mode without taking on codegen or graph
introspection. Three changes:

**1. Chain-patch paths live on `ContextBinding`.** A
`transform_path: str | None` field replaces the per-instrument
`transform_names` dict carried by `DetectorViewFactory`; when set, the
binding is chain-patch and carries its own author-declared
`log_key: type[ValueLog]` so multiple bindings can coexist on one
workflow without colliding on a shared parameter (the "Multiple dynamic
transforms per chain" item below is closed by this).
`Instrument.apply_dynamic_transforms(workflow, {source_name:
component_type, ...})` is the single seam: factories call it on the
workflow and the helper iterates the instrument's bindings, groups by
component type, and inserts one fused
`NeXusTransformationChain[T, SampleRun]` provider per type. The old
`transform_paths` kwarg dispatch through `WorkflowFactory.create` is
gone — factories that need chain patching call the helper directly.
Spec-scope chain-patch bindings are rejected at registration: the helper
reads only instrument-scope records, so a spec-scope declaration would
deliver an unused Sciline parameter (silent-wrong).

**2. Instrument scope is the default for motion; `skip_motion` opts
out.** LOKI's carriage and bifrost's two rotation streams move from
per-spec to instrument scope (one declaration each, with
`dependent_sources` naming the affected source). Specs that consume the
source but not the geometry (LOKI `tube_view`; bifrost
`detector_ratemeter`, `unified_detector_view`) call
`SpecHandle.skip_motion()` to drop out of the gate. The flag is a single
boolean — no per-stream list — because a spec spanning many sources
typically either needs all instrument-scope context or none of it.
`skip_motion=True` ignores *all* instrument-scope bindings for that spec;
spec-scope bindings (declared by the spec itself) are unaffected. Today
every instrument-scope binding is motion-related, hence the name; if a
non-motion instrument binding ever lands, rename.

**3. Failure-mode shift.** Before: forgetting to declare a binding on a
new spec consuming the source was silent (job runs without the dynamic
value). After: new specs pick up instrument-scope bindings by default;
forgetting to `skip_motion` on a non-consuming spec produces a *noisy*
gate-on-unused-stream (the job waits, doesn't process with stale data).
Silent-wrong is upgraded to noisy-slow.

### What this does *not* address

- **Direct-bind opt-out for non-motion.** `skip_motion` ignores all
  instrument-scope bindings indiscriminately. If a future instrument-
  scope binding is process context (not motion) and some spec wants to
  keep it while dropping motion, this needs finer-grained opt-out.
- **Orphan placeholders on monitors.** Some monitor chains carry empty
  NXlogs that no binding covers (LOKI `beam_monitor_m4`'s `trans_20`,
  tracked in ``tests/config/motion_binding_test.py``'s
  ``_KNOWN_ORPHAN_NXLOGS``). A workflow that loads such a monitor
  crashes at construction independently of the binding mechanism; the
  fix is either a shared NXlog declaration in the geometry generator
  or a separate f144 stream for the monitor position.

## 10. References

- [ADR 0002][adr-0002] — Context-stream gating at the JobManager.
- [ADR 0003][adr-0003] — Unified declaration model for workflow context
  inputs.
- [Block B implementation plan](./adr-0003-context-binding-implementation.md)
  — Sequencing of the ContextBinding migration.
- [adr-0002-0003-review.html](./adr-0002-0003-review.html) — Full review
  of both ADRs as landed on the `jobmanager-context-gate` branch.

### Key files in the branch

- `src/ess/livedata/config/stream.py` — `ContextBinding` record with
  `transform_path` and `log_key` fields.
- `src/ess/livedata/config/workflow_spec.py` — `WorkflowSpec.skip_motion`
  field.
- `src/ess/livedata/config/instrument.py` —
  `Instrument.add_context_binding` accepts `transform_path`/`log_key`;
  `Instrument.apply_dynamic_transforms` patches a workflow's chain
  providers from the instrument's bindings.
- `src/ess/livedata/config/instruments/loki/{specs,factories}.py` —
  instrument-scope carriage declaration; `_i_of_q_factory` calls
  `instrument.apply_dynamic_transforms`; `skip_motion` on tube_view.
- `src/ess/livedata/config/instruments/bifrost/{specs,factories}.py` —
  instrument-scope rotation declarations; `skip_motion` on
  detector_ratemeter and unified_detector_view.
- `src/ess/livedata/handlers/dynamic_transforms.py` —
  `build_patched_chain_provider` / `add_dynamic_transforms` (the fused
  per-component chain-patch provider; consumed by
  `Instrument.apply_dynamic_transforms`).
- `src/ess/livedata/handlers/value_log.py` — `ValueLog` base class plus
  `synthesise_provider` (the `exec`-based factory needed to build a
  Sciline provider with one parameter per binding).
- `src/ess/livedata/handlers/detector_view/factory.py` —
  `DetectorViewFactory` takes `instrument` and calls
  `instrument.apply_dynamic_transforms`.
- `src/ess/livedata/handlers/workflow_factory.py` —
  `SpecHandle.skip_motion`; `SpecHandle.add_context_binding` rejects
  chain-patch bindings (must live at instrument scope).
- `src/ess/livedata/core/job_manager.py` — `JobFactory.create` filters
  instrument-scope bindings via `skip_motion` and resolves
  `workflow_key` via `_resolve_workflow_key`.
- `tests/config/motion_binding_test.py` — CI walk of every chain-patch
  `transform_path` against the registered NeXus artifact, plus
  orphan-NXlog ledger.
