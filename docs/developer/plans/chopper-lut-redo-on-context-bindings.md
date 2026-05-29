# Redoing the chopper-LUT workflow on the context-binding mechanism

Issue: #894 (original chopper-workflow scope)

## Purpose

Assess whether the `chopper-workflow` branch's dynamic wavelength-LUT work can be
redone on the context-binding mechanism landed on `jobmanager-context-gate`
(ADRs 0002/0003), and on what scope. Conclusion: **yes, on a substantially
smaller scope, with no new context-wiring concepts.** This doc records the
mapping, the redo scope, and the residual seams.

The original chopper-workflow design and the upstream/producer-side pieces
(synthesizer internals, NeXus geometry script) are out of scope here — see
`dynamic-tof-lookup-table.md`. This doc only covers the wiring.

## Verdict

Three of the wiring pieces `chopper-workflow` invented are now generic and
already present on this branch. The only residual machinery is *not*
context-wiring: the synthesizer (producer-side) and a trigger primary stream
(firing model). The redo is mostly deletion plus a handful of
`add_context_binding` calls.

## Mapping: chopper-workflow piece -> new mechanism

| `chopper-workflow` piece | New-branch equivalent | Status |
|---|---|---|
| Per-chopper setpoint as aux cached via `is_context=True` accumulator, assembled into `DiskChoppers` outside sciline, replayed on job start | `SpecHandle.add_context_binding(stream_name=<chopper>_setpoint, workflow_key=<ScilineKey>)` — direct-param flavour, one call per chopper from `factories.py` | Subsumed |
| "All choppers locked" gating inside the synthesizer | `Job.gating_streams` + `_gate_pending_context` (ADR 0002), derived from the bindings | Subsumed, and better |
| Orchestrator empty-batch context-enrichment for the chopperless one-shot tick | `peek_pending_streams` + `get_context` empty-batch branch | Already landed |
| `resolve_stream_names` declaring per-chopper input PV streams | `gather_source_names` includes context bindings, expanded via device substreams | Subsumed |
| Static `DiskChoppers` assembled outside sciline by overriding NXlogs | Ordinary sciline provider consuming the `<chopper>_setpoint` context keys | Simplified (workflow-internal) |

### Evidence

- **Direct-param binding + gating round-trips today.** bifrost binds
  `InstrumentAngle[SampleRun]` / `SampleAngle[SampleRun]` at instrument scope
  (`bifrost/factories.py:77`); these flow through the gate end-to-end. Spec
  scope differs only in storage and the `dependent_sources` default;
  `JobFactory.create` merges instrument- and spec-scope bindings identically
  (`job_manager.py:174`).
- **Gating is "all present, consume-driven."** `Job.missing_context` requires
  every gating stream seen before the gate opens -> "all choppers ready" is
  automatic. `pipeline_required_keys` (`stream_processor_workflow.py`,
  `context_workflow_builder.py:85`) gates only specs whose pipeline actually
  reads the setpoints.
- **The one-shot tick path is generic and already here.**
  `orchestrating_processor.py:235-247` — the comment names "the chopperless
  wavelength_lut tick that fires only once." `peek_pending_streams` includes
  primary `source_names` for scheduled jobs, so a cached trigger activates the
  job once and is not refilled after activation.
- **Routing is automatic.** `route_derivation.py:45` adds spec-scope context
  bindings to the subscription set by stream name (expanded via device
  substreams), so per-chopper setpoint bindings self-subscribe — no manual
  route declaration.

## The one seam chopper must still bring

The gate opens once; the job then fires **on primary-data ticks only**
(`_jobs_with_primary_data`). A `set_context` update does **not** trigger a
recompute. So to refire the LUT when a setpoint changes mid-run, chopper still
needs the synthetic `chopper_cascade` as the job's **primary** stream, with
`allow_bypass=True` so the trigger value bypasses accumulation.

This is not a context-wiring concept and is small:

- Cascade-as-primary composes cleanly with gating: setpoints gate (sticky
  `_seen_context_streams`); the cascade primary drives the fire once the gate is
  open. No conflict — a stream is either primary or aux, never both.
- `allow_bypass` is an existing `ess.reduce.StreamProcessor` kwarg, forwardable
  via `ContextWorkflowBuilder.processor_kwargs`.
- The synthetic source itself is the synthesizer's job (producer-side, out of
  scope).

ADR 0002 and ADR 0003 (§"Param-dependent context") deliberately leave the
firing model primary-driven; chopper's trigger lives outside that boundary.

## Redo scope

Delete (now generic or synthesizer-internal):

- Bespoke per-chopper aux caching/replay via `is_context=True`.
- Synthesizer-side "all locked" gating as the workflow-activation mechanism
  (keep plateau detection; drop its role as the job gate).
- Manual per-chopper route declarations in `resolve_stream_names`.
- Any local orchestrator empty-batch patch — it is on the branch already.

Add:

- One `SpecHandle.add_context_binding(...)` per chopper from the LUT spec's
  `factories.py`, mapping `<chopper>_setpoint` -> its sciline key
  (`RotationSpeedSetpoint[chopperN]`, etc.; `workflow_key: Any` accepts
  parametrised keys).
- A `DiskChoppers` sciline provider consuming those context keys (replacing the
  outside-sciline assembly).
- The factory returns a `ContextWorkflowBuilder` (the signal that it consumes
  context; see `workflow_factory.py:_infer_consumes_context`), forwarding
  `allow_bypass=True` for the cascade trigger via `processor_kwargs`.

Keep (out of scope, unchanged):

- `ChopperSynthesizer` and its `outer_source_wrapper` plumbing.
- The synthetic `chopper_cascade` source and its `_SYNTHETIC_LOG_ATTRS` entry.

## Risks / notes

- **First production user of spec-scope `add_context_binding`.** Covered by unit
  tests (`workflow_factory_test.py:941-1022`) but unexercised in any instrument
  — production bindings today are all instrument-scope. The direct-param half is
  proven via bifrost; the spec-scope storage path is mechanically equivalent, so
  the risk is low but real. Add an instrument-level integration test for the
  spec-scope path as part of the redo.
- **Chain-patch is irrelevant to chopper.** `ValueLog` / `apply_dynamic_transforms`
  patches `depends_on` transform chains, not NXlog values inside `NXdisk_chopper`.
  Chopper setpoints are direct-param; spec scope rejects `ValueLog` keys anyway
  (`workflow_factory.py:319`). Do not route setpoints through chain-patch.
- **N choppers -> N bindings**, each a distinct sciline key. Mechanical, not a
  limitation.

## Addendum: component-scope auto-wiring instead of spec-scope

Question explored: rather than each chopper-consuming spec declaring its setpoint
bindings, could chopper streams auto-wire at the *component* level — the way the
branch auto-binds motion context whenever a detector/monitor is in use?

**Answer: yes, and it is arguably the more idiomatic fit. It needs no new
mechanism** — the instrument-scope direct-param path already does exactly this for
bifrost's rotation angles. The choice between spec-scope and component-scope is a
declaration-locality trade, not a capability gap.

### How motion auto-wiring actually works (two paths)

Motion context reaches a workflow by two distinct routes:

1. **Chain-patch** (geometry into the `depends_on` chain). The factory tells the
   instrument which component it loaded:
   `apply_dynamic_transforms(workflow, {source_name: NXdetector})`
   (`detector_view/factory.py:273`). Matching instrument-scope `ValueLog` bindings
   are fused into the chain. This path is **component-keyed**.
2. **Direct-param** (a value straight into `set_context`). bifrost's
   `InstrumentAngle[SampleRun]` / `SampleAngle[SampleRun]` are declared at
   instrument scope with `dependent_sources={'unified_detector'}`
   (`bifrost/factories.py:77`). `JobFactory.create` resolves them by source
   membership; `ContextWorkflowBuilder.build` then delivers/gates **only** the keys
   the pipeline actually consumes (`pipeline_required_keys`). The cut workflows
   gate; the detector view and ratemeter, sharing the same source but not reading
   the angles, do not. This path is **source-keyed + consume-driven**.

Chopper setpoints are NXlog values inside `NXdisk_chopper`, not `depends_on`
transforms — so they are direct-param, i.e. path 2. No `apply_dynamic_transforms`,
no `components` map.

### The component-scope chopper design

Declare the per-chopper bindings **once at instrument scope**, driven by
`Instrument.choppers` (the single source of truth chopper-workflow already
introduced):

```python
for chopper in instrument.choppers:
    instrument.add_context_binding(
        stream_name=f'{chopper}_rotation_speed_setpoint',
        dependent_sources=<all sources that may consume choppers>,
        workflow_key=RotationSpeedSetpoint[chopper],
    )
    # ... and the delay setpoint
```

A chopper-consuming spec then declares **nothing** about chopper streams. It only
builds a pipeline that consumes the chopper keys (the `DiskChoppers` provider) and
returns a `ContextWorkflowBuilder`. Consume-driven delivery does the rest:

- chopper-LUT workflow consumes the keys -> receives them, gates on them;
- any other workflow on the same source that does not -> neither.

This is strictly **less per-spec wiring** than the spec-scope variant, and it scales
for free to future chopper-consuming workflows (e.g. a reduction that builds its TOF
table directly from chopper geometry).

### Where the motion analogy holds — and where it bends

Holds:

- Consume-driven gating is identical to the loki precedent ("xy projection / i_of_q
  gate; tube view does not"). A spec is gated on choppers iff its pipeline reads
  chopper keys.

Bends — worth a decision:

- **`dependent_sources` is meaningful for motion, vacuous for choppers.** Motion
  attaches to one detector, so `dependent_sources={that detector}` is a real first
  filter. Choppers are in the beam for *everything*, so the honest value is "all
  sources" and gating collapses to the consumption stage alone. The mechanism
  supports it, but the source-membership filter carries no information for choppers
  — semantically this is consumption-scoped context wearing a source-scoped record.
  A small helper deriving `dependent_sources` from all registered specs avoids
  hand-enumeration; `_validate_binding_dependent_sources` requires every listed
  source be advertised by some spec (so the synthetic `chopper_cascade` source must
  be registered).
- **Component-scope commits to a canonical chopper-key vocabulary.**
  `RotationSpeedSetpoint[chopper]` etc. become a shared contract across every
  chopper-consuming workflow. Spec-scope lets each workflow pick its own keys.
  DRY favours the shared vocabulary; it is a commitment, not free.
- **The binding supplies live values + gating only — not the provider.** "Uses
  choppers" means *the pipeline already declares the chopper keys* (the
  `DiskChoppers` provider is in the graph) and the factory returns a
  `ContextWorkflowBuilder`. The binding feeds existing keys; it does not inject the
  provider. Same contract as motion (the chain-patch binding feeds the chain; it
  does not create the detector).
- **Firing model unchanged.** Component-scope does not remove the cascade primary
  (see "The one seam" above). It changes *where context is declared*, not *what
  drives a recompute*.

### Recommendation

- If chopper geometry will only ever be consumed by the LUT workflow: spec-scope is
  the smaller, more local declaration — keep the main plan.
- If more than one workflow will consume choppers (LUT now, reduction workflows
  later): prefer **component/instrument scope**, driven by `Instrument.choppers`,
  with `dependent_sources` spanning all sources and consume-driven delivery doing
  the filtering. This mirrors motion, declares the chopper contract once, and lets
  chopper-consuming specs stay silent about wiring.

Either way the conclusion of the main doc stands: no new context-wiring concepts —
component-scope reuses the exact instrument-scope direct-param path bifrost already
exercises.
</content>
</invoke>
