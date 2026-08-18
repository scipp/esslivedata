# ADR 0010: Wavelength lookup tables as streamed context inputs

- Status: proposed
- Deciders: Simon
- Date: 2026-08-11

## Context

`wavelength_lut_workflow` computes a wavelength lookup table from the live chopper
cascade: it consumes each chopper's rotation-speed and delay setpoints as context
(ADR 0003), rebuilds the cascade whenever `ChopperSynthesizer` emits a `chopper_cascade`
tick, and publishes the table as an ordinary workflow result. Nothing consumes that
result. It exists to be looked at.

Meanwhile every backend workflow converting time-of-arrival to wavelength loads a table
from a file fixed at import time. `LookupTableFilename` is assigned bare-generic in each
instrument's `factories.py`, so one simulation-derived table serves every component in
the job, and it describes a *nominal* chopper configuration. Run in a different
configuration and the reduction is silently wrong. That is the motivation for computing
the table live.

Three properties of the file-based tables shape the decision.

**Out-of-range lookups fail silently.** A table is indexed by `distance`; a lookup outside
its range yields `NaN`, those events fall outside every histogram bin, and the component
renders empty with no error anywhere. The shipped tables already miss monitors on DREAM,
LOKI and BIFROST, which therefore produce nothing today.

**One table per job wastes almost all of its rows.** Per-component `Ltotal` spreads are
under a metre — often a scalar, for monitors and for indirect geometry — while the shipped
tables span tens of metres. At the default resolution that is hundreds of distance rows
where a handful are used. The cost is not bandwidth but recompute: the polygon
rasterization runs over every distance row on every chopper change.

**The range is a user parameter**, whose default covers no instrument correctly. An
operator starting the LUT workflow with defaults can silently blank every detector.

The mechanisms this builds on exist: context bindings and JobManager gating (ADR 0002,
ADR 0003), uniform stream-name keying (ADR 0004), and the NICOS derived-device mirror
(ADR 0006), which republishes selected workflow outputs onto a dedicated topic under names
that deliberately exclude job identity.

What does not exist, and is the new concept here: **a workflow output republished as an
input stream for other workflows.** The NICOS mirror is publish-only — nothing consumes a
mirrored topic — and no backend service subscribes to `livedata_data`. This is the first
cross-service feedback edge.

## Decision

### One LUT job, one output per component

The LUT workflow keeps its single `chopper_cascade` source and gains one output per
component, each covering exactly that component's `Ltotal` range. The range parameter is
removed. Per-component messages follow automatically: `UnrollingSinkAdapter` already
splits a multi-output result into one message per output.

One job, not one per component. A job's identity is `(workflow_id, source_name)` where
`source_name` *is the stream it consumes*; the LUT job consumes the chopper cascade and
nothing per-component. Per-component jobs would require inventing synthetic trigger
streams so `ChopperSynthesizer` could fan one tick into many — manufacturing input identity
to express output identity — and would recompute the same cascade once per component, with
one chance per job to drift on the shared parameters. Consistency across components is the
point of the feature.

The outputs class is built per instrument, since instruments do not share components. This
mirrors the existing per-instrument parameter override and stays dashboard-safe: no
science imports.

Per-component tables are ordinary outputs. Nothing auto-plots them; a user who wants one
adds a plot. That visibility is worth having — "why is this component empty" is answered by
its table, which today is unreadable because the relevant rows are buried in tens of metres
of empty distance.

### Ranges are derived generically, padded for motion

Each component's range is computed from the registered geometry artifact by one generic
walk to its pixels. The `Ltotal` definition a consumer uses at lookup time does vary —
scattering geometry (source to sample to pixel) for most detectors, a straight line for
monitors, and source-to-sample alone for indirect geometry, where the analyser and
detector legs are excluded — but the differences are metres against flight paths of tens
to hundreds of metres. Padding absorbs them. Declaring the rule per component would buy
exactness the table's distance resolution does not reward, at the cost of a per-component
declaration nobody re-checks.

Motion is the one thing the artifact cannot supply, and it withholds more than expected:
a live f144-driven transform is stored as an *empty* NXlog, so neither the travel envelope
nor the component's resting position can be recovered from it. Instruments therefore
declare both, as one `MotionEnvelope` per moving axis keyed by NeXus transform path. Which
components ride an axis stays derived — a component is affected precisely when the axis
appears in its `depends_on` chain — so one hung off it later inherits the envelope
instead of silently getting a nominal-only range.

A component riding an axis nobody declared cannot be placed at all, and gets no table.
Nothing binds such a component: gating on a stream that is never published would leave the
job waiting forever, and for an aux-selectable monitor it would take down every job that
merely *could* have selected it.

Over-padding is cheap and under-padding is silent, so the bias is deliberate. Widening a
range adds distance rows at fixed `DistanceResolution`; it costs recompute in the LUT job
and nothing else, and the rows are never wrong. A range that is too narrow yields `NaN`
for every event outside it, which is the failure this ADR exists to remove.

The range is static and covers the full envelope. The LUT workflow does not consume motion:
a live range would gate the LUT job on motion, couple motion to LUT-driven clearing, and
race the consumer's own geometry patch.

A guard test recomputes each range from the registered geometry artifact and fails if a
declared envelope no longer contains it, so regenerating an artifact cannot silently move
a component out of its table.

### A generic mirror: workflow outputs republished as input streams

Modelled on ADR 0006 but named for the seam rather than for the LUT, because an output fed
back as another workflow's input recurs — a fitted beam centre, a live calibration.

A `context_outputs` field on `WorkflowSpec` maps an output field name to a stream-name
template, validated exactly as `device_outputs` is. A new `StreamKind` and topic carry it,
routed to the existing da00 serializer, with the wire `source_name` being the rendered
stream name. A dedicated topic rather than `livedata_data`: backend services must not
subscribe to every detector image in the facility to receive a lookup table.

The **ingest** half has no ADR 0006 analogue. It is a da00 adapter with no stream lookup
table, so the internal stream name is the da00 source name — the shape the ROI route
already uses — plus a preprocessor case returning `LatestValueAccumulator`. That
accumulator is already marked as context, so the context cache and the JobManager gate need
no change.

Stream names are prefixed (`wavelength_lut/…`), keeping the namespace greppable and
collision-free against device names. Job identity is excluded, per ADR 0006: a
`ContextBinding` is declared at import time and cannot know a job number, and excluding it
is what lets a relaunched LUT job transparently resume feeding its consumers.

### Consumers bind the LUT as gated context

A spec-scope `ContextBinding` per consuming source, using `dependent_sources` to select
which component's stream that source receives. No new sciline type parameter is needed:
`Component` is fixed for detectors and carries no per-bank identity, but a job already
handles one bank, so per-component delivery is a routing concern rather than a typing one.
A reduction job binds its detector and monitor tables as distinct keys.

The bound key is essreduce's public `LookupTable[RunType, Component]` dataclass, so
`Component` distinguishes a job's detector and monitor tables with no new type parameter.
The wire value is a single `DataArray` carrying the table plus its provenance scalars as
coords, because da00 serializes a `DataArray` and the dataclass has non-array fields
(`pulse_stride` is an `int`, `choppers` a `DataGroup`). A small provider reassembles the
dataclass from those coords. It does not route through
`load_lookup_table_from_file`, whose matching branch is a backwards-compatibility shim for
tables predating the dataclass — depending on it would tie us to a deprecated path and
silently drop `choppers`, which that format cannot carry. Chopper provenance travels in
the identity coord instead. da00 round-trips variances, which the uncertainty mask
requires unconditionally.

A reduction needs one table per sciline `Component` — the detector plus the *incident*
and *transmission* monitor roles — and which physical monitor fills a role is a per-job aux
selection that an import-time binding cannot know. Every candidate monitor therefore binds
its own context key, and the factory, which does see the selection, maps the chosen ones
onto the roles. The unselected keys are dead parameters, which `set_context` stores and
computes nothing from.

Gating on every candidate rather than on the selected one costs nothing: all tables come
from the same LUT job, split out of one result, so they arrive together and the gate opens
at the same instant either way. The over-gating concern below is about a *second,
independent* producer and does not apply. This supersedes an earlier plan to bind the
default monitors and raise at job creation when the selection differed, which would have
made the aux selector a lie in wavelength mode.

### The gate is resolved from the job's parameters

Coordinate mode is a parameter, so a spec that offers both modes must not gate its TOA
jobs on a table they never read. ADR 0003 accepted exactly this over-gate for motion and
prescribed splitting the spec if it ever became real. It has become real, and both of that
decision's premises have expired.

The cost of over-gating is far higher here than for motion. Motion streams are
control-system PVs pushed continuously, arriving within seconds of any service start. The
LUT is published by an operator-started livedata job that re-emits only on chopper change.
Over-gating TOA on it makes TOA — the mode you fall back to when everything else is broken
— depend on the most fragile link in the chain.

And the prescribed remedy is worse than the disease. Splitting doubles the workflow list,
forces the operator to start and stop two jobs, and, because the dashboard's data plane is
keyed by `DataKey` = (`workflow_id`, `source_name`, `output_name`), gives the two modes
*different output identities*. A plot cannot follow a mode switch; the user reconfigures it
or keeps two. Coordinate mode is a property of how you are currently looking at a detector,
not of which detector you are looking at, and the spec split makes the workflow list encode
the wrong one.

So the gate becomes parameter-dependent instead: a `ContextBinding` carries a predicate
over the job's validated params. A TOA job resolves an empty gating set; a wavelength job
gates on its component's table. The workflow stays unified and coordinate mode stays a
parameter.

This is affordable because the gate is already a per-job quantity. It is resolved at a
single call site, inside job creation, which already holds the full `WorkflowConfig`;
nothing consumes the gating set earlier. Kafka subscriptions are derived statically per
spec and are a superset of any per-job gate, so narrowing the gate needs no subscription
change, and the context cache is keyed by stream name alone. The declaration-level
collision validators run on declarations rather than resolved sets, so a predicate that
only *removes* bindings keeps them conservative.

The condition is stated once, in the predicate. The factory wires the table's context key
unconditionally, because declaring a context key that no provider consumes is a verified
no-op: `StreamProcessor` registers every context key as a graph node, so a key on a branch
the targets never reach maps to an empty recompute set, and `set_context` stores a dead
parameter and computes nothing. A TOA job that never receives the stream does not even
reach that point, since the workflow only forwards context keys present in the batch. Had
the factory branched too, the two conditions would have had to agree with nothing to catch
a disagreement — the existing guard only checks that a workflow with resolved context keys
implements `SupportsContext`, not that it consumes them.

### Consumers clear when the LUT's identity changes

A new LUT means the chopper phasing changed, so the wavelength for a given time-of-arrival
changed: data accumulated either side of it are not the same measurement. Consumers clear.

Consumers do not compare tables. The producer stamps an **identity coord** alongside the
provenance scalars, derived from the inputs that determine the table — the chopper
setpoints, pulse period and stride, resolutions, source offset and component range — each
rounded to a declared precision. Consumers clear when that scalar changes.

Clearing on *any* received LUT would be simpler, and correctly locates the decision at the
producer rather than having consumers second-guess it. It fails on one case that matters:
restarting the LUT job re-emits an unchanged table, and that restart is the recovery action
for a lost stream. Recovery must not destroy every consumer's statistics. The same
objection rules it out for the liveness heartbeat discussed below.

Deriving the fingerprint from inputs rather than from the table's bytes keeps it stable
across producer restarts — the setpoints replay from retained context, so the same cascade
yields the same fingerprint in a fresh process — and avoids a float hash that would not
survive a library or platform change. The rounding precision is the noise-rejection knob,
at the source, backing up the plateau filtering `ChopperSynthesizer` already applies.

Clearing is opt-in per binding. Universal clearing is arguably more correct — a carriage
move does invalidate an accumulated histogram, since the pixels moved — but that is a
question about motion bindings, independent of this feature, and is left to be decided on
its own merits. Motion and LUT clearing are orthogonal here by construction: because the
range is static, a carriage move does not change the table and cannot cause a LUT-driven
clear.

No new plumbing is needed to act on it: `Job.clear()` is already driven by an external
event on run transitions, and this is a second trigger on that path.

## Alternatives considered

| Option | Notes |
|---|---|
| **Per-component tables, one job, many outputs (chosen)** | Removes the range parameter, cuts recompute by one to two orders of magnitude depending on how much motion padding a component carries, makes each table readable, and maps onto the per-output mirror unchanged. |
| One instrument-wide table, range = union over all components | Also removes the range parameter, with far less plumbing, but spans an order of magnitude in distance with almost all rows empty. Rejected. |
| One job per component | Requires synthetic trigger streams to give each job a `source_name`, recomputes the cascade per component, and multiplies the ways shared parameters can drift. Rejected. |
| **Derive every range generically from the geometry artifact, padded (chosen)** | The `Ltotal` rule does differ across geometries, but by metres against flight paths of tens to hundreds of metres, and padding is free apart from recompute. One derivation, no per-component declarations. |
| Declare the `Ltotal` rule per component | Exact where padding is merely sufficient, and buys that exactness with a per-component declaration that nobody re-checks when the geometry changes. Rejected. |
| Hand-declare every range | A dozen-plus numbers per instrument that nobody re-checks when an artifact is regenerated. Rejected. |
| **LUT workflow consumes component motion streams** | Tables would always describe where the component actually is, and the static travel envelope — the one number this design needs from the instrument team — would disappear. It does *not* remove motion from consumers, which still need pixel positions for scattering geometry and for the per-pixel `Ltotal` that indexes the table. Costs: motion joins the LUT job's gating set, so a dead motion PV stops all wavelength reduction; every sample during a move re-emits and clears; and the LUT job's motion value can lag the one the consumer patched into its geometry, so padding is still needed. The strongest alternative; recorded to revisit. |
| Route the LUT as ungated aux with the file as default (ROI precedent) | Survives a cold start, but leaves reducing with a stale or nominal table as a silent mode — the thing the feature exists to prevent. Rejected in favour of the gate plus explicit limitations. |
| **Split the spec into TOA-only and wavelength variants** | ADR 0003's prescribed remedy for the over-gate. Rejected on UX: two specs are two entries in the workflow list, two jobs to run, and — because `DataKey` embeds `workflow_id` — two unrelated output streams, so a plot cannot follow a mode switch. It also duplicates every per-instrument params override. |
| Reuse `livedata_data` instead of a dedicated topic | Backend services would subscribe to every detector image in the facility. Rejected. |
| Reset on any received LUT, with no identity | Simplest, and puts the decision at the producer. But a LUT-job restart re-emits an unchanged table, and that restart is the recovery action; a liveness heartbeat would also clear on every beat. Rejected. |
| ADR 0006's `start_time` generation marker | Changes on a restart with identical configuration, wiping statistics mid-run. Rejected. |
| Content comparison at the consumer | Workable, but is one comparison per consumer rather than one stamp, and puts the noise tolerance far from the setpoints it filters. `sc.identical` is not usable at all: it returns `False` for bit-identical arrays containing `NaN`, which every LUT has. Rejected. |
| Fingerprint hashed from the table's bytes rather than its inputs | Not stable across a library or platform change, and says nothing about *why* the table changed. Rejected. |
| Naming the mirror for the LUT rather than for the seam | Same code either way; the generic name is the honest one. Rejected. |

## Consequences

- `WorkflowSpec` gains `context_outputs`; `ContextBinding` gains an opt-in clear flag; the
  LUT carries an identity coord alongside its provenance scalars.
- A new `StreamKind`, topic, sink route, ingest route and preprocessor case. The ingest
  half is the genuinely new mechanism, with no precedent in ADR 0006.
- Route derivation will gather the LUT stream names and then drop them, since they appear
  in no stream lookup table. Harmless, because the context route is added unconditionally
  rather than derived from the mapping, but it needs a test pinning that — it reads as a
  bug otherwise.
- `ContextBinding` gains a predicate over the job's params, and the gate resolution
  takes the validated params model. This supersedes ADR 0003's param-dependent-context
  non-goal, which was decided on YAGNI grounds. Params validation moves up into job
  creation and `WorkflowFactory.create` takes the validated model rather than the raw
  `WorkflowConfig`, so the order is validate → resolve gate → build with one validation
  site. Resolution splits in two: `declared_context_keys` ignores predicates and serves
  the static callers (route derivation, the workflow visualizer) that need the superset;
  `resolve_context_keys` filters it by predicate for the job path.
- Specs are unchanged: coordinate mode stays a parameter on one workflow, so output
  identity and any plot built on it survive a mode switch.
- The file-based table stops being reachable on a migrated instrument, so the streamed
  table cannot be A/B'd against it side by side within one instrument. The LUT job
  publishes its tables as ordinary outputs, which is the comparison surface instead.
- `Instrument` gains `motion_envelopes` and, once the LUT factory is attached, the set of
  components it could place. Consumers bind against that set rather than against the full
  component list.
- DREAM and LOKI migrate first; they are the only instruments offering wavelength mode
  today. The detector- and monitor-view factories lose their lookup-table filename
  argument outright rather than defaulting it, so a view has no file path left to fall
  back to, and LOKI's I(Q) reduction migrates with them. What remains of
  `LookupTableFilename` is set explicitly by the unmigrated BIFROST and ESTIA reduction
  pipelines at their own call sites — an input, not a fallback. Removing the type
  entirely waits on migrating those two.
- The LUT workflow loses its range parameter.

### Standing limitations

These follow from the decision and are not resolved by it.

- **Nothing guarantees the LUT job is running.** Under gating it is a hard prerequisite for
  all wavelength reduction on the instrument: if nobody starts it, or someone stops it,
  every wavelength workflow blocks on missing context. Recovery is to restart the LUT job.
- **A backend restart loses the LUT.** Consumers pin to the high watermark — no replay, no
  compaction, no seed (ADR 0002) — and the cascade tick fires only when a chopper input
  changes while all choppers are locked, which does not happen during a steady run.
- **The gate protects startup, not steady state.** Gate graduation is one-way and context
  is sticky, so stopping the LUT job does not re-block consumers; they keep reducing with
  the last table indefinitely, silently.
- **Two concurrent LUT jobs would flap**, both publishing the same stream names. The
  duplicate-name check is static across specs and cannot see two live jobs of one spec.

The first three are one problem — the LUT stream has no liveness — with one shape of
answer, borrowed from how a physical device reports to NICOS: an alarm status. Identity-coord
clearing is what makes it affordable, since a re-emitted identical table clears nothing, so
the LUT job can heartbeat: recent re-emission is liveness, prolonged silence is an alarm on
every dependent job, and the heartbeat doubles as recovery after a backend restart. A
compacted topic consumed from the earliest offset is the more principled recovery mechanism
and remains the long-term answer, but it needs a per-topic exception to the unconditional
high-watermark pin and message keys the sink does not currently set.

Alongside it: a dashboard guard before a running LUT job is reconfigured or stopped, since
the blast radius is every wavelength workflow on the instrument.
