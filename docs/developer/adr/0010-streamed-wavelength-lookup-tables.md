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

### One LUT job, two tables, laid out as blocks

The LUT workflow keeps its single `chopper_cascade` source and gains two outputs — a
detector table and a monitor table. The range parameter is removed. One message per output
follows automatically: `UnrollingSinkAdapter` already splits a multi-output result.

Two, rather than one per component, because a table is a function of `distance` and
`event_time_offset` alone. It carries no component identity: a per-component table is
merely a *restriction* of the same function to that component's rows. What varies across
components is therefore only which stretch of beamline must be covered, and components
that share a stretch can share a table with nothing lost.

But a single grid spanning every component is mostly empty, which is the objection this
ADR opened with: monitors sit tens to hundreds of metres upstream of the detectors. So a
table is a **concatenation of uniform blocks**, one per group of components that sit close
together. Detectors share one dense block — they surround the sample, so their flight
paths cluster within a couple of metres, and the gaps between banks cost less than a block
each would. Each monitor gets its own block, a handful of rows at its flight path and
nothing in between. On LOKI, four monitors strung over seventeen metres cost twenty-six
rows in total.

The concatenation is deliberately not a uniform grid, and essreduce's interpolator assumes
one: `interpolator_numba` locates a row as `int((ltotal - first) / (distance[1] -
distance[0]))`, reading the wrong row silently — and only under numba, since the scipy
fallback handles an uneven axis correctly. A consumer must therefore select its own block
before the table reaches essreduce, and the invariant "a multi-block table never reaches
`WavelengthInterpolator`" is enforced in one place and tested. Blocks are recoverable from
the wire because the producer keeps them more than one resolution step apart, merging
ranges that would land closer; `distance_resolution`, already carried as a coord, is the
marker that separates them again.

One job, not one per component. A job's identity is `(workflow_id, source_name)` where
`source_name` *is the stream it consumes*; the LUT job consumes the chopper cascade and
nothing per-component. Per-component jobs would require inventing synthetic trigger
streams so `ChopperSynthesizer` could fan one tick into many — manufacturing input identity
to express output identity — and would recompute the same cascade once per component, with
one chance per job to drift on the shared parameters. Consistency across components is the
point of the feature.

The outputs class is static and shared by every instrument, since the outputs are the two
groups rather than an instrument's component list.

The tables are ordinary outputs. Nothing auto-plots them; a user who wants one adds a plot.
That visibility is worth having — "why is this component empty" is answered by its table,
which today is unreadable because the relevant rows are buried in tens of metres of empty
distance. Two outputs is also what makes them usable in the UI: a list that grew with the
component count was a plot picker nobody wanted to read. The monitor table plots as an
overlay of one curve per row, the same shape as the chopper-cascade-bands diagnostic.

### Ranges are derived generically, padded for motion

Each component's range is computed from the registered geometry artifact by one generic
walk to its pixels; the ranges are what the blocks are laid out to cover, and what a
consumer matches its own `Ltotal` against to find its block. Both sides derive them from
the same artifact with the same essreduce providers, in different services — an agreement
they never exchange, and which a test therefore pins by re-deriving every range the way a
consumer would and asserting a block covers it.

The `Ltotal` definition a consumer uses at lookup time does vary —
scattering geometry (source to sample to pixel) for most detectors, a straight line for
monitors, and source-to-sample alone for indirect geometry, where the analyser and
detector legs are excluded — but the differences are metres against flight paths of tens
to hundreds of metres. Padding absorbs them. Declaring the rule per component would buy
exactness the table's distance resolution does not reward, at the cost of a per-component
declaration nobody re-checks.

Motion is the one thing the artifact cannot supply: a live f144-driven transform is stored
as an *empty* NXlog, so the component riding it has no position at all until something
supplies an axis value. Instruments therefore declare one `AxisRange` per moving axis,
keyed by NeXus transform path. Both bounds are axis values in the axis's own units, so the
transform keeps supplying the direction and sense of the motion: the range is derived by
evaluating the geometry at the bounds, not by assuming which way along the beam the axis
travels. Which components ride an axis stays derived — a component is affected precisely
when the axis appears in its `depends_on` chain — so one hung off it later inherits the
range instead of silently being placed as if the axis did not exist.

This brackets translations and refuses rotations. Pixel positions are affine in a
translation's value, so `Ltotal` is convex and its maximum is attained at one corner of the
box the bounds span; evaluating the corners is therefore exact at the top end and the
padding absorbs the interior minimum a component traversing the sample plane could have.
An angle has no such property, so a live rotation axis is rejected rather than covered
approximately, and the component falls into the no-table case below. The first instrument
with one will have to design for it deliberately instead of inheriting a table too narrow
for its swing.

A component riding an axis nobody declared cannot be placed at all, so no block covers it.
Nothing binds such a component: its jobs would open their gate on the group's table and
then fail at every recompute, having no block to select. An aux-selectable monitor in that
state is rejected by the factory at job creation instead, since the job's own source may
well be placeable.

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

A `context_outputs` field on `WorkflowSpec` maps an output field name to a stream name,
plain and fixed at declaration time: a context stream carries no job identity (ADR 0006),
so a spec declaring context outputs has exactly one source name, and unlike `device_outputs`
the name is never templated over the sources. A new `StreamKind` and topic carry it, routed
to the existing da00 serializer, with the wire `source_name` being that stream name. A
dedicated topic rather than `livedata_data`: backend services must not subscribe to every
detector image in the facility to receive a lookup table.

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

One spec-scope `ContextBinding` per spec and group, with `dependent_sources` naming the
jobs that gate on it — the spec's placeable sources. A reduction binds both groups.

The binding no longer selects *which* table a source receives, because there is one per
group; what a job still has to select is its block, and it selects that by flight path. The
provider takes the job's own `DetectorLtotal` or `MonitorLtotal` — already in its graph,
computed from geometry rather than from stream data — and takes the block containing its
midpoint. Midpoint rather than full containment: a pixel beyond the table's range is a
documented `NaN`, and demanding full coverage would turn one stray pixel into a failed job.

Selecting by flight path rather than by name is what keeps component identity off the wire
entirely. The job already knows its `Ltotal`, the table already carries its distances, and
the two meet without either side naming a bank or a monitor.

The bound key is essreduce's public `LookupTable[RunType, Component]` dataclass, so
`Component` distinguishes a job's detector and monitor tables with no new type parameter.
The wire value is a single `DataArray` carrying the table plus its scalar fields as coords,
because da00 serializes a `DataArray` and the dataclass has non-array fields (`pulse_stride`
is an `int`, `choppers` a `DataGroup`). A small provider selects the job's block and
reassembles the dataclass from that block's coords. Every coord is taken from the built
table, never from the job's parameters: these fields describe the table, and the two can
differ — the stride may be guessed from the choppers, and the builder honours the requested
time resolution only up to fitting a whole number of bins into the frame period. Parameter
provenance rides on the identity coord below. The provider does not route through
`load_lookup_table_from_file`, whose matching branch is a backwards-compatibility shim for
tables predating the dataclass — depending on it would tie us to a deprecated path and
silently drop `choppers`, which that format cannot carry. Chopper provenance travels in the
identity coord instead. da00 round-trips variances, which the uncertainty mask requires
unconditionally.

A reduction needs one table per sciline `Component` — the detector plus the *incident* and
*transmission* monitor roles — and which physical monitor fills a role is a per-job aux
selection that an import-time binding cannot name. Under the shared monitor table it does
not have to: both roles bind the one monitor stream, and a provider generic in
`MonitorType` serves all three monitor roles an instrument has — the plain `NXmonitor` of a
monitor view and a reduction's two. Sciline instantiates it per role, and each instance
picks its block via that role's own `MonitorLtotal`. Which monitor fills a role is settled
by geometry the job already holds, not by the stream it binds.

A selection whose monitor has no block — an unplaceable one — is rejected by the factory at
job creation. The gate would open on the shared table's arrival and the job would then fail
at every recompute, which is a worse failure than never starting.

This supersedes three earlier shapes: binding the *default* monitors and raising when the
selection differed, which made the aux selector a lie in wavelength mode; binding every
candidate monitor to its own synthesized per-component key, which left every unselected key
a dead parameter and restated the candidate list in the factory; and aux-templated stream
names (`wavelength_lut/{incident_monitor}`), where the binding named the aux field and gate
resolution rendered it against the job's selection. The template mechanism worked and was
the most intricate thing in this ADR — declared-versus-resolved names, route derivation
expanding a template over an aux field's declared choices to keep subscriptions a superset,
and a collision check for two roles selecting one monitor. Sharing the table removes the
problem it solved rather than solving it better, and the mechanism, having lost its only
user, is gone.

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

The condition is stated once, in the predicate, and so is the stream-name-to-key mapping,
on the binding: consuming factories declare no context keys of their own, since the
routing layer injects the resolved bindings into the workflow after creation. A factory
that repeated the mapping would be a second declaration that has to agree with the binding
with nothing to catch a disagreement — the same drift ADR 0003 removed between routing and
graph wiring. What the factory does contribute is the reassembly provider, inserted
unconditionally: a provider whose input never arrives sits on a branch the targets never
reach, so for a TOA job it is dead graph and computes nothing.

### Consumers clear when the LUT's identity changes

A new LUT means the chopper phasing changed, so the wavelength for a given time-of-arrival
changed: data accumulated either side of it are not the same measurement. Consumers clear.

Consumers do not compare tables. The producer stamps an **identity coord** alongside the
scalar fields, derived from the inputs that determine the table — the chopper
setpoints, pulse period and stride, resolutions, source offset and block ranges — each
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
| **Two tables laid out as blocks — detectors dense, a block per monitor (chosen)** | Removes the range parameter, keeps the rows a per-component table would have had, and reduces the outputs, streams, bindings and sciline keys from one per component to one per group. A table carries no component identity, so sharing one costs nothing; the consumer selects its block by flight path. |
| One table per component, one job, many outputs | The first shape of this decision, and it worked. But it generates the outputs model per instrument, multiplies streams and bindings by the component count, gives the dashboard a plot list that grows with the instrument, and forces per-job table selection to be expressed in *stream names* — the aux-templating machinery below. All to publish N restrictions of one function. Superseded. |
| One instrument-wide table, one uniform grid | Also removes the range parameter, with far less plumbing, but spans an order of magnitude in distance with almost all rows empty: BIFROST's monitors alone are 155 m apart. Rejected — and it is what the block layout exists to avoid. |
| Merge every component into one *block* set, monitors included | One stream instead of two. Rejected: a monitor job would receive the detectors' dense block, which at a fine resolution is the megabyte-scale payload, to read a few rows of its own. |
| Cluster detectors by gap rather than one dense block | Would adapt to an instrument whose banks are far apart, using the same merge primitive the monitors use. Rejected for now: the layout would silently reshuffle when a geometry artifact is regenerated, and no current instrument needs it. The primitive is there if one does. |
| One row per monitor (a monitor is a point) | Attractive, and wrong: `WavelengthInterpolator` needs two nodes *bracketing* the flight path, and a single node is a zero-width grid that returns `NaN` for every lookup. The upstream builder also pads every block by two resolution steps of its own, so a five-row block is the floor without a bespoke second build path. Rejected — the saving is kilobytes. |
| One job per component | Requires synthetic trigger streams to give each job a `source_name`, recomputes the cascade per component, and multiplies the ways shared parameters can drift. Rejected. |
| **Derive every range generically from the geometry artifact, padded (chosen)** | The `Ltotal` rule does differ across geometries, but by metres against flight paths of tens to hundreds of metres, and padding is free apart from recompute. One derivation, no per-component declarations. |
| Declare the `Ltotal` rule per component | Exact where padding is merely sufficient, and buys that exactness with a per-component declaration that nobody re-checks when the geometry changes. Rejected. |
| Hand-declare every range | A dozen-plus numbers per instrument that nobody re-checks when an artifact is regenerated. Rejected. |
| **LUT workflow consumes component motion streams** | Tables would always describe where the component actually is, and the static travel envelope — the one number this design needs from the instrument team — would disappear. It does *not* remove motion from consumers, which still need pixel positions for scattering geometry and for the per-pixel `Ltotal` that indexes the table. Costs: motion joins the LUT job's gating set, so a dead motion PV stops all wavelength reduction; every sample during a move re-emits and clears; and the LUT job's motion value can lag the one the consumer patched into its geometry, so padding is still needed. The strongest alternative; recorded to revisit. |
| Route the LUT as ungated aux with the file as default (ROI precedent) | Survives a cold start, but leaves reducing with a stale or nominal table as a silent mode — the thing the feature exists to prevent. Rejected in favour of the gate plus explicit limitations. |
| **Both monitor roles bind the shared table and select by `MonitorLtotal` (chosen)** | One binding, one generic provider serving every monitor role, and no per-job identity on the wire. Which monitor fills a role is settled by geometry the job already holds. |
| Aux-templated stream names for per-job table selection | One binding per role, the placeholder naming the aux field, gate resolution rendering it from the job's selection, and route derivation expanding templates over the field's declared choices to keep subscriptions a superset. It worked, and was the most intricate mechanism in this ADR. Sharing the monitor table removes the problem instead of solving it; the mechanism now has no user. Superseded. |
| Bind every candidate monitor to its own per-component key | Works, since all tables arrive together, but synthesizes a key per candidate, leaves the unselected ones as dead parameters, and restates the candidate list in the factory's role mapping. Superseded. |
| **Split the spec into TOA-only and wavelength variants** | ADR 0003's prescribed remedy for the over-gate. Rejected on UX: two specs are two entries in the workflow list, two jobs to run, and — because `DataKey` embeds `workflow_id` — two unrelated output streams, so a plot cannot follow a mode switch. It also duplicates every per-instrument params override. |
| Reuse `livedata_data` instead of a dedicated topic | Backend services would subscribe to every detector image in the facility. Rejected. |
| Reset on any received LUT, with no identity | Simplest, and puts the decision at the producer. But a LUT-job restart re-emits an unchanged table, and that restart is the recovery action; a liveness heartbeat would also clear on every beat. Rejected. |
| ADR 0006's `start_time` generation marker | Changes on a restart with identical configuration, wiping statistics mid-run. Rejected. |
| Content comparison at the consumer | Workable, but is one comparison per consumer rather than one stamp, and puts the noise tolerance far from the setpoints it filters. `sc.identical` is not usable at all: it returns `False` for bit-identical arrays containing `NaN`, which every LUT has. Rejected. |
| Fingerprint hashed from the table's bytes rather than its inputs | Not stable across a library or platform change, and says nothing about *why* the table changed. Rejected. |
| Naming the mirror for the LUT rather than for the seam | Same code either way; the generic name is the honest one. Rejected. |

## Consequences

- `WorkflowSpec` gains `context_outputs`; `ContextBinding` gains an opt-in clear flag; the
  LUT carries an identity coord alongside its scalar fields.
- A published table is a concatenation of uniform blocks, and handing a multi-block table
  to essreduce's interpolator is silently wrong under numba while correct under the scipy
  fallback. `numba` is not a declared dependency, so the test suite exercises the tolerant
  path: the guard is the block selection itself plus a test asserting the selected block is
  uniform, not a test that would fail on the mistake. An upstream `searchsorted` fallback
  for non-uniform axes would remove the trap and let the selection go away.
- The aux-templated stream-name mechanism (`render_stream_name`, placeholder expansion in
  route derivation) lost its only user and is removed. Stream names are plain again on both
  sides of the mirror, which is what the statically derived Kafka subscriptions want.
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
  site. Resolution splits in two: `declared_context_keys` ignores predicates and leaves
  templates unrendered, serving the static callers (route derivation, the workflow
  visualizer) that need the superset; `resolve_context_keys` filters by predicate and
  renders stream-name templates for the job path, which therefore also receives the
  job's rendered aux selections.
- Specs are unchanged: coordinate mode stays a parameter on one workflow, so output
  identity and any plot built on it survive a mode switch.
- The file-based table stops being reachable on a migrated instrument, so the streamed
  table cannot be A/B'd against it side by side within one instrument. The LUT job
  publishes its tables as ordinary outputs, which is the comparison surface instead.
- `Instrument` gains `axis_ranges` and, once the LUT factory is attached, the set of
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
