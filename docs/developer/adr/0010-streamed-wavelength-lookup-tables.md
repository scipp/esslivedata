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

### Ranges are derived, from two declarations

The range must use the same `Ltotal` definition the consumer uses at lookup time, and
those differ: scattering geometry (source to sample to pixel) for most detectors, a
straight line for monitors, and source-to-sample alone for indirect geometry, where the
analyser and detector legs are excluded. A generic walk to each pixel would put an indirect
instrument's table tens of metres away from where it is queried, and every event would
`NaN`. The rule is therefore *declared* per component, so the physics distinction stays
visible instead of hiding inside a hand-written constant.

Motion is a second, independent declaration. Which components ride a moving axis is already
known from `Instrument.chain_patch_bindings`; only the travel envelope is underivable. So
instruments declare one number per moving axis and the affected component set is derived —
a component later hung off that axis inherits the padding instead of silently getting a
nominal-only range.

The range is static and covers the full envelope. The LUT workflow does not consume motion:
a live range would gate the LUT job on motion, couple motion to LUT-driven clearing, and
race the consumer's own geometry patch.

A guard test recomputes each range from the registered geometry artifact and fails if a
declared range no longer contains it, so regenerating an artifact cannot silently move a
component out of its table.

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

The wire value is the table plus its provenance scalars as coords — deliberately the legacy
on-disk layout that essreduce's file loader already rehydrates — so a small provider
inverts it. da00 round-trips variances, which the uncertainty mask requires
unconditionally.

Monitors are treated as statically known. The user-selectable aux-monitor mechanism is not
designed around; a check at job creation raises when the chosen monitor differs from the
bound one, rather than silently supplying a table for the wrong distance.

### Wavelength specs are split from TOA specs

Gating is resolved per `(workflow_id, source_name)` and never per parameters:
`Instrument.resolve_context_keys` takes no parameters, and the job's gating set is its key
set. Coordinate mode is a parameter. A spec offering a TOA/wavelength choice would
therefore gate TOA-mode jobs on a table they never read.

ADR 0003 accepted exactly this over-gate for motion and prescribed the remedy for when it
became real: split the spec. It has become real, and the cost is far higher than for
motion. Motion streams are control-system PVs pushed continuously and arrive within seconds
of any service start. The LUT is published by an operator-started livedata job that
re-emits only on chopper change. Over-gating TOA on it means TOA — the mode you fall back
to when everything else is broken — depends on the most fragile link in the chain.

Wavelength-variant specs are therefore added *additively*, carrying the LUT binding, and
existing specs are untouched. TOA keeps working unconditionally, with no version bumps and
no saved-config migration, and the file-based path remains available until the streamed one
is trusted. The new path carries no file default, so reducing with a table for the wrong
chopper configuration is impossible by construction rather than discouraged by a warning.

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
| **Per-component tables, one job, many outputs (chosen)** | Removes the range parameter, cuts recompute by two orders of magnitude, makes each table readable, and maps onto the per-output mirror unchanged. |
| One instrument-wide table, range = union over all components | Also removes the range parameter, with far less plumbing, but spans an order of magnitude in distance with almost all rows empty. Rejected. |
| One job per component | Requires synthetic trigger streams to give each job a `source_name`, recomputes the cascade per component, and multiplies the ways shared parameters can drift. Rejected. |
| Derive every range generically from the geometry artifact | Silently wrong for indirect geometry, which is the exact failure this ADR exists to remove. Rejected. |
| Hand-declare every range | A dozen-plus numbers per instrument that nobody re-checks when an artifact is regenerated. Rejected. |
| **LUT workflow consumes component motion streams** | Tables would always describe where the component actually is, and the static travel envelope — the one number this design needs from the instrument team — would disappear. It does *not* remove motion from consumers, which still need pixel positions for scattering geometry and for the per-pixel `Ltotal` that indexes the table. Costs: motion joins the LUT job's gating set, so a dead motion PV stops all wavelength reduction; every sample during a move re-emits and clears; and the LUT job's motion value can lag the one the consumer patched into its geometry, so padding is still needed. The strongest alternative; recorded to revisit. |
| Route the LUT as ungated aux with the file as default (ROI precedent) | Survives a cold start, but leaves reducing with a stale or nominal table as a silent mode — the thing the feature exists to prevent. Rejected in favour of the gate plus explicit limitations. |
| Parameter-dependent gating on coordinate mode | Explicitly ruled out by ADR 0003. Rejected. |
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
- Wavelength-variant specs are added; existing TOA and file-based specs are unchanged.
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
