# ADR 0010: Workflow outputs republished as context input streams

- Status: proposed
- Deciders: Simon
- Date: 2026-08-11

## Context

`wavelength_lut_workflow` computes a wavelength lookup table from the live chopper
cascade and publishes it as an ordinary workflow result. Nothing consumes that result.
Meanwhile every workflow converting time-of-arrival to wavelength loads a table from a
file fixed at import time, describing a *nominal* chopper configuration: run in any other
configuration and the reduction is silently wrong. The live table has to reach those
workflows.

The mechanisms this builds on exist: context bindings and JobManager gating (ADR 0002,
ADR 0003), uniform stream-name keying (ADR 0004), and the NICOS derived-device mirror
(ADR 0006), which republishes selected workflow outputs onto a dedicated topic under
names that deliberately exclude job identity.

What does not exist is **a workflow output republished as an input stream for other
workflows**. The NICOS mirror is publish-only, and no backend service subscribes to
`livedata_data`. This is the first cross-service feedback edge, and it recurs: a fitted
beam centre or a live calibration has the same shape. This ADR decides the generic
mechanism; ADR 0011 decides how the lookup table itself is laid out on it.

## Decision

### A generic mirror, modelled on ADR 0006

A `context_outputs` field on `WorkflowSpec` maps an output field name to a stream name.
The name is plain and fixed at declaration time: a context stream carries no job identity
(ADR 0006), so a spec declaring context outputs has exactly one source name and, unlike
`device_outputs`, the name is never templated over the sources. Excluding job identity is
also what lets a relaunched producer transparently resume feeding its consumers.

A new `StreamKind` and a dedicated topic carry it, through the existing da00 serializer,
with the wire `source_name` being the stream name. A dedicated topic rather than
`livedata_data`, because a backend service must not subscribe to every detector image in
the facility to receive one table.

The **ingest** half has no ADR 0006 analogue: a da00 adapter with no stream lookup table,
so the internal stream name is the da00 source name (the shape the ROI route already
uses), plus a preprocessor case returning `LatestValueAccumulator`. That accumulator is
already context, so the context cache and the JobManager gate are unchanged.

Stream names are prefixed (`wavelength_lut/…`) and checked for collisions across the
whole registry at startup, since they share a namespace with device and motion streams.
The mechanism is named for the seam, not for its first user.

### Consumers request a stream by key, and the gate is read off the built graph

The instrument *offers* a context stream per workflow key (`Instrument.offer_context_stream`).
An offer names no spec, source or params. A consuming factory inserts a provider whose
argument is that key, and the workflow build takes the offer up when the key is an
ancestor of a target key, which is exactly the condition under which the job could not
compute without it. The gate is the set of streams the job requested.

Deriving the gate rather than declaring it keeps insertion and gating from disagreeing.
They are two statements of one fact, and as separate declarations they drift silently,
in the direction that leaves a job `pending_context` forever. It also removes the need
for a spec to name which of its sources or aux selections read the stream.

Offers do not replace bindings. A `ContextBinding` is still required where the key reaches
the graph only because the binding injects it (a chain patch), where the stream filling a
key varies per source, or where a stream private to one spec should not be subscribed by
every service. The `offer_context_stream` docstring records the line between the two.

### The gate is per job and depends on the job's parameters

Coordinate mode is a parameter, so a spec offering both modes must not gate its
time-of-arrival jobs on a table they never read. ADR 0003 accepted exactly this over-gate
for motion and prescribed splitting the spec if it ever became real. Both premises have
expired.

The cost of over-gating is far higher here than for motion. Motion streams are pushed
continuously by the control system; the table is published by an operator-started job
that re-emits only on chopper change. Over-gating makes the fallback mode depend on the
most fragile link in the chain.

The prescribed remedy was tried and reverted. Splitting the spec doubles the workflow list
and, because the dashboard keys its data plane by `(workflow_id, source_name,
output_name)`, gives the two modes different output identities: a plot cannot follow a
mode switch. Coordinate mode is a property of how you look at a detector, not of which
detector you look at.

So the gate is parameter-dependent, without anything saying so. The factory inserts the
provider unconditionally; a time-of-arrival job's params build a graph that never reaches
it, so the job requests nothing. This is affordable because the gate is resolved once, at
job creation, which already builds the workflow; Kafka subscriptions are derived
statically per spec and are a superset of any per-job gate; and the context cache is
keyed by stream name alone.

## Alternatives considered

| Option | Notes |
|---|---|
| **Generic mirror on a dedicated topic, offered streams, gate derived from the built graph (chosen)** | One declaration per fact: the spec names its outputs, the instrument names the stream per key, the graph names what it consumes. |
| Reuse `livedata_data` instead of a dedicated topic | Every backend service would subscribe to every detector image in the facility. Rejected. |
| A per-spec `ContextBinding` carrying a predicate over the job's params | The first shape, and it worked. It restates in a declaration what the graph already knows, so the two can disagree silently, and a spec reading a stream on behalf of other sources (LOKI's I(Q) reading the monitor table on detector sources) had to declare that mismatch. Superseded. |
| Split the spec into time-of-arrival and wavelength variants | ADR 0003's prescribed remedy. Tried and reverted: two workflow-list entries, two jobs to run, and two output identities so no plot survives a mode switch. |
| Route the stream as ungated aux with the file as default (ROI precedent) | Survives a cold start but makes reducing with a stale or nominal table a silent mode, which is what the feature exists to prevent. Rejected. |
| Name the mechanism for the lookup table rather than for the seam | Same code either way; the generic name is the honest one. Rejected. |

## Consequences

- `WorkflowSpec` gains `context_outputs`; a new `StreamKind`, topic, sink route, ingest
  route and preprocessor case carry it. The ingest half is the genuinely new mechanism.
- `Instrument` gains offered context streams alongside `ContextBinding`. This supersedes
  ADR 0003's param-dependent-context non-goal.
- Job creation becomes validate → build → read the gate off the workflow. Nothing
  consumed the gating set before the build, so the reordering costs nothing.
- Route derivation gathers the offered stream names and then drops them, since they
  appear in no stream lookup table. Harmless, because the context route is added
  unconditionally, and pinned by a test because it reads as a bug otherwise.
- Specs are unchanged: coordinate mode stays a parameter on one workflow, so output
  identity and any plot built on it survive a mode switch.

### Standing limitations

A context output has no liveness and no replay. These follow from the decision and are
not resolved by it; the lookup table is the first stream to feel them.

- **Nothing guarantees the producer job is running.** Under gating it is a hard
  prerequisite for every consumer; if nobody starts it, or someone stops it, consumers
  block on missing context. Recovery is to restart the producer.
- **A backend restart loses the stream.** Consumers pin to the high watermark, with no
  replay, compaction or seed (ADR 0002), and a producer that re-emits only on change
  stays silent during a steady run.
- **The gate protects startup, not steady state.** Gate graduation is one-way and context
  is sticky, so stopping the producer does not re-block consumers; they keep reducing with
  the last value indefinitely, silently.
- **Two concurrent producer jobs would flap**, both publishing the same stream names. The
  duplicate-name check is static across specs and cannot see two live jobs of one spec.

The first three are one problem with one shape of answer, borrowed from how a device
reports to NICOS: the producer heartbeats, recent re-emission is liveness, prolonged
silence is an alarm on every dependent job, and the heartbeat doubles as recovery after
a restart. That is affordable only once consumers clear on a change of *identity* rather
than on every message (issue 1248). A compacted topic consumed from the earliest offset
is the more principled recovery and remains the long-term answer, but needs a per-topic
exception to the unconditional high-watermark pin and message keys the sink does not
set. Alongside either: a dashboard guard before a running producer is reconfigured or
stopped, since the blast radius is every consumer on the instrument.
