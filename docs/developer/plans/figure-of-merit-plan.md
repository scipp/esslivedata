# Figure of Merit (FOM) — Preliminary Implementation Plan

Companion to `figure-of-merit-concept.md`. That document fixes the
operational picture; this one sketches the technical design.

## Goals

- Let NICOS consume a workflow-derived scalar under a stable address.
- Let NICOS reset that scalar at scan-point boundaries.
- Reuse existing schemas (`WorkflowConfig`, `JobCommand`, `ResultKey`)
  unchanged. Keep FOM concerns in a thin separable layer.
- Leave room for: dedicated FOM topic, scheduled reset/start, multiple
  FOM slots — without designing those right now.

## Non-goals (initial version)

- Authoring new workflows specifically for FOM use; existing workflows
  whose outputs include a scalar are reused.
- Embedding FOM config into NeXus / experiment metadata.
- Authentication beyond generic Kafka topic ACLs.
- Atomic gap-less reconfigure. A brief outage on swap is acceptable.

## What the existing backend already gives us

- **`JobId = (source_name, job_number: UUID)`**, generated per
  `WorkflowConfig`, carried into `ResultKey` and into output messages.
- **`JobCommand`** with `reset`, `stop`, `pause`, `resume` actions,
  fully wired through `JobManager.job_command()`.
- **Commands topic** (`{instrument}_livedata_commands`) consumed by
  every backend service; **responses topic** carries ACKs correlated
  by `message_id`.
- **`ResultKey` reaches consumers via the Da00 payload.**
  `Da00Serializer` puts `message.stream.name` (which is
  `ResultKey.model_dump_json()`) into the Da00 `source_name` field
  (sink_serializers.py:81). The Kafka message key is `None` for
  Da00 messages — the field is free.
- **Heartbeats** for service and per-job state already publish to
  `{instrument}_livedata_heartbeat` every 2 s.
- **Jobs are in-memory only.** Service restart loses all configured
  jobs.

## The shape of the proposal

A FOM is **not a kind of job**; it is a **binding** that designates
"output X of job Y is currently fom-N." The binding lives in a small
per-service registry, populated by `Bind` / `Unbind` messages from
the dashboard. When a workflow result whose `(job_id, output_name)`
matches a binding is emitted, a parallel **mirror** message is also
published, carrying the same Da00 payload but tagged for FOM
consumption.

NICOS only sees this mirror stream. It reads the `ResultKey` from the
Da00 payload (already there today), extracts `job_id`, and uses
ordinary `JobCommand` for resets. No FOM-specific command schema, no
extension to `JobCommand` or `WorkflowConfig`, no UUID5 conventions.

## Key design decisions

### D1. FOM is a stream-alias binding, not a job property

`WorkflowConfig` describes a workflow run; that concept is complete
and should not be polluted with FOM-ness. The binding is a separate
small mechanism — and a generic one: it aliases a chosen *stream
name* to a particular output of a particular job. FOM is the first
use case, by convention using stream names `fom-0`, `fom-1`, ...,
but the mechanism itself has nothing FOM-specific in it.

```python
class Bind:        # dashboard → backend, on the commands topic
    message_id: str | None
    stream_name: str           # e.g. "fom-0"
    job_id: JobId
    output_name: str

class Unbind:      # dashboard → backend, on the commands topic
    message_id: str | None
    stream_name: str
```

Each backend service runs a small dispatcher that consumes these
messages and maintains a registry: `stream_name → (JobId, output_name)`.
A service stores the binding only if it holds the named job; a
service holding the binding clears it on `Unbind`. The actor ACKs
exactly once on the responses topic; non-actors stay silent. The
dashboard waits for the ACK or a short timeout (timeout ⇒ no
service was the actor).

**Bind never replaces.** A `Bind` for an already-bound stream name
is rejected (the holder ACKs an error). Reconfigure always goes
through `Unbind` first (D4), so each command has at most one actor
and therefore at most one ACK — predictable across the cluster.

### D2. Output translation: mirror by binding

Add `StreamKind.LIVEDATA_FOM`. The result-emission path, when it
emits a `JobResult` for `(job_id, output_name)` matching a local
binding, also emits a parallel message with this kind. The original
`LIVEDATA_DATA` emission is unchanged.

The mirror message:

- **Stream name (`message.stream.name`)** is the underlying
  `ResultKey.model_dump_json()` — unchanged. This propagates into
  the Da00 `source_name`, so NICOS reads `ResultKey` and extracts
  `job_id` exactly as today's consumers do for any result.
- **Kafka message key** is set to the bound `stream_name` bytes
  (e.g., `b"fom-0"`). `Da00Serializer` currently emits `key=None`;
  for `LIVEDATA_FOM` messages we set it. This is the marker NICOS
  filters on in phase 1 (shared topic) and is harmless in phase 2
  (dedicated topic).
- **Stream kind** is `LIVEDATA_FOM`, which routes to the topic per
  `stream_kind_to_topic`.

**Phase 1.** `LIVEDATA_FOM` maps to the existing
`{instrument}_livedata_data` topic. NICOS subscribes there and
filters messages by Kafka key prefix `fom-`.

**Phase 2.** `LIVEDATA_FOM` maps to a dedicated
`{instrument}_livedata_fom` topic. One-line change in
`stream_kind_to_topic`. NICOS switches subscription. The Kafka key
becomes redundant for filtering but remains useful for partitioning
and is harmless to keep.

The bound stream name needs to reach the FOM serializer to set the
Kafka key. Cleanest plumbing: the mirror layer attaches the
stream name to the emitted `Message` (a small new optional field,
or via a thin wrapper type for FOM messages); the serializer
reads it. Confined to the emission path — doesn't leak into
`JobManager`.

### D3. NICOS interface: existing schemas, nothing new

NICOS:

- Subscribes to the data topic (phase 1) or FOM topic (phase 2).
- Filters by Kafka key in phase 1 (`startswith("fom-")`); ignores
  it in phase 2.
- Decodes Da00 → reads `source_name` → parses as `ResultKey` → has
  `(job_id, output_name)`.
- Issues `JobCommand(job_id=..., action=reset)` on the commands
  topic for scan-point resets. ACK via the existing responses topic.

NICOS has no knowledge of slots-as-internal-concepts, bindings, or
any new schema. The slot is just a string in the Kafka key it
filters on. Reset goes through the existing `JobCommand` path with
no FOM-aware code in the dispatcher.

### D4. Configure / reconfigure flow

Configure (`fom-N` is currently free):

1. Dashboard sends `WorkflowConfig` for the chosen workflow on the
   chosen source. The job is created in some service.
2. Dashboard sends `Bind(stream_name="fom-N", job_id=..., output_name=...)`
   and awaits the ACK. Mirror emission begins on next result.

Reconfigure (`fom-N` currently bound to job A on service S):

1. Dashboard sends `Unbind(stream_name="fom-N")` and awaits the
   ACK. Mirror emission stops; the publisher is flushed before
   ACK so no in-flight `fom-N` message can race the new binding
   (D5).
2. Dashboard *optionally* sends `JobCommand(job_id=A, action=stop)`
   if job A is no longer needed. (Job A continues to publish on
   `LIVEDATA_DATA` if not stopped.)
3. Dashboard sends `WorkflowConfig` for the new workflow.
4. Dashboard sends `Bind(stream_name="fom-N", job_id=B, output_name=...)`.

`Bind` never replaces (D1); the explicit `Unbind` step is
mandatory, not optional, when the slot is currently held. Both
stop and reconfigure go behind the dashboard's confirmation guard
described in the concept document.

### D5. Sharding and cross-service coordination

Each service consumes the commands topic; bindings live only in
the service holding the bound job. Cross-service correctness for
reconfigure relies on the existing `message_id` / responses-topic
ACK mechanism, plus the "no replace" rule (D1) that keeps each
command to a single actor and a single ACK.

`Unbind` processing on the holding service:

1. Removes the binding entry.
2. Flushes the result publisher to drain any in-flight FOM mirror
   messages for that stream name.
3. Sends an ACK on the responses topic.

`Bind` processing on the holding service:

1. If the stream name is already bound locally, ACKs an error.
2. Otherwise stores the binding and ACKs success.

Services that aren't the actor (don't hold the named job, don't
hold the named stream) stay silent. The dashboard waits for the
single expected ACK or a short timeout (timeout ⇒ no actor ⇒
"nobody held it" for `Unbind`, "no such job" for `Bind`). Service
crashes mid-command are absorbed by the timeout — the crashed
service's binding state is gone with it.

This protocol does not depend on global ordering on the commands
topic.

### D6. Persistence: none

Backend restart drops jobs and bindings together. The dashboard,
on detecting reconnection, surfaces "FOM not configured" and the
operator re-engages. NICOS's existing time-based fallback covers
the gap.

Auto-replay on restart (e.g. via log-compacted commands topic) is
out of scope for v1: it opens a class of bugs (replay racing live
commands, ghost bindings, stale workflow versions) without solving
a real operational problem here, since restarts are rare and
observable.

### D7. FOM slot naming convention

The binding mechanism (D1) takes arbitrary stream names. For FOM
specifically, the dashboard and NICOS agree on the convention
`fom-0`, `fom-1`, ..., up to a small fixed maximum (e.g. 8). This
agreement is documentation, not schema: there is no enum or enum
range in the binding messages. Other use cases of the binding
mechanism (e.g. dashboard-pinned named outputs) can use any other
naming scheme without colliding.

## Implementation phases

**Phase 0 — agreement.** Confirm D1–D7. Write down the FOM naming
convention and the `Bind` / `Unbind` schemas as a short spec NICOS
can refer to.

**Phase 1 — core mechanism.**
- Define `Bind` and `Unbind` message types (generic stream-alias,
  not FOM-specific) and wire them through the commands-topic
  message adapter.
- Add a per-service binding registry plus the dispatcher that
  consumes `Bind` / `Unbind`. Enforce "no replace" (D1).
- Add `StreamKind.LIVEDATA_FOM`, mapped to the existing data topic.
- Extend the result-emission path: when a `JobResult`'s
  `(job_id, output_name)` matches a local binding, emit a parallel
  message with kind `LIVEDATA_FOM`, bound stream name attached.
- Extend the serializer for `LIVEDATA_FOM` to set the Kafka key
  to the bound stream name bytes (Da00 `source_name` stays as
  `ResultKey` JSON).
- Verify and, if needed, add a flush-on-`Unbind` step so that the
  ACK is emitted only after in-flight FOM mirrors have been
  produced.
- Tests at JobManager / OrchestratingProcessor / serializer level
  exercising bind / mirror / unbind, the "no replace" rejection,
  and a multi-service rebind scenario.

**Phase 2 — dedicated FOM topic.**
- Switch the topic mapping for `LIVEDATA_FOM` to
  `{instrument}_livedata_fom`. NICOS switches subscription. No
  other backend changes.

**Phase 3 — scheduled reset/start.**
- Wire NICOS-issued scheduled commands into the existing
  deferred-reset machinery once timing requirements warrant it.

## Dashboard side (sketch — separate plan needed)

The dashboard needs a place to:

- Choose a workflow and parameters for a slot, then submit
  `WorkflowConfig` + `Bind` as the configure operation.
- See current FOM state (slot bound, current value, errors).
- Stop or reconfigure the slot, behind a confirmation guard.

Two reasonable homes:

- **Dedicated FOM control panel.** Conceptually clean, but pulls
  workflow-selection UI out of where users currently expect it.
- **"Mark as FOM" affordance on the existing workflow-configuration
  widget.** Reuses the existing UI; binding becomes a follow-up
  action on a normal workflow configuration.

Decision deferred to a dashboard-focused plan.

## Wider-architecture sanity check

- **Device-level FOMs** (raw counters, monitor outputs) are already
  consumed directly by NICOS and do not need this mechanism. The FOM
  layer is for *workflow-derived* quantities.
- **A dedicated FOM service** separate from ESSlivedata would
  duplicate `JobManager`, the topic plumbing, and the workflow host.
  The binding-mirror approach achieves the same separation
  conceptually without a new service.
- **NICOS as workflow orchestrator** would push workflow internals
  into the control system — exactly what this design avoids.

ESSlivedata hosting FOM jobs (with FOM-ness expressed as a binding,
not a job property) is the right fit. Risk to manage: ESSlivedata
becomes part of the experiment-control critical path. NICOS's
time-based fallback bounds this.

## Risks and unknowns

- **No persistence.** Backend restart drops bindings and jobs.
  Dashboard must clearly surface "FOM not configured"; NICOS falls
  back to time-based scans without alarm. Deliberate v1 choice (D6).
- **Flush-on-unbind correctness.** D5 hinges on `Unbind`
  processing actually flushing in-flight FOM mirrors before
  ACK'ing. Needs verification against the producer/sink behaviour;
  add a flush if missing. Without it, a reconfigure can briefly
  emit overlapping mirror messages from old and new bindings.
- **Serializer slot plumbing.** The Da00 serializer is currently
  pure (depends only on stream name / timestamp / value). Setting
  the Kafka key for FOM requires the slot to reach the serializer,
  which is the one place "FOM-awareness" leaks into encoding.
  Confined and small, but worth a clean abstraction.
- **Stale `job_id` race during reconfigure.** NICOS holds the
  most recently observed `job_id`; if it issues a reset between
  unbind and the first message of the new binding, it targets
  the old job (no-op if stopped, otherwise resets a job NICOS no
  longer cares about). Same brief-outage class we already accept.
- **FOM naming convention is documentation, not schema.** A
  dashboard bug could publish under a name NICOS doesn't watch,
  or vice versa. Acceptable now (small deployment, two cooperating
  systems); if the cooperation grows, a per-instrument config
  enumerating canonical names with human labels is the obvious
  next step.

## What this plan does *not* settle

- FOM slot count and exact naming (`fom-N`, `fom_N`, ...).
- Dashboard widget placement.
- Concrete NICOS-side interface — owned by the NICOS team and
  needs a separate conversation.
