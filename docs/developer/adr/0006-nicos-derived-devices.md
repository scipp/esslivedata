# ADR 0006: Expose designated workflow outputs to NICOS as derived devices

- Status: accepted
- Deciders: Simon
- Date: 2026-06-29

## Context

NICOS, the ESS experiment control software, runs alongside ESSlivedata. Through
NICOS (UI and scripts) users count or drive multi-day scans against *derived*
quantities — total counts in a monitor or detector bank, a fitted peak statistic
— that ESSlivedata computes. NICOS consumes the results but does the stream
processing nowhere itself.

NICOS wants each such quantity to look like an ordinary device, the way the EPICS
PVs it connects to directly already do: a **static, git-tracked device list**,
each device addressed by a **stable identifier**, delivered on a **low-volume
dedicated topic**, and protected from accidental reset / stop / reconfigure.
NICOS dislikes runtime-published configuration and does not re-accumulate values
— it reads the current cumulative reading.

The obstacle was thought to be naming. ESSlivedata's internal result stream is
keyed by `ResultKey = WorkflowId + JobId + output_name`, and `JobId` carries a
random `job_number` regenerated on every (re)start, so the external key is
unstable. Three options were circled for months (a fixed `job_number`; replicate
the result under a fixed id; hoist `job_number` out of `ResultKey` wholesale).
The reframing that unblocked it: NICOS does not need to read the internal result
stream at all. A dedicated, stable-keyed extraction of designated outputs lets
the `job_number` simply drop out of the external identity, and then *no consumer
ever needs to detect a job transition from the data stream* — the signal the
naming schemes were straining to encode.

A separate prototype (a dedicated `NicosOrchestrator`, a curated
`config/nicos_workflows.py` list, fixed `job_number`s, a parallel control plane)
predates this decision and is superseded by it.

## Decision

### Stable-keyed device extraction onto a dedicated topic

The backend extracts designated outputs of running jobs and republishes them on a
dedicated `StreamKind.LIVEDATA_NICOS_DATA` topic (`{instrument}_livedata_nicos_data`),
keyed by a stable **device name**. `DeviceExtractor` (`core/nicos_devices.py`)
selects the contracted outputs from each `JobResult` and emits one message per
device.

- **Identity**: `WorkflowId (instrument + name + version) + source_name +
  output_name`, mapped to a device name by the contract. The random `job_number`
  is not part of the external identity.
- **Wire schema**: `da00`. It carries the signal, its error bar, and the
  `start_time` coordinate as named variables. `f144` is a lone scalar and cannot
  carry the marker; it is not used.
- **Eligibility**: decided by the workflow registry (`WorkflowSpec.device_outputs`).
  The extractor trusts the contract and emits whatever it designates; there is no
  runtime scalar/cumulative check. Devices are scalar cumulative outputs by
  authoring convention, not by an enforced gate.

### `start_time` as the generation marker

A stable identity does not by itself signal when an accumulation was reset (job
start/stop, reconfigure, run transition, backend restart): a cumulative value
dropping to zero is otherwise indistinguishable from a genuine low reading. A
cumulative output already carries a 0-D `start_time` coordinate, stamped by the
`JobManager` as the time of the first data accumulated in the current generation.
It is constant for a generation's lifetime and re-armed on reset, so a change
tells NICOS the prior accumulation ended and a fresh series began. The extractor
republishes it untouched — there is no separate generation token.

Because the marker is the first-data time (ns since epoch), a later generation
has a later marker: it advances monotonically across resets and backend restarts
with no stored counter. It is the same coordinate the dashboard uses to label a
plot's accumulation interval, so the reset signal cannot drift from the value's
provenance.

### Registry-derived, committed contract

The `(workflow, source, output) → device` mapping is **derived from the workflow
registry**, not hand-authored. A `WorkflowSpec` declares `device_outputs` (output
field → device-name template); `DeviceContract` (`config/device_contract.py`)
expands that over the spec's `source_names`. The registry is the single source of
truth, read directly by the backend (what to extract) and the dashboard (which
outputs are devices).

Each instrument's `device_contract.yaml` is a **generated, committed export** for
NICOS's static-list preference; nothing reads it back. Regenerate with
`python -m ess.livedata.config.device_contract`; a test fails if a committed
export drifts from the registry. `WorkflowId.version` is part of device identity,
so a breaking change retires the old device name — it goes silent on the wire and
NICOS notices — rather than silently changing semantics under a stable name.

### A single unified job per `(workflow, source)`, gated in the dashboard

There are no dedicated NICOS jobs, no separate orchestrator, no fixed
`job_number`. One job per `(workflow, source)` is shared by dashboard and NICOS,
and is **live as a device iff its owning job is running** — the backend extracts
every contracted output of a running job. There is no per-output activation
toggle, no persisted activation state, and no dashboard→backend control channel;
exposure is declared once, in the registry/yaml, and the dashboard only reflects
and protects it.

The dashboard's role is read-only plus protection (`dashboard/derived_devices.py`
is the single source of truth for the derivation, reused by every surface):

- a **device-bearing badge** on a running workflow's card when it bears a device;
- a **per-output device marker** on output chips;
- a **confirmation gate** intercepting reset / stop / reconfigure on device-bearing
  workflows, naming the affected devices; *staging* the next config is allowed,
  *commit* is gated;
- a read-only **derived-devices overview** (contract ∩ running jobs).

The gate re-reads live job state inside the orchestrator method, not only in the
widget, so an action that races a job restart cannot act on stale state.

### NICOS-initiated reset

NICOS resets the accumulation behind a device by producing a `JobCommand` JSON on
the `{instrument}_livedata_commands` topic, keyed by the `WorkflowId` it already
holds from the contract export:

```json
{"kind": "job_command", "action": "reset",
 "workflow_id": "bifrost/monitor_histogram/1"}
```

- **Granularity is whole-workflow.** Omitting `job_id` selects every running job
  of the workflow; all its sources reset together. This matches NICOS usage — a
  single device, or several counted in sync (e.g. a difference of two monitors) —
  and there is no per-source selector. Reset acts on the job, so it zeros all of
  that job's outputs together.
- **No acknowledgement.** `message_id` is omitted, so no service acks (a
  workflow_id-keyed command with a `message_id` would be echoed by every backend
  service). The `start_time` generation jump on the device topic is the
  confirmation.
- `WorkflowId` accepts its `instrument/name/version` string form on the wire, so
  NICOS sends the contract string verbatim; the dashboard keeps sending the nested
  object form.

## Alternatives considered

| Option | Notes |
|---|---|
| **Stable-keyed extraction, `job_number` dropped from external identity (chosen)** | No consumer detects job transitions from data; naming becomes a non-problem. |
| Fixed `job_number` for NICOS jobs | Keeps `job_number` in the identity; needs a parallel orchestrator, a curated fixed-id yaml, a dedicated stream kind. Rejected. |
| Replicate the result under a fixed id in `ResultKey` | Still bakes `job_number` semantics into the key and duplicates the result. Rejected. |
| Hoist `JobId` out of `ResultKey` (dashboard-wide refactor) | Not forced by NICOS; the extraction decouples NICOS from `ResultKey` entirely. An optional internal cleanup judged on dashboard merits alone. Deferred. |
| Delta stream, NICOS accumulates | Elegant for additive quantities, but non-additive peak devices cannot be deltas, and NICOS prefers not to accumulate. Rejected. |
| `ep01` as a reset/epoch signal | Its native meaning ("transient loss, value persists") is the opposite of a reset, forcing fragile sequence interpretation. The explicit `start_time` field replaces it. Rejected. |
| Separate generation token variable | Redundant with `start_time`, which already exists and already drives plot labelling. Rejected. |
| Dedicated NICOS jobs / `NicosOrchestrator` (prototype) | Physical isolation guarantees nothing a UI gate doesn't, while duplicating heavy detector-stream processing and maintaining a permanent parallel code path. Superseded. |
| Runtime per-output activation toggle | Adds a control channel and persisted state the design avoids, and fights the static-list premise — liveness would depend on operator UI state. The low-volume extraction is harmless when unscanned. Rejected. |
| Per-`source_name` reset selector | Not needed given whole-workflow usage, and avoids perturbing dashboard bookkeeping that has no per-source reset state. Rejected. |
| Dedicated inbound NICOS command topic keyed by device name | Cleaner decoupling, but a new topic plus a contract-resolution translation point. Reusing `livedata_commands` needs no new wiring and NICOS already holds the `WorkflowId`. Rejected for v1. |

## Key design choices

### `start_time` is the single source of truth for the reset signal

The marker and the value share one coordinate, so the reset signal cannot drift
from the value's provenance, and the backend stores no extra counter. The mark
advances across backend restarts for free.

### The contract cannot drift from the registry

Deriving the contract from `device_outputs` means an output that does not exist,
or a source a spec does not declare, simply cannot appear. The only remaining
failure mode — a template rendering a duplicate device name — fails loud at
construction. `version` in the identity makes a breaking change fail-fast (the
device goes silent) rather than silently re-defining a stable name.

### The gate and the marker are complementary

The dashboard gate guards against *accidental* disruption; *intentional*
transitions and backend restarts still reset the accumulation, and NICOS detects
those via `start_time` regardless of gating. The dashboard cannot know whether
NICOS is actively scanning a device (one-way publish, separate system), so the
gate states the change is disruptive *iff* NICOS happens to be using the device.

### Reset is whole-workflow and unacknowledged

A job produces all its outputs together, so lifecycle operations exist only at job
granularity; "reset device X" means "reset the job feeding X". Keying by
`WorkflowId` and omitting `message_id` keeps NICOS decoupled from `job_number` and
from our acknowledgement plane, with the device topic itself carrying the
confirmation.

## Consequences

- A new `StreamKind.LIVEDATA_NICOS_DATA` and its topic; `DeviceExtractor` runs in
  the result/serialization path.
- `device_contract.yaml` is generated and committed per instrument, guarded by a
  drift test; regeneration is a single command.
- Dashboard derived-device surfaces are read-only reflections plus a confirmation
  gate; no new orchestrator, persisted state, or control channel.
- `WorkflowId` validates from its `instrument/name/version` string form on the
  wire, so external producers (NICOS) can address commands by the exported string.
- Backend restart resets cumulative values to zero; NICOS does not re-accumulate.
  The marker makes the reset detectable but recovery of the lost accumulation is
  manual — we bank on backend stability. The bounded escalation, if restarts prove
  frequent, is accumulator checkpointing for simple additive devices only, never
  generic raw-data replay (intractable for reduction workflows whose normalization
  needs windows longer than the update period).
- **Out of v1**: liveness (NICOS can later subscribe to the existing job
  heartbeat; `ep01` is not used) and a separate validity channel (the `da00` error
  bar carries the uncertainty; NICOS scripts its own statistic). The internal
  `ResultKey` / `job_number` refactor is off this critical path.
