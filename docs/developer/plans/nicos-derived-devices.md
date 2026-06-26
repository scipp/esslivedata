# NICOS derived devices: mechanism

NICOS drives counting and multi-day scans against derived quantities computed by
ESSlivedata (monitor/detector total counts, fitted peak statistics). It needs
these as a **static list of devices**, addressed by a **stable** identifier, on a
**dedicated low-volume topic**, and must not have them reset/stopped/reconfigured
by accident.

The dashboard UI/UX (activation, gating, warnings) is a separate, composable
plan: `nicos-derived-devices-ui.md`.

## Projection onto a dedicated, stable-keyed topic

NICOS does **not** consume the internal result stream (keyed by `ResultKey` =
`WorkflowId + JobId + output_name`, where `JobId` carries a random `job_number`).
ESSlivedata publishes a **projection** of designated outputs onto a dedicated
topic:

- **Scope**: scalar, cumulative outputs only (e.g. `counts_total`), already 0-D
  `DataArray`s. NICOS reads the current value; it does not re-accumulate (its
  stated preference).
- **Identity**: `WorkflowId (instrument + name + version) + source_name +
  output_name`. The random `job_number` is not part of the external identity.
  Including `version` makes drift fail-fast (see Contract).
- **Wire schema**: `da00`. It carries the signal, its error bar, and the
  `start_time` coordinate (below) as named variables. `f144` cannot carry the
  extra variable (it is a lone scalar) and is not used.
- **Topic**: dedicated low-volume topic (a one-line addition to
  `stream_kind_to_topic`).

The stable identity gives NICOS a static device list. It does **not** by itself
signal when the accumulation was reset (job start/stop, config change, backend
restart): a cumulative value dropping to zero is otherwise indistinguishable from
a real low reading. The `start_time` coordinate carries that signal.

## Generation marker: start_time

A cumulative output already carries a 0-D `start_time` coordinate, stamped by the
`JobManager` as the time of the first data accumulated in the current generation
(`core/job.py`). It is constant for the lifetime of a generation and is re-armed
on reset (job start/stop, reconfigure, run transition), so a change tells NICOS
the prior accumulation was reset and the value begins a fresh series, not a
continuation. The projector republishes it untouched -- no separate token.

- **Derivation**: the start time of the first active data batch (ns since epoch).
  A later generation has a later first-data time, so the marker advances across
  resets and backend restarts with no stored counter.
- **No schema change**: `da00` is a list of named variables; `start_time` is one
  more beside signal/errors. NICOS already surfaces non-signal variables (e.g.
  `reference_time`).
- **Single source of truth**: the same coordinate the dashboard uses to label a
  plot's accumulation interval, so the reset signal cannot drift from the value's
  provenance.

Backend restart resets cumulative values to zero; NICOS does not re-accumulate.
`start_time` makes the reset **detectable** (it jumps), so NICOS does not mistake
the reset value for a real reading. Recovering the lost accumulation is manual; we
bank on backend stability. If restarts prove frequent, the bounded escalation is
**accumulator checkpointing** for simple additive devices only -- never generic
raw-data replay, which is intractable for reduction workflows whose normalization
needs longer windows than the update period.

## Liveness and validity

- **Liveness**: not in v1. If NICOS needs to distinguish "job alive, value static"
  from "job gone", it subscribes to the existing job heartbeat. `ep01`
  connection-status is not used.
- **Validity**: no separate channel. The `da00` error bar carries the uncertainty;
  NICOS scripts whatever statistic/threshold it needs (e.g. rejecting a peak with
  too few counts). `al00`/`ep01` do not apply -- NICOS receives EPICS PVs directly,
  not via Kafka.

## Unified jobs, gated in the UI

One job per `(workflow, source)`, shared by dashboard and NICOS. No dedicated NICOS
jobs, no separate orchestrator, no fixed `job_number`.

Even a dedicated NICOS job must remain reconfigurable, so physical isolation gives
no guarantee a UI gate doesn't, while costing duplicate processing of the same raw
stream (heavy for detector banks) and a permanent parallel code path. Protection
against accidental reset/stop/reconfigure is a **gate** in the dashboard, scoped to
running workflows that bear a device per the yaml contract (UI plan).

The gate and the marker are complementary: the gate guards against *accidental*
disruption, while *intentional* transitions and restarts still reset the
accumulation and NICOS detects those via `start_time` regardless of gating.

## Contract

The `(workflow, source, output) -> device` mapping is **derived from the workflow
registry**: a `WorkflowSpec` declares `device_outputs` (output field -> device-name
template), and the contract expands that over the spec's `source_names`. The
registry is the single source of truth, read directly by the backend and the
dashboard. A shared default on the monitor workflow registration makes every
instrument's cumulative monitor total a device, with no per-instrument repetition.

- NICOS still wants a static, git-tracked device list (it dislikes runtime-published
  config). Each instrument's `device_contract.yaml` is a **generated, committed
  export** of the registry-derived contract -- nothing reads it back. Regenerate
  with `python -m ess.livedata.config.device_contract`; a test fails if a committed
  export drifts from the registry.
- Because the contract is generated from the registry it **cannot drift** from it:
  an output that does not exist, or a source the spec does not declare, simply
  cannot appear. The validation that a hand-authored yaml needed against the
  registry is gone. The one remaining failure mode -- a template rendering a
  duplicate device name -- fails loud at construction.
- `WorkflowId.version` is part of device identity, so a breaking change retires the
  old device name (it goes silent on the wire, NICOS notices) rather than silently
  changing semantics behind a stable name.
- A device is **live whenever its owning `(workflow, source)` job is running**. The
  backend projects every declared device output of a running job; there is no
  runtime activation toggle and no dashboard->backend control channel. Liveness is
  predictable from "contract intersect running jobs", matching NICOS's static-list
  preference.

## Implementation

Producing the projection: derive the stable `source_name`, extract the scalar
value (with its `start_time` coordinate riding along as the generation marker),
and publish to the dedicated topic. The natural home is the sink/serialization path
that already turns results into `da00`. The exact seam (inline in
`orchestrating_processor` vs. a dedicated projector) is settled at build time; it
does not affect the wire contract above.

## Rejected alternatives

- **Fixed `job_number` for NICOS jobs**: keeps `job_number` in the identity, needs
  a parallel orchestrator, a curated fixed-id yaml, and a dedicated stream kind.
  The projection makes `job_number` irrelevant to NICOS.
- **Replicate the result with a fixed id part in `ResultKey`**: still bakes
  `job_number` semantics into the key and duplicates the result.
- **Move `JobId` out of `ResultKey` (dashboard-wide refactor)**: not forced by
  NICOS -- the projection decouples NICOS from `ResultKey` entirely. It is an
  optional internal cleanup judged on dashboard merits alone, off this critical
  path.
- **Delta stream, NICOS accumulates**: elegant for additive quantities, but
  non-additive peak devices cannot be expressed as deltas, and NICOS prefers not to
  accumulate.
- **`ep01` as a reset/epoch signal**: its native meaning ("transient loss, value
  persists") is the opposite of a reset, forcing fragile sequence interpretation.
  The `start_time` coordinate is an explicit field instead.
