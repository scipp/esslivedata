# NICOS derived devices: mechanism

Status: design agreed, not implemented. Several pieces are explicitly unsolved
(see Open dependencies). This document records the decision tree and, equally
important, the rejected alternatives and why -- so it is not re-litigated.

The dashboard UI/UX (activation, gating, warnings) is a separate, composable
plan: `nicos-derived-devices-ui.md`.

## Problem

NICOS drives counting and multi-day scans against derived quantities computed by
ESSlivedata (monitor/detector total counts, fitted peak statistics). NICOS needs
these as a **static list of devices**, addressable by a **stable** identifier,
delivered on a **low-volume dedicated topic** (so it need not subscribe to the
heavy result-data topic). These jobs must not be reset/stopped/reconfigured by
accident.

The historical blocker was device naming: results are identified by `ResultKey`
(`WorkflowId + JobId + output_name`) embedded in the message payload, and `JobId`
carries a random `job_number` (UUID). External keys are therefore unstable across
job restarts.

## Decision: a dedicated stable-keyed projection

NICOS does **not** consume the internal result stream. ESSlivedata publishes a
**projection** of designated outputs onto a dedicated topic:

- **Scope**: scalar, cumulative outputs only (e.g. `counts_total`). These already
  exist as 0-D `DataArray`s. NICOS reads the current value; it does not
  re-accumulate (its stated preference).
- **Identity**: `WorkflowId (instrument + name + version) + source_name +
  output_name`. The random `job_number` is dropped from the external identity.
  Including `version` makes drift fail-fast (see Contract).
- **Wire schema**: `da00` or `f144` -- NICOS consumes both. Free, deferrable
  choice; `f144` makes a scalar device byte-identical to an EPICS PV.
- **Topic**: dedicated low-volume topic (a one-line addition to
  `stream_kind_to_topic`).

Because the projection is dedicated and stable-keyed, **no consumer needs to
detect a job transition from the data stream**. This is the crux: the naming
problem dissolves rather than being chosen.

### Liveness and validity (sibling signals, EPICS-native)

- **Liveness**: NICOS uses **heartbeat staleness** (it already does this for real
  devices). `ep01` connection-status is **not** used -- its native meaning
  ("transient loss, value persists") is the opposite of what a reset would mean,
  so overloading it for "forget the past" would force fragile sequence
  interpretation against the grain of how NICOS reads `ep01` for PVs.
- **Validity**: `al00` alarm `INVALID` marks a value not to be trusted (peak with
  too few counts). Optional, deferrable, native to NICOS.

### No stream-borne reset signal

Deliberate reconfiguration is **gated** in the dashboard (UI plan) and does not
happen mid-scan. Unexpected reset = backend restart mid-scan, which is out of
scope (manual recovery). Hence no epoch/generation/tombstone field is needed on
the wire. Should that change, it must be an **explicit** field, never inferred
from `start_time` (which is per-window for sliding accumulators and thus not a
uniform epoch signal) or from `ep01`.

## Decision: unified jobs + UI gating (not isolation)

One job per `(workflow, source)`, shared by dashboard and NICOS. No dedicated
NICOS jobs, no separate orchestrator, no fixed `job_number`.

Rationale: even a "dedicated" NICOS job must remain reconfigurable, so physical
isolation provides no guarantee a UI gate doesn't. Dedicated jobs also cost
duplicate processing of the same raw stream (heavy for detector banks) and a
permanent parallel code path. Protection against accidental reset/stop/reconfigure
is therefore a **gate** in the dashboard, scoped to outputs currently exposed as
devices. See the UI plan.

## Decision: static versioned contract

The `(workflow, source, output) -> device` mapping is a **shared, versioned yaml**
(NICOS dislikes runtime-published config). Two layers compose:

- The **yaml** defines the *namespace* of devices that may exist -- NICOS's static
  list. `WorkflowId.version` is part of the device identity, so a breaking change
  retires the old device name (it goes silent, NICOS notices via staleness) rather
  than silently changing semantics behind a stable name. **Fail-fast.**
- **Activation** (UI plan) controls whether a given device's projection is
  currently *published*. The yaml says what can exist; activation says what is
  live right now.

An entry that cannot be resolved against the instrument's registry (wrong
workflow/version) must be surfaced **loudly** (dashboard-visible), not dropped to
a `log.info`, since it is a misconfiguration of a scan-critical contract.

## Open dependencies (unsolved -- do not present as done)

1. **Stable-keyed liveness (core dependency).** The current job heartbeat on
   `{instrument}_livedata_heartbeat` (x5f2, `core/orchestrating_processor.py`
   `_report_status`, `kafka/x5f2_compat.py`) is keyed `source_name:job_number` and
   its payload carries the full `JobId`. A stable-keyed projection consumer will
   **not** match it. NICOS-via-heartbeat-staleness requires emitting job liveness
   under the stable identity (`WorkflowId + source_name + output_name`, no
   `job_number`). Unimplemented; the scheme does not work without it.
2. **Backend restart resilience.** Cumulative values reset on restart; NICOS does
   not re-accumulate. Decision: bank on backend stability, recover manually.
   *Measure restart frequency in production* before building anything. If
   warranted, escalate to **accumulator checkpointing** (snapshot/reload state)
   for simple additive devices only -- never generic raw-data replay, which is
   intractable for reduction workflows whose normalization needs longer windows
   than the update period.
3. **`al00` validity** for non-additive (peak) devices.
4. **Projection placement.** Where the projection is produced (inline in
   `orchestrating_processor` vs. a dedicated projector) and how it derives the
   stable `source_name` and the scalar value. Implementation detail; resolve at
   build time.

## Rejected alternatives

- **(a) Fixed `job_number` for NICOS jobs** (current branch): keeps `job_number`
  in the identity, needs a parallel `NicosOrchestrator`, curated fixed-id yaml, and
  a dedicated stream kind. Replaced -- the projection makes `job_number`
  irrelevant to NICOS.
- **(b) Replicate result with a fixed id part in `ResultKey`**: still bakes
  `job_number` semantics into the key and duplicates the result; same selection
  problem as (a).
- **(c) Move `JobId` out of `ResultKey` (dashboard-wide refactor)**: was thought
  to be forced by NICOS. It is **not** -- the projection decouples NICOS from
  `ResultKey` entirely. (c) becomes an optional internal cleanup judged on
  dashboard merits alone, off the critical path.
- **Delta stream, NICOS accumulates**: elegant for additive quantities (dissolves
  epoch/restart entirely) but non-additive peak devices cannot be expressed as
  deltas, and NICOS prefers not to accumulate. Parked as a possible future
  optimization, not the spine.
- **`ep01` as a reset/epoch signal**: awkward and fragile (see Liveness).

## Net effect

The months-long identity puzzle (a/b/c) is dissolved, not chosen: a dedicated,
stable-keyed, low-volume projection plus heartbeat-based liveness means no
consumer needs a transition signal in the data. The remaining work is real but
bounded, and the genuine risk now lives in one named place -- stable-keyed
liveness -- rather than diffused across the schema.
