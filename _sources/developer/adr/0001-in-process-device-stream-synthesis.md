# ADR 0001: In-process synthesis of merged device streams

- Status: accepted
- Deciders: Simon
- Date: 2026-05-19

## Context

Motorised devices on ESS instruments publish three independent f144 substreams per device with non-synchronised timestamps:

- **RBV** — readback (actual measurement)
- **VAL** — target (setpoint)
- **DMOV** — "done moving" idle flag

Before this change only RBV was consumed anywhere in the stack — VAL and DMOV arrived from Kafka but no workflow or dashboard view used them. The system had no notion of a "device" as a unit, so a workflow that wanted, for example, to filter out readings taken while the motor was still moving had no way to access DMOV alongside RBV. The setpoint was similarly invisible to consumers.

Goal: make each device available as one logical stream and one DataArray — RBV as data, target/idle attached as time-dim coords on a shared `time` axis — so that any consumer (timeseries plotting, Sciline-based science workflows) can opt into using the extra fields without re-implementing cross-stream alignment. This change does not by itself introduce any consumer that filters on `idle` or overlays `target`; it unblocks those developments.

## Decision

Merge per-device substreams **in-process via a `MessageSource` decorator** (`DeviceSynthesizer`), matching the precedent of `ChopperSynthesizer`. The synthesizer:

1. Maintains per-device state (last-seen value + time per substream).
2. Emits `LogData` messages (with optional `target` / `idle` fields populated per device configuration) on a new `StreamKind.DEVICE` once bootstrap is complete (every configured substream observed at least once).
3. Suppresses configured substream messages from forwarding — for configured devices, raw VAL/DMOV/RBV no longer reach downstream preprocessors.
4. Emits on **every** input event for a configured substream (union-anchored).
5. Stamps each emit with `max(rbv_time, val_time, dmov_time)` across the synthesizer's last-seen substream times.

A new `Device(Stream)` record sits in `Instrument.streams` alongside its `F144Stream` substreams; `name_streams` auto-detects devices by EPICS source-suffix classification. The existing `ToNXlog` preprocessor (dispatched on `StreamKind.DEVICE` with `has_target` / `has_idle` flags set per device) grows a `time`-indexed `sc.DataArray` carrying RBV as data and optional `target` / `idle` coords.

## Alternatives considered

| Layer | Device first-class? | Notes |
|---|---|---|
| **MessageSource decorator (chosen)** | Real stream | Matches `ChopperSynthesizer`; one merged stream visible to every consumer |
| Per-stream preprocessor | n/a | Abstraction violation — preprocessors are single-`StreamId` by construction |
| Sciline provider per workflow | Graph artifact | Per-workflow state replication; non-Sciline consumers (timeseries) get no merged view |
| Dedicated aggregator service → new Kafka topic | Real (observable in Kafka) | Heavy: new deployment + schema versioning per instrument |

The cross-stream coupling belongs at the layer where streams are first-class. Anywhere else, the device is either a graph artifact (Sciline) or a multiplexed preprocessor (abstraction violation).

## Key design choices

### Union-anchored emission

Emit on every input event for a configured substream, not on RBV-only or change-only. RBV cadence dominates anyway (~10s–100s Hz during motion); VAL adds ~1 event per setpoint change and DMOV ~2 per motion. Union makes motion-start and settle transitions addressable as rows in the merged DataArray. Downsampling to RBV-only is trivial for any consumer that wants it; recovering transition rows after the fact is not.

### `max(rbv_time, val_time, dmov_time)` timestamp policy

Each row represents "device state as known up to time T". Monotonic by construction — out-of-order substream messages fold into state without producing dropped emits. The alternative ("use the triggering input's time") would drop any emit whose triggering event has an older timestamp than the last, losing both the row and the state update.

Edge case: a substream message older than the current max updates state correctly but emits at the unchanged max-time, which `ToNXlog` drops as a duplicate. The new substream value becomes visible at the next emit that advances max. State is never lost; only the addressability of that one transition as its own row.

### Bootstrap before first emit

Suppress emits until every configured substream has been observed at least once. Worst-case latency is bounded by the f144 forwarder's ~10s republish cadence. Producing partial samples with `Optional`-padded fields would force every downstream consumer to defensively handle missingness.

### `idle` is a coord, not a mask

A mask would silently exclude moving samples from `sum`/`mean`/etc., biasing reductions — but in-motion readings are not invalid, they measure a different thing. Whether to filter on `idle` is a workflow-level analysis choice, not a data-quality verdict.

### Substream suppression is unconditional

The synthesizer suppresses raw VAL/DMOV/RBV messages for configured devices, so downstream code sees only the merged `Device` stream. This keeps the routing model simple — the merged DataArray is the single canonical view of the device — and forecloses a class of confusion where a consumer might bind a workflow key to a raw substream when the merged view was intended. The merged DataArray carries every substream as a coord, so no information is lost.

### Auto-detection by EPICS source suffix

`name_streams` classifies each f144 substream by the suffix of its `source` attribute (the PV name written by the f144 forwarder): `.RBV` → readback, `.VAL` → target, `.DMOV` → idle. Substreams co-located under one NeXus parent group are then bundled into a `Device` if classified RBV is present **and** at least one of classified VAL / DMOV is present. Parent `nx_class` is not consulted — conforming devices exist on `NXcollimator` and `NXlog` parents at ESS; restricting to `NXpositioner` would miss them.

EPICS suffixes are the primary signal rather than NeXus child names because they are fixed by the EPICS motor-record convention while NeXus child names drift across instruments (e.g. tbl uses `position_readback`/`position_setpoint` while bifrost/loki use `value`/`target_value`). The NeXus filewriter template is per-instrument and operator-editable; the source attribute is what the f144 forwarder publishes for the PV, and the IOC-side `.RBV` / `.VAL` / `.DMOV` suffixes are stable across the facility.

Substreams whose `source` is `None` are not classifiable and are not grouped into Devices. Production f144 streams always carry a source; only synthesised or hand-crafted test fixtures can omit it. Test fixtures must set realistic sources to participate in device synthesis.

Non-motor readback variants (`-PosReadback` from piezo encoders, etc.) are deliberately not in the suffix allowlist. They either appear on their own (and remain plain streams) or appear as extra siblings of a canonical `.RBV`/`.VAL`/`.DMOV` group (in which case they remain plain streams alongside the synthesised `Device`).

### State persists across `RunStart` / `RunStop`

Run boundaries are file-writing markers, not physical events — a motor's position survives the boundary. The synthesizer holds no run-lifecycle hooks; `ToNXlog` (the same accumulator used for plain f144 logs) carries no-clear-on-run behaviour through unchanged.

## Consequences

- For configured devices, the dashboard exposes one merged per-device timeseries entry instead of an RBV-only stream entry; the underlying VAL and DMOV substreams remain inaccessible as standalone plots (they only ever surface as coords on the merged DataArray).
- A `Device` entry in `Instrument.streams` is a valid `LogContextBinding` target via the same dict-membership check as `F144Stream` — this is the migration path Bifrost uses for `InstrumentAngle` / `SampleAngle`.
- The extra `target` / `idle` coords ride along passively through `sc.lookup` and broadcasting; no downstream API changes were required.
- Detection is conservative: lone-RBV groups (no setpoint / no idle flag) are left as plain streams. Future need for setpoint-less sensor-style devices can relax the rule.
