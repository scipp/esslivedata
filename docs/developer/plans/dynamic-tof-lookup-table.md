# Dynamic TOF lookup table workflow

Issue: #894

## Goal

Replace the static "select a precomputed lookup table from a dropdown" model
with a workflow that **computes** a TOF-to-wavelength lookup table from
chopper settings, so that arbitrary commissioning configurations are supported
without shipping a precomputed file for each one.

This plan covers only the producer side: a new ESSlivedata workflow whose
result is the lookup-table `sc.DataArray`. How downstream services consume the
result is deferred to a follow-up.

## Non-goals

- **Downstream consumption.** How the published table reaches detector,
  monitor, or reduction workflows (sciline binding, context-key plumbing,
  cross-service consistency) is out of scope here. Solving the producer first
  lets us iterate on consumption independently.
- **Replacing the existing precomputed-table path immediately.** This workflow
  lands alongside today's filename-based `LookupTable` loading. We can deprecate
  the latter once the producer/consumer ends meet.
- **Standalone "chopperless fallback" path** (#894 as originally framed).
  Once the dynamic producer exists, "chopperless" is just the degenerate input
  case (`DiskChoppers = {}`); no separate workflow logic is needed. (A small
  JobManager extension *is* needed to fire `finalize` once and emit the result
  synchronously when the job's primary source resolves to zero physical
  streams — see the Chopperless instruments section — but that generalizes
  beyond this workflow rather than being a chopperless-specific code path.)

## Approach

The upstream `LookupTable` from `ess.reduce.unwrap.lut` is a dataclass with
five fields:

```python
@dataclass
class LookupTable:
    array: sc.DataArray             # output of the simulation
    pulse_period: sc.Variable       # input parameter
    pulse_stride: int               # input parameter
    distance_resolution: sc.Variable  # input parameter
    time_resolution: sc.Variable    # input parameter
    choppers: sc.DataGroup | None   # provenance, not needed here
```

Of these, only `array` is *produced* by the simulation; the rest are
*inputs*. That maps cleanly onto ESSlivedata's existing pattern: a pydantic
params model carries the inputs, the workflow returns a single `sc.DataArray`,
and the existing da00 sink path publishes it.

There is no new wire format or serialiser: a `sc.DataArray` round-trips
through `scipp_to_da00` / `da00_to_scipp` with values, coords, variances, and
units intact (verified).

## Workflow definition

### Inputs (pydantic params model)

`sc.Variable` is not a valid field type for UI-generated forms. The existing
convention (see `parameter_models.py`: `RangeModel`, `EdgesModel`,
`WavelengthRange`, etc.) decomposes a scalar-with-unit into a `float value`
plus a `StrEnum unit`, with a getter that constructs the `sc.Variable` on
demand.

The UI requires `LookupTableParams` to nest pydantic models exactly one level
deep, so even unitless scalars need a wrapper model rather than a bare
`int`/`float` field on `LookupTableParams`.

Small wrapper models, added to `parameter_models.py` alongside the existing
patterns:

```python
class PulsePeriod(BaseModel):
    value: float = Field(default=1000.0/14, description="Pulse period.")
    unit: TimeUnit = Field(default=TimeUnit.MS, description="Unit.")

    def get(self) -> sc.Variable:
        return sc.scalar(self.value, unit=self.unit.value)


class DistanceResolution(BaseModel):
    value: float = Field(default=0.1, description="Distance bin resolution.")
    unit: LengthUnit = Field(default=LengthUnit.METER, description="Unit.")

    def get(self) -> sc.Variable:
        return sc.scalar(self.value, unit=self.unit.value)


class TimeResolution(BaseModel):
    value: float = Field(default=250.0, description="Time bin resolution.")
    unit: TimeUnit = Field(default=TimeUnit.MICROSECOND, description="Unit.")

    def get(self) -> sc.Variable:
        return sc.scalar(self.value, unit=self.unit.value)


class LtotalRange(RangeModel):
    """Range of total flight paths covered by the lookup table."""
    start: float = Field(default=5.0, description="Shortest L_total.")
    stop: float = Field(default=200.0, description="Longest L_total.")
    unit: LengthUnit = Field(default=LengthUnit.METER, description="Unit.")


class Simulation(BaseModel):
    pulse_stride: int = Field(default=1, ge=1, description="Pulse stride.")
    num_simulated_neutrons: int = Field(
        default=10_000_000, ge=1_000,
        description="Neutrons in simulation. Lower for faster turnaround during commissioning or tests.",
    )
```

The unitless ints share a single sub-model rather than each getting its own
wrapper, and need no `get()` method — consumers access `.pulse_stride` /
`.num_simulated_neutrons` directly.

`LookupTableParams` composes them:

```python
class LookupTableParams(BaseModel):
    pulse_period: PulsePeriod = Field(default_factory=PulsePeriod)
    distance_resolution: DistanceResolution = Field(default_factory=DistanceResolution)
    time_resolution: TimeResolution = Field(default_factory=TimeResolution)
    Ltotal_range: LtotalRange = Field(default_factory=LtotalRange)
    simulation: Simulation = Field(default_factory=Simulation)
```

Per-instrument overrides for `Ltotal_range` defaults are expected
(LtotalRange varies with detector layout). Quality knobs
(`distance_resolution`, `time_resolution`, `num_simulated_neutrons`) start
relaxed during commissioning and tighten once the wiring is proven, so they
need to be UI-editable.

`pulse_period` is fixed in principle (ESS source frequency), but commissioning
may need to override it; kept UI-editable. `pulse_stride` could in principle be
derived from chopper info, but for v1 the operator sets it by hand; also kept
UI-editable.

### Chopper data

Chopper data is a **separate input** (not part of `LookupTableParams`) and is
assembled at runtime from two sources, combined into the `DiskChoppers` value
the upstream simulation expects:

1. **Geometry from the NeXus geometry file.** Slit positions, distances,
   radii, and other static chopper attributes. Use the upstream helpers
   documented at
   <https://scipp.github.io/scippneutron/user-guide/chopper/processing-nexus-choppers.html#NeXus-chopper-data>;
   do not reinvent. The geometry file is consumed via the existing pooch
   registry that other reduction workflows already use — no new runtime
   loading mechanism. The change is to ensure
   `src/ess/livedata/scripts/make_geometry_nexus.py` emits sufficient chopper
   info into the registry artifact (today it likely does not).
2. **Synthetic per-chopper setpoint streams**, produced in-process by a
   **chopper signal synthesizer** that wraps the existing `MessageSource`
   (decorator pattern, same protocol — downstream sees a regular message
   source). The synthesizer is internally stateful and consumes raw chopper
   f144 messages from the `${instrument}_choppers` Kafka topic (which carries
   multiple chopper PVs distinguished by `source_name`; tdct edge timestamps
   may also appear and are ignored for v1). No new flatbuffer schema is
   required — `rotation_speed_setpoint` and `phase` NXlogs are f144, already
   supported. Per message type:

   - `<chopper>_rotation_speed_setpoint` (clean upstream f144) flows through
     unchanged. Each becomes a **per-chopper aux source** for the workflow,
     cached via the existing `is_context=True` accumulator mechanism and
     replayed on job start.
   - `<chopper>_phase` (noisy f144 readback) is consumed by a per-chopper
     plateau detector built on top of
     `scippneutron.chopper.filtering.find_plateaus` +
     `collapse_plateaus`. When a stable region is identified, the synthesizer
     emits a synthetic `<chopper>_phase_setpoint` f144 message — the
     **per-chopper aux** the workflow needs (cached via the same mechanism).
     Raw phase samples are not forwarded.
   - When every chopper has both `rotation_speed_setpoint` and a detected
     `phase_setpoint` available and stable, the synthesizer emits a synthetic
     **primary tick** on logical source `'choppers'` (`setpoints_reached`).
     During instability the synthesizer simply does not emit; the workflow
     does not refire and the most recent published table stays as the last
     known-good output. The primary tick is what drives JobManager to
     (re-)fire the workflow.
   - For **chopperless instruments** (zero choppers configured), the
     synthesizer emits a vacuous `setpoints_reached` at startup — no
     plateau detection, no waiting. Combined with the latched-primary
     caching below, this makes chopperless a degenerate case of the normal
     flow with no special-case code path.
   - Other raw messages (`rotation_speed` readback, tdct, etc.) feed the
     synthesizer as state inputs only or are dropped — they are not workflow
     inputs.

**Why a `MessageSource` wrapper, not an adapter or a preprocessor.**
`MessageAdapter` is for stateless byte→domain transforms; the synthesizer is
stateful and N-input→M-output. `MessagePreprocessor` / `Accumulator` is bound
1:1 input-`StreamId`→output-`StreamId` and cannot cleanly express
cross-stream aggregation (`setpoints_reached` is global) or
emit-other-`StreamId` (synthetic `phase_setpoint` differs from the raw
`phase` input). Wrapping `MessageSource` preserves every existing
abstraction's semantics. Future migration to a producer-side service that
publishes the synthetic streams to Kafka is a near-trivial relocation: the
service drops the wrapper and uses a plain source.

`DiskChoppers` assembly happens **inside the workflow** (a sciline provider
that combines static NeXus chopper geometry with the cached aux setpoints),
mirroring the detector pattern. Specifics resolved by TODO 2 in "Provider".

Workflow execution is **user-triggered**: the operator sets params and
starts the job manually (no always-on / auto-start). Once running, the
synthesizer's primary `setpoints_reached` ticks drive re-execution. Both
the aux setpoints AND the primary `'choppers'` tick are cached via the
existing `is_context=True` accumulator mechanism and replayed on job-start
— `MessagePreprocessor.get_context()` already covers primary streams as
well as aux (see `orchestrating_processor.py:240-247`; the timeseries
service relies on this). If the synthesizer has emitted a
`setpoints_reached` before the job started — including the chopperless
case where it emits a vacuous tick at startup — the workflow fires
immediately on job-start from the cached value. If the operator changes
params and wants a new table, they stop and restart the job.

**Plateau-detection thresholds and the "stable enough" criteria** are
hardcoded inside the synthesizer for v1 — not exposed as UI params. Whether
they need to become user-tunable is something we revisit only if reality
forces it.

**Observability.** The synthesizer logs structured lines at each state
transition: `subscribed to N PVs`, `first message from <chopper>`,
`<chopper> phase locked`, `all locked, emitting setpoints_reached`,
`<chopper> phase lost`. The "operator clicked start, output topic empty"
state is then diagnosable from `journalctl` alone.

End-to-end validation runs on the CODA staging environment.

TODO:
- `park_angle` exists in NXdisk_chopper but its use is unclear; out of scope
  for v1 unless a concrete need surfaces.

### Output

`LookupTableArray = NewType('LookupTableArray', sc.DataArray)` — the `array`
field of the upstream `LookupTable`. Published via the existing sink path as a
single da00 message (~1–2 MB typical, well under the 100 MB Kafka limit).

Output source key: **`'tof_lookup_table'`** — distinct from the ingress
logical source name (`'choppers'`) to avoid namespace overlap when consumers
or future readers grep for either.
Per-instrument scoping is inherited from the service instance. A/B comparison
of two param sets would need distinct keys; YAGNI for v1.

The four scalar input params should be attached to the output array as 0-D
coords (`pulse_period`, `pulse_stride`, `distance_resolution`,
`time_resolution`). This makes the published message self-describing, so a
consumer can reconstruct a full `LookupTable` from the message alone without
out-of-band coordination on parameter values. (This is the same legacy on-disk
shape that `ess.reduce.unwrap.workflow.load_lookup_table` already consumes;
not a separate format invention.)

### Provider

Wraps `ess.reduce.unwrap.lut.LookupTableWorkflow`. The workflow takes the
params + choppers, computes the table, and returns the array with the four
scalar params attached as coords. A couple of additional providers may need
to be added to this base workflow, e.g., for turning the `LookupTable`
output into the `LookupTableArray` described above.

TODO:
- workflow needs to insert chopper stream info into static info obtained from NeXus file. Mirror the approach taken for detectors and detector events fed via stream. Small difference: We need to deal with multiple choppers (combined from multiple groups in NeXus).
- look into GenericNeXusWorkflow, it can apparently load choppers, but this is currently not connected to the LookupTableWorkflow. Something like `choppers = workflow.compute(RawChoppers[SampleRun])`, then figure out how to translate to the chopper object the LUT workflow expects.

## Service placement

The lookup-table workflow lives in a **new `tof_table` service** from v1, not
in `data_reduction`. An earlier draft considered a stepping-stone in
`data_reduction`, but the `MessageSource`-wrapper-based synthesis architecture
(see "Chopper data" above) makes that stepping-stone unattractive: installing
the wrapper in `data_reduction` would make every consumer of that service's
source chopper-aware, hold plateau-detection state at the source level, and
force the service to subscribe to `${instrument}_choppers` purely to feed one
workflow. Detector, monitor, and reduction workflows do not need any of that.
A dedicated service keeps the wrapper contained and gives a clean future
migration path for moving synthesis into its own producer-side service (the
wrapper is replaced by a plain Kafka source with no other code change).

**Architectural fit.** Unification of N chopper PVs into a single primary
source happens at the **`MessageSource`-wrapper layer** (the synthesizer
described under "Chopper data" above), not at the adapter or routing layer.
The workflow declares `'choppers'` as its primary and receives synthetic
ticks; per-chopper aux setpoints (`<chopper>_rotation_speed_setpoint`,
synthetic `<chopper>_phase_setpoint`) are normal physical f144 streams routed
through the existing aux-stream cache mechanism. JobManager stays 1:1
`(source, workflow)`, the orchestrator stays 1:1, and there is no
Bifrost-style many-physical-to-one-logical rewrite at the adapter layer for
chopper messages.

**Per-chopper identity** is preserved trivially: per-chopper aux streams
each have their own `StreamId`, and `DiskChoppers` (an `sc.DataGroup` keyed by
chopper name, matching upstream `LookupTable.choppers`) is assembled inside
the workflow from those aux entries plus static NeXus geometry — nothing
flattened across choppers, no bespoke per-message state in routing.

`config/route_derivation.py::resolve_stream_names` declares the per-instrument
**input** chopper-PV streams that the synthesizer subscribes to; the
workflow's primary `'choppers'` is itself synthetic and not mapped to physical
streams. The Bifrost compatibility block is unaffected by this plan.

### Chopperless instruments

Chopperless instruments require **no special handling** under the synthesizer
architecture. The synthesizer for chopperless emits a vacuous
`setpoints_reached` at startup (zero choppers ⇒ all in phase trivially); the
existing context-accumulator cache retains it. When the operator starts a
job, the cached tick is replayed via `MessagePreprocessor.get_context()` (see
`orchestrating_processor.py:240-247`, which already covers primary streams as
well as aux — the timeseries service uses this same pattern). The workflow
fires once and publishes. Stop/restart-to-recompute also works without
changes: the cached tick is still present, so the new job fires immediately
on activation.

This collapses what would otherwise be a chopperless-specific cluster of
changes (a `Job.is_one_shot` flag, inline `schedule_job` finalize, action
signature changes to `list[Message]`, `ConfigProcessor` aggregation) into one
configuration choice in the workflow factory: register the `'choppers'`
primary stream with a context accumulator (e.g., `LatestValueHandler`)
instead of a non-context one. No changes to `JobManager`,
`JobManagerAdapter`, or `ConfigProcessor`.

If the operator changes params and wants a new table, they stop and restart
the job — a new job, a new fire (cached tick replays on activation). No
"reset → re-finalize" on the same job instance for v1.

## Files expected to change

- `src/ess/livedata/parameter_models.py` — new scalar/range models for the params.
- `src/ess/livedata/handlers/lookup_table_workflow.py` — workflow factory + provider. Includes a sciline provider that assembles `DiskChoppers` from static NeXus chopper geometry + cached aux setpoints (specifics resolved by TODO 2 in "Provider"). Registers `'choppers'` primary stream with a context accumulator (`LatestValueHandler`) so the synthesizer's emit is cached and replayed on job-start via the existing `MessagePreprocessor.get_context()` path — this is what lets chopperless instruments use the normal flow with no special handling.
- `src/ess/livedata/handlers/lookup_table_workflow_specs.py` — `LookupTableParams` and `LookupTableArray` type.
- `src/ess/livedata/kafka/chopper_synthesizer.py` (new) — `ChopperSynthesizer`, a stateful wrapper around `MessageSource` (decorator pattern, same protocol). Consumes raw chopper f144 messages from `${instrument}_choppers`, runs per-chopper plateau detection on phase NXlogs, emits synthetic per-chopper `<chopper>_phase_setpoint` aux messages and synthetic primary `setpoints_reached` ticks on logical `'choppers'`. Emits a vacuous `setpoints_reached` at startup for chopperless instruments. Pass-through for `<chopper>_rotation_speed_setpoint`.
- `src/ess/livedata/kafka/message_adapter.py` — extend f144 routing to cover the chopper topic if not already; no `stream.name` rewrite, no new flatbuffer.
- `src/ess/livedata/config/route_derivation.py` — declare per-instrument chopper-PV streams (the synthesizer's input PVs); when the resolved set is empty, the synthesizer runs in chopperless mode (vacuous emit at startup, no PV subscription). No Bifrost-style logical→physical generalization needed for the workflow's *primary* (`'choppers'` is synthetic).
- `src/ess/livedata/services/tof_table.py` (new) — dedicated service hosting the lookup-table workflow. Wraps its `MessageSource` with `ChopperSynthesizer` for chopper-equipped instruments; uses a plain source for chopperless.
- `src/ess/livedata/config/instruments/<instrument>/factories.py` — register workflow per instrument; supply per-instrument `Ltotal_range` defaults; declare chopper PV streams (or omit for chopperless).
- `setup-kafka-topics.sh` — add the choppers topic for local dev.
- `tests/handlers/lookup_table_workflow_test.py`, `tests/kafka/chopper_synthesizer_test.py`, and tof_table service integration tests.

## Tests

- Unit: provider returns a `sc.DataArray` whose dims are `(distance, event_time_offset)` and whose values match `ess.reduce.unwrap.lut.LookupTableWorkflow` output.
- Unit: the four input params appear as 0-D coords on the output array, with correct dtypes and units.
- Unit: chopperless input (`DiskChoppers = {}`) produces a table without raising; values are the chopperless degenerate.
- Round-trip: `scipp_to_da00(output)` followed by `da00_to_scipp` preserves values, variances, dims, and the four scalar coords (regression check).
- Integration (chopperless instrument): instantiate the synthesizer for an instrument with zero chopper PVs configured; verify a vacuous `setpoints_reached` is emitted at startup and cached. Start a job; verify the workflow fires once via context replay and the published result matches the precomputed `loki-wavelength-lookup-table-no-choppers.h5` reference within numerical tolerance.
- Integration (chopper-equipped, reference fixture): for one chopper-equipped instrument, commit a small precomputed reference table generated offline from a known minimal chopper config; the integration test feeds the same chopper config and asserts the array agrees with the fixture within numerical tolerance. Catches regressions in the chopper synthesizer, the workflow's `DiskChoppers` assembly, and upstream `LookupTableWorkflow` integration before staging.
- Integration (chopper-equipped, gating behavior): feed raw chopper f144 messages into the wrapped `MessageSource`; verify the synthesizer emits `setpoints_reached` only after every chopper has a stable `phase_setpoint` and a `rotation_speed_setpoint` cached, and that the workflow re-runs accordingly.
- Chopper synthesizer unit: feeding a sequence of small phase fluctuations does *not* emit `setpoints_reached`; a step change followed by a new plateau does. `<chopper>_rotation_speed_setpoint` messages pass through unchanged. Synthetic per-chopper `<chopper>_phase_setpoint` messages carry the correct `StreamId`, schema, and stable value. tdct messages on the topic are ignored.
- Latched-primary unit: registering `'choppers'` with a context accumulator (e.g., `LatestValueHandler`) causes the synthesizer's emit to be cached and replayed via `MessagePreprocessor.get_context()` on job activation; the workflow fires once on the replayed value. No `is_one_shot` flag, no inline `schedule_job` finalize, no action-signature changes required.
- `resolve_stream_names` unit: per-instrument chopper-PV set resolves correctly (empty for chopperless instruments, populated for chopper-equipped); existing detector/monitor logical-name expansion is unchanged.

## Open questions

(None blocking implementation. Remaining details — per-instrument
`Ltotal_range` defaults, plateau-detection thresholds, and the static-chopper
geometry source resolved by TODO 2 — are answerable at impl time without
changing this plan's overall shape.)

