# Dynamic wavelength lookup-table workflow

Issue: #894

## Goal

Replace the static "select a precomputed lookup table from a dropdown" model
with a workflow that **computes** a wavelength lookup table from chopper
settings, so that arbitrary commissioning configurations are supported
without shipping a precomputed file for each one.

## Non-goals

- **Downstream consumption.** How the published table reaches detector,
  monitor, or reduction workflows (sciline binding, context-key plumbing,
  cross-service consistency) is out of scope.
- **Replacing the existing precomputed-table path immediately.** This
  workflow lives alongside today's filename-based loading; deprecate once
  the producer/consumer ends meet.

## Architecture

### Service hosting

The workflow is hosted by the **`timeseries`** service. It is conceptually
just an f144-driven workflow; if the producer published the chopper-cascade
trigger directly it would be a normal `timeseries` sibling. The
`ChopperSynthesizer` (below) is scaffolding for that upstream-side gap, not
a reason for a separate service.

The synthetic `chopper_cascade` source is special-cased in
`LogdataHandlerFactory` via a small `_SYNTHETIC_LOG_ATTRS` table —
registering it in `f144_attribute_registry` would auto-create an unwanted
timeseries workflow for it. The synthetic value is a boolean "at setpoint"
tick and carries no unit (distinct from `'dimensionless'`).

The synthesizer attaches via `DataServiceBuilder.outer_source_wrapper`, a
generic hook usable by any service that needs to inject synthetic messages
or filter raw streams at the domain level.

### Workflow shape

The workflow is a `StreamProcessorWorkflow` (the generic adapter from
`ess.reduce.streaming.StreamProcessor` to the livedata `Workflow` protocol)
wrapping `ess.reduce.unwrap.lut.LookupTableWorkflow()`. Composition adds:

1. **`DiskChoppers` provider.** For chopperless instruments,
   `_empty_choppers` returns `DataGroup({})`. For chopper-equipped
   instruments, the provider takes per-chopper aux setpoints
   (`<chopper>_rotation_speed_setpoint`, synthetic
   `<chopper>_phase_setpoint`) plus the static `RawChoppers[SampleRun]`
   from `GenericNeXusWorkflow`, and calls
   `scippneutron.chopper.DiskChopper.from_nexus()` per chopper.
   `from_nexus` requires a scalar `rotation_speed_setpoint` which
   `RawChoppers` does **not** contain (NeXus only has the time-dependent
   NXlog) — supplied from the cached aux. The static `delay` (from NeXus)
   and the synthetic `phase_setpoint` (from plateau detection) thread
   through orthogonally: `delay` is a calibration constant,
   `phase_setpoint` is the operator-set angle.
2. **Provenance provider.** Attaches the four scalar input parameters
   (`pulse_period`, `pulse_stride`, `distance_resolution`,
   `time_resolution`) as 0-D coords on the output array, pulling from the
   user-facing params (the upstream pipeline may unit-convert internally).
3. **Direct sciline parameter wiring** for `PulsePeriod`, `PulseStride`,
   `DistanceResolution`, `TimeResolution`, `LtotalRange`,
   `NumberOfSimulatedNeutrons`, `SourcePosition`.

`SourcePosition` is set to a placeholder vector for chopperless
instruments — the upstream simulator only uses it inside the per-chopper
loop. For chopper-equipped instruments it comes from `GenericNeXusWorkflow`
(loaded from the pooch geometry artifact) alongside `RawChoppers[SampleRun]`.

### Trigger plumbing

The synthetic `chopper_cascade` stream is exposed to sciline as a **dynamic
key** (`ChopperCascadeTrigger`) consumed by the `DiskChoppers` provider.
`allow_bypass=True` lets it flow directly to the provider rather than
through an accumulator — its value is ignored, only its presence drives a
refire.

The `OrchestratingProcessor` runs context enrichment and `process_jobs` on
every cycle, including empty batches: `peek_pending_streams` predicts which
jobs will activate, and `get_context` fills missing streams from cache.
This decouples "data is flowing" from "scheduled jobs can activate" — the
chopperless case relies on it because the synthesizer's cached one-shot
tick must drive the job without ongoing data flow. The mechanism
generalizes beyond this workflow.

The spec is registered with `reset_on_run_transition=False` so the table
persists across run starts/stops, matching `timeseries` semantics.

## Inputs

`sc.Variable` is not a valid field type for UI-generated forms. The
convention (see `parameter_models.py`: `RangeModel`, `EdgesModel`, etc.)
decomposes scalars into `value` + `unit` with a `get()` method.

Param models live in `wavelength_lut_workflow_specs.py` (single caller,
and avoids name clash with identically-named scipp keys in
`ess.reduce.unwrap.lut`):

```python
class Pulse(BaseModel):
    frequency: float = 14.0  # Hz, hardcoded — no realistic kHz/MHz case
    stride: int = 1
    def get_period(self) -> sc.Variable: ...

class DistanceResolution(BaseModel):
    value: float = 0.1
    unit: LengthUnit = LengthUnit.METER

class TimeResolution(BaseModel):
    value: float = 250.0
    unit: TimeUnit = TimeUnit.MICROSECOND

class LtotalRange(RangeModel):
    start: float = 5.0
    stop: float = 30.0
    unit: LengthUnit = LengthUnit.METER

class Simulation(BaseModel):
    num_simulated_neutrons: int = 1_000_000

class WavelengthLutParams(BaseModel):
    pulse: Pulse
    distance_range: LtotalRange
    distance_resolution: DistanceResolution
    time_resolution: TimeResolution
    simulation: Simulation
```

Operators enter source frequency in Hz rather than a high-precision period
in milliseconds. `pulse.stride` lives on `Pulse` since it is a property of
the source, not a simulation knob. Provenance coord names on the output
(`pulse_period`, `pulse_stride`) preserve the upstream-expected shape.

Per-instrument overrides for `LtotalRange` defaults are expected (varies
with detector layout). Quality knobs (`distance_resolution`,
`time_resolution`, `num_simulated_neutrons`) are UI-editable and start
relaxed during commissioning. `pulse.frequency` is fixed in principle but
commissioning may need overrides; kept editable. `pulse.stride` could in
principle be derived from chopper info; for now the operator sets it by
hand.

## Output

`WavelengthLut = NewType('WavelengthLut', sc.DataArray)` — the `array`
field of the upstream `LookupTable`. Published via the existing sink path
as a single da00 message (~1–2 MB typical, well under the 100 MB Kafka
limit).

Output source key: **`'wavelength_lut'`** — matches the spec name.
Per-instrument scoping is inherited from the service instance. A/B
comparison of two param sets would need distinct keys; YAGNI.

The four scalar input params are attached as 0-D coords on the output. This
makes the published message self-describing: a consumer can reconstruct a
full `LookupTable` from the message alone, without out-of-band coordination
on parameter values. Same on-disk shape that
`ess.reduce.unwrap.workflow.load_lookup_table` already consumes.

## Chopper data

Chopper data feeds the `DiskChoppers` provider above, assembled from two
sources:

1. **Static chopper geometry from the NeXus geometry artifact.** Slit
   positions/edges, axle position, radius, and `delay` per chopper. Loaded
   via `GenericNeXusWorkflow`'s `RawChoppers[SampleRun]` provider — the
   same path detector workflows use to load static geometry. The pooch
   geometry artifact must be regenerated to include `NXdisk_chopper`
   groups; `make_geometry_nexus.py`'s per-`nx_class` filter does not copy
   them today (~10 lines to add). Per-instrument artifacts must then be
   re-published with bumped consuming hashes.
2. **Synthetic per-chopper setpoint streams**, produced in-process by
   `ChopperSynthesizer` — a stateful `MessageSource` decorator wrapping the
   timeseries service's source. Consumes raw chopper f144 messages from
   `${instrument}_choppers` (multiple chopper PVs distinguished by
   `source_name`; tdct edge timestamps may also appear and are ignored).
   No new flatbuffer schema is required — `rotation_speed_setpoint` and
   `phase` NXlogs are f144, already supported.

### Synthesizer

Per message type:

- `<chopper>_rotation_speed_setpoint` (clean upstream f144) flows through
  unchanged. Each becomes a per-chopper aux source for the workflow,
  cached via the existing `is_context=True` accumulator mechanism and
  replayed on job start.
- `<chopper>_phase` (noisy f144 readback) feeds a per-chopper plateau
  detector built on `scippneutron.chopper.filtering.find_plateaus` +
  `collapse_plateaus`. When stable, the synthesizer emits a synthetic
  `<chopper>_phase_setpoint` f144 message — the per-chopper aux the
  workflow needs (cached via the same mechanism). Raw phase samples are
  not forwarded.
- When every chopper has both `rotation_speed_setpoint` and a stable
  `phase_setpoint` available, the synthesizer emits a synthetic primary
  tick on `chopper_cascade`. During instability the synthesizer simply
  does not emit; the workflow does not refire and the most recent
  published table stays as the last known-good output.
- For chopperless instruments (zero choppers configured), the synthesizer
  emits one vacuous `chopper_cascade` tick at startup and is otherwise a
  passthrough. The orchestrator's empty-batch context-enrichment path
  ensures the cached tick activates the job once on first start.
- Other raw messages (`rotation_speed` readback, tdct, etc.) feed
  synthesizer state only, or are dropped — they are not workflow inputs.

Plateau-detection thresholds and the "stable enough" criteria are
hardcoded inside the synthesizer; whether they need to become user-tunable
is revisited only if reality forces it.

**Why a `MessageSource` wrapper, not an adapter or a preprocessor.**
`MessageAdapter` is for stateless byte→domain transforms; the synthesizer
is stateful and N-input→M-output. `MessagePreprocessor` / `Accumulator` is
bound 1:1 input-`StreamId`→output-`StreamId` and cannot cleanly express
cross-stream aggregation (`chopper_cascade` is global) or
emit-other-`StreamId` (synthetic `phase_setpoint` differs from the raw
`phase` input). Wrapping `MessageSource` preserves every existing
abstraction's semantics. Future migration to a producer-side service that
publishes the synthetic streams to Kafka is a near-trivial relocation:
drop the wrapper and use a plain source.

### Execution model

User-triggered: the operator sets params and starts the job manually
(no always-on / auto-start). Once running, the synthesizer's primary
`chopper_cascade` ticks drive re-execution. If the operator changes params
and wants a new table, they stop and restart the job.

Chopperless instruments require no special handling: zero choppers ⇒ all
in phase trivially; the synthesizer's vacuous startup tick (cached, then
served via the orchestrator's empty-batch context-enrichment path) drives
one fire on job activation. Stop/restart-to-recompute also works without
changes.

**Per-chopper identity** is preserved trivially: per-chopper aux streams
each have their own `StreamId`, and `DiskChoppers` (an `sc.DataGroup`
keyed by chopper name, matching upstream `LookupTable.choppers`) is
assembled inside the workflow from those aux entries plus static NeXus
geometry — nothing flattened across choppers, no bespoke per-message state
in routing.

`config/route_derivation.py::resolve_stream_names` declares the
per-instrument input chopper-PV streams that the synthesizer subscribes
to; the workflow's primary `chopper_cascade` is itself synthetic and not
mapped to physical streams.

**Observability.** The synthesizer logs structured lines at each state
transition: `subscribed to N PVs`, `first message from <chopper>`,
`<chopper> phase locked`, `all locked, emitting chopper_cascade`,
`<chopper> phase lost`. The "operator clicked start, output topic empty"
state is then diagnosable from `journalctl` alone.

End-to-end validation runs on the CODA staging environment.

## Files

In tree:

- `src/ess/livedata/handlers/wavelength_lut_workflow.py` — workflow
  factory. Composes `LookupTableWorkflow()` with `_empty_choppers` and the
  provenance provider.
- `src/ess/livedata/handlers/wavelength_lut_workflow_specs.py` — pydantic
  param models, `WavelengthLutParams`, `WavelengthLutOutputs`, and
  `register_wavelength_lut_workflow_spec`.
- `src/ess/livedata/handlers/stream_processor_workflow.py` — generic
  adapter from `ess.reduce.streaming.StreamProcessor` to the livedata
  `Workflow` protocol.
- `src/ess/livedata/handlers/timeseries_handler.py` —
  `_SYNTHETIC_LOG_ATTRS` table for the synthetic `chopper_cascade` source.
- `src/ess/livedata/kafka/chopper_synthesizer.py` — synthesizer (currently
  the chopperless stub: vacuous startup tick + passthrough).
- `src/ess/livedata/services/timeseries.py` — wires the synthesizer via
  `outer_source_wrapper`.
- `src/ess/livedata/service_factory.py` — generic `outer_source_wrapper`
  hook on `DataServiceBuilder`.
- `src/ess/livedata/core/orchestrating_processor.py` — empty-batch path
  runs context enrichment + `process_jobs`, allowing cached-context job
  activation without ongoing data flow.
- `src/ess/livedata/config/instruments/{loki,dummy}/specs.py` and
  `factories.py` — register the spec and attach the chopperless factory.

Chopper-equipped extension:

- `wavelength_lut_workflow.py` — replace `_empty_choppers` with a provider
  taking per-chopper aux setpoints and `RawChoppers[SampleRun]` and
  calling `DiskChopper.from_nexus()` per chopper. Wire `Filename[SampleRun]`
  and `RawChoppers[SampleRun]` from `GenericNeXusWorkflow`; replace the
  placeholder `SourcePosition`.
- `chopper_synthesizer.py` — plateau detection, per-chopper
  `phase_setpoint` synthesis, conditional `chopper_cascade` emission once
  all choppers locked.
- `src/ess/livedata/scripts/make_geometry_nexus.py` — extend the
  per-`nx_class` filter to also copy `NXdisk_chopper` groups (~10 lines).
  Regenerate per-instrument geometry artifacts and re-publish to pooch
  with bumped consuming hashes.
- `src/ess/livedata/config/route_derivation.py` — declare per-instrument
  chopper-PV streams. Empty resolved set ⇒ chopperless mode (no PV
  subscription, vacuous startup tick).
- `src/ess/livedata/kafka/message_adapter.py` — extend f144 routing to
  cover the chopper topic if not already; no `stream.name` rewrite, no new
  flatbuffer.
- `setup-kafka-topics.sh` — add the choppers topic for local dev.
- `config/instruments/<instrument>/factories.py` for chopper-equipped
  instruments — supply per-instrument `LtotalRange` defaults and declare
  chopper PV streams.

## Tests

In tree:

- `tests/handlers/wavelength_lut_workflow_test.py` — provider returns a
  `sc.DataArray` whose dims are `(distance, event_time_offset)` and whose
  values match `LookupTableWorkflow` output; the four input params appear
  as 0-D coords with correct dtypes/units; chopperless input produces a
  table without raising.
- `tests/handlers/stream_processor_workflow_test.py` — generic wrapper.
- `tests/kafka/chopper_synthesizer_test.py` — vacuous startup tick,
  passthrough thereafter.
- `tests/services/wavelength_lut_test.py` — service-level integration on
  a chopperless instrument: a single fire via context replay; result
  matches the precomputed `loki-wavelength-lookup-table-no-choppers.h5`
  reference within numerical tolerance.

Chopper-equipped extension:

- Round-trip: `scipp_to_da00(output)` followed by `da00_to_scipp`
  preserves values, variances, dims, and the four scalar coords.
- Integration (reference fixture): for one chopper-equipped instrument,
  commit a small precomputed reference table generated offline from a
  known minimal chopper config; the test feeds the same chopper config
  and asserts the array agrees within numerical tolerance. Catches
  regressions in the synthesizer, the `DiskChoppers` assembly, and
  upstream `LookupTableWorkflow` integration before staging.
- Integration (gating behavior): feed raw chopper f144 messages into the
  wrapped `MessageSource`; verify the synthesizer emits `chopper_cascade`
  only after every chopper has a stable `phase_setpoint` and a
  `rotation_speed_setpoint` cached, and that the workflow re-runs
  accordingly.
- Synthesizer unit: small phase fluctuations do *not* emit
  `chopper_cascade`; a step change followed by a new plateau does.
  `<chopper>_rotation_speed_setpoint` messages pass through unchanged.
  Synthetic `<chopper>_phase_setpoint` messages carry the correct
  `StreamId`, schema, and stable value. tdct messages on the topic are
  ignored.
- `DiskChoppers` provider unit: given a `RawChoppers[SampleRun]`
  DataGroup (realistic NXdisk_chopper fields per chopper) plus
  per-chopper scalar `rotation_speed_setpoint` and `phase_setpoint`
  inputs, returns a `DiskChoppers` DataGroup keyed by chopper name with a
  fully-formed `DiskChopper` per entry. Verifies the cached aux setpoints
  thread through to the resulting `DiskChopper.frequency` / phase fields.
- `resolve_stream_names` unit: per-instrument chopper-PV set resolves
  correctly (empty for chopperless, populated for chopper-equipped);
  existing detector/monitor logical-name expansion is unchanged.

## Open questions

- `park_angle` exists in NXdisk_chopper but its use is unclear; out of
  scope unless a concrete need surfaces.
- Per-instrument `LtotalRange` defaults and plateau-detection thresholds
  are answerable at impl time without changing this plan's overall shape.
