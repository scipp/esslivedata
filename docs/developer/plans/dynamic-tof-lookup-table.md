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

1. **`DiskChoppers` provider.** For chopperless instruments, returns
   `DataGroup({})`. For chopper-equipped instruments, takes per-chopper
   aux setpoints (`<chopper>_rotation_speed_setpoint`, synthetic
   `<chopper>_delay_setpoint`) plus the static `RawChoppers[SampleRun]`
   loaded from the geometry artifact via `GenericNeXusWorkflow`, and
   calls `scippneutron.chopper.DiskChopper.from_nexus()` per chopper.
   The geometry artifact carries `rotation_speed_setpoint` and `delay`
   as length-0 NXlog placeholders for the streamed quantities; the
   provider overrides them with the cached scalar setpoints before
   calling `from_nexus`. `from_nexus` (scippneutron 26.4) prefers
   `rotation_speed_setpoint` over `rotation_speed` and derives `phase`
   from `delay` and the rotation speed (`phase = 2π·f·delay`) when no
   `phase` field is present — production NeXus files have none.
   `beam_position` is also injected as a constant 0 deg, matching the
   upstream essreduce convention (its tests, docs, and fakes all use 0):
   slit_edges are stored as already measured from the beam-crossing
   point, so the offset is zero.
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
   `delay` NXlogs are f144, already supported. (Plateau detection runs
   on `delay`, not `phase`: production NXdisk_chopper carries no `phase`
   field; `from_nexus` derives phase from `delay` downstream.)

### Synthesizer

Per message type:

- `<chopper>_rotation_speed_setpoint` (clean upstream f144) flows through
  unchanged. Each becomes a per-chopper aux source for the workflow,
  cached via the existing `is_context=True` accumulator mechanism and
  replayed on job start.
- `<chopper>_delay` (noisy f144 readback) feeds a per-chopper rolling-
  window stability detector. When stable, the synthesizer emits a
  synthetic `<chopper>_delay_setpoint` f144 message — the per-chopper
  aux the workflow needs (cached via the same mechanism). Raw delay
  samples are forwarded as a tap (see deviation note below).
- When every chopper has both `rotation_speed_setpoint` and a stable
  `delay_setpoint` available, the synthesizer emits a synthetic primary
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
emit-other-`StreamId` (synthetic `delay_setpoint` differs from the raw
`delay` input). Wrapping `MessageSource` preserves every existing
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
`<chopper> delay locked`, `all locked, emitting chopper_cascade`,
`<chopper> delay lost`. The "operator clicked start, output topic empty"
state is then diagnosable from `journalctl` alone.

End-to-end validation runs on the CODA staging environment.

## Files

In tree:

- `src/ess/livedata/handlers/wavelength_lut_workflow.py` — single
  chopper-count-independent factory `create_wavelength_lut_workflow`. The
  internal `_WavelengthLutWorkflow` (a thin `StreamProcessorWorkflow`
  subclass) loads `RawChoppers[SampleRun]` and the NXsource position
  once at construction via `GenericNeXusWorkflow`, caches per-chopper
  aux setpoint scalars, and assembles a `DiskChoppers` `DataGroup`
  outside sciline by overriding the empty NXlog placeholders with the
  cached scalars and calling `DiskChopper.from_nexus()` per chopper. A
  closure provider exposes the cached `DiskChoppers` to the pipeline
  keyed off the trigger.
- `src/ess/livedata/handlers/wavelength_lut_workflow_specs.py` — pydantic
  param models, `WavelengthLutParams`, `WavelengthLutOutputs`, and
  `register_wavelength_lut_workflow_spec` (auto-derives `aux_sources` from
  `Instrument.choppers`).
- `src/ess/livedata/handlers/stream_processor_workflow.py` — generic
  adapter from `ess.reduce.streaming.StreamProcessor` to the livedata
  `Workflow` protocol.
- `src/ess/livedata/handlers/timeseries_handler.py` —
  `_SYNTHETIC_LOG_ATTRS` table for the synthetic `chopper_cascade` source.
- `src/ess/livedata/kafka/chopper_synthesizer.py` — synthesizer with
  rolling-window stability detector and per-chopper plateau locking.
- `src/ess/livedata/services/timeseries.py` — wires the synthesizer via
  `outer_source_wrapper`, reading `Instrument.choppers` for chopper names.
- `src/ess/livedata/service_factory.py` — generic `outer_source_wrapper`
  hook on `DataServiceBuilder`.
- `src/ess/livedata/core/orchestrating_processor.py` — empty-batch path
  runs context enrichment + `process_jobs`, allowing cached-context job
  activation without ongoing data flow.
- `src/ess/livedata/config/instrument.py` — `Instrument.choppers: list[str]`
  field; single source of truth consumed by the spec, the service, and the
  factory.
- `src/ess/livedata/config/instruments/{loki,dummy}/specs.py` and
  `factories.py` — register the spec and attach the factory. LOKI's
  `factories.py` passes its geometry artifact path to the factory;
  the factory loads `RawChoppers[SampleRun]` and the NXsource position
  via `GenericNeXusWorkflow`.

Remaining (real chopper PVs / pooch):

- Geometry artifacts: regenerate from a recent `coda_loki_*.hdf` /
  `coda_<instrument>_*.hdf` source via `make_geometry_nexus.py` and
  re-publish to pooch with bumped consuming hashes for each chopper-
  equipped instrument.
- `src/ess/livedata/kafka/message_adapter.py` — extend f144 routing to
  cover the chopper topic if not already; no `stream.name` rewrite, no new
  flatbuffer.
- `setup-kafka-topics.sh` — add the choppers topic for local dev.
- `config/instruments/<instrument>/{specs,streams,factories}.py` for
  chopper-equipped instruments — declare `Instrument.choppers`, chopper PV
  stream aliases, and `LtotalRange` defaults via a `WavelengthLutParams`
  subclass.

## Tests

In tree:

- `tests/handlers/wavelength_lut_workflow_test.py` — output dims/units,
  provenance coords; multi-chopper assembly with locked setpoints; partial
  chopper data falls back to the empty cascade; missing geometry raises;
  da00 round-trip preserves dims/coords.
- `tests/handlers/stream_processor_workflow_test.py` — generic wrapper.
- `tests/kafka/chopper_synthesizer_test.py` — synthesizer unit tests
  including plateau detection, change re-emit gating, and chopperless
  mode.
- `tests/services/wavelength_lut_test.py` — service-level integration on
  a chopperless instrument: a single fire via context replay; result
  matches the precomputed `loki-wavelength-lookup-table-no-choppers.h5`
  reference within numerical tolerance.

Remaining test work:

- Round-trip: `scipp_to_da00(output)` followed by `da00_to_scipp`
  preserves values, variances, dims, and the four scalar coords. (Partial
  coverage in tree; expand to variances.)
- Integration (reference fixture): for one chopper-equipped instrument,
  commit a small precomputed reference table generated offline from a
  known minimal chopper config; the test feeds the same chopper config
  and asserts the array agrees within numerical tolerance. Catches
  regressions in the synthesizer, the `DiskChoppers` assembly, and
  upstream `LookupTableWorkflow` integration before staging.
- Integration (gating behavior): feed raw chopper f144 messages into the
  wrapped `MessageSource`; verify the synthesizer emits `chopper_cascade`
  only after every chopper has a stable `delay_setpoint` and a
  `rotation_speed_setpoint` cached, and that the workflow re-runs
  accordingly.
- `DiskChoppers` assembly unit: given a `RawChoppers[SampleRun]`
  DataGroup (realistic NXdisk_chopper fields per chopper) plus
  per-chopper scalar setpoints, the workflow produces a `DiskChopper`
  per entry with the cached aux values threaded into
  `rotation_speed_setpoint`/`delay`. (Covered in tree by
  `tests/handlers/wavelength_lut_workflow_test.py::TestMultiChopperWorkflow`.)

## Status

Shipped:

- `ChopperSynthesizer` with rolling-window stability detector
  (mean/std-based lock, exact-equality change detection on
  `_rotation_speed_setpoint`, noise-floor compare on locked delay mean).
  Plateau-detects on `<chopper>_delay`, emits
  `<chopper>_delay_setpoint` (production NXdisk_chopper carries no
  `phase` field; `from_nexus` derives phase from delay downstream).
  Chopperless mode preserved.
- Single chopper-count-independent factory `create_wavelength_lut_workflow`
  parameterised by a chopper-name list and a NeXus geometry filename.
  Empty list ⇒ chopperless (no NeXus needed). Per-chopper aux setpoints
  are cached, merged with the static `RawChoppers[SampleRun]` from
  `GenericNeXusWorkflow` (`rotation_speed_setpoint` and `delay` empty
  NXlogs overridden, `beam_position` injected as 0 deg per upstream
  convention), and assembled into `DiskChoppers` via
  `DiskChopper.from_nexus()` *outside* sciline. A closure provider
  exposes the cached `DiskChoppers` to the pipeline keyed off the
  trigger. The sciline graph shape does not depend on chopper count.
- `Instrument.choppers: list[str]` is the single source of truth: the
  spec derives `aux_sources` from it, the timeseries service passes it
  to `ChopperSynthesizer`, and the factory uses it to drive provider
  assembly.
- LOKI declares `choppers=['bw_chopper1', 'bw_chopper2', 'fo_chopper1',
  'fo_chopper2']` (matching the four NXdisk_chopper groups in production
  NeXus files); the factory loads the published geometry artifact via
  `GenericNeXusWorkflow`. `HardcodedChopperGeometry` is gone.
- `make_geometry_nexus.py` correctly handles NXdisk_chopper groups
  (PR #916): copies static fields verbatim and trims NXlogs to length-0
  placeholders. A regenerated LOKI artifact (from a recent `coda_loki_*.hdf`
  source) carries `slit_edges`, `radius`, `slits`, and transformations
  per chopper; needs to be published to pooch with a bumped consuming
  hash before the new factory works in production.
- `log_producer_widget` extended with optional `noise_stddev` /
  `publish_rate_hz` per slider, and publishes initial values on creation
  so a default-position slider still emits one message.

Deviation from this plan: raw `<chopper>_delay` is forwarded (tap, not
filter) so the noisy readback is plottable in dev mode. Decide before
productionising whether to revert to drop.

## Remaining work

### NeXus chopper geometry — pooch publish

- `scripts/make_geometry_nexus.py` extended to copy `NXdisk_chopper`
  groups, trimming NXlogs to length-0 placeholders. -> PR #916 (merged).
- New provider taking `RawChoppers[SampleRun]` from `GenericNeXusWorkflow`
  plus per-chopper aux setpoints, calling `DiskChopper.from_nexus()` per
  chopper. Source position from NeXus, `beam_position=0` injected. -> Done.
- Regenerate per-instrument geometry artifacts (e.g. from a recent
  `coda_<instrument>_*.hdf` source), publish to pooch, bump consuming
  hashes. **Outstanding** — LOKI artifact `geometry-loki-2026-05-08.nxs`
  generated locally from `coda_loki_999999_00026352.hdf`; not yet
  uploaded.

### Per-instrument UI param defaults

Per-instrument-fixed defaults for user-facing params (`LtotalRange` —
5–30 m suits LOKI; DREAM/BIFROST will differ — and possibly
`pulse.frequency` for commissioning) need to surface as initial values in
the UI form. Resolution: instrument passes a subclass of
`WavelengthLutParams` with overridden field defaults to
`register_wavelength_lut_workflow_spec` (which already accepts
`params: type[WavelengthLutParams]`). The other previously-listed values
turned out to belong elsewhere: chopper geometry is now a factory-time
argument at the instrument's call site (gone when NeXus lands), and
plateau-detection thresholds remain on the synthesizer.

### Real chopper PVs (off the dev/log-producer path)

- Chopper topic (e.g. `${instrument}_choppers`) added in
  `setup-kafka-topics.sh` and the f144 routing in `message_adapter.py`.
- Per-instrument chopper-PV stream declarations alongside
  `f144_log_streams`. The chopper-name list lives on `Instrument.choppers`
  already; `streams.py` adds the PV→logical-name mapping per chopper.

### Per-instrument adoption

LOKI is wired today. DREAM / BIFROST / ODIN / NMX each need:

- `Instrument.choppers` populated with their chopper names,
- chopper PV stream declarations,
- `LtotalRange` defaults for the instrument's detector layout (via the
  param-subclass mechanism above),
- a regenerated geometry artifact carrying `NXdisk_chopper` groups
  published to pooch (same recipe as LOKI: run
  `make_geometry_nexus.py` on a recent `coda_<instrument>_*.hdf`).

### Tests

- Reference fixture: precomputed table for one chopper-equipped instrument,
  integration test that feeds chopper f144s and asserts agreement.
- Round-trip `scipp_to_da00` ↔ `da00_to_scipp` preserves values + the four
  scalar coords.
- Service-level integration: gating behaviour (no cascade until both
  setpoints), refire on plateau change.

### Producer/consumer end meeting (synthesizer's exit)

Once an upstream service publishes `chopper_cascade_reached` directly, drop
`ChopperSynthesizer`, the `outer_source_wrapper` plumbing, and
`_SYNTHETIC_LOG_ATTRS`. The workflow becomes a plain f144 consumer.

Then deprecate the static `loki_lookup_table_no_choppers` path: LOKI's
monitor / I(Q) workflows currently load via `LookupTableFilename`; switch
them to consume the dynamically-published `wavelength_lut` da00 message.

## Backpressure and rate limiting

The synthesizer can in principle emit cascades faster than a Monte-Carlo
recompute can finish (~seconds at 1M neutrons). The current design has
two implicit dampers and one open verification gap:

1. **Plateau detector debounce.** A window of stable delay samples is
   required before a new lock; rapid operator nudges produce no
   intermediate cascades. Once locked, only a *new* mean (drift > `atol`)
   re-fires.
2. **One compute per orchestrator cycle, on latest values.** The cascade
   stream is f144 → `ToNXlog` (`is_context=True`); the provider reads
   `[time, -1].data` from the chopper logs. N cascades arriving in one
   cycle's batch collapse into one compute on the latest setpoints. The
   gating in `JobManager.process_jobs` (`_jobs_with_primary_data`) ensures
   no spurious refires when no new primary data arrives — the
   `test_subsequent_steps_do_not_republish` test already covers this for
   the chopperless case.
3. **Open: behaviour under sustained backpressure.** If recompute is
   slower than the trigger rate, Kafka messages pile up in the consumer
   queue (`consumer_lag` grows in metrics) and each cycle processes a
   larger batch — still one compute per cycle, on the latest values, so
   the queue eventually drains. Not yet load-tested. Worth adding a
   stress test that fires cascades faster than the compute can finish and
   asserts (a) no compute backlog (output count ≤ cycle count, not
   trigger count), and (b) each emitted LUT reflects the latest setpoints
   at compute start (not stale ones from when the trigger arrived).

## Open questions

- `park_angle` exists in NXdisk_chopper but its use is unclear; out of
  scope unless a concrete need surfaces.
- Per-instrument `LtotalRange` defaults and plateau-detection thresholds
  are answerable at impl time without changing this plan's overall shape.
- Should the synthesizer drop or forward raw `<chopper>_delay`? Plan
  originally said drop; prototype forwards. Decision driven by whether
  operators want the noisy readback plotted.
