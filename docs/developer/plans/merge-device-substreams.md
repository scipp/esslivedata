# Merge VAL/DMOV/RBV substreams into a single device stream

## Problem

Motorised devices on ESS instruments publish independent f144 streams with
non-synchronised timestamps:

- **RBV** — readback value (actual measurement)
- **VAL** — target value (setpoint)
- **DMOV** — "done moving" idle flag (boolean)

Today these are exposed as three independent streams. The system has no
notion of a "device" as a single entity, so:

- The dashboard's timeseries view shows three disconnected plots per
  motor — users see a value trace but cannot tell whether the device was
  settled when each reading was taken.
- There is no way for any workflow (display or science) to *drop* or
  flag data taken while a setpoint had not yet been reached.

## Goal

Present each device as one logical stream and one DataArray, carrying
the readback as data with target/settled state attached as coords. Both
timeseries plotting and any downstream science workflows consume the
merged view via the same mechanism.

## v0 scope

All services that consume motion streams wrap with `DeviceSynthesizer`:
`timeseries`, `data_reduction`, `monitor_data`, `detector_data`. The
suppression/synthesis is uniform across the system; no service sees raw
VAL/DMOV/RBV for configured devices.

**Validating consumers:**

- **Timeseries service** — plots one trace per *device* (instead of
  per PV), with the merged DataArray carrying target/settled as coords.
  Display-layer mask injection (see [Mask vs coord](#mask-vs-coord))
  lets the plot grey out in-motion samples.
- **Bifrost data_reduction** — the only instrument with active
  `LogContextBinding`s today (two, both pointing at motor RBV
  substreams). Both are migrated to `Device`-stream targets in v0,
  exercising the full science-workflow path end-to-end and validating
  the API shape. `InstrumentAngle` and `SampleAngle` are already
  `sciline.Scope[..., sc.DataArray]`, so the binding migration is two
  string changes — the extra `target`/`settled` coords ride along
  passively through the existing `sc.lookup` consumption pattern.

Other instruments have no `LogContextBinding`s today; their services
gain the wrapping (passthrough until they configure devices) but no
binding migration is needed.

## Decomposition

The merge is two distinct operations:

1. **Cross-stream coupling** — combine three independent input streams
   into a coherent per-device view. Stateful per device.
2. **Accumulation into a DataArray** — grow a multi-coord DataArray over
   time. Matches the existing `LogData → ToNXlog` pattern.

The architectural choice is purely *where operation 1 lives*. Operation
2 is a standard preprocessor parameterised over a new payload type.

## Where the merge happens

**A `MessageSource` decorator (DeviceSynthesizer), matching the
`ChopperSynthesizer` precedent.**

Each consuming service wraps its adapted message source with a
DeviceSynthesizer. The synthesizer maintains per-device state, emits
synthetic `DeviceSample` messages on a new `StreamKind.DEVICE` stream
per device, and suppresses the underlying VAL/DMOV/RBV messages from
forwarding (scoped to configured devices — see
[Suppression](#suppression)).

### Layer comparison

| Layer | Device is first-class? | State location | Notes |
|---|---|---|---|
| **MessageSource decorator (chosen)** | Real stream | Per-service (one instance) | Matches ChopperSynthesizer; same merged stream for every consumer |
| Per-stream preprocessor | n/a | n/a | Abstraction violation — preprocessors are single-`StreamId` by construction |
| Sciline provider per workflow | Graph artifact | Per workflow per device | Sciline-only; timeseries doesn't use Sciline; per-workflow state replication |
| Dedicated aggregator service → new Kafka topic | Real (observable in Kafka) | Single source of truth | Heavy: new deployment + schema versioning per instrument |

The synthesizer wins because the cross-stream coupling belongs at the
layer where streams are first-class. Anywhere else, the device is
either a graph artifact (Sciline) or a multiplexed preprocessor
(abstraction violation). One synthesizer instance per service holds the
state once; downstream code routes a single `DeviceSample` stream per
device.

## Representation

### `DeviceSample` payload

```python
@dataclass(frozen=True, slots=True)
class DeviceSample:
    time: Timestamp
    value: float                 # RBV
    target: float | None         # VAL — None if device has no setpoint substream
    settled: bool | None         # DMOV — None if device has no idle substream
```

One `DeviceSample` per device event. The accumulator grows it into a
DataArray. Type-safe dispatch via `StreamKind.DEVICE`.

`time` is `Timestamp` (not raw `int`) because `DeviceSample` is an
in-process domain object — there is no upstream schema to mirror.
`LogData.time` is `int` only to decouple from the `logdata_f144`
upstream schema. The closer precedent here is `Message.timestamp`,
which is already `Timestamp`.

### DeviceSample timestamp policy

`DeviceSample.time = max(rbv_time, val_time, dmov_time)` across the
synthesizer's last-seen times for that device's configured substreams.
Each row semantically represents "device state as known up to time T".
Monotonic by construction; out-of-order f144 messages across
independent substreams fold into state without producing dropped
emits.

Alternative ("use the triggering input's time") would drop any emit
whose input arrives with an older `data.time` than the last seen,
losing both the row and the state update. Z2-max-timestamp updates
state always and emits monotonically.

Known edge case: a substream message whose `data.time` is older than
the current max (e.g. a delayed DMOV transition predating the most
recent RBV) updates state correctly but produces an emit at the
unchanged max-time, which `ToDeviceLog` drops as duplicate. The new
substream value becomes visible at the next emit that advances max.
Acceptable — state is never lost, only the addressability of that one
transition as its own row.

### Republish handling

Upstream forwarders republish `(time, value)` pairs every ~10s even
when unchanged. The synthesizer does **not** dedup these itself; it
processes every input and emits naively. `ToDeviceLog` inherits
`ToNXlog`'s contract — duplicate-`data.time` and out-of-order
timestamps are silently dropped at `add()`. Verified at
`to_nxlog.py:78-94`. Synthesizer-level dedup would be a CPU
micro-optimisation, not a correctness requirement; defer until
profiling motivates it.

### Merged DataArray (accumulator output)

Single `sc.DataArray`:

- **`time` dim coord**: monotonic; each row's time is the
  `DeviceSample.time` carried by the synthesizer (= max of substreams'
  last-seen times at emit).
- **Data**: RBV value at that row's time. Forward-filling is implicit
  — the synthesizer already carries the last-known RBV in each
  DeviceSample, so the accumulator just appends.
- **`target` coord** (optional): VAL at that row's time. Same
  forward-fill mechanism (synthesizer-side, not accumulator-side).
- **`settled` coord** (optional, bool): DMOV at that row's time. Same.

`ToDeviceLog` is structurally a `ToNXlog` variant: pre-allocate, grow
in chunks, append rows, dedup duplicate/out-of-order timestamps. No
`sc.lookup` resampling involved — every row already carries the full
triple from the synthesizer.

Why union-anchored is the row policy: see [Anchoring](#anchoring).

### Mask vs coord

`settled` is deliberately a **coord, not a mask**. A mask would
silently exclude moving samples from `sum`/`mean`/etc., biasing
reductions — but in-motion readings are not invalid, they measure a
different thing. Whether to filter by `settled` is a workflow-level
analysis choice, not a data-quality verdict.

The canonical merged DataArray stays mask-free. Dashboard rendering
(motion-region highlighting, target overlay, partial-device handling)
is a display-layer concern, deferred to wherever the timeseries plot
handler chooses to do it.

## Anchoring

**Union-anchored** (synthesizer emits on every input VAL/DMOV/RBV
event):

- RBV cadence dominates (continuous during motion, ~10s–100s Hz; idle
  when settled).
- VAL ≈ one event per setpoint change.
- DMOV ≈ two events per motion.

Union adds at most a handful of messages per motion on top of the RBV
stream. Trivial cost in exchange for transition timestamps becoming
addressable rows ("when did motion start?", "when did the device
settle?") without going back to the suppressed raw streams. Downsampling
to RBV-only is trivial for any consumer that wants it; recovering
transition rows after the fact is not.

## Bootstrap

**Delayed start.** The synthesizer suppresses device emits until *every
configured substream* has been observed at least once. For a device
with all three substreams (RBV+VAL+DMOV), all three must arrive; for an
estia-style device with no DMOV, only RBV+VAL.

Worst-case latency is bounded by the upstream forwarder's republish
cadence — currently ~10s repeats of `(time, value)` even when
unchanged. So a fresh consumer with `auto.offset.reset=latest` sees
all three substreams within ~10s of subscribe. The republish
guarantee is assumed to apply to VAL and DMOV as it does to RBV; if a
specific PV turns out not to republish, the mitigation paths
(`auto.offset.reset=earliest` for motion topics, or eager emission
with `Optional` fields) remain available as targeted fixes.

## Run lifecycle

The synthesizer **persists state unconditionally** across `RunStart`
and `RunStop`. Run boundaries are file-writing markers, not physical
events — a motor's position survives the boundary. No reset hook in
the synthesizer; no run-lifecycle plumbing. `ToDeviceLog` mirrors
`ToNXlog`'s behaviour and does not clear on run transitions either.

## Suppression

The synthesizer **suppresses** the raw VAL/DMOV/RBV messages from
forwarding, scoped to streams that are part of a configured `Device`.
Non-device f144 streams pass through unchanged.

Consequence: per-PV timeseries plots for VAL/DMOV/RBV of configured
devices disappear from the dashboard. The information is not lost — the
merged DataArray carries everything as coords, and the plot handler can
render target/settled as overlay traces if needed.

Rationale: per-PV plots and per-device plots showing the same
underlying data confuse users ("which one is real?"). One canonical
view per device.

## Stream-registry shape

### Units propagation

`Device.units` is populated by detection from the RBV substream's
`F144Stream.units`. Detection cross-checks `val_substream.units ==
rbv_substream.units` and raises on mismatch — a motor with setpoint in
mm and readback in degrees is a real upstream data bug, and silent
tolerance would produce meaningless DataArrays. DMOV's unit is
ignored; the `settled` coord is always dimensionless bool.

`ToDeviceLog` allocates the data variable and the `target` coord with
`Device.units`. The `settled` coord and the `time` coord use their
respective conventional units (dimensionless bool, ns since epoch).

### `Device(Stream)` subclass in `Instrument.streams`

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Device(Stream):
    value: str                       # required: streams-dict name of the RBV stream
    target: str | None = None        # optional: VAL stream name
    settled: str | None = None       # optional: DMOV stream name
    units: str | None = None
    writer_module: str = 'device'    # marker
    nx_class: str = 'NXpositioner'   # parent group's NeXus class
    # nexus_path inherited — points to parent group
    # topic/source inherited as None (synthesised in-process)
```

`F144Stream` entries for the substreams stay in `Instrument.streams` —
they drive Kafka subscription (the synthesizer's *input*). `Device`
entries sit alongside as `topic=None` records.

Existing routing layers filter cleanly:
- `instrument.f144_streams` uses `isinstance(s, F144Stream)` — `Device` filtered out.
- `StreamLUT` construction filters on `topic is not None` — `Device` filtered out.
- New: `instrument.devices` returns `{name: s for name, s in self.streams.items() if isinstance(s, Device)}` — used only by the synthesizer.

### New `StreamKind.DEVICE`

The synthesizer emits `Message(stream=StreamId(kind=DEVICE, name=device_name), value=DeviceSample(...))`.

Preprocessor factories dispatch on `StreamKind` via `match` (existing
pattern). New arm:

```python
case StreamKind.DEVICE:
    return ToDeviceLog(...)
```

`ToDeviceLog` is a new accumulator parameterised over device metadata
(units, optional-field presence). It consumes `DeviceSample` and grows
the merged DataArray described above.

### Pointer fields vs convention-based resolution

The audit (see `.scratch/device-detection-inconsistencies.md`) shows
the NeXus child-name convention is not universal:

- estia: 14 devices have no `idle_flag` (RBV+VAL only).
- tbl: 8 devices use `position_readback`/`position_setpoint` instead
  of `value`/`target_value`.
- loki: `NXPositioner` casing typo on 6 parents.
- bifrost/loki: lone-readback edge cases with non-`.RBV` source suffixes.

Pure convention-based runtime resolution (synthesizer scans `streams`
for siblings of expected names) breaks tbl entirely. Storing explicit
substream-name pointers on the `Device` record keeps the runtime
resolution rule trivial and uniform; all alt-naming complexity lives in
the *detection* logic that populates those pointers.

## Auto-detection in `name_streams`

```python
streams = name_streams(PARSED_STREAMS, rename={...})
```

Detection runs unconditionally after rename resolution, before the
result dict is returned. Produces `Device` entries with auto-populated
pointer fields. No opt-out keyword in v0 — no concrete need has
surfaced, and the structural detection rule is conservative enough
that false positives are improbable. If a need arises (overrides for
non-standard structures, or a global off-switch), add the knob then
with a real use case driving its shape.

**Device name selection:** `suggest_names` runs once on the union of
parent-group paths (for device entries) and substream paths (for
F144Stream entries). The shared pass guarantees uniqueness across
devices + substreams in one shot — no collision possible by
construction. User-supplied `rename` entries that collide with an
auto-detected device name raise a clear error.

Detection rule — **structural pattern, not class-based**:

1. For each parent group, examine children. Identify by NeXus name
   with alt-name fallbacks:
   - RBV signal: `value` || `position_readback`
   - VAL signal: `target_value` || `position_setpoint`
   - DMOV signal: `idle_flag`
2. **Emit a `Device` only if RBV is present *and* at least one of VAL
   or DMOV is present.** A lone RBV is just a stream — nothing to
   merge, no need to wrap. Whichever of VAL/DMOV is absent → the
   corresponding pointer field is `None`.
3. Cross-check EPICS source suffix (`.RBV`/`.VAL`/`.DMOV` and the
   bifrost `PosReadback` variant) as a secondary validation. Mismatch
   logs a warning but does not block detection.

`parent_nx_class` is deliberately **not** a detection criterion. The
structural pattern is the device-ness signal; restricting to
`NXpositioner` would miss conforming devices on `NXcollimator` (loki, 2
cases) and `NXlog` (estia, 1 case) parents, while helping with no edge
cases. False positives from structural pattern alone are improbable —
`value`+`target_value`+`idle_flag` together is a strong f144 motor
convention — and the cost of a false positive (a Device that downstream
ignores) is much lower than a false negative (a real device the
consumer has no automatic way to wire up).

Rationale for the `name_streams` seam (vs codegen):

- **Separation of concerns**: `streams_parsed.py` is purely "what's in
  the NeXus file" — path-keyed, auto-generated. Device synthesis is a
  wiring decision and should not be baked into the NeXus mirror.
- **Naming coherence**: `name_streams` already owns the path→name
  transformation. The device name and its substream pointers must come
  from the same naming pass to avoid drift.
- **No regeneration churn**: improving detection benefits every existing
  `streams_parsed.py` at the next import.
- **Hand-written specs benefit**: any caller of `name_streams` gets
  device detection automatically.

## Related cleanup (not blocking)

`streams_parsed.py` codegen currently omits `parent_nx_class` even
though `F144Stream` supports the field. Detection no longer needs it
(see [Auto-detection](#auto-detection-in-name_streams)), so this is
decoupled from the device-synthesizer work. Worth landing for general
debugging/inspection value; can go before, alongside, or after.

1. Extend `nexus_helpers.py` codegen to emit `parent_nx_class` on every
   `F144Stream`.
2. Regenerate all per-instrument `streams_parsed.py` files.

## Implementation sketch

In order of how much each touch point changes:

1. **New** `Device(Stream)` dataclass in `config/stream.py`. Fields per
   the [Device record](#devicestream-subclass-in-instrumentstreams)
   section.
2. **New** `DeviceSample` dataclass in `handlers/accumulators.py` (or
   adjacent to `LogData`).
3. **New** `StreamKind.DEVICE` in `core/message.py`.
4. **`name_streams`** (`config/stream.py:78`): implement detection;
   runs unconditionally after rename resolution. No new keyword.
5. **New** `ToDeviceLog` accumulator in `handlers/`, parameterised over
   device metadata. Output: multi-coord DataArray.
6. **New** `DeviceSynthesizer` in `kafka/device_synthesizer.py`, modeled
   on `chopper_synthesizer.py`. Decorator wrapping the adapted
   `MessageSource[Message]`. Holds per-device state; emits
   `DeviceSample` on union-anchored trigger; suppresses configured
   substreams from forwarding.
7. **Service wiring**: wrap all four motion-stream-consuming services
   (`timeseries`, `data_reduction`, `monitor_data`, `detector_data`)
   with `DeviceSynthesizer` via `outer_source_wrapper=` in the
   respective `service_factory` setup. Timeseries already wraps with
   `ChopperSynthesizer`; compose as
   `lambda src: DeviceSynthesizer(ChopperSynthesizer(src), devices=...)`.
   The two wrappers don't interact (chopper tick isn't a device
   substream; devices aren't a chopper concern), so order is stylistic.
8. **Preprocessor factory dispatch**: add
   `case StreamKind.DEVICE: return ToDeviceLog(...)` in each handler
   factory (`DetectorHandlerFactory`, `MonitorHandlerFactory`,
   `DataReductionHandler`, `TimeseriesHandlerFactory`).
9. **`LogContextBinding` validation** (`config/instrument.py:128`):
   accept either `F144Stream` or `Device` records as binding targets.
   No new merge-provider machinery needed — the device stream produces
   a DataArray downstream of the accumulator, identical in shape to the
   context-injection contract today.
10. **Bifrost binding migration** (`bifrost/factories.py:73,78`): change
    the two `stream_name=` arguments from RBV substream names to the
    detected Device names. Verify end-to-end that the bifrost reduction
    workflow tolerates the extra `target`/`settled` coords on the
    incoming `InstrumentAngle`/`SampleAngle` DataArrays.

## Test strategy

Three layers, iterate as implementation surfaces issues:

1. **`DeviceSynthesizer` unit tests** — fake wrapped source, scripted
   LogData sequences. Cover bootstrap gate, Z2 timestamp computation,
   selective suppression, out-of-order substream handling.
2. **`ToDeviceLog` unit tests** — scripted DeviceSamples; assert
   DataArray shape, growth, dedup. Parallel to existing `ToNXlog` tests.
3. **Bifrost end-to-end** — real synthesizer + real `ToDeviceLog` + real
   Sciline pipeline; only the message source is a fake feeding f144
   sequences. Validates detection output, binding migration, and that
   the reduction workflow tolerates the extra coords on
   `InstrumentAngle`/`SampleAngle`.

Specific scenarios worth explicit tests: bootstrap (no emit on RBV-only),
bootstrap completion timing, Z2 (b) edge case, republish dedup, partial
device (missing `target` or `settled` coord on the DataArray).

## Open questions

- **Verification risk for bifrost migration**: extra `target`/`settled`
  coords on `InstrumentAngle`/`SampleAngle` DataArrays. Expected
  benign — `sc.lookup` and broadcasting tolerate inert non-dim coords.
  If something breaks, a small Sciline shim provider that extracts
  `da.data` recovers the pre-merge shape. Verify by running the
  bifrost reduction workflow as part of v0.
- **Unify with `ChopperSynthesizer` once v1 chopper handling lands.**
  V1 chopper (plateau detection on phase NXlogs + per-chopper
  `phase_setpoint` synthesis, per the existing comment in
  `chopper_synthesizer.py`) has the same structural shape as device
  merging: multiple f144 substreams in, synthesized event out, stateful
  cross-stream coupling. Worth a refactor pass after v1 chopper is
  designed; premature now because v0 chopper is just a one-shot tick
  and unifying would contort one of them.
- **Value-only `NXsensor`-style groups** (e.g. loki's `PT100_*`
  temperature readings): cleanly ignored by the rule "RBV + (VAL or
  DMOV)". If a future need surfaces for treating these as
  setpoint-less devices, the rule can be relaxed.
- **Device-level metadata beyond units** (description, axis label,
  setpoint range): out of scope; revisit when a consumer needs it.
- **`DeviceSample.value` typing**: currently `float`. Some f144 streams
  carry int values; consider `int | float` if a real case surfaces.
