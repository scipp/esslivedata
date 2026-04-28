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
  JobManager extension *is* needed to fire `finalize` once for jobs with zero
  input streams — see the Chopperless instruments section — but that
  generalizes beyond this workflow rather than being a chopperless-specific
  code path.)

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
2. **Live rotation/phase from the `${instrument}_choppers` Kafka topic.**
   This route does not exist in ESSlivedata yet; it needs:
   - A new `MessageAdapter` and stream-mapping entry routing the topic to the
     workflow.
   - Possibly a new flatbuffer schema (likely `tdct`, TBC) — depends on which
     fields the upstream system publishes. Concrete schema and fields are
     determined at impl time by inspecting existing `coda_*.hdf` files in the
     project root, cross-checked against the upstream chopper class needs.
   - An update to the local-dev `setup-kafka-topics.sh` script.

End-to-end validation runs on the CODA staging environment.

A small **chopper preprocessor** owns the combination: it ingests live
rotation/phase samples, holds the static NeXus geometry, and emits a
`DiskChoppers` value when chopper state has changed *substantially*.

Workflow execution is **user-triggered**: the operator sets params and starts
the job manually (no always-on / auto-start). Once the job is running, the
preprocessor's debouncing controls *re-execution*; it does not control whether
the job exists.

On job start, the workflow waits for the preprocessor's first stable emit —
the output topic stays empty until choppers lock. This is the simplest path
and is consistent with "no recompute unless inputs are stable."

The "substantial change" gating thresholds are **hardcoded inside the
preprocessor** for v1 — not exposed as UI params. The preprocessor compares
against setpoint when available (so subscribing to *both* readback and
setpoint is the working assumption; readback drives gating, setpoint provides
the reference). Whether this stays contained in the preprocessor or grows
into something user-tunable is something we revisit only if reality forces it.

**Observability:** the preprocessor logs structured lines at each state
transition — `subscribed to N PVs`, `first message from <chopper>`,
`<chopper> locked`, `all locked, emitting`. The "operator clicked start,
output topic empty" state is then diagnosable from `journalctl` alone, with
no behavior change.

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

## Service placement

Two viable options:

1. **Register the workflow in the existing `data_reduction` service.**
   Reuses the `ReductionHandlerFactory`, sciline pipeline machinery, sink path,
   and orchestrator. No new docker container, no new lifecycle to manage.
2. **New `tof_table` service.** Conceptually cleaner separation (the workflow
   has different cardinality from per-source reduction jobs — one table per
   instrument config, not one result per detector source).

Either way, the workflow itself is the same; service choice is purely about
process boundaries.

**Production trajectory:** option 2 is expected to be the production shape.
v1 in `data_reduction` is a stepping-stone; we accept that recompute blocks
other workflows in the shared service because *nothing else can produce
useful output without a fresh table after a param change anyway*. The
preprocessor's job is to ensure recomputes are truly necessary — that, not
async/threading, is what makes blocking tolerable.

**Architectural fit:** resolved by reusing the **Bifrost many-physical-to-one-logical
pattern at the routing layer**. A new `MessageAdapter` for the chopper
flatbuffer rewrites `stream.name` to a single logical source name
(`'choppers'`) on ingress, just as `Ev44ToDetectorEventsAdapter(merge_detectors=True)`
does for Bifrost's 45 detector banks. The workflow spec lists `'choppers'` as
its **primary** source (not aux); preprocessor gating drives re-execution as
designed.
`config/route_derivation.py::resolve_stream_names` expands the logical name
to the physical chopper PVs for Kafka subscription.

This means JobManager stays 1:1 `(source, workflow)`, the orchestrator stays
1:1, and routing N→1 happens entirely in the adapter. Option 1
(`data_reduction`) is therefore tractable for v1 without abstraction changes
beyond a small generalization of `resolve_stream_names` — see below.

**Important distinction from the Bifrost case:** the adapter rewrites the
*routing key* per message but does **not** combine messages from different
choppers into a flattened log. For ev44, Bifrost's accumulators flatten
events because events from any detector bank are interchangeable
(per-event `event_id` is unique). For choppers, **per-chopper identity must
be preserved**: each chopper has its own rotation/phase time series, and
flattening would destroy that. The chopper preprocessor is therefore *not*
an ordinary accumulator — it keeps a `dict[chopper_name, ChopperState]`
internally, updating the relevant slot per incoming message (chopper
identity comes from the payload, e.g., `tdct.source_name` field). Its
emitted `DiskChoppers` is an `sc.DataGroup` keyed by chopper name (matching
the upstream `LookupTable.choppers: sc.DataGroup | None` shape) — structure
preserved, nothing flattened.

`resolve_stream_names` currently has a Bifrost-specific compatibility block
("safe because Bifrost is the only instrument with this many-physical-to-one-logical
pattern", `route_derivation.py:65`). Adding choppers makes that statement
false. Either add a `chopper` category alongside `detector`/`monitor`, or
replace the special-case with a generic "logical → physical" mapping. Modest
cleanup, not architectural change.

### Chopperless instruments

Instruments where no chopper PVs are configured in the stream mapping have
**zero input streams** for this workflow. JobManager has no message-driven
trigger and would never call `finalize`. Resolution: extend JobManager so
that a workflow with **zero input streams** has its `finalize` called once at
job creation, and the job exits immediately afterwards (it can never be
triggered again). The workflow then computes a chopperless table
(`DiskChoppers = {}`) and publishes it once.

This generalizes naturally beyond this workflow — any future zero-input
"computed once from params" workflow gets the same treatment. We deliberately
do **not** extend this to "zero primary, some aux" workflows: defining when
such a workflow first runs (first message on every aux? first on any?) is
trickier than we need.

If the operator changes params and wants a new table, they stop and restart
the job — a new job, a new fire. No "reset → re-finalize" on the same
job instance for v1.

## Files expected to change

- `src/ess/livedata/parameter_models.py` — new scalar/range models for the params.
- `src/ess/livedata/handlers/lookup_table_workflow.py` — workflow factory + provider.
- `src/ess/livedata/handlers/lookup_table_workflow_specs.py` — `LookupTableParams` and `LookupTableArray` type.
- `src/ess/livedata/handlers/chopper_preprocessor.py` — combines static NeXus chopper geometry with the live `${instrument}_choppers` stream into a gated `DiskChoppers` output.
- `src/ess/livedata/kafka/message_adapter.py` (and stream mapping) — route for `${instrument}_choppers` topic; new adapter that rewrites `stream.name` to logical `'choppers'` (mirrors `Ev44ToDetectorEventsAdapter(merge_detectors=True)`); possibly a new flatbuffer schema (`tdct`?).
- `src/ess/livedata/config/route_derivation.py` — generalize `resolve_stream_names` so the logical→physical expansion isn't Bifrost-only; add a `chopper` category (or replace the special-case with a generic mechanism).
- `src/ess/livedata/scripts/make_geometry_nexus.py` — emit sufficient chopper info into the registry artifact (today it likely does not). When this changes, the regenerated artifact must be republished to the pooch registry and consuming hashes bumped.
- `src/ess/livedata/config/instruments/<instrument>/factories.py` — register workflow per instrument; supply per-instrument `Ltotal_range` defaults.
- `src/ess/livedata/services/data_reduction.py` — register the new workflow factory (if option 1).
- `src/ess/livedata/core/job_manager.py` (or wherever the job lifecycle lives) — extension: jobs whose workflow declares **zero input streams** get `finalize` called once at creation, and exit. Preserves existing behavior for jobs with one or more input streams.
- `setup-kafka-topics.sh` — add the choppers topic for local dev.
- `tests/handlers/lookup_table_workflow_test.py` and chopper-preprocessor tests.

## Tests

- Unit: provider returns a `sc.DataArray` whose dims are `(distance, event_time_offset)` and whose values match `ess.reduce.unwrap.lut.LookupTableWorkflow` output.
- Unit: the four input params appear as 0-D coords on the output array, with correct dtypes and units.
- Unit: chopperless input (`DiskChoppers = {}`) produces a table without raising; values are the chopperless degenerate.
- Round-trip: `scipp_to_da00(output)` followed by `da00_to_scipp` preserves values, variances, dims, and the four scalar coords (regression check).
- Integration (chopperless instrument): instantiate workflow on an instrument with zero chopper PVs configured; verify JobManager fires `finalize` once on creation, output matches the precomputed `loki-wavelength-lookup-table-no-choppers.h5` reference within numerical tolerance, job exits.
- Integration (chopper-equipped, reference fixture): for one chopper-equipped instrument, commit a small precomputed reference table generated offline from a known minimal chopper config; the integration test feeds the same chopper config and asserts the array agrees with the fixture within numerical tolerance. Catches regressions in the chopper preprocessor, the adapter rewrite path, and upstream `LookupTableWorkflow` integration before staging.
- Integration (chopper-equipped, gating behavior): feed chopper messages through the adapter + preprocessor; verify the preprocessor gates as expected and the workflow re-runs only on substantial changes.
- Chopper preprocessor unit: feeding a sequence of small rotation/phase fluctuations does *not* trigger a downstream emit; a step change does. NeXus geometry is preserved across emits.
- JobManager unit: a job whose workflow declares zero input streams has `finalize` called exactly once at creation and then exits; jobs with ≥1 input stream behave as today.
- Message-adapter unit: the new chopper adapter rewrites `stream.name` to `'choppers'` regardless of the physical chopper PV, while preserving the chopper identity in the payload. Mirrors the existing patterns in `tests/kafka/message_adapter_test.py`.
- `resolve_stream_names` unit: logical `'choppers'` expands to the configured set of physical chopper PV stream names; existing detector/monitor logical-name expansion is unchanged.

## Open questions

(None blocking implementation. Remaining details — exact flatbuffer schema,
per-instrument `Ltotal_range` defaults, exact gating thresholds — are
answerable at impl time without changing this plan.)

