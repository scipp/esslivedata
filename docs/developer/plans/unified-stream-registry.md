# Unified per-stream registry

## Goal

A single source of truth per instrument for *streaming data declarations*
— one record per group that the ESS filewriter would write into a NeXus
file. Today the same logical entity is described by four parallel
structures kept in sync by hand and convention. This proposal collapses
them into one list of typed records, with all current usages derived
from it, and positions the registry to be auto-populated from a NeXus
geometry artifact with code-side overrides.

The proposal is independent of any work-in-progress feature branch. It
is motivated by two ongoing efforts that both bump into the current
fragmentation:

- A *dynamic-transforms* effort that drives entries in a detector
  `depends_on` chain from live f144 streams. To wire one stream to a
  workflow it must add entries in three places (per-instrument log
  dict, attribute registry, binding list) plus a typed Sciline-key
  declaration.
- A *chopper-workflow* effort that drives per-chopper geometry
  assembly from live f144 setpoint streams. It duplicates the same
  routing layer and adds a synthesised-stream concept that the other
  three structures handle awkwardly.

Both efforts work today, but the cost of adopting a new stream is
larger than the actual change, and the parallel structures invite
drift (LOKI already has a manual one-line splice in `specs.py` that
exists only because units live in two places).

## Non-goals

- No changes to message-flow semantics: routing/scoping/wrapping all
  keep their current behaviour. This is a config-layer reshape.
- No new feature for stream subscription, ROI, run control, etc.
- Not an implementation plan; mechanical steps for the rollout sketch
  appear at the end but the focus is on the choices and their reasoning.

## Current state

Four parallel structures (three if we do not count `log_context_bindings`
added in `worktree-issue-922-loki-dynamic-transform`) all keyed off the same
logical entity (one filewriter row → one streaming group in a NeXus file):

| Structure | Holds | Lives in | Consumed by |
|---|---|---|---|
| `f144_log_streams` dict | `{stream_name → {source, topic, units}}` | per-instrument `specs.py` | derives the next two |
| `Instrument.f144_attribute_registry` | `{stream_name → {units, ...}}` | `Instrument` | `ToNXlog(attrs=...)`; gates accepted LOG streams; drives auto-registration of timeseries workflow specs |
| `Instrument.log_context_bindings` | `[LogContextBinding(stream_name, log_key, dependent_sources, units, [nxlog_path])]` | `Instrument` | aux-source routing into specs; transform-chain patching; `StreamProcessorWorkflow` context wrapping |
| `StreamLUT[logs]` | `{(topic, source_name) → stream_name}` | per-instrument `streams.py` | message_adapter / Kafka source |

Additionally:

- `Instrument.source_metadata: dict[str, SourceMetadata]` carries
  `title` / `description` keyed by *source* name — overloaded across
  detectors, monitors, and f144 streams. LOKI uses entries keyed by
  both detector names (`loki_detector_0`) and f144 stream names
  (`detector_carriage`).
- `nexus_helpers.extract_stream_info()` already parses
  `(group_path, topic, source, nx_class, parent_nx_class, writer_module, units)`
  per streaming group from any NeXus file, as `StreamInfo`. There is
  a CLI generator that emits today's hard-coded `f144_log_streams`
  dict from a file. The NeXus-parsing future has a working extractor;
  what's missing is the in-memory abstraction it should parse into.

## What's wrong with the current shape

1. **Adoption cost.** Wiring one new f144 stream to a workflow touches
   three lists plus a typed Sciline-key class definition. Each entry
   repeats `stream_name`. Mistakes are silent (a missing
   attribute-registry entry causes messages to be dropped with a
   warning, not an error).
2. **Drift surface.** `units` lives in two places (the per-instrument
   dict and either the attribute registry or the binding). Today the
   `__post_init__` of `Instrument` auto-derives one from the other,
   but the entry point of truth is the per-instrument dict, and that
   itself is hand-typed.
3. **No room for synthesised streams.** Streams that exist only
   in-process (a synthesiser plateau-detecting on a noisy upstream
   readback, e.g. `<chopper>_delay_setpoint`) don't fit the
   "filewriter row" mental model; today they live in a separate
   `_SYNTHETIC_LOG_ATTRS` dict in `timeseries_handler.py`.
4. **Hand-coding doesn't scale.** Some instruments will eventually
   have O(100) f144 streams (DREAM is the largest case). A
   hand-maintained dict per instrument is fine at ~10 streams; it
   will not be at 100.

(Title/description for f144 streams also live in `source_metadata`
separately from the registry. This proposal *does not* address that
split — see Choice 4 for why.)

## Scope

The registry is f144-first. `Stream` is designed as the base for all
writer modules (ev44, tdct, ...) so detectors and monitors can later
become `Stream` subclasses, but that migration is out of scope for
this proposal. Initial rollout touches only the f144 fragmentation.
Display lookups (`get_source_title` etc.) stay routed through
`source_metadata` unchanged.

## Target shape

### A. Canonical record per stream + separate consumer bindings

One record describes the NeXus-facing facts of a streaming group: how
it arrives over Kafka, where it sits in the NeXus structure, and (for
f144) its units. *Runtime overlay* that wires a stream into a Sciline
workflow (workflow key, dependent-source scoping) lives in a separate
`log_context_bindings` list, referencing streams by name. *Display info*
(title, description) stays in `source_metadata`, the existing
side-dict that already covers detectors and monitors. The registry is
the noun for data plumbing; bindings are the verbs that point at it;
source_metadata is the display map.

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Stream:
    """Any streaming group in NeXus (or synthesised in-process)."""

    # Identity / routing
    stream_name: str                     # internal handle, not user-facing
    nexus_path: str | None               # full HDF5 path; None for synthesised
    topic: str | None                    # group's `topic` attr; None for synthesised
    source: str | None                   # group's `source` attr; None for synthesised
    writer_module: str                   # 'f144' | 'ev44' | 'tdct' | ...

    # NeXus shape (cheap to carry; rarely consulted directly)
    nx_class: str = ''                   # 'NXlog', 'NXevent_data', ...
    parent_nx_class: str | None = None   # 'NXdetector', 'NXdisk_chopper', ...


@dataclass(frozen=True, slots=True, kw_only=True)
class F144Stream(Stream):
    """f144 NXlog stream — value/time payloads."""
    units: str | None = None
    writer_module: str = 'f144'          # fixed at the class level
```

`Instrument.streams: dict[StreamName, Stream]` replaces
`f144_log_streams` and `f144_attribute_registry`. The dict is keyed by
`stream_name`, which is what every non-kafka-boundary consumer uses as
the routing handle. Display fields (`title`, `description`) stay in
`source_metadata` (see Choice 4) — the stream record is purely data
plumbing.

All current registry usages become derived views:

- `ToNXlog`-attrs lookup → `instrument.streams[name]`, pull `units`
  (and any other NXlog attrs we eventually want) off an `F144Stream`.
- Auto-registration of timeseries specs → iterate
  `[s for s in instrument.streams.values() if isinstance(s, F144Stream)]`
  and read `stream_name`.
- `StreamLUT[logs]` → built once in `streams.py` from
  `{(s.topic, s.source): s.stream_name for s in instrument.streams.values() if s.topic is not None}`.

The runtime overlay — wiring f144 streams into Sciline workflows for
both the dynamic-transforms work and the chopper-workflow work — lives
in a single flat list of `LogContextBinding` records:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class LogContextBinding:
    """One f144 stream feeding a Sciline workflow key, scoped to dependents."""
    stream_name: str                     # references Instrument.streams[stream_name]
    workflow_key: type[ValueLog]         # Sciline key type the value gets bound to
    dependent_sources: frozenset[str]    # spec source_names that pick up this binding
```

`Instrument.log_context_bindings: list[LogContextBinding]`. Bindings are thin:
they no longer mirror `units` or `nexus_path` — those live on the
stream record they reference. A binding is purely "stream X feeds
Sciline key K, scoped to dependents D".

One binding type covers both known consumers:

- *Dynamic-transforms* binding example:
  `LogContextBinding(stream_name='detector_carriage', workflow_key=DetectorCarriagePosition, dependent_sources=frozenset({'loki_detector_0'}))`.
  The detector spec for `loki_detector_0` picks up the live readback
  as an aux input, keyed by `DetectorCarriagePosition`, and patches it
  into the transform chain.
- *Chopper-workflow* binding example:
  `LogContextBinding(stream_name='bw_chopper1_delay_setpoint', workflow_key=BwChopper1DelaySetpoint, dependent_sources=frozenset({'chopper_cascade_in_phase'}))`.
  The chopper workflow spec is triggered by the synthetic
  `chopper_cascade_in_phase` stream; every chopper f144 stream
  declares itself as an aux input to that trigger and feeds a typed
  Sciline key. A `make_chopper_bindings(choppers, fields, trigger=...)`
  helper generates the `len(choppers) * len(fields)` entries from the
  per-instrument chopper list.

The aux-source routing mechanism (`LogContextAuxSources`) iterates
`instrument.log_context_bindings` once and filters by `dependent_sources &
spec.source_names`. The dispatch between "feed a detector's transform
chain" vs. "feed a chopper workflow" is driven by *the workflow_key
type* and the Sciline graph downstream, not by separate binding kinds.

Why one list, not per-consumer lists:

- `dependent_sources` is a property of the *consumer-stream binding*,
  not of the stream itself. Different bindings for the same stream
  may have different scoping. Putting bindings on the stream record
  (the original §1 fat-record idea) wedges this into a single global
  answer.
- One mechanism, one list: the same `LogContextAuxSources` plumbing
  serves both consumers. Adding a third Sciline-key-driven consumer is
  just more `LogContextBinding` entries — no new field on `Instrument`, no
  new routing class.
- The registry stays writer-module-symmetric: ev44 and tdct don't
  inherently need any runtime-overlay fields, so they don't gain a
  subclass just to default them to None. If a future ev44 consumer
  needs its own binding shape (different from `type[ValueLog]` keys),
  that gets its own binding type and field — the unification asserted
  here is across f144 Sciline-key consumers, not across all writer
  modules.

**Chopper-consumer precondition.** Today's chopper handler maps
`<chopper>_delay_setpoint` → `raw_choppers[<chopper>]['delay']` by
parsing stream names. To participate in this unified binding model,
the chopper workflow must consume its f144 inputs as typed `ValueLog`
subclasses (one per chopper f144 stream), with the destination mapping
moving into the Sciline graph. This is a real refactor of the chopper
consumer, not just a config-shape change; the simplification of the
overall config is conditional on it.

### B. Codegen-time parse, not runtime parse

For an instrument with O(100) NXlogs, the registry is bootstrapped
*offline* from the NeXus file by a CLI generator (extending the
existing `nexus_helpers` generator). The generator emits a
checked-in Python module containing a `list[F144Stream]` literal:

```python
# config/instruments/dream/streams_parsed.py  -- auto-generated
PARSED_STREAMS = [
    F144Stream(
        stream_name='sample_temperature',
        nexus_path='/entry/instrument/sample/temperature',
        topic='dream_sample', source='SAMPLE-Tmp:RBV', units='K',
    ),
    ...
]
```

The per-instrument hand-edited module composes the final registry:

```python
# config/instruments/dream/streams.py  -- hand-edited
from .streams_parsed import PARSED_STREAMS

streams = build_streams(
    PARSED_STREAMS,
    overrides={
        '/entry/.../wrong_units_path': dict(units='K'),  # filewriter bug fix
    },
    synthetics=[
        F144Stream(stream_name='delay_setpoint', nexus_path=None,
                   topic=None, source=None, ...),
    ],
)
```

`build_streams` returns the `dict[StreamName, Stream]`. Overrides are
expected to be rare (filewriter unit bugs, hand-chosen `stream_name`
renames); display fields live in `source_metadata` and are set there
separately. Override keys are accepted as either `stream_name` (for
hand-named entries) or `nexus_path` (the stable NeXus identity, for
entries that haven't been renamed). Collisions on `stream_name` raise
at codegen *and* construction time.

For instruments with few streams (LOKI, BIFROST today), the registry
is a hand-written dict literal. The parser/generator is opt-in.

**Why codegen, not runtime parsing:** the dashboard imports the same
instrument config module the backend does and currently has no NeXus
file access. Runtime parsing would force the dashboard into a Pooch
download path. The CLI-generator approach preserves the
"no-file-access" invariant for any process that imports the instrument
config, and makes the parser output reviewable as code. A future move
to dynamic (parse-on-backend, ship-to-dashboard) is local: the seam
is "who populates the dict at startup", and `Stream` is a plain frozen
dataclass that round-trips through any serialisation layer.

A CI check that re-runs codegen against the committed geometry file
and fails on diff is cheap insurance against drift.

### C. Synthesised streams

A synthesised stream (no NeXus row, emitted in-process by a
synthesiser) is `Stream(topic=None, source=None, nexus_path=None,
...)`. The predicate "synthesised" is `stream.topic is None`; no
separate flag needed.

A `__post_init__` invariant enforces that `topic`, `source`, and
`nexus_path` are all None or all non-None.

Consumers must be oblivious to the distinction: workflow harnesses,
timeseries UI, and binding resolution all index into
`Instrument.streams` by name and read the same fields regardless of
origin. The only place that filters by `topic is None` is the
kafka-boundary `StreamLUT` builder, which has no entry to emit for a
stream that never arrives over Kafka. Authoring may keep parsed and
synthetic entries visually separate (as in §B's `build_streams` call),
but the runtime registry is a single flat dict.

## Choices and reasoning

### 1. NeXus facts on the record, runtime overlay and display split off

**Choice.** The record (`Stream` / `F144Stream`) holds only
NeXus-derivable facts plus, for f144, `units`. *Runtime overlay* —
`workflow_key`, `dependent_sources` — lives in a separate
`log_context_bindings` list on `Instrument` that references streams by
name. *Display* — `title`, `description` — stays in
`source_metadata` (see Choice 4).

**Alternative considered (and rejected).** A single fat record per
stream carrying every facet: NeXus facts, display, workflow_key,
dependent_sources.

**Reasoning.** `dependent_sources` is a property of *a stream-binding*,
not of the stream itself. A given stream may participate in multiple
bindings with different scoping (e.g., the same f144 stream feeding
two different Sciline workflows triggered by different sources).
Bolting `dependent_sources` onto the stream forces a single global
answer, which is wrong as soon as there's a second binding. The same
argument applies to `workflow_key`.

`units` for f144 stays on the record because it is a filewriter-level
property of the stream — read from the NeXus file and rarely
overridden in code (only for filewriter bugs).

The drift problem that motivated the proposal still goes away: `units`
no longer lives in two places (registry + binding); it lives only on
the stream record. Bindings become thin (`stream_name`, `workflow_key`,
`dependent_sources`) and don't mirror anything.

### 2. `nexus_path` on the base, not the subclass

**Choice.** The full HDF5 group path is a base-class field on
`Stream`, applicable to any streaming kind. Named `nexus_path` (not
`nxlog_path`) because the base covers `NXevent_data`, `NXdisk_chopper`
children, etc., not just NXlog.

**Earlier alternative considered (and rejected during discussion).**
Putting the path only on a `DynamicTransformBinding` subclass,
because only the dynamic-transforms consumer reads it.

**Reasoning.** `nexus_path` is not a transform-specific detail — it
is the NeXus identity of the streaming group. Filewriter groups carry
the `topic`/`source`/`units` triple at a specific HDF5 path; the path
is the primary key in the file. Any consumer that wants to splice a
live value into a NeXus structure (transform chain, chopper DataGroup,
etc.) reads the path and dispatches on it. The auto-parser inevitably
produces `nexus_path` for every group it sees; pushing it down into a
subclass forces a downcast before the parser output is usable.

`nexus_path` is None only for synthesised streams.

### 3. Subclass over composition for writer-module-specific overlay

**Choice.** `F144Stream(Stream)` adds f144-specific fields by
subclassing. Future ev44/tdct overlays do the same when needed.

**Alternative considered.** Composition: `Stream` carries an optional
`f144: F144Meta | None` field. Or: flat record with everything
Optional.

**Reasoning.** Post-split (Choice 1), only `units` is f144-specific.
The subclass still earns its keep:

- Subclassing keeps the base type useful in its own right. Iteration
  filters by `isinstance(s, F144Stream)`. No extra indirection at
  call sites.
- A flat record with `units: str | None` everywhere would collapse
  the type distinction: an ev44 stream definitionally has no `units`
  in the f144 sense; allowing the field hides that.
- The trade-off — can't tack a new writer-module facet onto an
  existing parsed record — is harmless because the parser dispatches
  on `writer_module` and builds the right subclass directly.

If `units` ends up being the only f144-specific field forever, the
subclass is light. If more f144 fields appear later (e.g., a rate hint
for batching), they have a natural home.

### 4. Display stays in `source_metadata`, separate from the stream registry

**Choice.** `title` and `description` are *not* on `Stream`. They live
in `Instrument.source_metadata: dict[str, SourceMetadata]`, the same
side-dict that already holds display info for detectors and monitors.
`get_source_title(name)` / `get_source_description(name)` keep their
current shape, consulting `source_metadata` only.

**Alternative considered (and rejected).** Move display fields onto
`Stream` records, with `source_metadata` retained only for
detector/monitor entries during a transitional period.

**Reasoning.** `Stream` is "what arrives on the wire / what NeXus
says" — pure data plumbing. Display labels are a UI concern with a
different lifecycle and a different audience. Merging them onto the
record conflates the two: every override path, every parser fixture,
every record-equality test now also concerns itself with human-readable
strings.

Keeping display in `source_metadata`:

- Source_metadata's "keyed by source name across entity kinds" overload
  is the right shape, not a bug. It is the single display map for
  streams, detectors, monitors, and (eventually) anything else with a
  source name.
- The "two lookups to fully describe a stream" cost is one helper
  method internally; consumers call `instrument.get_source_title(name)`
  and don't see the split.
- No transitional state. Display for f144 entries is set in
  `source_metadata` the same way detector/monitor display already is.
- The override surface on `build_streams` shrinks to "filewriter unit
  bugs and stream_name renames" — small enough that the typed override
  class is overkill.
- Future Detector/Monitor records (when `Ev44Stream` lands) can stay
  data-only too; their display continues to live in
  `source_metadata`. No follow-up drain needed.

### 5. Override identity is `nexus_path` (or `stream_name`), not the dict key alone

**Choice.** The `overrides=` keyword to `build_streams` accepts either
`stream_name` or `nexus_path` as the key for each override entry. The
in-memory dict is keyed by `stream_name`.

**Reasoning.** Neither identity is intrinsically stable — both can
change if the geometry file or the naming convention changes. The
relevant difference is failure mode:

- A patch keyed by `nexus_path` fails loudly if the parser sees a
  different path next round (no entry to patch → error).
- A patch keyed by `stream_name` works fine until someone renames
  the entry, then silently misses.

For hand-coded instruments (LOKI), authors think in `stream_name` and
overrides are local. For parser-generated registries (DREAM-future),
`nexus_path` is the stable handle the generator emits. Accepting
both lets the per-instrument config use whichever reads better.

Synthesised streams are declared inline in `synthetics=`, so the
override-key question doesn't apply to them.

### 6. `StreamLUT` isolation is preserved

**Choice.** Internal code never reads `stream.topic` or
`stream.source`; only the kafka-boundary modules (`streams.py` per
instrument, `message_adapter`, kafka source) do. The `StreamLUT` is
*derived* from `instrument.streams` rather than maintained
separately.

**Reasoning.** Isolation is enforced by *who reads which field*, not
*which object holds the field*. Putting `topic`/`source` on the same
record lets the NeXus extractor produce one complete row per stream
(NeXus already collocates them as group attrs) and lets the
LUT-building code in `streams.py` be a one-liner. Modules outside
`streams.py` and `kafka/` continue to use `stream_name` as the
routing handle.

### 7. Registry is `dict[StreamName, Stream]`, single flat collection

**Choice.** `Instrument.streams: dict[StreamName, Stream]` holds
streams of all writer modules, keyed by `stream_name`. Per-kind views
are filters (`isinstance(s, F144Stream)`).

**Alternative considered.** `list[Stream]` (parser-natural ordering)
or split lists per writer module.

**Reasoning.** Lookup is the dominant access pattern (bindings,
NXlog-attrs, derived StreamLUT entries). A dict makes those O(1) and
lets bindings reference streams by name without scanning. The
parser still yields `list[Stream]`; the construction step
(`build_streams`) converts to a dict and raises on `stream_name`
collisions. Collisions are statically known per geometry — the rename
map in `build_streams(...)` is where they get resolved.

The "config representation flat across writer modules" framing is
preserved: per-kind splits stay in `StreamMapping` (which `kafka/`
needs) and are derived from the unified registry.

### 8. Synthesised streams marked by `topic is None`, not a flag

**Choice.** No `synthesized: bool` field. The predicate is `topic is
None` (equivalently, `nexus_path is None`). An invariant in
`__post_init__` couples the three "real" fields so they are all None
or all set.

**Reasoning.** A separate flag would be derivable from the others,
with a refactor risk if they disagree. The implicit predicate is
cheap, type-safe, and self-consistent.

Crucially, consumers are oblivious to whether a stream is synthesised:
only the kafka-boundary LUT builder filters on `topic is None`.
Workflow harnesses, display lookups, timeseries UI, and binding
resolution treat all entries uniformly.

### 9. Parse offline (codegen), not at import or at runtime

**Choice.** The parser is a CLI generator that emits a checked-in
`streams_parsed.py`. The hand-edited `streams.py` imports that list
and composes the final registry via `build_streams`. No process reads
the NeXus geometry at import or startup just to populate the registry.

**Alternatives considered.**

- *Parse at import time, dashboard reads geometry via Pooch.* Breaks
  the current invariant that the dashboard has no file-access
  dependency. Cold startups depend on cache state and network.
- *Backend parses, ships parsed registry to dashboard via runtime
  channel.* Adds a configuration-distribution mechanism that doesn't
  exist today. Heavy.

**Reasoning.** Both backend and dashboard import the same per-instrument
config module. Codegen is the simplest way to keep both file-free at
startup while still letting DREAM-scale geometries skip
hand-maintenance. A CI check that re-runs codegen and fails on diff
protects against drift between the committed `streams_parsed.py` and
the geometry file.

A future move to dynamic distribution is local: `Stream` is a frozen
dataclass that round-trips through any serialisation layer, and the
seam is "who populates `Instrument.streams` at startup".

### 10. Construction-time invariants, not test-time validation

**Choice.** Misconfiguration in the registry, bindings, or overrides
fails at process startup, not at pytest time. Specifically:

- `Instrument.__post_init__` raises if:
  - any `log_context_bindings[i].stream_name` is missing from
    `Instrument.streams`, or
  - any override key passed to `build_streams` did not match a parsed
    record (caught inside `build_streams` itself, which raises before
    returning).
- End of `Instrument.load_factories()` raises if any
  `log_context_bindings[i].dependent_sources` element does not match the
  `source_names` of some registered spec. This check has to run *after*
  factory loading because spec registration populates `source_names`.

Pytest checks remain only for *the invariant logic itself* (does the
validator correctly reject a malformed binding?), not as enumeration
of every binding as a regression test.

**Alternative considered.** Keep the existing test-time pattern
(`dynamic_transforms_registry_test.py` and analogues). Catches errors
in CI but not at the moment of misconfiguration; a typo only surfaces
when pytest runs.

**Reasoning.** Misconfiguration is a deployment-blocking error, not a
warning. The dashboard and backend both import `Instrument` at
startup; a typo in `streams.py` (wrong stream_name, missing override
target, unknown dependent source) should crash the process before
traffic starts. This matches the project's fail-fast preference and
removes a category of "tests pass, deploy fails" surprises.

Consumer-specific reachability checks (e.g., dynamic-transforms's
"every binding's `nexus_path` lies on a depends-on chain that some
detector loads") move into that consumer's own construction-time
check, not the registry's. The registry only validates registry
invariants; consumers validate consumer invariants.

## What this enables

Stating these as motivations rather than work items, so the proposal
can stand on its own.

- **Adoption of a new f144-driven workflow consumer becomes a one-line
  binding entry.** A new `LogContextBinding(stream_name=..., workflow_key=...,
  dependent_sources=...)` references an existing stream record. No
  duplication of `units` or `nexus_path`, no separate attribute
  registry, no manual splice.
- **NeXus-driven instruments scale to O(100) streams.** The
  hand-maintained per-instrument dict for DREAM-class instruments
  becomes a generated `streams_parsed.py` plus a small overrides dict
  in `streams.py` — overrides are expected to be rare (filewriter unit
  bugs, hand-chosen `stream_name` renames).
- **Both known Sciline-key consumers use the same binding mechanism.**
  Dynamic-transforms (detector-scope) and chopper-workflow (trigger-scope)
  produce entries in the same `log_context_bindings` list, distinguished
  only by their `workflow_key` types and `dependent_sources`. The
  chopper consumer must adopt typed `ValueLog`-keyed inputs (moving
  destination mapping out of name parsing) to participate.
- **Synthesised streams stop being a special case.** They are
  `F144Stream` records with `topic=None`. Consumers iterate the
  registry blind to the distinction; only the kafka-boundary LUT
  builder filters them out.
- **`StreamProcessorWorkflow.context_keys` for f144 streams becomes
  a derived view of the bindings.** Today's factories hand-code the
  stream-name → Sciline-key mapping (e.g.
  `bifrost/factories.py:_make_cut_stream_processor` literally
  declares `context_keys={'detector_rotation': InstrumentAngle[SampleRun], ...}`;
  `detector_view/factory.py` mixes f144 entries derived from
  `_dynamic_transforms` config with non-f144 ROI entries). Post-bindings,
  factories filter `instrument.log_context_bindings` by
  `dependent_sources & spec.source_names` and derive `context_keys`
  from the matched entries. The stream-name → Sciline-key mapping
  lives in one place — the binding declaration — instead of being
  duplicated in every factory that wires up an `f144`-consuming
  workflow. Non-f144 `context_keys` entries (UI control signals like
  `roi_rectangle: ROIRectangleRequest`) stay hand-coded in the factory;
  bindings cover the f144 half where the duplication actually existed.

## Open questions worth flagging

1. **Generalising overlay beyond f144.** When ev44/tdct streams gain
   per-stream runtime config (rate hints, decoder choices), each gets
   a `Stream` subclass. If a future ev44/tdct consumer needs its own
   binding shape (key types other than `type[ValueLog]`), a second
   binding type and field is added — `LogContextBinding` is asserted to
   unify f144 Sciline-key consumers, not all writer modules.
2. **Codegen drift protection.** A CI check that re-runs the
   generator and fails on diff against the committed
   `streams_parsed.py` is straightforward but needs a stable input
   (the geometry file in pooch, pinned). Worth confirming the
   geometry-file pinning workflow before relying on this.

## Rollout sketch (not a plan)

The change is naturally incremental and behaviour-preserving:

1. Promote `nexus_helpers.StreamInfo` into the config layer as
   `Stream`; add `F144Stream(Stream)` with `units` only (no consumer
   overlay).
2. Add `Instrument.streams: dict[StreamName, Stream]`. Derive
   `f144_attribute_registry` from
   `{name: {'units': s.units} for name, s in streams.items() if isinstance(s, F144Stream)}`
   in `__post_init__`; keep the existing field temporarily for the
   migration window.
3. Migrate per-instrument `f144_log_streams` dicts into
   `dict[StreamName, F144Stream]` literals. Mechanical, one instrument
   per PR.
4. Switch `_make_*_logs()` in `streams.py` files to iterate
   `instrument.streams` instead of importing the per-instrument dict.
5. Drop the redundant `f144_log_streams` dicts and
   `f144_attribute_registry` constructor kwargs.
6. Introduce `Instrument.log_context_bindings: list[LogContextBinding]` and
   migrate the dynamic-transforms work onto it. Bindings reference
   streams by `stream_name`; they no longer carry `units`.
7. Refactor factories that hand-code f144 `context_keys` to derive them
   from `instrument.log_context_bindings` filtered by spec source_names.
   Concrete sites: `bifrost/factories.py:_make_cut_stream_processor`
   (drops the hand-coded
   `{'detector_rotation': InstrumentAngle[SampleRun], 'sample_rotation': SampleAngle[SampleRun]}`),
   `detector_view/factory.py` (drops the f144 portion built from
   `_dynamic_transforms`; non-f144 ROI entries stay).
8. Refactor the chopper consumer to consume f144 inputs as typed
   `ValueLog` subclasses, moving the `<chopper>_<field>` destination
   mapping out of name parsing and into the Sciline graph. Once that
   lands, the chopper consumer's `make_chopper_bindings(...)` helper
   produces `list[LogContextBinding]` entries that go into the same
   `log_context_bindings` list as the dynamic-transform entries.
9. Extend the existing `nexus_helpers` CLI generator to emit a
   `streams_parsed.py` containing a `list[F144Stream]` literal. Add
   `build_streams(parsed, overrides, synthetics) -> dict[StreamName, Stream]`.
   Migrate one stream-heavy instrument to validate the override
   ergonomics.

Each step is a small reviewable PR. Consumers (the dynamic-transforms
work, the chopper work) become much smaller follow-ups once the
unified record and binding list are in place, because they just *read*
fields that already exist on the record instead of declaring their own
parallel lists.
