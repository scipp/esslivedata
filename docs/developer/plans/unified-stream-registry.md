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
3. **Display split.** Title/description for an f144 stream live in
   `source_metadata`, separate from the routing/units/path facets.
   Two lookups to fully describe a stream.
4. **No room for synthesised streams.** Streams that exist only
   in-process (a synthesiser plateau-detecting on a noisy upstream
   readback, e.g. `<chopper>_delay_setpoint`) don't fit the
   "filewriter row" mental model; today they live in a separate
   `_SYNTHETIC_LOG_ATTRS` dict in `timeseries_handler.py`.
5. **Hand-coding doesn't scale.** Some instruments will eventually
   have O(100) f144 streams (DREAM is the largest case). A
   hand-maintained dict per instrument is fine at ~10 streams; it
   will not be at 100.

## Scope

The registry is f144-first. `Stream` is designed as the base for all
writer modules (ev44, tdct, ...) so detectors and monitors can later
become `Stream` subclasses and `source_metadata` can be drained, but
that migration is out of scope for this proposal. Initial rollout
touches only the f144 fragmentation. Display lookups (`get_source_title`
etc.) consult both the streams registry and `source_metadata` in the
transitional period — small enough to be a non-issue.

## Target shape

### A. Canonical record per stream + separate consumer bindings

One record describes the NeXus-facing facts and display facets of a
streaming group: how it arrives over Kafka, where it sits in the NeXus
structure, how it is named/described in the UI, and (for f144) its
units. *Runtime overlay* that wires a stream into a specific consumer
(e.g., Sciline log keys, dependent-source scoping) lives in a separate
typed list per consumer, referencing streams by name. The registry is
the noun; bindings are the verbs that point at it.

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

    # Display
    title: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class F144Stream(Stream):
    """f144 NXlog stream — value/time payloads."""
    units: str | None = None
    writer_module: str = 'f144'          # fixed at the class level
```

`Instrument.streams: dict[StreamName, Stream]` replaces
`f144_log_streams`, `f144_attribute_registry`, and the f144 entries in
`source_metadata`. The dict is keyed by `stream_name`, which is what
every non-kafka-boundary consumer uses as the routing handle.

All current registry usages become derived views:

- `ToNXlog`-attrs lookup → `instrument.streams[name]`, pull `units`
  (and any other NXlog attrs we eventually want) off an `F144Stream`.
- Auto-registration of timeseries specs → iterate
  `[s for s in instrument.streams.values() if isinstance(s, F144Stream)]`
  and read `stream_name`.
- Display lookup → `stream.title` / `stream.description` (with
  `stream_name` as fallback).
- `StreamLUT[logs]` → built once in `streams.py` from
  `{(s.topic, s.source): s.stream_name for s in instrument.streams.values() if s.topic is not None}`.

The runtime overlay — the dynamic-transforms work, the chopper-workflow
work — lives in *consumer-owned* binding lists:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class LogContextBinding:
    """One f144 stream feeding a Sciline log key, scoped to dependents."""
    stream_name: str                     # references Instrument.streams[stream_name]
    log_key: type[ValueLog]
    dependent_sources: frozenset[str]
```

`Instrument.log_context_bindings: list[LogContextBinding]` (and
analogous per-consumer lists as they appear). Bindings are thin: they
no longer mirror `units` or `nexus_path` — those live on the stream
record they reference. A binding is purely "consumer X wants stream Y,
scoped to dependents Z".

Why split runtime overlay from the stream record:

- `dependent_sources` is a property of the *consumer-stream binding*,
  not the stream. Different consumers reading the same stream may have
  different scoping needs (e.g., the chopper-workflow consumer routes
  by chopper group name, not by detector scope).
- The registry stays writer-module-symmetric: ev44 and tdct don't
  inherently need any runtime-overlay fields, so they don't gain a
  subclass just to default them to None.
- The split makes adoption of a new consumer kind a local change in
  one place (a new binding type plus its consumer code), not a new
  field on every `Stream` subclass.

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
        'sample_temperature': dict(title='Sample temperature'),
        '/entry/.../wrong_units_path': dict(units='K'),  # filewriter bug fix
    },
    synthetics=[
        F144Stream(stream_name='delay_setpoint', nexus_path=None,
                   topic=None, source=None, title='Delay setpoint', ...),
    ],
)
```

`build_streams` returns the `dict[StreamName, Stream]`. Override keys
are accepted as either `stream_name` (for hand-named entries) or
`nexus_path` (the stable NeXus identity, for entries that haven't been
renamed). Collisions on `stream_name` raise at codegen *and*
construction time.

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
timeseries UI, display lookups, and binding resolution all index into
`Instrument.streams` by name and read the same fields regardless of
origin. The only place that filters by `topic is None` is the
kafka-boundary `StreamLUT` builder, which has no entry to emit for a
stream that never arrives over Kafka. Authoring may keep parsed and
synthetic entries visually separate (as in §B's `build_streams` call),
but the runtime registry is a single flat dict.

## Choices and reasoning

### 1. NeXus facts + display on the record, runtime overlay split off

**Choice.** The record (`Stream` / `F144Stream`) holds NeXus-derivable
facts plus optional display fields (`title`, `description`) and, for
f144, `units`. *Consumer-specific runtime overlay* — `log_key`,
`dependent_sources`, future binding kinds — lives in per-consumer
binding lists on `Instrument` that reference streams by name.

**Alternative considered (and rejected).** A single fat record per
stream carrying every facet: NeXus facts, display, log_key,
dependent_sources, and whatever future consumers need.

**Reasoning.** `dependent_sources` is a property of *a consumer-stream
binding*, not of the stream itself. A given stream may be read by
multiple consumers with different scoping rules (the dynamic-transforms
consumer scopes by detector; the chopper-workflow consumer routes by
chopper group). Bolting `dependent_sources` onto the stream forces a
single global answer, which is wrong as soon as there's a second
consumer. The same argument applies to `log_key` and any future
consumer-specific routing facet.

Putting display fields (`title`, `description`) on the record *is*
correct: they are properties of "this streaming entity", not of a
particular consumer's wiring. Likewise `units` for f144 — it is a
filewriter-level property of the stream, just sometimes wrong in the
file and needing override in code.

The drift problem that motivated the proposal still goes away: `units`
no longer lives in two places (registry + binding); it lives only on
the stream record. Bindings become thin (`stream_name`, `log_key`,
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

### 4. Display fields on the stream record

**Choice.** `title` and `description` are fields on `Stream` itself,
not in a parallel `source_metadata` dict.

**Reasoning.** Display info is a property of "this streaming entity",
just not derivable from NeXus. Same pattern as `log_key` *would have
been* if it weren't consumer-specific: a non-NeXus fact set in
per-instrument code. Putting it on the record:

- Eliminates the "two lookups to fully describe a stream" problem.
- Doesn't break detector/monitor metadata: those eventually become
  `Stream` subclasses (`Ev44Stream` with `parent_nx_class='NXdetector'`
  / `'NXmonitor'`) with their own display fields, draining
  `source_metadata` entirely. Out of scope for this proposal.

The codegen path leaves `title` / `description` as `None`; the
per-instrument `streams.py` is where human-readable text is set via
the overrides dict.

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
display, NXlog-attrs, derived StreamLUT entries). A dict makes those
O(1) and lets bindings reference streams by name without scanning. The
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

## What this enables

Stating these as motivations rather than work items, so the proposal
can stand on its own.

- **Adoption of a new f144-driven workflow consumer becomes a one-line
  binding entry.** A new `LogContextBinding(stream_name=..., log_key=...,
  dependent_sources=...)` references an existing stream record. No
  duplication of `units` or `nexus_path`, no separate attribute
  registry, no manual splice.
- **NeXus-driven instruments scale to O(100) streams.** The
  hand-maintained per-instrument dict for DREAM-class instruments
  becomes a generated `streams_parsed.py` plus a small overrides dict
  in `streams.py` — a few human-meaningful entries for streams that
  drive display, run consumer bindings, or fix file bugs.
- **The two ongoing efforts simplify symmetrically.** A consumer that
  patches `depends_on` chains reads `nexus_path` off `F144Stream`
  records via its own binding list. A consumer that builds chopper
  `DataGroups` reads the same fields with its own binding type and
  dispatches differently. Neither owns its own routing/attribute
  lists; both reference the registry by name.
- **Synthesised streams stop being a special case.** They are
  `F144Stream` records with `topic=None`. Consumers iterate the
  registry blind to the distinction; only the kafka-boundary LUT
  builder filters them out.

## Open questions worth flagging

1. **Generalising overlay beyond f144 — and whether other consumers
   need their own binding types.** When ev44/tdct streams gain
   per-stream runtime config (rate hints, decoder choices), each gets
   a `Stream` subclass. Separately, the chopper-workflow consumer
   will want its own binding type (analogous to `LogContextBinding`)
   that captures whatever scoping it needs. Worth nailing down the
   binding-type pattern once the second concrete consumer lands.
2. **`source_metadata` migration.** Display fields for f144 streams
   move onto the stream record; detector/monitor entries stay in
   `source_metadata` until those become `Ev44Stream`-style records.
   Worth deciding whether to do that drain incrementally or as a
   single follow-up.
3. **Validation timing.** Today's
   `dynamic_transforms_registry_test.py` walks NeXus depends-on
   chains and verifies every binding's path is reachable. With the
   unified record this generalises to two checks: (a) every binding's
   `stream_name` exists in `Instrument.streams`, (b) every override's
   key matches a parsed record. Both could move from tests to
   instrument-construction-time invariants. Worth deciding before the
   first big rollout PR.
4. **Codegen drift protection.** A CI check that re-runs the
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
6. Move `title` / `description` for f144 streams from
   `source_metadata` onto the records.
7. Introduce `Instrument.log_context_bindings: list[LogContextBinding]`
   (the consumer-side overlay), and migrate the dynamic-transforms
   work onto it. Bindings reference streams by `stream_name`; they no
   longer carry `units`.
8. Extend the existing `nexus_helpers` CLI generator to emit a
   `streams_parsed.py` containing a `list[F144Stream]` literal. Add
   `build_streams(parsed, overrides, synthetics) -> dict[StreamName, Stream]`.
   Migrate one stream-heavy instrument to validate the override
   ergonomics.

Each step is a small reviewable PR. Consumers (the dynamic-transforms
work, the chopper work) become much smaller follow-ups once the
unified record is in place, because they just *read* fields that
already exist on the record instead of declaring their own parallel
lists.
