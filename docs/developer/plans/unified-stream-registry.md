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

## Target shape

### A. Single canonical record per stream

One record describes everything about a streaming group: how it
arrives over Kafka, where it sits in the NeXus structure, how it is
displayed in the UI, and (optionally) how it is wired into Sciline
workflows.

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Stream:
    """Any streaming group in NeXus (or synthesised in-process)."""

    # Identity / routing
    stream_name: str                     # internal handle, not user-facing
    nxlog_path: str | None               # full HDF5 path; None for synthesised
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
    log_key: type[ValueLog] | None = None
    dependent_sources: frozenset[str] = frozenset()
    writer_module: str = 'f144'          # fixed at the class level
```

`Instrument.streams: list[Stream]` replaces `f144_log_streams`,
`f144_attribute_registry`, `log_context_bindings`, and the f144 entries
in `source_metadata`. All current usages become derived views:

- `ToNXlog`-attrs lookup → match by `stream_name`, pull `units` (and
  any other NXlog attrs we eventually want) off an `F144Stream`.
- Auto-registration of timeseries specs → iterate `F144Stream`s and
  read `stream_name`.
- Aux-source routing → iterate `F144Stream`s with non-empty
  `dependent_sources` and `log_key is not None`.
- Transform-chain patching (the dynamic-transforms work) → filter to
  `F144Stream`s whose `nxlog_path` lies on a chain the consumer
  loads.
- Display lookup → `stream.title` / `stream.description` (with
  `stream_name` as fallback).
- `StreamLUT[logs]` → built once in `streams.py` from
  `{(s.topic, s.source): s.stream_name for s in instrument.streams if s.topic is not None}`.

### B. Auto-parse with code-side overrides

For an instrument with O(100) NXlogs, the registry is bootstrapped
from the NeXus file and patched in code:

```python
streams = parse_streams_from_nexus(geometry_file)        # ~100 records
overrides = [
    F144StreamOverride(
        nxlog_path='/entry/instrument/sample/temperature',
        units='K',                                       # filewriter bug fix
        title='Sample temperature',
        log_key=SampleTemperatureLog,
        dependent_sources=frozenset({'mantle_detector'}),
    ),
    ...
]
instrument = Instrument(..., streams=apply_overrides(streams, overrides))
```

The override mechanism is *one* operation that handles both "fix a
wrong field in the file" and "attach the runtime overlay" — there are
not two override paths.

For instruments with few streams (LOKI, BIFROST today), the registry
can still be hand-written as a `list[F144Stream]` literal. The parser
is opt-in.

### C. Synthesised streams

A synthesised stream (no NeXus row, emitted in-process by a
synthesiser) is `Stream(topic=None, source=None, nxlog_path=None,
...)`. The predicate "synthesised" is `stream.topic is None`; no
separate flag needed.

A `__post_init__` invariant enforces that
`topic`, `source`, and `nxlog_path` are all None or all non-None
(real-and-parseable, or synthesised).

## Choices and reasoning

### 1. One record, not two (NeXus facts + runtime overlay)

**Choice.** Merge static (parsable) and runtime (livedata-specific)
facets onto a single dataclass.

**Alternative considered.** Keep them separate: `Stream` (parser
output) and `LivedataConsumer(stream_name, log_key,
dependent_sources, ...)` joined by stream name.

**Reasoning.** Separation looks cleaner in the abstract — parser
produces one type, code attaches another. In practice:

- Most overlay fields are already optional. Setting `log_key=None` on
  a record that is "just for timeseries plotting" is one line of
  mild clutter and zero ambiguity.
- The current world *already has* the duplication problem: today's
  `LogContextBinding.units` mirrors `f144_log_streams[*].units`. The
  two-list world makes drift possible; the one-list world makes it
  impossible.
- Fewer types means fewer joins at call sites. "Tell me everything
  about this stream" is one record lookup, not two.
- The parse-with-overrides pattern works on a single record type
  uniformly: every field defaults to whatever the parser produced and
  can be replaced by an override.

### 2. `nxlog_path` on the base, not the subclass

**Choice.** The full HDF5 group path is a base-class field on
`Stream`, applicable to any streaming kind.

**Earlier alternative considered (and rejected during discussion).**
Putting `nxlog_path` only on a `DynamicTransformBinding` subclass,
because only the dynamic-transforms consumer reads it.

**Reasoning.** `nxlog_path` is not a transform-specific detail — it
is the natural NeXus identity of the streaming group. Filewriter
NXlog groups already carry the `topic`/`source`/`units` triple at a
specific HDF5 path; the path is the primary key in the file. Any
consumer that wants to splice a live value into a NeXus structure
(transform chain, chopper DataGroup, etc.) reads the path and dispatches
on it. Hiding it inside a transform-only subclass is wrong on two
counts:

- The chopper-workflow consumer has the same conceptual need — it
  merges a stream's latest value into `raw_choppers[group][field]` —
  but today encodes the destination address implicitly through
  string-parsing of stream names plus hard-coded DataGroup keys.
  Lifting `nxlog_path` to the base means both consumers express the
  same coordinate in the same way.
- The auto-parser inevitably produces `nxlog_path` for every f144
  group it sees; pushing it down into a subclass forces a downcast
  before the parser output is usable.

`nxlog_path` is None only for synthesised streams.

### 3. Subclass over composition for writer-module-specific overlay

**Choice.** `F144Stream(Stream)` adds f144-specific fields by
subclassing. Future ev44/tdct overlays do the same when needed.

**Alternative considered.** Composition: `Stream` carries an optional
`f144: F144Meta | None` field (and analogous slots for other writer
modules). Or: flat record with everything Optional.

**Reasoning.** Today only f144 has a typed overlay; ev44/tdct streams
currently have no per-record runtime config. The pragmatic shape:

- Subclassing keeps the base type useful in its own right. The
  parser yields `list[Stream]`; the routing layer iterates and
  filters by `isinstance`. No extra indirection at call sites.
- Composition slots ("`Stream.f144`", "`Stream.ev44`", ...) is YAGNI
  while only one writer module has overlay fields. The slots would
  be `None` for every other stream forever.
- A flat record with Optional everywhere collapses the type
  distinctions. An ev44 stream definitionally has no `units` field
  in the f144 sense; allowing the field hides that.
- The trade-off — you can't tack a new writer-module facet onto an
  existing parsed record — is harmless because the parser dispatches
  on `writer_module` and builds the right subclass directly.

### 4. Display fields on the stream record

**Choice.** `title` and `description` are fields on `Stream` itself,
not in a parallel `source_metadata` dict.

**Earlier shape.** `Instrument.source_metadata` is a dict keyed by
source name, overloaded across detector names (`loki_detector_0`),
monitor names (`beam_monitor_m1`), and f144 stream names
(`detector_carriage`). Today it is the single display lookup for any
"thing" the UI surfaces.

**Reasoning.** Display info is another *overlay facet* — not
derivable from NeXus, set in code, optional. It fits the same
parse+override mechanism as `log_key` / `dependent_sources` with zero
extra machinery. Putting it on the record:

- Eliminates the "two lookups to fully describe a stream" problem.
- Doesn't break detector/monitor metadata: those should eventually
  get their own typed records (`Detector(name, title, description,
  detector_number, ...)`, `Monitor(...)`) with their own display
  fields, drained from `source_metadata` over time. Same pattern,
  different entity; the records don't merge.
- `source_metadata` can remain as a transitional dict covering the
  residual entities (detectors, monitors) until those are typed too.

The auto-parse path leaves `title` / `description` as `None`; the
per-instrument override step is where human-readable text is set.

### 5. Override key is `nxlog_path`, not `stream_name`

**Choice.** The override map is keyed by the NeXus path.

**Reasoning.** `stream_name` is a derived shorthand (the parser uses
something like `suggest_internal_name(group_path)` to pick it). If
the naming convention ever changes, overrides keyed by `stream_name`
silently miss; overrides keyed by `nxlog_path` don't. Path is the
stable NeXus identity. The user noted that `stream_name` is not
user-facing, which reinforces this: there is no UI cost to changing
the convention later if names are derived.

Synthesised streams have no `nxlog_path` and are declared in code,
not parsed, so the override-key question doesn't apply to them.

### 6. `StreamLUT` isolation is preserved

**Choice.** Internal code never reads `stream.topic` or
`stream.source`; only the kafka-boundary modules (`streams.py` per
instrument, `message_adapter`, kafka source) do. The `StreamLUT` is
*derived* from `instrument.streams` rather than maintained
separately.

**Reasoning.** The original isolation argument is sound — outside the
kafka package, nothing should care about Kafka coordinates. But the
isolation is enforced by *who reads which field*, not by *which
object holds the field*. Putting `topic`/`source` on the same record
as everything else lets the NeXus extractor produce one complete row
per stream (NeXus already collocates them as group attrs) and lets
the LUT-building code in `streams.py` be a one-liner. Modules
outside `streams.py` and `kafka/` continue to use `stream_name` as
the routing handle.

### 7. Single `streams` list, not multiple per-kind lists

**Choice.** `Instrument.streams: list[Stream]` holds streams of all
writer modules. Per-kind views are filters (`isinstance(s,
F144Stream)`).

**Reasoning.** Conceptually the registry mirrors what's in the NeXus
file: a flat collection of streaming groups across various
`writer_module`s. The current per-kind splits (`detectors`,
`monitors`, `area_detectors`, `logs` on `StreamMapping`) exist
because the kafka layer dispatches by topic kind; that split stays in
`StreamMapping` and is derived from the flat list. The config-layer
representation is flat because that's the natural NeXus view.

### 8. Synthesised streams marked by `topic is None`, not a flag

**Choice.** No `synthesized: bool` field. The predicate is `topic is
None` (equivalently, `nxlog_path is None`). An invariant in
`__post_init__` couples the three "real" fields so they are all None
or all set.

**Reasoning.** A separate flag would add a field that is always
derivable from the others, with a refactor risk if the flag and the
fields disagree. The implicit predicate is cheap, type-safe, and
self-consistent.

## What this enables

Stating these as motivations rather than work items, so the proposal
can stand on its own.

- **Adoption of a new f144-driven workflow consumer becomes a one-line
  override** on the parsed/declared stream — set `log_key` and
  `dependent_sources`. No additional list entry, no separate attribute
  registry, no manual splice for units.
- **NeXus-driven instruments scale to O(100) streams.** The
  hand-maintained per-instrument dict for DREAM-class instruments
  becomes a parser call plus a small override list — a few
  human-meaningful entries for streams that actually drive workflows,
  display, or fix file bugs.
- **The two ongoing efforts simplify symmetrically.** A consumer that
  patches `depends_on` chains reads `nxlog_path` and `log_key` off
  `F144Stream`s. A consumer that builds chopper `DataGroups` reads
  the same fields and dispatches differently. Neither owns its own
  routing/attribute/binding lists.
- **Synthesised streams stop being a special case.** They are
  `F144Stream` records with `topic=None`. Whatever consumer wires
  them up reads the same fields.

## Open questions worth flagging

1. **When to derive `stream_name` vs. set it explicitly.** Initial
   migration keeps `stream_name` an explicit field (allows
   hand-chosen names like `detector_carriage` to survive). Long-term,
   `stream_name = suggest_internal_name(nxlog_path)` could become the
   default with explicit names as an override. Worth deciding once the
   parser has run against several real geometries.
2. **Generalising overlay beyond f144.** When ev44 (event) or tdct
   (chopper TDC) streams want their own runtime overlay, each gets a
   `Stream` subclass. Whether those subclasses end up needing
   `dependent_sources` (the same scoping concept) is a question for
   when there's a concrete need; today's detector/monitor wiring
   doesn't use it.
3. **`source_metadata` migration.** Display fields for f144 streams
   move onto the stream record; detector/monitor entries stay in
   `source_metadata` until those entities get typed records of their
   own. Worth deciding whether to do that drain incrementally or as a
   single follow-up.
4. **Compile-time validation.** Today's
   `dynamic_transforms_registry_test.py` walks NeXus depends-on
   chains and verifies every binding's `nxlog_path` is reachable.
   With the unified record this generalises to "every override's
   `nxlog_path` matches a parsed record"; chains-reachable checks stay
   consumer-specific. Worth thinking about whether overrides should be
   validated at instrument-construction time rather than via tests.

## Rollout sketch (not a plan)

The change is naturally incremental and behaviour-preserving:

1. Promote `nexus_helpers.StreamInfo` into the config layer as
   `Stream`; add `F144Stream(Stream)` with the overlay fields stubbed
   even before any consumer reads them.
2. Add `Instrument.streams: list[Stream]`. Derive
   `f144_attribute_registry` from `[s for s in streams if
   isinstance(s, F144Stream)]` in `__post_init__`; keep the existing
   field temporarily for the migration window.
3. Migrate per-instrument `f144_log_streams` dicts into
   `list[F144Stream]` literals. Mechanical, one instrument per PR.
4. Switch `_make_*_logs()` in `streams.py` files to iterate
   `instrument.streams` instead of importing the per-instrument dict.
5. Drop the redundant `f144_log_streams` dicts and
   `f144_attribute_registry` constructor kwargs.
6. Move `title` / `description` for f144 streams from
   `source_metadata` onto the records.
7. Add `parse_streams_from_nexus` + `apply_overrides`. Migrate one
   stream-heavy instrument to validate the override ergonomics.

Each step is a small reviewable PR. Consumers (the dynamic-transforms
work, the chopper work) become much smaller follow-ups once the
unified record is in place, because they just *read* fields that
already exist on the record instead of declaring their own parallel
lists.
