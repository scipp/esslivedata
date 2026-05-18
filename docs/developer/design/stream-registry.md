# Per-instrument stream registry

## Overview

Each instrument carries a single source of truth for *streaming data declarations*:
one record per group that the ESS filewriter would write into a NeXus file.
The registry replaces what was previously several parallel structures kept in sync by hand.
It is positioned so that for NeXus-rich instruments the bulk of it can be auto-generated from a geometry artifact with code-side overrides.

The two consumers that drove the shape:

- A *dynamic-transforms* effort that drives entries in a detector `depends_on` chain from live f144 streams.
  To wire one stream to a workflow it must declare a stream record and a binding from that record to a Sciline workflow key.
- A *chopper-workflow* effort that drives per-chopper geometry assembly from live f144 setpoint streams.
  It needs the same routing layer and additionally needs *synthesised* streams (no NeXus row, emitted in-process).

## Shape

### `Stream` and `F144Stream` records

`Stream` is the base class for any streaming group; `F144Stream` adds the f144-specific `units` field.
The base intentionally carries the NeXus identity (`nexus_path`, `nx_class`, `parent_nx_class`) and the wire identity (`topic`, `source`) — these are the facts the filewriter collocates as group attrs at a specific HDF5 path.

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Stream:
    writer_module: str
    nexus_path: str | None = None
    topic: str | None = None
    source: str | None = None
    nx_class: str = ''
    parent_nx_class: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class F144Stream(Stream):
    units: str | None = None
    writer_module: str = 'f144'
    nx_class: str = 'NXlog'
```

`Instrument.streams: dict[str, Stream]` holds streams of all writer modules, keyed by the instrument-facing *stream name*.
The name is the routing handle every non-kafka-boundary consumer uses; the wire identity (`topic`/`source`) is read only by the Kafka boundary (`streams.py` per instrument, `message_adapter`, kafka source).

### Synthesised streams

A synthesised stream (no NeXus row, emitted in-process by a synthesiser) is `Stream(topic=None, source=None, nexus_path=None, ...)`.
The predicate "synthesised" is `stream.topic is None`; no separate flag.
`__post_init__` enforces the invariant that `topic` and `source` are both None or both set.

Consumers are oblivious to the distinction:
workflow harnesses, timeseries UI, and binding resolution all index into `streams` by name and read the same fields regardless of origin.
The only place that filters by `topic is None` is the Kafka-boundary `StreamLUT` builder, which has no entry to emit for a stream that never arrives over Kafka.

### `LogContextBinding`

The runtime overlay — wiring f144 streams into Sciline workflows — is a separate list on `Instrument`, referencing streams by name:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class LogContextBinding:
    stream_name: str            # references Instrument.streams[stream_name]
    workflow_key: Any           # Sciline key type (e.g. InstrumentAngle[SampleRun])
    dependent_sources: frozenset[str]  # spec source_names that pick up this binding
```

`Instrument.log_context_bindings: list[LogContextBinding]`.
Bindings are thin: they do not mirror `units` or `nexus_path` — those live on the stream record they reference.
A binding is purely "stream X feeds Sciline key K, scoped to dependents D".

Factories derive `StreamProcessorWorkflow.context_keys` by calling `instrument.get_context_keys(source_name)`, which filters the binding list by `dependent_sources`.

One binding type covers both known consumers:

- *Dynamic-transforms* binding: the detector spec picks up the live readback as an aux input, keyed by a `TransformValueLog` subclass, and patches it into the transform chain.
- *Chopper-workflow* binding: a chopper workflow spec is triggered by a synthetic `chopper_cascade_in_phase` stream; every chopper f144 stream declares itself as an aux input to that trigger and feeds a typed Sciline key.

The aux-source routing mechanism iterates `log_context_bindings` once and filters by `dependent_sources & spec.source_names`.
The dispatch between "feed a detector's transform chain" vs. "feed a chopper workflow" is driven by the `workflow_key` type and the Sciline graph downstream, not by separate binding kinds.

### Display info stays in `source_metadata`

`title` and `description` are *not* on `Stream`.
They live in `Instrument.source_metadata: dict[str, SourceMetadata]`, the same side-dict that already holds display info for detectors and monitors.
`Stream` is data plumbing; `source_metadata` is the UI map.
Merging the two would force every record-equality test, override path, and parser fixture to concern itself with human-readable strings.

## Codegen seam

For instruments with O(100) NXlogs the registry is bootstrapped *offline* from a NeXus file by a CLI generator (`python -m ess.livedata.nexus_helpers <geometry.nxs> --generate`).
The generator emits a checked-in `streams_parsed.py` containing a path-keyed dict literal:

```python
# config/instruments/<inst>/streams_parsed.py  -- auto-generated
PARSED_STREAMS: dict[str, F144Stream] = {
    '/entry/instrument/sample/temperature': F144Stream(
        nexus_path='/entry/instrument/sample/temperature',
        source='SAMPLE-Tmp:RBV', topic='<inst>_sample', units='K',
    ),
    ...
}
```

The hand-edited `specs.py` composes the final name-keyed registry via `name_streams`:

```python
from .streams_parsed import PARSED_STREAMS

streams = name_streams(
    PARSED_STREAMS,
    rename={
        '/entry/instrument/detector_carriage/value': 'detector_carriage_value',
    },
)
```

`name_streams` auto-suggests an instrument-facing name from the tail of each NeXus path (with generic container groups like `entry` / `instrument` / `sample` filtered out), and accepts a `rename=` map keyed by `nexus_path` to override individual suggestions.
It raises if a rename key matches no parsed entry, or if the resulting names are not unique.

**Why codegen, not runtime parsing.**
Both backend and dashboard import the same per-instrument config module.
Runtime parsing would force the dashboard into a file-access path (Pooch download, cold startup depends on cache state and network).
Codegen keeps both file-free at startup and makes the parser output reviewable as code.
A future move to dynamic distribution is local: `Stream` is a frozen dataclass that round-trips through any serialisation layer, and the seam is "who populates `Instrument.streams` at startup".

For instruments with few streams (today: `dummy`), the registry is a hand-written `dict[str, F144Stream]` literal and the codegen path is not used.

## Choices and reasoning

The shape above resolves several tensions that are not obvious from the code alone.
The reasoning is recorded here so that revisiting any of these does not require re-deriving the trade-off.

### 1. NeXus facts on the record, runtime overlay split off

The record (`Stream` / `F144Stream`) holds only NeXus-derivable facts plus, for f144, `units`.
Runtime overlay — `workflow_key`, `dependent_sources` — lives in a separate `log_context_bindings` list that references streams by name.

A fat record carrying every facet was considered and rejected.
`dependent_sources` is a property of *a stream-binding*, not of the stream itself:
a given stream may participate in multiple bindings with different scoping (e.g., the same f144 stream feeding two different Sciline workflows triggered by different sources).
Bolting `dependent_sources` onto the stream forces a single global answer, which is wrong as soon as there is a second binding.
The same argument applies to `workflow_key`.

`units` for f144 stays on the record because it is a filewriter-level property of the stream — read from the NeXus file and rarely overridden in code (only for filewriter bugs).

### 2. `nexus_path` on the base, not the subclass

The full HDF5 group path is a base-class field on `Stream`, applicable to any streaming kind.
Named `nexus_path` (not `nxlog_path`) because the base covers `NXevent_data`, `NXdisk_chopper` children, etc., not just NXlog.

`nexus_path` is not a transform-specific detail — it is the NeXus identity of the streaming group.
Filewriter groups carry the `topic`/`source`/`units` triple at a specific HDF5 path; the path is the primary key in the file.
Any consumer that wants to splice a live value into a NeXus structure (transform chain, chopper `DataGroup`, etc.) reads the path and dispatches on it.
The auto-parser inevitably produces `nexus_path` for every group it sees.

`nexus_path` is None only for synthesised streams.

### 3. Subclass over composition for writer-module-specific overlay

`F144Stream(Stream)` adds f144-specific fields by subclassing.
Future ev44/tdct overlays do the same when needed.

Subclassing keeps the base type useful in its own right:
iteration filters by `isinstance(s, F144Stream)`, no extra indirection at call sites.
A flat record with `units: str | None` everywhere would collapse the type distinction:
an ev44 stream definitionally has no `units` in the f144 sense; allowing the field hides that.

If a writer module ever needs binding shapes other than the f144 `(stream_name, workflow_key, dependent_sources)` triple, a second binding type is added — `LogContextBinding` unifies f144 Sciline-key consumers, not all writer modules.

### 4. Display stays in `source_metadata`

Already covered in [Shape](#shape).
`source_metadata`'s "keyed by source name across entity kinds" overload is the right shape, not a bug.
It is the single display map for streams, detectors, monitors, and (eventually) anything else with a source name.

### 5. Registry is `dict[str, Stream]`, single flat collection

`Instrument.streams: dict[str, Stream]` holds streams of all writer modules, keyed by name.
Per-kind views are filters (`instrument.f144_streams` returns `dict[str, F144Stream]`).

Lookup is the dominant access pattern (bindings, NXlog-attrs, derived `StreamLUT` entries).
A dict makes those O(1) and lets bindings reference streams by name without scanning.

### 6. Synthesised streams marked by `topic is None`, not a flag

A separate `synthesized: bool` flag would be derivable from the other fields, with a refactor risk if they disagree.
The implicit predicate is cheap, type-safe, and self-consistent.

### 7. `StreamLUT` isolation is preserved

Internal code never reads `stream.topic` or `stream.source`; only the Kafka-boundary modules (`streams.py` per instrument, `message_adapter`, kafka source) do.
The `StreamLUT` is *derived* from `instrument.streams` rather than maintained separately.

Isolation is enforced by *who reads which field*, not *which object holds the field*.
Putting `topic`/`source` on the same record lets the NeXus extractor produce one complete row per stream (NeXus already collocates them as group attrs) and lets the LUT-building code be a one-liner.

### 8. Construction-time invariants, not test-time validation

Misconfiguration in the registry, bindings, or overrides fails at process startup, not at pytest time.
Specifically:

- `Instrument.__post_init__` raises if any `log_context_bindings[i].stream_name` is missing from `Instrument.streams`.
- `name_streams` raises if a rename key matches no parsed entry, or if name collisions are produced.
- End of `Instrument.load_factories()` raises if any `log_context_bindings[i].dependent_sources` element does not match the `source_names` of some registered spec.
  This runs *after* factory loading because spec registration populates `source_names`.

The dashboard and backend both import `Instrument` at startup; a typo in `specs.py` (wrong stream_name, missing rename target, unknown dependent source) crashes the process before traffic starts.
Consumer-specific reachability checks (e.g., "every binding's `nexus_path` lies on a `depends_on` chain") move into that consumer's own construction-time check; the registry only validates registry invariants.

## Open questions

1. **Generalising overlay beyond f144.**
   When ev44/tdct streams gain per-stream runtime config (rate hints, decoder choices), each gets a `Stream` subclass.
   If a future ev44/tdct consumer needs its own binding shape (key types other than f144 Sciline keys), a second binding type and field is added.
2. **Codegen drift protection.**
   A CI check that re-runs the generator and fails on diff against the committed `streams_parsed.py` is straightforward but needs a stable input (the geometry file pinned in Pooch).
   The current drift test parameterises over instruments, regenerates from a local coda file when present, and skips otherwise.
