# Per-instrument stream registry

## Overview

Each instrument carries a single source of truth for streaming data declarations: one record per group that the ESS filewriter would write into a NeXus file.
NeXus-rich instruments auto-generate the bulk of the registry from a geometry artifact, with code-side overrides for the few names that need to differ.

Two routing-layer consumers shaped the design:

- *Dynamic detector transforms* drive entries in a `depends_on` chain from live f144 streams.
- *Chopper workflows* drive per-chopper geometry assembly from live f144 setpoint streams, including synthesised streams that are emitted in-process rather than read from Kafka.

Both need a stream → Sciline-key routing layer with per-consumer scoping.

## Shape

### `Stream` and `F144Stream` records

`Stream` is a frozen dataclass for any streaming group; `F144Stream` subclasses it for the f144 writer module and adds `units`.
The base carries the NeXus identity (`nexus_path`, `nx_class`) and the wire identity (`topic`, `source`) — the facts the filewriter collocates as group attrs at a specific HDF5 path.

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Stream:
    writer_module: str
    nexus_path: str | None = None
    topic: str | None = None
    source: str | None = None
    nx_class: str = ''


@dataclass(frozen=True, slots=True, kw_only=True)
class F144Stream(Stream):
    units: str | None = None
    writer_module: str = 'f144'
    nx_class: str = 'NXlog'
```

`Instrument.streams: dict[str, Stream]` holds streams of all writer modules, keyed by the instrument-facing *stream name*.
The name is the routing handle for every non-kafka-boundary consumer; `topic` / `source` are read only by the Kafka boundary (`streams.py` per instrument, `message_adapter`, kafka source).
Per-kind views are filters (`instrument.f144_streams` returns `dict[str, F144Stream]`).

`nexus_path` is on the base because consumers that splice live values into a NeXus structure — transform-chain patching, chopper `DataGroup` assembly — dispatch on it.
It is None only for synthesised streams.

### Synthesised streams

A synthesised stream (no NeXus row, emitted in-process by a synthesiser) is `Stream(topic=None, source=None, nexus_path=None, ...)`.
The predicate "synthesised" is `stream.topic is None`.
`__post_init__` enforces that `topic` and `source` are both None or both set.

Consumers are oblivious to the distinction: workflow harnesses, timeseries UI, and binding resolution all index into `streams` by name.
The only place that filters by `topic is None` is the Kafka-boundary `StreamLUT` builder, which has no entry to emit for a stream that never arrives over Kafka.

### `LogContextBinding`

The runtime overlay — wiring f144 streams into Sciline workflows — is a separate list referencing streams by name:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class LogContextBinding:
    stream_name: str            # references Instrument.streams[stream_name]
    workflow_key: Any           # Sciline key type (e.g. InstrumentAngle[SampleRun])
    dependent_sources: frozenset[str]  # spec source_names that pick up this binding
```

`Instrument.log_context_bindings: list[LogContextBinding]`.
A binding declares only "stream X feeds Sciline key K, scoped to dependents D"; it does not mirror `units` or `nexus_path`, which live on the stream record it references.

Bindings live separately from the record because `dependent_sources` and `workflow_key` belong to *a binding*, not to a stream: a single stream can participate in multiple bindings with different scoping (e.g., feeding two Sciline workflows triggered by different sources).
A field on the stream record would force one global answer.

Factories derive `StreamProcessorWorkflow.context_keys` by calling `instrument.get_context_keys(source_name)`, which filters the binding list by `dependent_sources`.
The aux-source routing mechanism iterates `log_context_bindings` once and filters by `dependent_sources & spec.source_names`.

The same binding type covers both known consumers:

- *Dynamic-transforms*: the detector spec picks up the live readback as an aux input, keyed by a `TransformValueLog` subclass, and patches it into the transform chain.
- *Chopper-workflow*: a chopper workflow spec is triggered by a synthetic `chopper_cascade_in_phase` stream; every chopper f144 stream declares itself as an aux input to that trigger and feeds a typed Sciline key.

Dispatch between the two is driven by the `workflow_key` type and the Sciline graph downstream, not by separate binding kinds.

### Display info: `source_metadata`, not `Stream`

`title` and `description` live in `Instrument.source_metadata: dict[str, SourceMetadata]`, the side-dict that already holds display info for detectors and monitors.
`Stream` is data plumbing; display labels have a different audience and lifecycle, and merging them onto the record would pull human-readable strings into every record-equality test, override path, and parser fixture.

## Codegen seam

For instruments with O(100) NXlogs the registry is bootstrapped *offline* by the CLI generator `python -m ess.livedata.nexus_helpers <geometry.nxs> --generate`.
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

`name_streams` auto-suggests an instrument-facing name from the tail of each NeXus path (generic container groups like `entry` / `instrument` / `sample` are filtered out) and accepts a `rename=` map keyed by `nexus_path`.
It raises if a rename key matches no parsed entry, or if the resulting names collide.

Codegen rather than runtime parsing: backend and dashboard both import the per-instrument config module at startup, and the dashboard has no file-access path.
Parsing at runtime would force a Pooch download with cold-start dependencies on cache state and network.
Codegen keeps both processes file-free at startup and makes the parser output reviewable as code.
`Stream` is a frozen dataclass that round-trips through any serialisation layer, so moving to dynamic distribution later is a local change to who populates `Instrument.streams`.

For instruments with few streams (today: `dummy`), the registry is a hand-written `dict[str, F144Stream]` literal and the codegen path is not used.

## Construction-time invariants

Misconfiguration in the registry, bindings, or overrides crashes the process at startup, not at pytest time:

- `Instrument.__post_init__` raises if any `log_context_bindings[i].stream_name` is missing from `Instrument.streams`.
- `name_streams` raises if a rename key matches no parsed entry, or if names collide.
- End of `Instrument.load_factories()` raises if any `log_context_bindings[i].dependent_sources` element does not match the `source_names` of some registered spec.
  This runs *after* factory loading because spec registration populates `source_names`.

The dashboard and backend both import `Instrument` at startup, so any typo in `specs.py` (wrong stream_name, missing rename target, unknown dependent source) crashes the process before traffic starts.

Consumer-specific reachability checks (e.g., "every binding's `nexus_path` lies on a `depends_on` chain") are the consuming module's responsibility, not the registry's.
The registry validates registry invariants only.

## Open questions

1. **Generalising overlay beyond f144.**
   When ev44/tdct streams gain per-stream runtime config (rate hints, decoder choices), each gets its own `Stream` subclass.
   A consumer that needs a binding shape other than the f144 triple `(stream_name, workflow_key, dependent_sources)` gets its own binding type — `LogContextBinding` unifies f144 Sciline-key consumers, not all writer modules.
2. **Codegen drift protection.**
   `streams_parsed.py` is regenerated by hand; staleness against the latest geometry file is not caught automatically.
   An earlier drift test was dropped because it skipped in CI (the geometry files are not in Pooch) and so gave false confidence without catching the realistic failure mode.
   A real check needs a stable geometry-file input pinned in Pooch first.
