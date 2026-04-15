# Automatic route derivation from workflow specs (v2)

Supersedes `auto-route-derivation.md`. Key change: operates on source names
and filtered `StreamMapping` instead of a `RouteKind` enum, which naturally
supports per-source service splits.

## Problem

Services hard-code which Kafka routes to subscribe to. This causes
unnecessary subscriptions and requires manual bookkeeping when adding
workflows that need new stream kinds (e.g., motion/f144 data).

Workflow specs already declare all their stream dependencies via
`source_names` and `aux_sources`. `StreamMapping` LUTs map
`(topic, source_name)` pairs to the same internal names used in specs.
Routes should follow from this.

## Design

### Gather needed source names from specs

```python
def gather_source_names(
    instrument: Instrument,
    namespace: str,
    source_subset: set[str] | None = None,
) -> set[str]:
    """Collect internal stream names that specs in `namespace` depend on.

    Parameters
    ----------
    source_subset:
        If given, only include specs whose primary sources overlap with this
        set. Their full dependencies (including aux sources) are still added.
        Use this for per-source service splits.
    """
    names: set[str] = set()
    for spec in instrument.workflow_factory.values():
        if spec.namespace != namespace:
            continue
        spec_sources = set(spec.source_names)
        if source_subset is not None and not (spec_sources & source_subset):
            continue
        names |= spec_sources
        if spec.aux_sources:
            for aux_input in spec.aux_sources.inputs.values():
                names.update(aux_input.choices)
    return names
```

### Resolve logical to physical stream names

Workflow specs use logical names (e.g. `unified_detector`), which may not
appear in the `StreamMapping` LUTs. For example, Bifrost merges 45 physical
streams (`arc0_triplet0`, ...) into one logical detector. When a logical
name is in `instrument.detector_names` (or `monitors`) but not in any LUT,
all physical names from that category are included.

```python
def resolve_stream_names(
    needed: set[str],
    instrument: Instrument,
    stream_mapping: StreamMapping,
) -> set[str]:
    known = stream_mapping.all_stream_names
    resolved = needed & known
    unknown = needed - known
    if not unknown:
        return resolved
    logical_detectors = set(instrument.detector_names)
    logical_monitors = set(instrument.monitors)
    if unknown & logical_detectors:
        resolved |= set(stream_mapping.detectors.values())
    if unknown & logical_monitors:
        resolved |= set(stream_mapping.monitors.values())
    return resolved
```

### Filter StreamMapping to needed names

```python
# StreamMapping
def filtered(self, needed: set[str]) -> StreamMapping:
    """Return a copy containing only entries whose internal names are in `needed`."""
    return StreamMapping(
        detectors={k: v for k, v in self.detectors.items() if v in needed},
        monitors={k: v for k, v in self.monitors.items() if v in needed},
        area_detectors={k: v for k, v in self.area_detectors.items() if v in needed},
        logs={k: v for k, v in self.logs.items() if v in needed} if self.logs else None,
    )
```

This is the core mechanism. The filtered mapping determines both *which
kinds* of routes to add and *which topics* to subscribe to.

### Build routes from non-empty LUTs

```python
# RoutingAdapterBuilder
def with_routes_from_mapping(self) -> Self:
    if self._stream_mapping.detector_topics:
        self.with_detector_route()
    if self._stream_mapping.area_detector_topics:
        self.with_area_detector_route()
    if self._stream_mapping.monitor_topics:
        self.with_beam_monitor_route()
    if self._stream_mapping.log_topics:
        self.with_logdata_route()
    return self
```

No enum dispatch — the mapping structure already encodes which adapter
type each LUT needs.

### Service usage (whole-instrument)

```python
def make_detector_service_builder(...) -> DataServiceBuilder:
    stream_mapping = get_stream_mapping(instrument=instrument, dev=dev)
    instrument_obj = instrument_registry[instrument]
    instrument_obj.load_factories()

    needed = gather_source_names(instrument_obj, 'detector_data')
    needed = resolve_stream_names(needed, instrument_obj, stream_mapping)
    scoped = stream_mapping.filtered(needed)

    adapter = (
        RoutingAdapterBuilder(stream_mapping=scoped, stream_counter=stream_counter)
        .with_routes_from_mapping()
        .with_livedata_commands_route()
        .with_run_control_route()
        .build()
    )
    ...
```

Infrastructure routes (`commands`, `run_control`) are always added
explicitly — every service needs them regardless of specs.

### Service usage (per-source split)

DREAM splits detector data across independent topics. A dedicated worker
for the mantle detector:

```python
needed = gather_source_names(
    instrument_obj, 'detector_data', source_subset={'mantle_detector'}
)
# {'mantle_detector', 'motion_cabinet_2', ...} — includes aux deps
needed = resolve_stream_names(needed, instrument_obj, stream_mapping)
scoped = stream_mapping.filtered(needed)
# scoped.detector_topics = {'dream_detector_mantle'} — only the relevant topic

adapter = (
    RoutingAdapterBuilder(stream_mapping=scoped, stream_counter=stream_counter)
    .with_routes_from_mapping()
    .with_livedata_commands_route()
    .with_run_control_route()
    .build()
)
```

The `source_subset` parameter is the only new knob. Topic selection and
adapter wiring follow from the filtered mapping.

## Detector sharding

For instruments like DREAM where detector data is split across independent
Kafka topics, a single `detector_data` (or `data_reduction`) service may
become a bottleneck. Sharding lets you run multiple workers, each handling
a subset of detectors.

### CLI interface

```
python -m ess.livedata.services.detector_data \
    --instrument dream --num-shards 3 --shard 1
```

`--num-shards 1 --shard 0` (the default) is the whole-instrument case —
existing deployments don't change.

Only `detector_data` and `data_reduction` support sharding, since only
detector topics are split per-source in practice.

### Shard assignment

The sharding key is `instrument.detector_names`, which provides a stable
ordered list of primary detector sources:

```python
def get_source_subset(
    detector_names: list[str],
    num_shards: int,
    shard: int,
) -> set[str]:
    return set(sorted(detector_names)[shard::num_shards])
```

This feeds directly into `gather_source_names(..., source_subset=...)`.
Aux dependencies (motion/f144 streams) are pulled in automatically — if
two shards need the same f144 topic, each subscribes independently
(Kafka handles fan-out).

### Effectiveness constraint

Sharding is effective only when the sharded sources live on separate Kafka
topics. If two detector sources share a topic, they must land in the same
shard (the consumer receives both regardless). The service should warn at
startup if sources in different shards share a topic.

### Dashboard compatibility

The dashboard subscribes to output topics, which are the same regardless
of how many shards produced the data. No dashboard changes needed.

## ROI streams

ROI aux sources (`roi_rectangle`, `roi_polygon`) are not in any
`StreamMapping` LUT — they live on a dedicated `livedata_roi` topic.

Recommended approach: detect ROI names in `gather_source_names` output
and add the ROI route explicitly. Making ROI first-class in `StreamMapping`
would require inventing a fake `InputStreamKey` for a topic that doesn't
follow the `(topic, source_name)` keying convention.

## What this replaces

- Per-service hard-coded route lists
- No `RouteKind` enum needed — `StreamMapping` LUT structure already
  encodes the adapter-type distinction
- Single source of truth: specs declare dependencies, routes follow

## Why not `RouteKind`?

A `RouteKind` enum (DETECTOR, MONITOR, LOGDATA, …) is a lossy
intermediate: it tells you *which category* of route to add but discards
*which sources within that category*. For whole-instrument services this
works, but per-source splits need the source-level information that
`RouteKind` throws away. Operating on source names + filtered
`StreamMapping` handles both cases with one mechanism.
