# ADR 0004: Key workflow inputs uniformly by on-disk stream name

- Status: accepted
- Deciders: Simon
- Date: 2026-06-02

## Context

A message's identifier passes through four naming schemes between Kafka and a
Sciline provider:

- `(topic, source_name)` — raw Kafka identity.
- **role** — the logical input a workflow author and a user reason in for
  auxiliary data ("the incident monitor used for normalisation"), which may be
  filled by any of several physical streams.
- **on-disk `stream_name`** — the canonical instrument-config name a stream is
  given once the `StreamLUT` has resolved it on ingress.
- **`sciline.Key`** — the graph's type-addressed input (e.g.
  `NeXusData[Incident, SampleRun]`).

The three kinds of workflow input (primary detector data, dynamic auxiliary
data, context — ADR 0002/0003) all arrive on the data path and all end up inside
the wrapped `StreamProcessor`, differing only in how their stream binding is
chosen and how they are fed into the graph.

Dynamic aux used to be keyed in `dynamic_keys` by its stable role (e.g.
`incident_monitor`), while the primary input and context inputs were keyed by
on-disk stream name. That asymmetry leaked the role scheme onto the data path:
the role→stream map had to be threaded into `wire_dynamic_transforms` (to match
bindings against on-disk `dependent_sources`) and into the `Job`'s runtime
routing (a `_stream_to_fields` remap turning incoming stream-named data back into
roles before handing it to the workflow). Two layers spoke a scheme that only
mattered at construction.

## Decision

Key **every** dynamic input — primary, aux, context — uniformly by its on-disk
`stream_name` on the data path. The role scheme is resolved to a stream name
exactly once, at workflow construction, and never enters the running workflow
object.

Translation between schemes happens at three points, only two of which are on
the data path:

| # | Boundary | From → To | Performed by | Path |
|---|---|---|---|---|
| 1 | Kafka ingress | `(topic, source_name)` → `stream_name` | `MessageAdapter.get_stream_id` + `StreamLUT` | data |
| 2 | Workflow construction | role → `stream_name` (building `dynamic_keys`) | instrument factory, inline `aux_source_names[role]` | build-time |
| 3 | Workflow runtime | `stream_name` → `sciline.Key` | `StreamProcessorWorkflow.accumulate` / `set_context` (dict lookup) | data |

`JobFactory.create` computes the role→stream map (`aux_source_names`) from
`AuxSources.render` — instrument defaults overlaid with the user's selection —
and passes it to the factory. The factory keys `dynamic_keys` by on-disk stream
name directly, resolving each role inline (`aux_source_names['incident_monitor']:
NeXusData[Incident, SampleRun]`). An unknown role raises `KeyError` at job
creation, visible in factory code. The primary input is the degenerate case: its
binding is the job's identity (`source_name`), so translation 2 is the identity
function and the factory keys it directly without `aux_source_names`. Context
wire names equal their stream names (ADR 0003), so translation 2 does not touch
them either — it concerns auxiliary inputs only.

Consequently the `Job` carries no mapping: `input_streams` and `gating_streams`
are plain `set[str]` of stream names, and `Job.add` forwards
`{**primary_data, **aux_data}` to the workflow unchanged. The only place a
`stream_name` becomes a `sciline.Key` is the dict lookup inside
`StreamProcessorWorkflow.accumulate`, where which dict the name is found in
(`dynamic_keys` vs `context_keys`) decides `accumulate` vs `set_context`.

Because the routing layer must inject per-job context and f144-driven transforms
*after* the factory has produced the workflow but *before* the wrapped
`StreamProcessor` is built, `StreamProcessorWorkflow` defers construction behind
`SupportsContext.build(context_keys, chain_patch_bindings)`. The factory passes
only its own internal context (e.g. ROI) to the constructor; `build` merges the
routing-layer `context_keys`, wires `chain_patch_bindings`, and bakes the graph.

One-stream-to-many-roles multiplexing is dropped. It was unused and is
incompatible with on-disk keying: a single incoming `stream_name` maps to exactly
one `dynamic_keys` entry.

## Alternatives considered

| Option | Notes |
|---|---|
| **Uniform on-disk keying, role resolved at construction (chosen)** | One scheme on the data path; `Job` is a plain name set; role lives only in factory code where an unknown role fails loudly at job creation. |
| Keep aux keyed by role; remap `stream_name` → role in `Job` | The prior state. Forces the role map into `wire_dynamic_transforms` and a `Job._stream_to_fields` remap on every batch; two data-path layers carry a scheme that only matters at build time. Rejected. |
| Resolve roles at runtime in `Job`/`JobManager` | Moves a construction-time concern onto the hot path and re-introduces the remap. Rejected. |
| Preserve one-stream-to-many-roles multiplexing | Unused, and irreconcilable with a one-to-one `stream_name` → `dynamic_keys` mapping. Rejected. |

## Key design choices

### Only one naming scheme is on the data path

Translation 2 happens at construction. Once messages flow, the entire stack from
`WorkflowData.data` down to the `StreamProcessorWorkflow` dict lookup speaks a
single language — `stream_name`. Only the innermost shell (`StreamProcessor` /
Sciline) speaks `sciline.Key`. This is the invariant that lets `Job` hold plain
name sets and forward data untouched.

### Role resolution is a build-time concern, co-located with the factory

The role scheme is the one a workflow author and a user reason in. Resolving it
in the factory that constructs the workflow — the same module that declares the
`dynamic_keys` — keeps role knowledge local and makes a typo a loud `KeyError` at
job creation rather than a silent mis-route.

### Two independent sources must agree on the on-disk name

The `StreamLUT` drives translation 1 on ingress; the `render` / `aux_source_names`
map drives translation 2 in the factory. Nothing cross-checks them: an aux source
whose rendered `stream_name` never appears on ingress simply never delivers data.
This is the price of decoupling routing from construction; it is acceptable
because both sides derive from the same instrument stream definitions.

## Consequences

- `Job.input_streams` / `gating_streams` are `set[str]`; `Job.add` forwards
  primary and aux data to the workflow unchanged, with no remap.
- `dynamic_keys` is keyed by on-disk stream name for every input; the role scheme
  never enters the workflow object.
- `wire_dynamic_transforms` takes only the workflow and the resolved
  `ChainPatchBinding` records — no `aux_source_names` — because it reads component
  types off `dynamic_keys`, which is already stream-name-keyed (ADR 0003).
- `StreamProcessorWorkflow` defers its `StreamProcessor` construction behind
  `SupportsContext.build`, so the routing layer can inject per-job context and
  chain-patch bindings before the graph is baked.
- One-stream-to-many-roles multiplexing is no longer supported.
- The naming model and its translation points are illustrated in
  :doc:`/developer/design/stream-keying`.
