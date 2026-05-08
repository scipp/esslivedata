# Dynamic NeXus transforms: design proposal (v3)

Issue #922 surfaces a structural problem larger than its specific failure: the
geometry artifact represents dynamic geometry as length-0 NXlog placeholders,
and any workflow that loads a NeXus chain through one of these placeholders
crashes at construction time unless explicit wiring replaces the empty NXlog
with a live f144 sample.

Today this wiring exists only inside `DetectorViewFactory`, keyed by
`source_name`. As a result:

* The I(Q) factory does not install it and crashes against the
  `2026-05-08` LOKI artifact (issue #922).
* The LOKI monitor wavelength path does not install it either; it crashes
  identically for `beam_monitor_m4` (the unfinished tail of #780/#782 — the
  monitor infrastructure landed in `8c423664`, but no LOKI factory uses it
  yet).
* Every future SANS/monitor variant that walks a depends_on chain through an
  NXlog placeholder will repeat the same trap, silent until enabled.

This document proposes a single model and a small set of helpers, replacing
the current per-source wiring.

## Goals

1. **One source of truth per instrument.** Which NXlog nodes are dynamic, and
   which f144 stream drives each — declared once, reused by every workflow
   that loads geometry from the artifact.
2. **Mirror the depends_on graph.** Wiring is keyed by NXlog path, not by
   `source_name`. If two components share an NXlog along their chain, one
   declaration covers both.
3. **No silent wrong values.** Empty / missing logs raise; baked-in defaults
   in the artifact are never used. Typo'd registry paths fail loudly.
4. **Multiple dynamic transforms in one workflow.** A single workflow can load
   several components, each with its own dynamic NXlog (e.g., a SANS workflow
   loading `loki_detector_0` _and_ `beam_monitor_m4`).
5. **No per-factory boilerplate beyond a single function call.**
6. **Loud failures on either side of the routing/wiring split.** If the
   factory-side helper is missing or the spec-side aux sources are missing,
   the symptom must be a clear error at finalize, never a silently wrong
   plot.

## What is rejected

* **Per-factory wiring as a long-term pattern.** Smallest delta today but
  invites whack-a-mole; coupling with three places. Issue #922 happens
  precisely because this pattern was followed and silently incomplete.
* **Preserving a single-sample default in the artifact.** Conflicts with
  Goal 3.
* **A single `Mapping`-typed Sciline parameter aggregating all logs**
  (considered in v2). Collides with `StreamProcessorWorkflow` (SPW;
  `handlers/stream_processor_workflow.py`) and its stateful per-Sciline-key
  context: a partial batch carrying only one stream would replace the whole
  mapping and nuke other streams' previously-set values. Either SPW grows
  per-element merge semantics for shared keys (a real behavioural change),
  or every binding gets its own Sciline key. The latter is simpler and
  preserves SPW unchanged.
* **Inferring loaded components from the pipeline's set `NeXusName[X]`
  parameters.** Set parameters survive in the Sciline graph regardless of
  whether the active providers actually consume them, so inference would
  spuriously patch chains nothing reads (and trip validation on irrelevant
  paths). Factories declare their loaded component types explicitly.

## Author-facing model

### Per-instrument registry

A `DynamicTransformBinding` describes the binding between an NXlog node along
a depends_on chain and the f144 stream that drives it. The author declares
one Sciline key per binding by subclassing the shared `TransformLog`
dataclass, so the key is grep-able, IDE-visible, and type-safe (no
`NewType`-with-union awkwardness).

```python
@dataclass(frozen=True, slots=True)
class TransformLog:
    """Latest NXlog samples for a dynamic-transform binding.

    Subclass to create a distinct Sciline key per binding. ``log`` is
    ``None`` before the first ``set_context`` call; otherwise it is the
    NXlog produced by ``ToNXlog`` (may still be empty if no f144
    message has arrived).
    """
    log: sc.DataArray | None = None

class DetectorCarriageLog(TransformLog): pass
class BeamMonitorM4PositionLog(TransformLog): pass

@dataclass(frozen=True, slots=True)
class DynamicTransformBinding:
    nxlog_path: str                   # absolute NeXus path of the placeholder NXlog
    stream_name: str                  # name of the f144 aux stream
    log_key: type[TransformLog]       # subclass used as Sciline key for the patched chain provider
    consumers: frozenset[str]         # source names whose chain walks through nxlog_path
```

`consumers` is declared explicitly by the author. It is the source of truth
for "which spec gets this aux source merged into it" (see Aux-source
derivation, below); the CI artifact validator confirms it matches the actual
depends_on graph.

`Instrument` carries

```python
dynamic_transforms: list[DynamicTransformBinding] = field(default_factory=list)
```

For LOKI in this PR (#922 scope):

```python
LOKI_DYNAMIC_TRANSFORMS = [
    DynamicTransformBinding(
        nxlog_path='/entry/instrument/detector_carriage/value',
        stream_name='detector_carriage',
        log_key=DetectorCarriageLog,
        consumers=frozenset({'loki_detector_0'}),
    ),
]
```

`beam_monitor_m4` wavelength is a known follow-up. The artifact represents
its position via a separate empty NXlog (`.../trans_20`) rather than
`detector_carriage/value`; resolving requires either updating
`make_geometry_nexus.py` so m4's depends_on shares the carriage NXlog (in
which case adding `'beam_monitor_m4'` to the existing carriage binding's
`consumers` is sufficient — no second binding) or registering a separate
m4 position f144 stream. Out of scope here; the new infrastructure is
designed so that follow-up is a one-line change to the registry, not a
re-architecture.

This is the only place an instrument author declares dynamic geometry. The
current `LOKI_DYNAMIC_TRANSFORMS` dict, the per-source wiring inside
`DetectorViewFactory(dynamic_transforms=...)`, and the
`DetectorROIAuxSources(dynamic_transforms=...)` plumbing all collapse into
this registry.

### Validation against the artifact

A registry entry that does not match the artifact (typo'd `nxlog_path`,
orphan `nxlog_path` not reached from any chain, `consumers` set diverging
from the actual chain walk) is a configuration bug. A CI-level test (one
parameterisation per instrument with `dynamic_transforms` populated):

* For each `binding` in the registry, walk depends_on starting from each
  declared consumer in the artifact. Confirm the binding's `nxlog_path`
  appears on every consumer's chain.
* Conversely, walk depends_on from every component in the artifact and
  flag any empty NXlog encountered that is not covered by a binding.
* Reject duplicate `log_key`s in the registry — Sciline collapses two
  parameters of the same key, silently merging two bindings.

Plain key-existence checks would let an orphan binding pass CI. No runtime
validation — duplicates the CI check on every workflow construction.

### Aux-source derivation

`AuxSources` entries for dynamic streams are derived from the registry,
scoped per spec by the binding's `consumers` set:

```python
def dynamic_transform_aux_sources(
    instrument: Instrument, source_names: Iterable[str]
) -> AuxSources:
    """AuxSources covering every binding whose consumers intersect source_names."""
```

`register_detector_view_spec` and `register_monitor_workflow_specs`
automatically merge this into the spec's aux sources, passing the spec's
`source_names`. The merge composes with any caller-supplied
`aux_sources` (e.g. LOKI's chopper f144 streams already passed to
`register_monitor_workflow_specs`); it is not a default-fill that gets
overridden when the caller passes their own. Each spec only carries the
streams its sources actually need — m0–m3 do not show a phantom
`detector_carriage` aux input in the dashboard. Authors do not thread
anything through spec construction.

This automatic merge guarantees Goal 6 from the routing side: if the
factory-side helper is wired, the routing layer will deliver the f144
messages.

### Workflow-side helper

```python
def apply_dynamic_transforms(
    workflow: sciline.Pipeline,
    *,
    instrument: Instrument,
    component_types: Iterable[type],
) -> dict[str, sciline.typing.Key]:
    """Patch the pipeline to drive matching NXlog placeholders from f144 streams.

    For each component type in ``component_types``, reads the corresponding
    ``NeXusName[T]`` value(s) set on ``workflow``, walks the depends_on
    chain in the geometry artifact (resolved via ``Filename[SampleRun]``
    set on the workflow), and intersects every walked transformation entry
    with ``instrument.dynamic_transforms``. For each component type with
    matches, replaces the ``NeXusTransformationChain[T, SampleRun]``
    provider with one that consumes the matched bindings'
    ``log_param``s and writes their latest sample into the chain.

    Returns the ``context_keys`` mapping (stream_name -> log_key) that
    the factory passes to ``StreamProcessorWorkflow``.

    If the chain walk for a given component type encounters any empty NXlog
    that is not matched by a registry binding, raises with a registry-aware
    message naming the offending NXlog path — operators should not chase a
    bare ``reject_time_dependent_transform`` from essreduce.
    """
```

`component_types` is explicit, not inferred — the factory already knows
which component types its workflow loads. For SANS:
`(NXdetector, NXmonitor)`. For detector view: `(NXdetector,)`. For monitor:
`(NXmonitor,)`.

The helper reads `Filename[SampleRun]` from the workflow rather than
re-accepting it, removing one drift surface.

### Factory shape (after)

```python
@specs.i_of_q_handle.attach_factory()
def _i_of_q_factory(source_name, params, aux_source_names):
    wf = _make_base_workflow()
    wf[NeXusDetectorName] = source_name
    wf[NeXusMonitorName[Incident]] = aux_source_names['incident_monitor']
    wf[NeXusMonitorName[Transmission]] = aux_source_names['transmission_monitor']
    ...
    context_keys = apply_dynamic_transforms(
        wf, instrument=instrument, component_types=(NXdetector, NXmonitor)
    )
    return StreamProcessorWorkflow(
        wf,
        dynamic_keys=_dynamic_keys(source_name),
        context_keys=context_keys,
        target_keys=target_keys,
        accumulators=_accumulators,
    )
```

`DetectorViewFactory` migrates to the same shape (and drops its
`dynamic_transforms` constructor argument). `_monitor_workflow_factory`
adopts it for wavelength mode, completing the #782 last mile.

## Failure-mode analysis (Goal 6)

Three places must agree per binding: registry entry, spec-side aux source,
factory-side helper call. The first is automatic if the registry is the
single source of truth; the second is automatic via
`register_*_workflow_specs`. Only the third is a manual touch per factory.

| State | Symptom |
|---|---|
| Helper called, aux source missing | f144 messages are never routed to the workflow → patched provider's `log_param` stays `None` → finalize raises `ValueError("No samples yet for ...")`. Loud. |
| Aux source wired, helper not called | The chain still walks the empty NXlog → essreduce's `to_transformation` raises `reject_time_dependent_transform`. Loud (this is exactly the issue #922 stack). |
| Both wired, no f144 messages yet | Patched provider raises `ValueError("No samples yet for ...")` at first finalize. Loud. |
| Helper called with `component_types` whose chain walks pass through an empty NXlog with no registry match | `apply_dynamic_transforms` raises immediately at workflow construction with a registry-aware message naming the offending NXlog. Loud, and operator is pointed at the right place. |

No path produces a silently wrong result.

## Implementation: multi-transform support

The current core mechanism in `detector_view/workflow.py` supports exactly
one dynamic transform per workflow (`TransformName` is a scalar
`NewType`). Generalising:

* For each requested component type `T` with at least one matching binding,
  `apply_dynamic_transforms` builds a **separately annotated** closure with
  concrete (non-generic) annotations and inserts it via `wf.insert(...)`.
  Sciline displaces essreduce's generic `get_transformation_chain` for that
  specific `(NeXusComponent[T, SampleRun], NeXusTransformationChain[T, SampleRun])`
  instantiation only. The closure form:

  ```python
  def _patched_chain(component, log_a, log_b, ...):
      chain = get_transformation_chain(component)
      patched = deepcopy(chain)
      for binding, container in zip(matched, [log_a, log_b, ...]):
          if binding.nxlog_path not in patched.transformations:
              continue
          # container is None before the first set_context
          # (essreduce.streaming.StreamProcessor.__init__ pre-sets every
          # context_key to None — see streaming.py:363-365). After
          # set_context, container is the binding's TransformLog
          # subclass with log either None or an NXlog DataArray.
          if container is None or container.log is None \
                  or container.log.sizes.get('time', 0) == 0:
              raise ValueError(
                  f"No samples yet for {binding.stream_name!r} "
                  f"(transform {binding.nxlog_path!r})"
              )
          log = container.log
          patched.transformations[binding.nxlog_path].value = \
              log['time', -1].data
      return patched

  _patched_chain.__annotations__ = {
      'component': NeXusComponent[T, SampleRun],
      **{f'log_{i}': b.log_key for i, b in enumerate(matched)},
      'return': NeXusTransformationChain[T, SampleRun],
  }
  ```

  Note that the closure must use **explicit positional parameters**
  (`def _patched(component, log_0, log_1, ...)`) — not `*logs`. Sciline
  resolves dependencies by annotation name, and a `*logs` variadic does
  not bind the `log_0`, `log_1`, ... entries in `__annotations__`. The
  helper builds the closure body via `exec` on a per-arity template, or
  via `functools.partial` / a closure-builder per-arity; either way the
  positional shape must match the annotation keys exactly.

  Annotations are set explicitly on the closure object — not via
  `from __future__ import annotations` strings — to avoid forward-reference
  resolution interacting badly with closure-captured types.
* The current single-transform code expresses the dataflow in two Sciline
  steps: `TransformValueLog → TransformValue → NeXusTransformationChain`.
  The patched closure above **fuses** the two: it extracts the latest
  scalar from each container's `.log` and writes it onto the chain in one
  pass. Per-binding subclasses are therefore needed only on the log side
  (`TransformLog` subclasses); a mirrored `TransformValue`-per-binding
  hierarchy is *not* introduced — the intermediate Sciline node would have
  exactly one consumer and adds nothing. If a future consumer ever needs
  the bare scalar (e.g. a status display of the carriage position
  independent of the patched chain), splitting it back is a small,
  binding-local refactor.
* Each binding's `log_key` is a distinct Sciline key (a `TransformLog`
  subclass). SPW gains a small wrapping rule in `accumulate`: if a
  `context_keys` value is **a class** and a `TransformLog` subclass, the
  raw NXlog payload is wrapped via `key(log=raw)` before `set_context`;
  otherwise the raw value is passed through unchanged (preserving the
  existing path for ROI request streams etc., which today are `NewType`
  aliases of `sc.DataArray`).

  ```python
  context[sciline_key] = (
      sciline_key(log=raw)
      if isinstance(sciline_key, type) and issubclass(sciline_key, TransformLog)
      else raw
  )
  ```

  The `isinstance(sciline_key, type)` guard is non-negotiable: `NewType`
  instances and parameterised generics are not classes and `issubclass`
  raises `TypeError` on them. This is *not* a behavioural change to
  stateful context semantics — each binding still has its own Sciline
  key, so the v2 partial-batch concern does not apply.
* One patched provider per `(T, SampleRun)` composes naturally for workflows
  loading multiple component types.

The existing `TransformValueStream`, `TransformValue`, `TransformName`,
`TransformValueLog` (the awkward `NewType('TransformValueLog', sc.DataArray | None)`),
`get_transformation_chain_with_value`, `transform_value_from_log`, and
`add_dynamic_transform` are deleted in the same PR. No external API used
these.

## Pre-merge verification

* Confirm essreduce's `get_transformation_chain` is the sole provider of
  `NeXusTransformationChain[X, SampleRun]` for any X used by LOKI workflows
  (so `wf.insert(patched)` cleanly replaces it). If a competing provider
  exists, the patched-replacement strategy fails.
* Confirm `wf[Filename[SampleRun]]` is reliably set before
  `apply_dynamic_transforms` runs in every factory. (LOKI factories already
  do; verify for any other instrument adopting this.)
* End-to-end smoke test of the closure pattern, **verified at proposal
  finalisation** (see `.scratch/`):
  * (a) `wf.insert(_patched)` displaces essreduce's
    `get_transformation_chain` for the specific `(T, SampleRun)`
    instantiation. **Confirmed** against `LokiWorkflow`.
  * (b) Sciline resolves the closure's `__annotations__` including
    closure-captured generic return type
    (`NeXusTransformationChain[NXdetector, SampleRun]`). **Confirmed.**
  * (c) `StreamProcessor.__init__` pre-sets every context key to
    `None` (`streaming.py:363-365`); the patched closure must therefore
    handle `container is None` *before* dereferencing `container.log`.
    **Discovered during smoke test, captured in pseudo-code above.**
    Sciline does *not* construct a zero-arg default of the declared
    type — the StreamProcessor's `workflow[key] = None` hack is the
    operative behaviour.
  * (d) Closure must use explicit positional parameters, not `*logs`,
    or Sciline does not bind the per-`log_N` annotations. **Discovered
    during smoke test.**

## Migration / scope

Single PR:

1. New `DynamicTransformBinding` type + `Instrument.dynamic_transforms`
   field.
2. New `TransformLog` base dataclass + `apply_dynamic_transforms` helper
   with multi-transform machinery + small SPW wrapping rule
   (~3 lines: wrap raw NXlogs into the binding's `TransformLog` subclass
   before `set_context`).
3. `register_detector_view_spec` and `register_monitor_workflow_specs`
   automatically merge `dynamic_transform_aux_sources(instrument)` into
   the spec's aux sources whenever `instrument.dynamic_transforms` is
   non-empty.
4. LOKI: replace `LOKI_DYNAMIC_TRANSFORMS` (the dict) with the registry
   carrying just the carriage binding (m4 is a documented follow-up).
   Rewire `_xy_projection` and `_i_of_q_factory` to call the helper.
   `_monitor_workflow_factory` rewires only when the carriage binding's
   `consumers` includes a monitor source (i.e. once the artifact /
   stream change for m4 is done — separate PR).
5. Delete the old single-transform machinery, including:
   * `TransformValueStream`, `TransformValue`, `TransformName`,
     `TransformValueLog` from `detector_view/types.py`.
   * `get_transformation_chain_with_value`, `transform_value_from_log`,
     `add_dynamic_transform` from `detector_view/workflow.py`.
   * `DetectorViewFactory(dynamic_transforms=...)` ctor argument.
   * `LOKI_DYNAMIC_TRANSFORMS` dict.
   * `DetectorROIAuxSources(dynamic_transforms=...)` field.
   * `TransformValueStream` re-export from `detector_view/__init__.py`.
   * `tests/handlers/detector_view/transform_value_test.py` (exercises
     deleted providers; replaced by new `apply_dynamic_transforms`
     tests).
6. Tests, in the same PR:
   * `apply_dynamic_transforms`: chain walking, multi-transform within
     and across component types, missing-log error, no-op when no
     bindings match.
   * SPW wrapping rule: `TransformLog` subclass keys get raw NXlog
     wrapped; non-`TransformLog` keys pass through; partial-batch
     semantics preserved per binding (one Sciline key per binding).
   * Roundtrip test for `loki/i_of_q/1` (issue #922).
   * Roundtrip / integration test for monitor wavelength on
     `beam_monitor_m4`.
   * Existing detector-view test coverage retained.
   * CI-level registry-vs-artifact validator test, parameterised per
     instrument with `dynamic_transforms` populated.

## Non-goals / explicitly deferred

* **Readiness gate at job start.** Worth doing — moves the "no f144 yet"
  failure from finalize to a clear error at job start. Orthogonal; track
  separately. The `ValueError("No samples yet for ...")` failure mode is
  loud enough that the gate is ergonomic improvement, not correctness.
* **Static-default fallback for dev / offline use.** Rejected by Goal 3.

## Performance notes

* `apply_dynamic_transforms` walks the artifact at workflow-build time. A
  single h5py open per construction is ~ms; acceptable. Pre-optimisation
  rejected.
* `deepcopy(chain)` per finalize. Chain has a handful of transformations;
  expected to be sub-millisecond. **Quick measurement before merging** to
  confirm; if it shows up, copy only the affected transformation node
  rather than the full chain.
