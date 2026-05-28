# Block B: Implement ADR 0003 (unified ContextBinding declaration model)

Builds on the Block A commits on `jobmanager-context-gate` (commits c16e5c24,
7424b02c, cda8ba5f) that implement the JobManager gate per ADR 0002 with
transitional carriers (`Workflow.context_keys` protocol member,
`AuxSources.initial_context_messages` hook, `DetectorROIAuxSources`,
`bifrost_aux_sources`). This plan replaces those transitional carriers with
first-class `ContextBinding` declarations as specified in ADR 0003.

Reviewed by senior-engineer-review (decisions baked in below; risk register
trimmed to what remains open).

## What changed during implementation

The plan below is the *pre-implementation* document. Three conceptual shifts
emerged while running the code:

1. **B3/B4 split.** The original B3 deleted legacy AuxSources while
   `JobFactory` still consumed them. Split into B3 (purely additive: declare
   new records) and B4 (atomic swap + cleanup). This split is reflected in
   the revised B3/B4 sections below.
2. **Spec-scope for motion context.** The ADR (and this plan) initially placed
   bifrost rotation and LOKI carriage at instrument scope. Implementation
   showed that "every workflow on this source" is the wrong filter — some
   specs on the same source legitimately do not consume motion (LOKI
   `tube_view`, bifrost detector view / ratemeter). Both moved to spec scope.
   ADR 0003 updated to match in a later commit.
3. **Routing-pickup folded into B4.** B7's `gather_source_names` extension was
   originally sequenced after B4. Implementation showed that leaves LOKI tests
   red in the interim (the old `aux_input.choices` pickup goes away in B4).
   The extension landed inside B4. Final commit count: 7, not 8.

A few smaller drops worth knowing: `Instrument.get_context_keys` had no
production callers after the bifrost migration and was deleted;
`SpecRequirements.requires_aux_sources` was YAGNI'd; `WorkflowFactory.create`
gained `context_keys` as a fourth opt-in kwarg (a fifth would warrant
rethinking the pattern itself).

Two follow-ups noted for separate PRs:
- Document `ContextBinding`'s resolver-purity assumption (name-suffixing) in
  its docstring — B8's collision check depends on it.
- Revisit the `WorkflowFactory.create` opt-in pattern if a fifth kwarg appears.

## Final-state target (post-Block B)

- `LogContextBinding` renamed to `ContextBinding` (`config/stream.py`),
  with optional `stream_resolver` and `seed_factory` fields.
- `Instrument.log_context_bindings` → `Instrument.context_bindings`;
  `Instrument.add_log_context_binding` → `Instrument.add_context_binding`.
- `WorkflowSpec.context_bindings: list[ContextBinding]`; `SpecHandle.add_context_binding`
  appends, defaulting `dependent_sources` to `frozenset(spec.source_names)`.
- `JobFactory.create` merges instrument + spec ContextBinding entries filtered by
  source membership, computes `context_keys` (workflow-facing field-name →
  workflow-key) and `context_stream_names` (wire-name set for the gate), and
  emits seed messages. `JobManager.schedule_job` (via `_seed_initial_context`)
  fires the seeds and marks the wire stream names in `_seen_context_streams`.
- The wire stream names are added to `Job.aux_source_names` (mapping field →
  wire) so the existing `_filter_data_for_job` and `_stream_to_fields` remap
  in `Job.add` handle routing without a second mechanism. `context_stream_names`
  remains an explicit `__init__` parameter (a strict subset of
  `aux_source_names.values()`) so the JobManager does not recompute it.
- `Workflow.context_keys` removed from the protocol. `StreamProcessorWorkflow`
  still accepts `context_keys` as a constructor argument (unchanged).
  `AreaDetectorView` and `TimeseriesStreamProcessor` lose their empty stubs.
- `DetectorROIAuxSources` deleted. ROI declared per-spec via
  `spec_handle.add_context_binding(...)` with a resolver
  (`lambda jid, name: f"{jid}/{name}"`) and a `seed_factory`.
- `bifrost_aux_sources` deleted; the three qmap-family `register_spec` calls
  lose the `aux_sources=…` kwarg. The existing instrument-level
  `add_context_binding` calls in bifrost's `factories.py` carry the rotation
  declarations.
- LOKI's `LOKI_DYNAMIC_TRANSFORMS` dual declaration collapses: one
  `instrument.add_context_binding(stream_name='detector_carriage',
  workflow_key=TransformValueLog, dependent_sources=frozenset({'loki_detector_0'}))`
  in `loki/factories.py`. The `transform_name` (NeXus path used by
  `add_dynamic_transform`) does not fit `ContextBinding` and lives in a small
  per-source `dict[str, str]` local to `loki/factories.py` — see ADR 0003 §
  "LOKI transform_name carrier". `DetectorViewFactory.dynamic_transforms`
  constructor parameter is removed; the factory receives `context_keys` as a
  kwarg from `WorkflowFactory.create` and consults the local LOKI dict only
  when `TransformValueLog` appears in `context_keys`.
- `AuxSources.initial_context_messages` hook removed; `AuxSources` slimmed to
  dynamic, user-selectable aux only.
- `Job.context_aux_stream_names` → `Job.context_stream_names`.
- `gather_source_names` learns to include the stream names of instrument-level
  `context_bindings` and spec-level `context_bindings` whose `stream_resolver` is
  `None` (so the bifrost rotation and LOKI carriage f144 streams get subscribed
  by the `detector_data` namespace). Spec-level entries with a resolver are
  job-scoped (ROI) and routed via a dedicated topic, not via
  `gather_source_names`.
- `SpecRequirements.requires_aux_sources` in `dashboard/plotter_registry.py:89-120`
  is removed as YAGNI — the only references are in the deleted test cases
  (see B3).
- Registration-time collision validation: a spec-level `ContextBinding` whose
  resolved wire name collides with an instrument-level one for any (spec,
  source) pair is a registration error.

## Commit sequence

### B1 — Introduce ContextBinding, deprecate LogContextBinding

Mechanical rename + add optional fields. No semantic change.

Files:
- `src/ess/livedata/config/stream.py` — rename `LogContextBinding` to
  `ContextBinding`; add `stream_resolver: Callable[[JobId, str], str] | None = None`
  and `seed_factory: Callable[[JobId], Message] | None = None` (both default
  preserve current behaviour). Keep the class `@dataclass(frozen=True, slots=True,
  kw_only=True)`.
- `src/ess/livedata/config/__init__.py` — update the re-export.
- `src/ess/livedata/config/instrument.py` —
  - rename field `log_context_bindings` → `context_bindings`;
  - rename method `add_log_context_binding` → `add_context_binding`;
  - update `_validate_binding_stream_name` body + error message
    ("ContextBinding references unknown stream…").
- `src/ess/livedata/config/instruments/bifrost/factories.py:74,79` — update the
  two `add_log_context_binding` call sites.
- `tests/config/instrument_test.py:330+` — rename `TestLogContextBindings` to
  `TestContextBindings`; update the six `add_log_context_binding` call sites
  (lines 334, 349, 369, 372, 402, 424). Update assertions on
  `instrument.log_context_bindings` to `instrument.context_bindings`.

Validation: `pytest -n auto -q`. Slow bifrost qmap test should pass unchanged.

### B2 — WorkflowSpec.context_bindings + SpecHandle.add_context_binding

Additive. No consumption yet.

Files:
- `src/ess/livedata/config/workflow_spec.py` — add
  `context_bindings: list[ContextBinding] = Field(default_factory=list)` to
  `WorkflowSpec`.
- `src/ess/livedata/handlers/workflow_factory.py` — extend `SpecHandle` with
  `add_context_binding(stream_name, workflow_key, *, dependent_sources=None,
  stream_resolver=None, seed_factory=None)` that constructs a `ContextBinding`
  and appends it to `self._factory[self.workflow_id].context_bindings`. When
  `dependent_sources` is None, default to `frozenset(spec.source_names)`.
- Unit tests for the new method (default `dependent_sources` substitution,
  appending semantics).

Validation: `pytest -n auto -q`.

### B3 — Add ContextBinding record declarations (additive only)

**Revised after the first implementation attempt.** The previous draft of B3
deleted `DetectorROIAuxSources` and `bifrost_aux_sources` while leaving
`JobFactory.create` on the legacy `Workflow.context_keys ∩ aux_source_names`
path. That path produces an empty gate set once the AuxSources subclasses are
removed, so bifrost crashes immediately. The fix is to split "add new
declarations" from "switch consumption + delete legacy" into two commits:

- **B3 (this commit):** Add the new ContextBinding records everywhere they need
  to land. NOTHING is deleted; the legacy AuxSources subclasses, kwargs, and
  factory parameters stay in place. The new records are inert because
  `JobFactory.create` still reads `Workflow.context_keys` and
  `AuxSources.initial_context_messages`. Behaviour is unchanged.
- **B4 (next commit):** Atomic swap. `JobFactory.create` consumes the new
  records, the legacy consumption paths are removed, and the now-unused
  AuxSources subclasses + kwargs + factory parameters are deleted.

Files (B3, purely additive):

ROI declarations:
- `src/ess/livedata/handlers/detector_view_specs.py` (around
  `register_detector_view_spec`, line 583+) — when `roi_support` is true,
  call `handle.add_context_binding(stream_name='roi_rectangle',
  workflow_key=ROIRectangleRequest,
  stream_resolver=lambda jid, name: f"{jid}/{name}",
  seed_factory=_roi_rectangle_seed)` and the polygon equivalent. Add
  `_roi_rectangle_seed` / `_roi_polygon_seed` module-level helpers that
  build the same `Message` produced today by
  `DetectorROIAuxSources.initial_context_messages` at lines 562-577.
- Do NOT delete `DetectorROIAuxSources` or the `aux_sources=...` kwarg yet.
  Do NOT change `DetectorViewFactory`'s signature yet. Both stay live; both
  are deleted in B4.

LOKI carriage declaration:
- `src/ess/livedata/config/instruments/loki/factories.py` — add
  `instrument.add_context_binding(stream_name='detector_carriage',
  workflow_key=TransformValueLog,
  dependent_sources=frozenset({'loki_detector_0'}))`.
- Do NOT yet delete `LOKI_DYNAMIC_TRANSFORMS` or change `DetectorViewFactory`.
  The new instrument-level record sits alongside the existing dual
  declaration; B4 removes the duplication.

Bifrost declarations:
- Already in place from B1 (the `add_log_context_binding` → `add_context_binding`
  rename). Bifrost's `instrument.add_context_binding(...)` calls in
  `bifrost/factories.py:74,79` now contribute to `instrument.context_bindings`,
  ready for B4 to consume.

Tests:
- Add focused unit tests asserting the new records are present on the
  relevant specs/instruments after registration. Light tests only —
  behavioural coverage stays on the existing AuxSources tests until B4
  rewires consumption.

Validation: full fast suite (no behaviour changes) + bifrost slow test
(passes via the legacy `Workflow.context_keys ∩ aux_source_names` path).

### B4 — Atomic swap: JobFactory consumes ContextBinding; delete legacy

Switches the source of the gate set + seed messages AND removes the legacy
declarations + factory parameters whose roles are now covered by ContextBinding.
This is the largest commit and the one that requires the most care, but it
remains a single coherent change: "the system now uses ContextBinding".

Files:

JobFactory + JobManager swap:
- `src/ess/livedata/core/job_manager.py` (`JobFactory.create`) —
  - build `matching` as the union of `instrument.context_bindings` and
    `spec.context_bindings` filtered by `job_id.source_name in ci.dependent_sources`;
  - `context_keys = {ci.stream_name: ci.workflow_key for ci in matching}`;
  - `wire_for = {ci.stream_name:
        ci.stream_resolver(job_id, ci.stream_name) if ci.stream_resolver
        else ci.stream_name
        for ci in matching}`;
  - splice `wire_for` into `rendered_aux_names` (field → wire) so the existing
    `_filter_data_for_job` and `_stream_to_fields` remap handle routing. This
    is the recommended (a) approach: `aux_source_names` is the field → wire
    vehicle for ALL inputs, dynamic or context (this matches what
    `DetectorROIAuxSources.render` does today; we just compose the dict from
    a different source);
  - `context_stream_names = set(wire_for.values())` — strict subset of
    `rendered_aux_names.values()`;
  - `seed_messages = [ci.seed_factory(job_id) for ci in matching
        if ci.seed_factory is not None]`;
  - pass `context_keys` to the workflow factory via the existing
    `WorkflowFactory.create` opt-in kwarg pattern (line 213-230 — add
    `if 'context_keys' in sig.parameters: kwargs['context_keys'] = context_keys`);
  - return `(Job, list[Message])` from `JobFactory.create` so
    `JobManager.schedule_job` can forward seeds (and mark `_seen_context_streams`).
- `src/ess/livedata/core/job_manager.py` (`JobManager._seed_initial_context`)
  — replace the `AuxSources.initial_context_messages(...)` call. Receive the
  seed list from `JobFactory.create`, call `on_schedule_seed(messages)`, and
  populate `_seen_context_streams[job_id]` from the message stream names —
  same bookkeeping as today, just from a different source.
- `src/ess/livedata/core/job.py` — no field→wire mapping changes needed
  (approach (a) reuses `aux_source_names`). The `context_aux_stream_names`
  parameter stays.
- `src/ess/livedata/handlers/workflow_factory.py` — extend the opt-in kwargs
  in `WorkflowFactory.create` to pass `context_keys` when the factory declares
  it. The receiving side already exists in `monitor_workflow.create_monitor_workflow`
  (line 176) and the detector_view factory once its signature changes (below).

Detector view factory refactor:
- `src/ess/livedata/handlers/detector_view/factory.py` — drop the
  `dynamic_transforms` parameter from `DetectorViewFactory.__init__`; the
  factory gains a `context_keys: dict[str, type]` kwarg consumed via the
  `WorkflowFactory.create` opt-in pattern. Pass `context_keys` through to
  `StreamProcessorWorkflow`. Replace the inline `_dynamic_transforms.get`
  branch (lines 262-265) with: read `context_keys` for entries whose value
  is `TransformValueLog`; for each, resolve the `transform_name` from the
  LOKI-local per-source `dict[str, str]` and call
  `add_dynamic_transform(workflow, transform_name=…)`. Drop the inline ROI
  context_keys assembly (lines 247-252) — those now arrive in the
  `context_keys` kwarg.

LOKI transform_name carrier:
- `src/ess/livedata/config/instruments/loki/factories.py` — replace
  `LOKI_DYNAMIC_TRANSFORMS` (the old `TransformValueStream` dict) with a
  stripped-down `dict[str, str]` (source_name → transform_name). Pass it to
  `DetectorViewFactory` via a new constructor kwarg `transform_names:
  dict[str, str]`. The `aux_stream` field of `TransformValueStream` is no
  longer needed (its role is taken by `ContextBinding.stream_name`).
- `src/ess/livedata/config/instruments/loki/specs.py:43-51,27,277` — delete
  `LOKI_DYNAMIC_TRANSFORMS`, the `DetectorROIAuxSources` import, and the
  `aux_sources=DetectorROIAuxSources(dynamic_transforms=LOKI_DYNAMIC_TRANSFORMS)`
  kwarg.

Legacy deletions (now safe because the new path is live):
- `src/ess/livedata/handlers/detector_view_specs.py` — delete the entire
  `DetectorROIAuxSources` class (lines 485-577). In `register_detector_view_spec`
  (line 583+), drop the `aux_sources` keyword argument and the default
  `aux_sources if aux_sources is not None else DetectorROIAuxSources()`
  (line 685). Reword the docstring at lines 609-612.
- `src/ess/livedata/config/instruments/dream/specs.py:26,206` — drop the
  `DetectorROIAuxSources` import and the `aux_sources=DetectorROIAuxSources()`
  kwarg.
- `src/ess/livedata/config/instrument.py:411-428` — drop the
  `DetectorROIAuxSources()` instantiation in the `_register_detector_views`
  helper.
- `src/ess/livedata/config/instruments/bifrost/specs.py:183-198` — delete
  `bifrost_aux_sources`.
- `src/ess/livedata/config/instruments/bifrost/specs.py:317,328,339` — drop
  `aux_sources=bifrost_aux_sources` from the three qmap-family `register_spec`
  calls.

YAGNI cleanup:
- `src/ess/livedata/dashboard/plotter_registry.py:89-120` — delete
  `SpecRequirements.requires_aux_sources`. No production `register_plotter`
  call uses it; the only references are in the test cases deleted below.
  Flag in the commit message.

Test updates:
- `tests/handlers/detector_view_specs_test.py` — delete `TestDetectorROIAuxSources`
  (line 230+) and the four `DetectorROIAuxSources()` direct-construction sites
  (lines 102, 162, 198, 210). Behavioural coverage of ROI seeding lands in
  the new JobFactory-level tests in this commit.
- `tests/dashboard/plotting_controller_test.py:140-205` — delete the four
  `SpecRequirements(requires_aux_sources=[DetectorROIAuxSources])` test cases.
- `tests/services/data_reduction_test.py:218-263`
  (`test_bifrost_qmap_drops_events_until_rotation_arrives`) — at line 247
  the test passes `spec.aux_sources.get_defaults()` into
  `WorkflowConfig.aux_source_names`. After `bifrost_aux_sources` deletion,
  `spec.aux_sources is None` and the test should pass `aux_source_names={}`.
  The rotation streams now flow into the gate via
  `instrument.context_bindings` (in place since B1).
- New tests in `tests/core/job_manager_test.py` (or a dedicated file)
  covering: `JobFactory.create` building `context_keys` from instrument +
  spec ContextBinding; seed_factory firing through `on_schedule_seed`; the
  ROI cold-start case opening the gate at tick 1.

Validation: full fast suite + `test_bifrost_qmap_drops_events_until_rotation_arrives`
slow test + ROI/detector-view tests. This commit is large by design; the
plan keeps it as one atomic change to avoid an intermediate state where
ContextBinding is consumed AND legacy AuxSources still fires (which would
cause double-seeding for ROI).

### B5 — Rename Job.context_aux_stream_names → Job.context_stream_names

Cosmetic, scoped, mechanical. Moved up per senior review so B6's cleanup uses
the final name directly.

Files:
- `src/ess/livedata/core/job.py` — rename constructor parameter, attribute,
  and docstring (no longer "subset of `aux_source_names`" — well, it is, but
  via the routing reuse decision; tighten the wording to "wire stream names
  whose context value must be available before the workflow runs").
- `src/ess/livedata/core/job_manager.py` — `JobFactory.create` and the gate
  logic reference the renamed parameter.
- `tests/core/job_manager_test.py` — `FakeJobFactory.create` already constructs
  `context_aux_stream_names=set(aux.values())` (line 60). Update name.
- `tests/core/job_test.py` — `FakeProcessor` and any direct constructions.
- Run `rg context_aux_stream_names` afterwards to confirm nothing left.

Validation: `pytest -n auto -q`.

### B6 — Remove transitional carriers

After B4, nothing reads `Workflow.context_keys` or
`AuxSources.initial_context_messages`. Confirmed via review: only consumers
were `JobFactory.create` (replaced in B4) and the test surface (replaced in
B3/B5).

Files:
- `src/ess/livedata/handlers/workflow_factory.py` — remove the `context_keys`
  property from the `Workflow` protocol.
- `src/ess/livedata/handlers/stream_processor_workflow.py` — remove the
  `context_keys` read-only property (constructor argument stays). Rename
  `_context_keys_map` back to `_context_keys` if the underscore name now
  matches the public-vs-private intent.
- `src/ess/livedata/handlers/area_detector_view.py`,
  `src/ess/livedata/handlers/timeseries_handler.py` — remove the empty
  `context_keys` stubs and the now-unused `Any` import in `timeseries_handler.py`.
- `src/ess/livedata/config/workflow_spec.py` — remove
  `AuxSources.initial_context_messages` and the `from
  ess.livedata.core.message import Message` import.
- `tests/handlers/monitor_workflow_test.py` — drop the `workflow.context_keys`
  assertions or replace with `workflow._context_keys` if the test still wants
  to assert on wrapped state.
- `tests/core/job_test.py` — drop `FakeProcessor.context_keys`.

Validation: `pytest -n auto -q`.

### B7 — Routing-pickup extension for ContextBinding

Required for bifrost rotation and LOKI carriage f144 streams to be subscribed
by the `detector_data` namespace. Without this, the gate stays closed
indefinitely for those workflows in production.

Files:
- `src/ess/livedata/config/route_derivation.py` (`gather_source_names`,
  line 14-46) — after the existing aux-source pickup at lines 36-38, add:

  ```python
  # Spec-level ContextBinding: include only if not job-scoped (resolver is None).
  # Job-scoped entries (ROI) route via a dedicated topic.
  for ci in spec.context_bindings:
      if ci.stream_resolver is None:
          names.add(ci.stream_name)
  ```

  And outside the spec loop (instrument-wide, no namespace filter — they
  apply to every spec whose source is in `dependent_sources`):

  ```python
  for ci in instrument.context_bindings:
      # Pickup if any spec in this namespace shares a source with the binding.
      relevant = any(
          spec.group.name == namespace
          and (set(spec.source_names) & ci.dependent_sources)
          for spec in instrument.workflow_factory.values()
      )
      if relevant:
          names.add(ci.stream_name)
  ```

- `tests/config/route_derivation_test.py:76+` — extend coverage for the two
  new pickup paths.

Validation: `pytest -n auto -q` + the bifrost slow test, which should now
ALSO be extensible to confirm the recovery path (rotation arrives → gate
opens → events flow) — but only if Kafka mocking is in place. Defer the
recovery-path assertion to a follow-up if non-trivial.

### B8 — Registration-time collision validation

Files:
- `src/ess/livedata/config/instrument.py` (`Instrument.__post_init__` or a
  helper invoked at the end of registration) — for every (spec, source) pair
  where both spec-level and instrument-level ContextBinding entries apply,
  compute the resolved wire-stream names for the spec-level entries (using
  a placeholder JobId — the resolver must be pure of JobId-specific data
  beyond name suffixing; if not, use the unresolved `stream_name`) and assert
  no collision with the instrument-level wire names. Raise `ValueError`.
- Unit test for the error case.

Validation: `pytest -n auto -q`.

## Sequencing rationale

- B1-B2 are additive and risk-free. Each is safe to land alone.
- B3 is purely additive: new ContextBinding records are added alongside the
  legacy AuxSources subclasses. JobFactory still consumes the legacy path; the
  new records are inert. Behaviour is unchanged.
- B4 is the atomic swap. JobFactory switches to consume ContextBinding records,
  the legacy `Workflow.context_keys` / `AuxSources.initial_context_messages`
  consumption is removed, and the now-unused AuxSources subclasses + kwargs +
  factory parameters are deleted in one commit. Splitting "switch" from
  "delete" would create a state where legacy seeding still fires alongside
  the new path, causing double-seeded ROI accumulators.
- B5 renames the field — done after B4 so the renamed code lands once,
  cleanly.
- B6 deletes the dead protocol/method stubs (`Workflow.context_keys`,
  `AuxSources.initial_context_messages`) that nothing consumes after B4.
- B7 unblocks routing for non-ROI ContextBinding (the bifrost/LOKI
  production-readiness blocker, flagged in ADR 0003 as a hard co-requirement,
  not a follow-up).
- B8 adds the validation backstop.

## What we are NOT doing

- Param-dependent gating (TOA vs wavelength). ADR 0003 § "Param-dependent
  context" explicitly punts.
- Changes to the JobManager gate mechanism itself. Block A landed it; we only
  swap the source of the gate set.
- Any changes to how the dashboard publishes ROI.
- Changes to `Workflow` protocol's other methods (`accumulate`/`finalize`/`clear`).
- The recovery-path assertion for the bifrost gate (rotation arrives → events
  flow). Worth doing as a follow-up after B7.

## Remaining open risks

1. **`LOKI transform_name` carrier location.** ADR 0003 § "LOKI transform_name
  carrier" accepts a small per-source `dict[str, str]` in `loki/factories.py`
  as a workaround for `ContextBinding` not carrying the NeXus path. The
  alternative — encode `transform_name` into the workflow_key wrapper — is
  uglier. Implementation should confirm the dict approach is clean enough; if
  it grows beyond LOKI to other instruments, revisit.
2. **`WorkflowFactory.create` opt-in proliferation.** Adding a fourth opt-in
  kwarg (`context_keys`) is fine on today's three-kwarg pattern but suggests
  the pattern itself may not scale. Not a blocker; flag if a fifth shows up
  in the same PR.
3. **`Instrument.__post_init__` collision check (B8) and resolver purity.**
  The check assumes `stream_resolver(jid, name)` is a name-suffixing operation;
  a resolver that does anything else would make collision detection unsound.
  The ROI resolver is `lambda jid, name: f"{jid}/{name}"` so this holds today.
  Document the assumption in `ContextBinding`'s docstring.

## Validation gates

- `pytest -n auto -q` (fast suite) after each commit.
- Slow tests after B3, B4, B6, B7:
  `pytest tests/services/data_reduction_test.py::test_bifrost_qmap_drops_events_until_rotation_arrives -v`
- Final: `pytest -n auto -m "not integration" -q` (full slow set).

## Out-of-scope follow-ups (separate PRs)

- Logical-name aliasing for spec-level wire names (would let ROI declare
  `roi_rectangle` and let routing auto-prefix per job).
- Param-dependent gating.
- Moving the dashboard's ROI seeding to the producer side (no longer needed
  with backend seeding in place).
- Recovery-path assertion in the bifrost slow test.
- Removal of the `WorkflowFactory.create` opt-in pattern in favour of a single
  context-aware factory hook, if more kwargs accrete.
