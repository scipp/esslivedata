# Block B: Implement ADR 0003 (unified ContextInput declaration model)

Builds on the Block A commits on `jobmanager-context-gate` (commits c16e5c24,
7424b02c, cda8ba5f) that implement the JobManager gate per ADR 0002 with
transitional carriers (`Workflow.context_keys` protocol member,
`AuxSources.initial_context_messages` hook, `DetectorROIAuxSources`,
`bifrost_aux_sources`). This plan replaces those transitional carriers with
first-class `ContextInput` declarations as specified in ADR 0003.

Reviewed by senior-engineer-review (decisions baked in below; risk register
trimmed to what remains open).

## Final-state target (post-Block B)

- `LogContextBinding` renamed to `ContextInput` (`config/stream.py`),
  with optional `stream_resolver` and `seed_factory` fields.
- `Instrument.log_context_bindings` → `Instrument.context_inputs`;
  `Instrument.add_log_context_binding` → `Instrument.add_context_input`.
- `WorkflowSpec.context_inputs: list[ContextInput]`; `SpecHandle.add_context_input`
  appends, defaulting `dependent_sources` to `frozenset(spec.source_names)`.
- `JobFactory.create` merges instrument + spec ContextInput entries filtered by
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
  `spec_handle.add_context_input(...)` with a resolver
  (`lambda jid, name: f"{jid}/{name}"`) and a `seed_factory`.
- `bifrost_aux_sources` deleted; the three qmap-family `register_spec` calls
  lose the `aux_sources=…` kwarg. The existing instrument-level
  `add_context_input` calls in bifrost's `factories.py` carry the rotation
  declarations.
- LOKI's `LOKI_DYNAMIC_TRANSFORMS` dual declaration collapses: one
  `instrument.add_context_input(stream_name='detector_carriage',
  workflow_key=TransformValueLog, dependent_sources=frozenset({'loki_detector_0'}))`
  in `loki/factories.py`. The `transform_name` (NeXus path used by
  `add_dynamic_transform`) does not fit `ContextInput` and lives in a small
  per-source `dict[str, str]` local to `loki/factories.py` — see ADR 0003 §
  "LOKI transform_name carrier". `DetectorViewFactory.dynamic_transforms`
  constructor parameter is removed; the factory receives `context_keys` as a
  kwarg from `WorkflowFactory.create` and consults the local LOKI dict only
  when `TransformValueLog` appears in `context_keys`.
- `AuxSources.initial_context_messages` hook removed; `AuxSources` slimmed to
  dynamic, user-selectable aux only.
- `Job.context_aux_stream_names` → `Job.context_stream_names`.
- `gather_source_names` learns to include the stream names of instrument-level
  `context_inputs` and spec-level `context_inputs` whose `stream_resolver` is
  `None` (so the bifrost rotation and LOKI carriage f144 streams get subscribed
  by the `detector_data` namespace). Spec-level entries with a resolver are
  job-scoped (ROI) and routed via a dedicated topic, not via
  `gather_source_names`.
- `SpecRequirements.requires_aux_sources` in `dashboard/plotter_registry.py:89-120`
  is removed as YAGNI — the only references are in the deleted test cases
  (see B3).
- Registration-time collision validation: a spec-level `ContextInput` whose
  resolved wire name collides with an instrument-level one for any (spec,
  source) pair is a registration error.

## Commit sequence

### B1 — Introduce ContextInput, deprecate LogContextBinding

Mechanical rename + add optional fields. No semantic change.

Files:
- `src/ess/livedata/config/stream.py` — rename `LogContextBinding` to
  `ContextInput`; add `stream_resolver: Callable[[JobId, str], str] | None = None`
  and `seed_factory: Callable[[JobId], Message] | None = None` (both default
  preserve current behaviour). Keep the class `@dataclass(frozen=True, slots=True,
  kw_only=True)`.
- `src/ess/livedata/config/__init__.py` — update the re-export.
- `src/ess/livedata/config/instrument.py` —
  - rename field `log_context_bindings` → `context_inputs`;
  - rename method `add_log_context_binding` → `add_context_input`;
  - update `_validate_binding_stream_name` body + error message
    ("ContextInput references unknown stream…").
- `src/ess/livedata/config/instruments/bifrost/factories.py:74,79` — update the
  two `add_log_context_binding` call sites.
- `tests/config/instrument_test.py:330+` — rename `TestLogContextBindings` to
  `TestContextInputs`; update the six `add_log_context_binding` call sites
  (lines 334, 349, 369, 372, 402, 424). Update assertions on
  `instrument.log_context_bindings` to `instrument.context_inputs`.

Validation: `pytest -n auto -q`. Slow bifrost qmap test should pass unchanged.

### B2 — WorkflowSpec.context_inputs + SpecHandle.add_context_input

Additive. No consumption yet.

Files:
- `src/ess/livedata/config/workflow_spec.py` — add
  `context_inputs: list[ContextInput] = Field(default_factory=list)` to
  `WorkflowSpec`.
- `src/ess/livedata/handlers/workflow_factory.py` — extend `SpecHandle` with
  `add_context_input(stream_name, workflow_key, *, dependent_sources=None,
  stream_resolver=None, seed_factory=None)` that constructs a `ContextInput`
  and appends it to `self._factory[self.workflow_id].context_inputs`. When
  `dependent_sources` is None, default to `frozenset(spec.source_names)`.
- Unit tests for the new method (default `dependent_sources` substitution,
  appending semantics).

Validation: `pytest -n auto -q`.

### B3 — Migrate ROI, LOKI dynamic_transforms, bifrost_aux_sources (single shot)

This is the largest commit by far. It introduces the new declarations
everywhere they need to land. The old consumption (`Workflow.context_keys`
piggy-back, `AuxSources.initial_context_messages`) stays live until B4 — so
ROI/bifrost/LOKI workflows continue to work via the legacy path through this
commit.

Files:

ROI migration:
- `src/ess/livedata/handlers/detector_view/factory.py` — drop the
  `dynamic_transforms` parameter from `DetectorViewFactory.__init__`; the
  factory gains a `context_keys: dict[str, type]` kwarg (consumed via the
  `WorkflowFactory.create` opt-in pattern at line 213-230). Pass `context_keys`
  through to `StreamProcessorWorkflow`. Replace the inline
  `_dynamic_transforms.get(source_name)` lookup at lines 262-265 with: read
  `context_keys` for entries whose value is `TransformValueLog`; for each,
  resolve the `transform_name` from the LOKI-local per-source dict (see
  below) and call `add_dynamic_transform(workflow, transform_name=…)`.
- `src/ess/livedata/handlers/detector_view_specs.py` —
  - delete the entire `DetectorROIAuxSources` class (lines 485-577);
  - in `register_detector_view_spec` (line 583+), drop the `aux_sources`
    keyword argument and the `aux_sources if aux_sources is not None else
    DetectorROIAuxSources()` default (line 685). When `roi_support` is true
    (the existing predicate), call
    `handle.add_context_input(stream_name='roi_rectangle',
    workflow_key=ROIRectangleRequest,
    stream_resolver=lambda jid, name: f"{jid}/{name}",
    seed_factory=_roi_rectangle_seed)` (and the polygon equivalent).
    `_roi_*_seed` builds the same `Message` produced today by
    `DetectorROIAuxSources.initial_context_messages` at lines 562-577.
  - Reword the docstring at lines 609-612 — no longer mentions subclassing
    `DetectorROIAuxSources`.
- `src/ess/livedata/config/instruments/dream/specs.py:26,206` — drop the
  `DetectorROIAuxSources` import and the `aux_sources=DetectorROIAuxSources()`
  kwarg.
- `src/ess/livedata/config/instruments/loki/specs.py:27,277` — drop the
  `DetectorROIAuxSources` import and the kwarg. `LOKI_DYNAMIC_TRANSFORMS`
  itself is moved (see LOKI section below).
- `src/ess/livedata/config/instrument.py:411-428` — drop the
  `DetectorROIAuxSources()` instantiation in the `_register_detector_views`
  helper.

LOKI dynamic_transforms migration (consolidated here per senior review):
- `src/ess/livedata/config/instruments/loki/factories.py` — add
  `instrument.add_context_input(stream_name='detector_carriage',
  workflow_key=TransformValueLog,
  dependent_sources=frozenset({'loki_detector_0'}))`.
  Move `LOKI_DYNAMIC_TRANSFORMS` from `specs.py` to a stripped-down
  per-source `dict[str, str]` here (source_name → transform_name);
  the `aux_stream` field of `TransformValueStream` is no longer needed
  (its old role is taken by the `ContextInput.stream_name`). Pass this
  dict to `DetectorViewFactory` via a new constructor kwarg
  `transform_names: dict[str, str]` so the factory can look up the NeXus
  path when wiring `add_dynamic_transform`.
- `src/ess/livedata/config/instruments/loki/specs.py:43-51` — delete
  `LOKI_DYNAMIC_TRANSFORMS` and the import. The spec no longer references it.

bifrost_aux_sources cleanup:
- `src/ess/livedata/config/instruments/bifrost/specs.py:183-198` — delete
  `bifrost_aux_sources`.
- `src/ess/livedata/config/instruments/bifrost/specs.py:317,328,339` — drop
  `aux_sources=bifrost_aux_sources` from the three qmap-family `register_spec`
  calls.

Tests:
- `tests/handlers/detector_view_specs_test.py` — delete `TestDetectorROIAuxSources`
  (line 230+) and the four `DetectorROIAuxSources()` direct-construction sites
  (lines 102, 162, 198, 210). Behaviour under test (ROI seeding) moves to
  per-spec coverage in the JobFactory-level tests added in B4.
- `tests/dashboard/plotting_controller_test.py:140-205` — delete the four
  `SpecRequirements(requires_aux_sources=[DetectorROIAuxSources])` test cases.
  These are dead test machinery (no production `register_plotter` call uses
  `requires_aux_sources`). Also delete
  `SpecRequirements.requires_aux_sources` from
  `src/ess/livedata/dashboard/plotter_registry.py:89-120` as YAGNI — flag
  in commit message that this is removed because no production caller used
  it.
- `tests/services/data_reduction_test.py:218-263`
  (`test_bifrost_qmap_drops_events_until_rotation_arrives`) — at line 247
  the test passes `spec.aux_sources.get_defaults()` into
  `WorkflowConfig.aux_source_names`. After `bifrost_aux_sources` deletion,
  `spec.aux_sources` is `None` and `aux_source_names = {}`. The test should
  pass `aux_source_names={}` directly and rely on the instrument-level
  `add_context_input` (already in place via bifrost's `factories.py`) for
  the gate to fire.

Validation: full fast suite + `pytest tests/services/data_reduction_test.py::test_bifrost_qmap_drops_events_until_rotation_arrives -v`.
At the END of B3 ROI still works because the AuxSources path is still live
for the seed and `Workflow.context_keys` is still consulted for the gate set.

### B4 — JobFactory consumes ContextInput records

Switches the source of the gate set + seed messages. Stop using
`Workflow.context_keys` and `AuxSources.initial_context_messages`.

Files:
- `src/ess/livedata/core/job_manager.py` (`JobFactory.create`) —
  - build `matching` as the union of `instrument.context_inputs` and
    `spec.context_inputs` filtered by `job_id.source_name in ci.dependent_sources`;
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
  (line 176) and the detector_view factory after B3.

Validation: full fast suite + `test_bifrost_qmap_drops_events_until_rotation_arrives`
slow test + ROI/detector-view tests.

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

### B7 — Routing-pickup extension for ContextInput

Required for bifrost rotation and LOKI carriage f144 streams to be subscribed
by the `detector_data` namespace. Without this, the gate stays closed
indefinitely for those workflows in production.

Files:
- `src/ess/livedata/config/route_derivation.py` (`gather_source_names`,
  line 14-46) — after the existing aux-source pickup at lines 36-38, add:

  ```python
  # Spec-level ContextInput: include only if not job-scoped (resolver is None).
  # Job-scoped entries (ROI) route via a dedicated topic.
  for ci in spec.context_inputs:
      if ci.stream_resolver is None:
          names.add(ci.stream_name)
  ```

  And outside the spec loop (instrument-wide, no namespace filter — they
  apply to every spec whose source is in `dependent_sources`):

  ```python
  for ci in instrument.context_inputs:
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
  where both spec-level and instrument-level ContextInput entries apply,
  compute the resolved wire-stream names for the spec-level entries (using
  a placeholder JobId — the resolver must be pure of JobId-specific data
  beyond name suffixing; if not, use the unresolved `stream_name`) and assert
  no collision with the instrument-level wire names. Raise `ValueError`.
- Unit test for the error case.

Validation: `pytest -n auto -q`.

## Sequencing rationale

- B1-B2 are additive and risk-free. Each is safe to land alone.
- B3 introduces all the new declarations and removes the deprecated aux-sources
  classes that overlap with them, but does NOT yet swap the gate-set source.
  ROI/bifrost continue to work via the legacy path. The diff is large but
  the bulk is mechanical deletions.
- B4 is the one behavioural swap: gate set + seeds now come from ContextInput.
  Tests still pass because B3 already declared every entry through the new
  path.
- B5 renames the field — moved earlier than the previous draft so B6's
  cleanup uses the final name.
- B6 deletes the dead carriers.
- B7 unblocks routing for non-ROI ContextInput (this is the bifrost/LOKI
  production-readiness blocker — flagged in ADR 0003 as a hard co-requirement,
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
  as a workaround for `ContextInput` not carrying the NeXus path. The
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
  Document the assumption in `ContextInput`'s docstring.

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
