# ADR 0002: Gate workflow execution at the JobManager on context-stream readiness

- Status: proposed
- Deciders: Simon
- Date: 2026-05-21

## Context

Sciline-based workflows parametrise their graph on context streams (motor positions, ROIs, choppers) delivered alongside detector data. `StreamProcessor` holds a hidden `None` seed for every context input. If a workflow runs before a context message has arrived, providers downstream of that input receive `None` and either crash (bifrost's `process_chunk_workflow` reads rotation inside `accumulate`, raising `'NoneType' object has no attribute 'ndim'`) or — worse — silently produce detector data attributed to a default geometry. Both failure modes have surfaced in production.

The system therefore needs a gate that prevents workflow invocation until every context input the workflow depends on has a value. The design question is where this gate lives.

ROI context streams shape the answer. The dashboard publishes a zero-length concatenated DataArray when the user deletes the last ROI — a normal steady state. At cold startup it publishes nothing at all, so a naive "all context_keys must arrive" gate would never open for a detector-view workflow that observes ROI but does not require one to be selected. The gate needs to distinguish "absence is a fault" (rotation, carriage_log) from "absence is a steady state" (ROI).

## Decision

Gate at `JobManager`, not at the workflow execution unit. Activation remains time-driven via `should_start`. On each tick, after `OrchestratingProcessor` enriches `WorkflowData` with cached context via `MessagePreprocessor.get_context`, `JobManager` checks per active job: are all entries of `job.aux_source_names` present in the enriched `WorkflowData`? If any are missing, the job is skipped for this tick — no `job.add()` call, no work item, no finalize — and a `pending_context_warning(missing)` is recorded in the job warning state.

Context-stream readiness is established at the preprocessor layer, not the workflow layer:

- `LatestValueHandler` carries an optional `default` constructor argument. When set, `get()` returns the default while no message has been added, so `get_context` includes the stream from tick 1.
- For `StreamKind.LIVEDATA_ROI`, the detector-data factory (`handlers/detector_data_handler.py`) constructs the handler with the empty `RectangleROI` / `PolygonROI` DataArray as default — byte-identical to the steady state the dashboard publishes when the user deletes all ROIs.
- F144-backed context (rotation, carriage_log) receives no default. `ToNXlog.get()` and undefaulted `LatestValueHandler.get()` raise; `MessagePreprocessor.get_context` catches `RuntimeError`/`ValueError` and omits the key, so the JobManager sees the aux as missing and gates.

`StreamProcessorWorkflow` knows nothing about gating: it is a pure "given dynamic + context, produce result" unit. `Job` exposes `missing_aux(available: set[str]) -> set[str]` so the JobManager call site stays declarative and the warning formatter has one source for the missing-name set.

## Alternatives considered

| Layer | Notes |
|---|---|
| **JobManager-level (chosen)** | Gating is a scheduling decision. The workflow execution unit stays pure: no "drop dynamic while pending" branch, no tracking of which context keys have ever arrived. Warning emission is unified at one layer. |
| Inside `StreamProcessorWorkflow`, per-workflow opt-in | Each instrument factory must opt in by listing the required context keys; SPW carries a "drop dynamic while pending" branch. Couples scheduling to workflow execution and spreads the gate-configuration knowledge across factories. |
| Inside `StreamProcessorWorkflow`, all context keys auto-gated | Removes the per-factory opt-in but keeps the scheduling/execution coupling. SPW still tracks per-key arrival state, still discards dynamic data while pending, and still bubbles warnings up through the workflow → job interface. |
| Defer job construction until context is ready | Conflicts with time-based scheduling — schedules fire at configured times, not when data arrives — and moves the sciline-pipeline build cost from config-load to first-data-arrival, introducing user-visible latency between context arrival and first output. |

## Key design choices

### Time-based activation is preserved

`should_start(current_time)` continues to drive the scheduled → active transition. The aux-readiness gate is layered on top: an active job with missing aux receives no data this tick and emits a warning. Schedule semantics stay independent of stream arrival timing.

The gate is implemented as a single predicate evaluated once per active job per tick, immediately before `_filter_data_for_job`. There is no new control-flow path inside the job-processing loop: the gated branch simply records a warning and continues to the next job. Activation-step gating (deferring the scheduled → active transition until aux is ready) was considered but rejected as it would entangle the schedule contract with stream arrival; the per-tick check carries no equivalent cost.

### Job state semantics

A job that is time-active but context-gated is conceptually distinct from one whose last processing attempt produced an error. Modelling this with a dedicated `JobState` (e.g. `pending_context`) is preferred over overloading the existing `warning` state, since the two states carry different operator expectations (a context-pending job is healthy and waiting; a warning job has produced an error and may need attention). The ADR does not block on this — `JobState.warning` is an acceptable interim if introducing a new state proves disruptive — but the dedicated state is the target.

### Primary data arriving before context is dropped

If primary data arrives at tick K while an aux stream is still missing, the primary is filtered out of the job's input for that tick and not buffered. The workflow processes the next primary batch that arrives after the gate opens. This is conceptually correct: detector events whose geometry context is unknown cannot be meaningfully reduced, and silent attribution to a default geometry is the exact failure mode the gate exists to prevent. "Dropped while waiting" is the desired behaviour, not a regression.

### Default values express "no message yet is a valid steady state"

Whether the absence of a context message is a fault or a steady state is a property of the stream, not of the workflow. ROI absence is a steady state ("no selection"); rotation absence is a fault. `LatestValueHandler(default=…)` encodes this at the accumulator, where the stream's nature is already known. Workflows declare their inputs without enumerating which are required vs optional — that distinction lives where the data enters the system.

### `missing_aux` lives on `Job`, not on the workflow

The check needs `Job.aux_source_names` (derived from the workflow spec's routing) and the set of available stream names from `WorkflowData`. The workflow execution unit has neither, and — given the decision above — shouldn't. A method on `Job` keeps the JobManager call site declarative and gives the warning formatter a single source of truth.

## Consequences

- `StreamProcessorWorkflow` stays focused on workflow execution; it carries no state about which context keys have arrived, and no factory needs to opt into gating.
- Any workflow that depends on motion or geometry context (bifrost cuts, future detector-geometry-dependent reductions) gets correct startup behaviour automatically.
- Warning emission for context-pending jobs is consolidated inside `JobManager`. There is no warning-message channel between the workflow execution unit and the job layer.
- Test coverage for the gate lives at `JobManager` level, exercising the readiness predicate against various combinations of aux arrival and defaulted accumulators. End-to-end assertions ("events drop until rotation arrives" for bifrost; "LOKI starts cleanly because carriage log precedes events") remain valid observables.
- No new coupling between layers: the preprocessor → orchestrator → JobManager data path is unchanged. JobManager reads `aux_source_names` (already on `Job`) and `WorkflowData` keys (already received via `process_jobs`).
