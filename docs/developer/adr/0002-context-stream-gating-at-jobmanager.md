# ADR 0002: Gate workflow execution at the JobManager on context-stream readiness

- Status: proposed
- Deciders: Simon
- Date: 2026-05-21

## Context

Sciline-based workflows parametrise their graph on context inputs — motor positions, ROIs, choppers — delivered alongside detector data. `StreamProcessor` holds a hidden `None` seed for every context input. If a workflow runs before a context message has arrived, providers downstream of that input receive `None` and either crash (bifrost's `process_chunk_workflow` reads rotation inside `accumulate`, raising `'NoneType' object has no attribute 'ndim'`) or — worse — silently produce detector data attributed to a default geometry. Both failure modes have surfaced in production.

The system needs a gate that prevents workflow invocation until every **context** input the workflow depends on has a value. The design questions are where this gate lives and how it knows which inputs to gate on.

### Aux vs context — disambiguation

The codebase uses **auxiliary sources** (`AuxSources`, `Job.aux_source_names`) for a routing concept: the per-job set of additional input streams beyond the primary data stream. Users select these in the UI (e.g. "which monitor stream feeds this job's normalization?"). Auxiliary sources are not all of one kind:

- **Dynamic aux** — inputs that accumulate over time inside the workflow (e.g. LOKI's incident/transmission monitors). They flow through `StreamProcessor.accumulate` and tolerate late arrival; events processed against an empty accumulator still produce a useful (uncorrected) result that the consumer can ignore.
- **Context aux** — inputs that parametrise the workflow graph (e.g. ROI selection, motor rotation). They flow through `StreamProcessor.set_context` and are not tolerated as `None` — providers downstream either crash or produce silently-wrong output.

Only the second category needs gating. The first must not be gated, or dynamic accumulation breaks (events arrive before monitors → events dropped → no result ever even after monitors arrive).

The workflow factory already encodes this distinction via `StreamProcessorWorkflow`'s `dynamic_keys` and `context_keys` arguments. The Job and JobManager need to learn it too.

### ROI cold start

ROI is a context input with a meaningful "no message yet" steady state: the user has not (yet) selected an ROI. The dashboard publishes a zero-length concatenated DataArray when the user deletes the last ROI, but at cold startup it publishes nothing — so a gate that waits for the first ROI message would never open for a detector-view workflow whose user has not yet drawn one.

The `MessagePreprocessor` populates accumulators lazily — only on first message arrival (`orchestrating_processor.py:62-68`). An accumulator that does not exist contributes no value, so a defaulted `LatestValueHandler` is useless on its own: without a triggering message, it never gets constructed and its default never reaches the gate.

## Decision

Gate at `JobManager`, not at the workflow execution unit. Two design pieces fall out of the aux-vs-context disambiguation:

1. **Gate on context streams only.** The workflow factory exposes the subset of `aux_source_names` that correspond to `StreamProcessorWorkflow.context_keys`. `Job` carries this subset as `context_aux_stream_names: set[str]`. `Job.missing_context(available)` returns the context aux not present in the (post-`get_context`) `WorkflowData`. The JobManager skips and warns when this set is non-empty.
2. **Seed context accumulators at job scheduling.** Workflow factories that have context inputs with a steady-state default value (currently ROI) declare them as initial-aux messages. The JobManager forwards these to the preprocessor at scheduling time as ordinary `Message` objects, creating the accumulator and seeding it with the default. Subsequent dashboard publishes overwrite the seed normally.

Activation remains time-driven via `should_start`. On each tick, after `OrchestratingProcessor` enriches `WorkflowData` with cached context via `MessagePreprocessor.get_context`, JobManager checks per active job: is `job.missing_context({s.name for s in data.data})` empty? If not, skip the job for this tick and record a `pending_context_warning(missing)`.

`StreamProcessorWorkflow` exposes a read-only `context_keys` mapping so the factory can derive `context_aux_stream_names`. SPW itself remains a pure "given dynamic + context, produce result" unit — it does not consult the gate, does not track which keys have arrived, does not drop dynamic data.

## Alternatives considered

| Option | Notes |
|---|---|
| **Gate at JobManager on context-only aux, with backend-seeded ROI defaults (chosen)** | The gate respects the workflow's own context-vs-dynamic distinction. Seeding decouples "context has a meaningful default" from "a message has arrived from the dashboard". |
| Gate at JobManager on all aux | Conflates routing with execution; gates dynamic aux (monitors) that legitimately accumulate without context. Forces tests and production to pre-publish dynamic aux before events. Rejected. |
| Gate inside `StreamProcessorWorkflow` with per-workflow opt-in | Couples scheduling to workflow execution; requires factories to opt in explicitly and forces SPW to carry a "drop dynamic while pending" branch. The opt-in spreads gate configuration across factories. |
| Have the dashboard publish "no ROI" at job startup | Solves cold-start ROI for the production happy path but does not cover tests, headless deployments, or restart-without-dashboard. The backend has all the information needed; relying on the dashboard adds a coupling we do not need. |
| Defer job construction until context ready | Conflicts with time-based scheduling and moves the sciline-pipeline build cost from config-load to first-data-arrival, introducing user-visible latency. |

## Key design choices

### Aux vs context, explicitly

`Job.aux_source_names` keeps its existing routing semantics — every aux stream the job consumes, regardless of how the workflow uses it. A new `context_aux_stream_names: set[str]` carries the gating subset, populated by the factory from `StreamProcessorWorkflow.context_keys`. `Job.missing_context(available)` uses the subset; `Job.missing_aux` does not exist.

This keeps the workflow factory as the single source of truth for which inputs are context vs dynamic. Adding a new workflow that needs gating requires no changes to JobManager — only the standard declaration of `context_keys` in the factory.

### Time-based activation is preserved

`should_start(current_time)` continues to drive the scheduled → active transition. The context-readiness gate is layered on top: an active job whose context aux is missing receives no data this tick and emits a warning. Schedule semantics stay independent of stream arrival timing.

The gate is implemented as a single predicate evaluated once per active job per tick, immediately before `_filter_data_for_job`. There is no new control-flow path inside the job-processing loop: the gated branch records a warning and continues to the next job. Activation-step gating (deferring scheduled → active until context is ready) was considered but rejected as it would entangle the schedule contract with stream arrival.

### Primary data arriving before context is dropped

If primary data arrives at tick K while a context aux is still missing, the primary is filtered out of the job's input for that tick and not buffered. The workflow processes the next primary batch that arrives after the gate opens. This is conceptually correct: detector events whose geometry context is unknown cannot be meaningfully reduced, and silent attribution to a default geometry is the exact failure mode the gate exists to prevent.

For dynamic aux (e.g. monitors arriving late), the gate does not fire — those inputs accumulate normally and the workflow produces output as soon as both primary and dynamic aux have flowed through.

### Backend seeding for defaulted context streams

A workflow factory whose context input has a meaningful "no message yet" default (currently ROI) returns a list of initial `Message` objects from a hook on the factory base. The JobManager hands these to the preprocessor at `schedule_job` time via the same `preprocess_messages` path used by Kafka-delivered messages. This creates the accumulator and stores the default as its first value; subsequent dashboard-published messages overwrite it normally.

This replaces the earlier idea of constructing `LatestValueHandler` with an in-memory `default` argument — that approach failed because the accumulator was not constructed at all without a message. Seeding via the message path guarantees the accumulator exists from scheduling onward.

### Job state semantics

A job that is time-active but context-gated is conceptually distinct from one whose last processing attempt produced an error. Modelled with a dedicated `JobState.pending_context` rather than overloading the existing `warning` state: a context-pending job is healthy and waiting, a warning job has produced an error and may need attention. The ADR does not block on this — `JobState.warning` is an acceptable interim if introducing a new state proves disruptive — but the dedicated state is the target.

## Consequences

- `StreamProcessorWorkflow` exposes `context_keys` read-only; otherwise it stays focused on workflow execution and carries no gate state.
- The workflow-factory base grows an optional `initial_context_messages` hook for seeding defaulted context streams. ROI is the only current user.
- Any workflow that declares motion or geometry context (bifrost cuts, future detector-geometry-dependent reductions) gets correct startup behaviour automatically. No per-instrument opt-in.
- Dynamic aux (LOKI monitors) flows through unchanged — the gate does not see it.
- Warning emission for context-pending jobs is consolidated inside `JobManager`. There is no warning-message channel between the workflow execution unit and the job layer.
- Tests at JobManager level cover: dynamic aux not gated; context aux with seed (ROI) ready from tick 1; context aux without seed (rotation) blocks then unblocks; primary dropped while gated.
- The preprocessor → orchestrator → JobManager data path is unchanged. JobManager reads `aux_source_names` (already on `Job`) and the new `context_aux_stream_names`. Seeding goes through the existing `preprocess_messages` entry point.
