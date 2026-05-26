# ADR 0002: Gate workflow execution at the JobManager on context-stream readiness

- Status: accepted
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

Context streams are declared separately from `AuxSources`, at spec-level or instrument-level (see ADR 0003). The Job and JobManager consume the resolved set at job creation time.

### ROI cold start

ROI is a context input with a meaningful "no message yet" steady state: the user has not (yet) selected an ROI. The dashboard publishes a zero-length concatenated DataArray when the user deletes the last ROI, but at cold startup it publishes nothing — so a gate that waits for the first ROI message would never open for a detector-view workflow whose user has not yet drawn one.

The `MessagePreprocessor` populates accumulators lazily — only on first message arrival (`orchestrating_processor.py:62-68`). An accumulator that does not exist contributes no value, so a defaulted `LatestValueHandler` is useless on its own: without a triggering message, it never gets constructed and its default never reaches the gate.

## Decision

Gate at `JobManager`, not at the workflow execution unit. Two design pieces fall out of the aux-vs-context disambiguation:

1. **Gate on context streams.** `Job` carries `context_stream_names: set[str]` — the set of wire stream names whose values the workflow consumes via `set_context`. `Job.missing_context(available)` returns the context streams not present in the (post-`get_context`) `WorkflowData`. The JobManager skips and warns when this set is non-empty. The set is populated at `JobFactory.create` from spec-level and instrument-level context-input declarations (see ADR 0003 for the declaration mechanism).
2. **Seed context accumulators at job scheduling.** Context inputs with a steady-state default value (currently ROI) carry a seed alongside their declaration. The JobManager forwards seed messages to the preprocessor at scheduling time as ordinary `Message` objects, creating the accumulator and seeding it with the default. Subsequent dashboard publishes overwrite the seed normally.

Activation remains time-driven via `should_start`. On each tick, after `OrchestratingProcessor` enriches `WorkflowData` with cached context via `MessagePreprocessor.get_context`, JobManager checks per active job: is `job.missing_context({s.name for s in data.data})` empty? If not, skip the job for this tick and record a `pending_context_warning(missing)`.

`StreamProcessorWorkflow` remains a pure "given dynamic + context, produce result" unit — it does not consult the gate, does not track which keys have arrived, does not drop dynamic data.

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

`Job.aux_source_names` keeps its existing routing semantics — every aux stream the job consumes, regardless of how the workflow uses it. A new `context_stream_names: set[str]` carries the gating set, populated at `JobFactory.create` from context-input declarations resolved for the specific (spec, source). `Job.missing_context(available)` uses this set; `Job.missing_aux` does not exist. The declaration model is settled in ADR 0003.

This keeps spec/instrument-level declarations as the single source of truth for which inputs are context. Adding a new workflow that needs gating requires no changes to `JobManager` — only the standard declaration in the spec or on the instrument binding.

### Time-based activation is preserved

`should_start(current_time)` continues to drive the scheduled → active transition. The context-readiness gate is layered on top: an active job whose context aux is missing receives no data this tick and emits a warning. Schedule semantics stay independent of stream arrival timing.

The gate is implemented as a single predicate evaluated once per active job per tick, immediately before `_filter_data_for_job`. There is no new control-flow path inside the job-processing loop: the gated branch records a warning and continues to the next job. Activation-step gating (deferring scheduled → active until context is ready) was considered but rejected as it would entangle the schedule contract with stream arrival.

### Primary data arriving before context is dropped

If primary data arrives at tick K while a context aux is still missing, the primary is filtered out of the job's input for that tick and not buffered. The workflow processes the next primary batch that arrives after the gate opens. This is conceptually correct: detector events whose geometry context is unknown cannot be meaningfully reduced, and silent attribution to a default geometry is the exact failure mode the gate exists to prevent.

For dynamic aux (e.g. monitors arriving late), the gate does not fire — those inputs accumulate normally and the workflow produces output as soon as both primary and dynamic aux have flowed through.

### Backend seeding for defaulted context streams

Context inputs with a meaningful "no message yet" default (currently ROI) carry a seed alongside the declaration. The JobManager hands seed messages to the preprocessor at `schedule_job` time via the same `preprocess_messages` path used by Kafka-delivered messages. This creates the accumulator and stores the default as its first value; subsequent dashboard-published messages overwrite it normally. The declaration mechanism is settled in ADR 0003.

This replaces the earlier idea of constructing `LatestValueHandler` with an in-memory `default` argument — that approach failed because the accumulator was not constructed at all without a message. Seeding via the message path guarantees the accumulator exists from scheduling onward.

### Job state semantics

A job that is time-active but context-gated is conceptually distinct from one whose last processing attempt produced an error. Modelled with a dedicated `JobState.pending_context` rather than overloading the existing `warning` state: a context-pending job is healthy and waiting, a warning job has produced an error and may need attention. The ADR does not block on this — `JobState.warning` is an acceptable interim if introducing a new state proves disruptive — but the dedicated state is the target.

## Consequences

- `StreamProcessorWorkflow` stays focused on workflow execution and carries no gate state.
- Context-input declarations may carry an optional seed (see ADR 0003 for the declaration model). ROI is the only current user.
- Any workflow that declares motion or geometry context (bifrost cuts, future detector-geometry-dependent reductions) gets correct startup behaviour automatically. No per-instrument opt-in.
- Dynamic aux (LOKI monitors) flows through unchanged — the gate does not see it.
- Warning emission for context-pending jobs is consolidated inside `JobManager`. There is no warning-message channel between the workflow execution unit and the job layer.
- Tests at JobManager level cover: dynamic aux not gated; context aux with seed (ROI) ready from tick 1; context aux without seed (rotation) blocks then unblocks; primary dropped while gated.
- The preprocessor → orchestrator → JobManager data path is unchanged. JobManager reads `input_stream_names` (the routing-combined view over user-selected aux and framework-injected context, see ADR 0003 "Job carries aux and context as separate maps") and `context_stream_names` for gating. Seeding goes through the existing `preprocess_messages` entry point.
