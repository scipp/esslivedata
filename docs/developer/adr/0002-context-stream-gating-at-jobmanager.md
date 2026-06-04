# ADR 0002: Gate workflow execution at the JobManager on context-stream readiness

- Status: accepted
- Deciders: Simon
- Date: 2026-05-21

## Context

Sciline-based workflows parametrise their reduction graph on *context* values —
motor positions, sample geometry, choppers — delivered as their own streams
alongside detector data. `StreamProcessor` holds a hidden `None` seed for every
such context binding. If a workflow runs before its context stream has produced a
value, the providers downstream of that input receive `None` and either crash
(bifrost's `process_chunk_workflow` reads rotation inside `accumulate`, raising
`'NoneType' object has no attribute 'ndim'`) or — worse — silently produce
detector data attributed to a default geometry. Both failure modes have surfaced
in production.

The system needs a gate that prevents workflow invocation until every context
input *that has no safe default* has a value. The design questions are where this
gate lives and how it learns which inputs to gate on.

### What needs gating, and what does not

A job consumes several kinds of input beyond its primary data stream. They differ
in whether the workflow tolerates the input being absent:

- **Dynamic aux** — inputs that accumulate over time inside the workflow (e.g.
  LOKI's incident/transmission monitors). They flow through
  `StreamProcessor.accumulate` and tolerate late arrival: events processed against
  an empty accumulator still produce a useful (uncorrected) result the consumer can
  ignore. Gating these would break accumulation — events arriving before monitors
  would be dropped, and no result would ever appear even after the monitors do.
- **Context with a safe default** — inputs that parametrise the graph but map a
  missing value to a well-defined result. ROI is the example: the Sciline providers
  treat "no request" and "empty request" identically as *no ROI selected*
  (`detector_view/roi.py`). There is no failure mode to prevent, so these are not
  gated either; ROI is routed as an ordinary auxiliary source.
- **Context with no safe default** — inputs that parametrise the graph and whose
  absence is unrepresentable: motion/geometry (bifrost rotation, LOKI carriage).
  `None` crashes or yields silently-wrong geometry. **Only this category is gated.**

The gating axis is therefore *"does this input tolerate absence?"*, not
*"`set_context` vs `accumulate`"* — ROI flows through `set_context` yet is not
gated. The set of gated streams for a job is derived from `ContextBinding`
declarations at job-creation time; the declaration model is settled in ADR 0003.

## Decision

Gate at `JobManager`, not at the workflow execution unit.

`Job` carries `gating_streams: set[str]` — the wire stream names whose values the
workflow needs before it can run. `Job.missing_context(available)` returns the
gating streams not present in `available`. The set is populated at
`JobFactory.create` from the `ContextBinding` declarations that resolve for the
specific `(spec, source)` (ADR 0003).

A gated job occupies a dedicated stage **between** *scheduled* and *active*, held in
its own `_pending_context_jobs` bucket. When `should_start` fires, `_start_job`
routes the job by `gating_streams`: empty → it becomes `active` at once; non-empty →
it enters `pending_context` with a `pending_context_warning`. Only active jobs are
processed — a pending job is never fed primary data.

`JobManager` tracks a **sticky** per-pending-job set of seen context streams
(`_seen_context_streams`). Context values arrive intermittently — a motor position
is published once and then is quiet — so a stream seen on any tick stays available.
Each tick, `_open_context_gates` folds the tick's available streams into each pending
job's sticky set and checks `job.missing_context(seen)`:

- non-empty → the job stays pending; its `pending_context_warning(missing)` is
  refreshed to the still-missing streams.
- empty → the job graduates to `active` (`_activate_pending_job`), dropping its
  warning and sticky set; its own processing outcome drives its state from there.

Graduation is **one-way**: `gating_streams` is fixed and `seen` only grows, so once a
job is active it has left the gate permanently and is never re-checked, and a job with
no context never enters the stage at all. The per-tick gate therefore scans only the
small, transient pending bucket — not every active job.

Time-based activation is preserved: `should_start(current_time)` drives the job *out
of* `scheduled` on time, independent of stream-arrival timing. The readiness gate
governs only the subsequent pending → active transition, so a job blocked on a dead
producer stays visible as `pending_context` rather than silently rewriting its
schedule.

`StreamProcessorWorkflow` remains a pure "given dynamic + context, produce result"
unit. It does not consult the gate, does not track which keys have arrived, and does
not drop dynamic data.

There is no seeding mechanism. The gate stays closed until the producer publishes.
If the producer is down, the workflow does not run — which is the correct behaviour
for a context with no safe default.

## Alternatives considered

| Option | Notes |
|---|---|
| **Gate at JobManager on context streams with no safe default (chosen)** | The gate respects the workflow's own tolerates-absence distinction. The set is derived from declarations, so adding a gated workflow needs no JobManager change. |
| Gate at JobManager on all aux | Conflates routing with execution; gates dynamic aux (monitors) that legitimately accumulate without context. Forces tests and production to pre-publish dynamic aux before events. Rejected. |
| Gate inside `StreamProcessorWorkflow` with per-workflow opt-in | Couples scheduling to workflow execution; forces `StreamProcessorWorkflow` to carry a "drop dynamic while pending" branch and spreads gate configuration across factories. Rejected. |
| Defer job construction until context ready | Conflicts with time-based scheduling and moves the Sciline-pipeline build cost from config-load to first-data-arrival, introducing user-visible latency. Rejected. |
| Hold gated jobs in `scheduled` rather than a distinct stage | Entangles the schedule contract with producer liveness — a dead motion producer would keep a job "scheduled" past its start, and the dashboard could not tell "not due yet" from "started but blocked". A dedicated `pending_context` stage keeps the two axes separate. Rejected. |

## Key design choices

### The gate set is derived, not declared per workflow

`Job.gating_streams` is populated at `JobFactory.create` from the `ContextBinding`
records resolved for the `(spec, source)`. Declarations are the single source of
truth for which inputs gate; adding a workflow that needs gating requires only the
standard declaration (ADR 0003), no `JobManager` change.

### Sticky seen-set, not per-tick presence

Gating compares against an accumulated `_seen_context_streams` set, not the streams
present *this* tick. A context value seen once stays available — this matches how
the preprocessor caches context for `set_context` and lets the gate open on a quiet
tick once the last stream has been seen.

### A distinct stage, not a per-tick skip

A gated job is held in its own `_pending_context_jobs` bucket, not left among the
active jobs with a per-tick "skip if gated" branch. Because `gating_streams` is fixed
and `seen` only grows, readiness is monotonic: a job graduates once and never
re-gates. Modelling that as a one-way stage move means the processing loop iterates
only runnable jobs, the per-tick gate scans only the (small) pending bucket, and a
job that never needed context never participates in gating at all.

### Primary data arriving before context is dropped

A pending job is not among the active jobs a batch is dispatched to, so primary data
arriving while a gating stream is still missing simply is not delivered to it and
nothing is buffered. It processes the first primary batch on or after the tick its
gate opens. This is conceptually correct: detector events whose geometry context is
unknown cannot be meaningfully reduced, and silent attribution to a default geometry
is the exact failure mode the gate exists to prevent. Dynamic aux is unaffected — it
is not in the gate set and accumulates normally.

### Dedicated `JobState.pending_context`

A job held in the pending stage is conceptually distinct both from one not yet due
(`scheduled`) and from one whose last processing attempt errored (`warning`).
`JobState.pending_context` is healthy and waiting on a producer; surfacing it
distinctly lets an operator tell a dead motion producer from a job that simply has
not started.

## Consequences

- `StreamProcessorWorkflow` stays focused on workflow execution and carries no gate
  state.
- Any workflow that declares motion or geometry context gets correct startup
  behaviour automatically — the only requirement is the `ContextBinding`
  declaration (ADR 0003). No per-instrument gate opt-in.
- Dynamic aux (LOKI monitors) flows through unchanged — the gate does not see it.
- ROI is routed as an auxiliary source, not gated: "no ROI" is a safe default, so
  there is no readiness to wait on and no seed to inject.
- Warning emission for context-pending jobs is consolidated inside `JobManager`.
  There is no warning channel between the workflow execution unit and the job layer.
- `JobState.pending_context` is added to the job state machine, backed by a
  `_pending_context_jobs` bucket between `_scheduled_jobs` and `_active_jobs`. Job
  lookup, control, reset, status, and finish handling all account for the new bucket.
- Tests at JobManager level cover: dynamic aux not gated; context (motion) blocks
  then unblocks once its stream is seen; the sticky seen-set keeps the gate open on
  quiet ticks; primary dropped while gated; gates independent across jobs.
- The declaration model that feeds `gating_streams` and `context_keys`, and the
  routing that ensures the gated streams are actually subscribed, are settled in
  ADR 0003.
