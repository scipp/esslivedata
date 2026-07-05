# ADR 0007: Single-writer state with versioned pull as the dashboard concurrency model

- Status: proposed
- Deciders: Simon
- Date: 2026-07-05

## Context

The dashboard runs on three kinds of threads: the shared ingestion thread
(`DashboardServices._update_loop`, draining Kafka and driving orchestrator
updates at ~50 ms), each session's Tornado/Bokeh IOLoop (UI callbacks and the
100 ms session poll, under that session's document lock), and the polling
thread paths that run layer activation. Session-bound objects (`Pipe`,
`DynamicMap`, Bokeh documents) may only be mutated on their own session's
IOLoop — a hard constraint established in ADR 0005.

Historically, state changes propagated between these threads by two competing
mechanisms:

1. **Synchronous push**: callbacks fanning out from the mutating thread,
   carrying data or lifecycle events — `DataService` extracting and
   delivering per subscriber inside ingestion (`_notify`), the
   `JobOrchestrator` subscription registry notifying `LayerSubscription` /
   `PlotOrchestrator` on job changes, and `PlotOrchestrator` lifecycle
   notifications rebuilding every session's grid widgets from the mutating
   session's thread.
2. **Versioned pull**: a single writer updates authoritative state and bumps a
   version counter; each consumer polls the version on its own clock and pulls
   a snapshot when it changed — the pattern prescribed in
   `.claude/rules/dashboard-widgets.md`, followed by the workflow-status
   widgets, and (since ADR 0005) by the frame-gated session flush.

The 2026-07 audits (#1039, #1044; standalone issues #1040, #955; assessment
#1042) showed the two mechanisms have sharply different defect profiles.
Every confirmed cross-thread race, session leak, and category of compute waste
was located in push machinery: registration racing ingestion, notify loops
iterating live sets mutated by the UI thread, cross-session document mutation
outside the document lock, the viewer gate and stash bolted onto the push path
along with their documented races, and the rebinding chain that exists only to
push "the job key changed" to consumers. The code built on versioned pull was
audited clean. The same failure was fixed once before (#1007), in push code.

The structural reason: under push, thread-safety is a per-call-site
obligation. Each new entry point (add-layer-while-running, heartbeat
deactivation, the stale-session reaper) must remember every guard, and the
audit trail shows each one independently missed some. Under versioned pull,
safety is a property of the pattern: the writer is single, the version bump is
atomic, and consumers read snapshots on their own thread.

## Decision

Single-writer state with versioned pull is the dashboard's concurrency and
delivery model. Push is the exception and requires explicit justification.

Concretely:

- **Every piece of shared state has exactly one writing thread.** Other
  threads request changes by enqueueing for the owner or mutating via a
  narrowly-scoped, documented lock — never by direct multi-step mutation.
- **State changes are observed by version, not by callback.** A writer bumps a
  monotonic version (per key, grid, or subsystem as granularity requires);
  consumers poll the version on their own clock (session poll, frame flush)
  and pull an immutable snapshot when it advanced. Callbacks, where they
  survive, carry no data — at most "something changed" wake-ups that
  degrade to the next poll tick if lost.
- **Data flows by pull at display cadence.** Ingestion records what changed
  (dirty keys); extraction, copy, and compute happen when a consumer with a
  viewer pulls — not eagerly per delta on the ingestion thread. "No viewers,
  no work" is the default, not a gate.
- **Session-bound mutation happens only on the owning session's IOLoop**
  (reaffirming ADR 0005), including teardown paths such as the stale-session
  reaper.
- **Subscription identity is stable.** Keys a consumer subscribes under must
  not embed ephemeral tokens (e.g. per-commit `job_number`); generation
  discrimination is a filter at the ingest boundary, not part of consumer
  identity. (Assessment: #1042, Option B.)

## Alternatives considered

- **Keep push, add locks** (the direction of the standalone lock/gate fixes,
  e.g. serializing `DataService`, guarding `PlotOrchestrator` topology).
  Rejected as the end state: each lock protects one call-site pattern, the
  lock-ordering surface grows (`ingestion_guard` → service lock → document
  lock), and nothing prevents the next feature from bypassing the guards —
  the audited failure mode. Locks survive, but narrowly, protecting buffer
  writes rather than delivery.
- **Marshal all mutation onto session IOLoops** (`add_next_tick_callback`
  everywhere). Rejected: Panel cannot resolve session context from background
  threads (ADR 0005), fan-out to N sessions multiplies work, and shared
  (non-session) state still needs an owner.
- **Full actor model / message-passing rewrite.** Rejected as
  disproportionate: versioned pull achieves the same isolation for this
  system's shapes (one ingestion writer, per-session readers) with the
  patterns already proven in the codebase.

## Consequences

- The three push planes converge on the model (tracked in #1046):
  data delivery inverts to pull-on-frame (#1045, subsuming the viewer gate
  #1043 and shrinking the #1009 lock to buffer writes); the job-identity
  rebinding chain is deleted via stable keys (#1042 Option B); grid widgets
  move to version polling and reaper teardown moves to the owning IOLoop
  (#1040, #955).
- Audited races #1039-2/3/9 and the session-leak items #1039-5/11 are removed
  structurally rather than guarded; field symptoms #789 and #714 are expected
  to resolve (verify after landing).
- New features inherit safety by construction: a new consumer polls a version
  and pulls a snapshot; a new producer bumps a version. Review can enforce a
  simple rule — "who is the single writer, and where is the version?" —
  instead of re-deriving interleavings per PR.
- Worst-case freshness is bounded by the consumer's poll cadence (100 ms
  session tick; see ADR 0005 for why this does not add visible latency over
  push).
- Cost: dirty-flag/version bookkeeping replaces eager delivery, and
  during migration both mechanisms coexist — the migration is sequenced so
  each step deletes more push machinery than it adds pull machinery.
