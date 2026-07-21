# ADR 0008: Job adoption from heartbeat observation replaces persisted job identity

- Status: accepted
- Deciders: Simon
- Date: 2026-07-21

## Context

Since the stable-data-key rework (#1042), the dashboard's data plane is keyed
by stable `DataKey`s and the per-commit `job_number` acts only as a generation
filter and provenance stamp. What remains job-scoped is the dashboard's
knowledge of *which backend job is the current generation*: established
locally at commit time in `ActiveJobRegistry`, persisted as `current_job` in
the config store, and re-established after a dashboard restart by
`ActiveJobRegistry.restore()` reading that store. The last remaining step of
#1042 replaces this persistence-as-authority with *adoption*: deriving the
current generation from live signals. This ADR decides how.

Persistence-based re-recognition is trust in a local record, not observation.
Its failure modes are the ones seen in pre-production (#714): jobs running on
the backend with no dashboard-side owner after a restart with a lost or stale
store, and — analysed on #1046 during the pending-command-expiry work
(#1067) — a stop command swallowed by the producer, which leaves an orphan
that no UI affordance can reach: local state says "stopped", so the stop
button is gone and a recommit skips the stop-old-jobs branch.

That analysis also settled what adoption must enable: **desired/observed
reconciliation**. UI state derives from desired state (user intent, held
locally) compared against observed state (what the backend reports); commands
are actuators, re-issued while the two differ. Command acknowledgements stay
what #1067 made them — fast user feedback (toasts), never a state driver —
because a lost ACK is indistinguishable from a lost command. Reconciliation
needs observed run-state; it does not need config content.

`JobStatus` heartbeats already carry `job_id` (source name + `job_number`),
`workflow_id`, run state, and start time. That is sufficient to observe *which
job runs as which generation*. What heartbeats do not carry is the job's
configuration — needed only for provenance (`resolve_config`) and for
labelling the running job in the UI.

Deployment reality, load-bearing for this decision: exactly one dashboard
instance writes commands, and no external actor (e.g. NICOS) starts jobs.
Every running job was started by this dashboard or a previous incarnation of
it, from a config this dashboard staged and persisted.

## Decision

The dashboard adopts running jobs from heartbeat observation; the persisted
job record is demoted from identity-authority to provenance cache. No wire
protocol change.

- **Identity by observation.** At startup and continuously thereafter, a
  non-stale `JobStatus` for a known workflow establishes that workflow's
  observed generation. `ActiveJobRegistry`'s current generation is initialized
  from observation; `ActiveJobRegistry.restore()` and the use of `current_job`
  as the source of truth for "what is running" are deleted. If several
  job_numbers heartbeat for one workflow (a stop-lagged predecessor alongside
  its replacement), the record resolves which is current; absent a record,
  the latest start time wins and the others surface as orphans.
- **Config by record, demoted.** The generation→config record (what
  `current_job` persistence carries today) continues to be written at commit,
  but is consulted only to label an observed job: provenance
  (`resolve_config`), UI display of the running parameters. It never decides
  whether a job runs.
- **Record misses degrade, they do not lie.** An observed job whose
  `job_number` is not in the record (crash between send and persist, store
  loss) is surfaced as running-with-unknown-config: visible, stoppable,
  counted in reconciliation, without params provenance. The next commit
  supersedes it.
- **Reconciliation drives run-state UI.** Desired "stopped" with observed
  running heartbeats re-issues the stop (bounded, logged); desired "running"
  with no observed heartbeats within the status-staleness window surfaces
  through the existing PENDING-plus-expiry path. Lost sends, lost ACKs, and
  orphans collapse into one recovery mechanism.

## Alternatives considered

- **Keep persisted `current_job` as authority** (status quo): the record
  cannot see reality — every #714 scenario is a record/reality divergence,
  and the swallowed-stop analysis showed the divergence can be unreachable
  from the UI. Rejected; this is what adoption replaces.
- **Commands-topic replay**: reconstruct identity by re-reading the command
  stream at startup. Rejected: replay reconstructs *intent*, not *reality* —
  a swallowed send never reached the topic, and a job whose worker died still
  replays as "started". Correctness would hinge on topic retention
  configuration, it adds a second consumer with a startup phase, and it only
  helps at startup, while heartbeats are continuous.
- **Config-fingerprint echo in `JobStatus`**: the dashboard mints a
  fingerprint of the committed config, sends it in `WorkflowConfig`, and the
  backend echoes it opaquely in every heartbeat; the dashboard recovers a
  running job's config by content instead of by record. Rejected for now:
  with a single dashboard instance the record covers every case except the
  crash-window/store-loss misses, which degrade gracefully above — a backend
  protocol change buying only that corner. Content is also not identity:
  byte-identical recommits share a fingerprint, so `job_number` remains the
  generation key regardless. Revisit if a second command writer (another
  dashboard instance, externally started jobs) becomes real — that breaks
  this ADR's "record ≈ my own history" premise and makes self-describing
  heartbeats worth the protocol change.
- **Full-config echo in `JobStatus`**: superset of the fingerprint; same
  rejection plus params payload in every heartbeat.

## Consequences

- Dashboard-only change; the backend and wire formats are untouched.
- `ActiveJobRegistry.restore()` is deleted; the `is_known_job` ingest window
  is fed by adoption instead of restoration; the generation→config record
  keeps being written but nothing reads it to decide run-state. This is the
  final step of #1042; its completion closes that issue.
- #714 closes when adoption plus reconciliation land: orphans become visible
  and stoppable, and the lost-stop scenario recovers via re-issued stops
  instead of requiring a backend restart.
- The single-writer deployment assumption is recorded here and must hold: a
  second dashboard instance or external job starts invalidate the premise,
  and the fingerprint-echo alternative is the designated escape hatch.
- Adoption latency is bounded by heartbeat cadence plus the status-staleness
  window: after a restart the dashboard may briefly show a running workflow
  as pending until its first heartbeat arrives. This replaces trusting a
  possibly-wrong record instantly with being provably right shortly.
