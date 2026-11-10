# ADR 0012: Desired-state topic for job control

- Status: accepted
- Deciders: Simon
- Date: 2026-11-10

## Context

Job control was edge-triggered. The dashboard sent a start (`WorkflowConfig`) or a
stop (`JobCommand`) on `{instrument}_livedata_commands`, and the worker kept the
result only in memory (`JobManager._jobs`). Every consumer, including the one on the
commands topic, starts at the high watermark, so a restarted worker never sees
earlier commands.

A worker restart therefore lost every job:

- After a crash, the dashboard still wanted the job, but nothing re-sent the start.
  Reconciliation (ADR 0008) re-issued only stops. A start was never re-issued, partly
  because `JobManager.schedule_job` overwrote an existing `JobId` and its data.
- After a graceful shutdown, the worker's final heartbeat reported every job as
  `stopped`. The dashboard took this as the end of the job and cleared its own desired
  state (`StoppedReason.backend_shutdown`). After the restart the dashboard no longer
  wanted the job either.

The component that acts kept its state in memory, and the durable copies of the
intent (the YAML `ConfigStore`, the commands topic) were not readable by it. This ADR
moves the comparison of desired and observed state into the worker, against a durable
record of the desired state (#1336, approach D of #856).

## Decision

The desired state lives on a compacted Kafka topic, `{instrument}_livedata_desired`.
Workers reconcile their jobs against it. The dashboard writes it and reads it back.

**Record.** One record per workflow. The key is the workflow id string
`instrument/name/version`. The value is the job set of one commit
(`core/desired_state.py`):

```python
class DesiredWorkflow(pydantic.BaseModel):
    workflow_id: WorkflowId
    job_number: JobNumber  # every commit mints a new one
    jobs: dict[str, DesiredJob]  # by source name; DesiredJob = params + aux sources
```

A commit writes one record. A stop writes one tombstone (null value). Reading the
topic from offset 0 and folding the records in order gives the *desired table*,
`dict[WorkflowId, DesiredWorkflow]`. `JobId(source_name, job_number)` and `ResultKey`
are unchanged.

**Worker.** Each worker reads the topic through its own consumer
(`KafkaDesiredStateReader`), starting at offset 0, without offset commits and without
the bounded `BackgroundMessageSource` queue. Before its first processing cycle it reads
to the partition EOF. In every cycle it applies new records and calls
`JobReconciler.reconcile`:

```python
wanted = {JobId(source, want.job_number) for want in table.values()
          if owns(want.workflow_id) for source in want.jobs}
stop every job not in wanted          # tombstoned, superseded, or source dropped
build at most one job in wanted that neither runs nor was rejected
```

Reconciling twice against the same table changes nothing, so replayed records need no
deduplication. A job that fails to build is *rejected*: its error is reported in the
heartbeat as a `JobStatus` with state `error` for as long as the table wants it, and it
is retried only when its record changes or the process restarts. A record of this
instrument whose workflow no worker has registered is logged once.

**Dashboard.** A commit or stop writes one record or tombstone through an idempotent
producer (`KafkaDesiredStateWriter`) and waits for the delivery report. Only a
confirmed write flips the generation or clears buffers. A failed write is reported
and changes nothing. The dashboard reads the topic back from offset 0 and derives each
workflow's current generation from the table. The status display compares the table
(desired) with heartbeats (observed) per job (`project_job_states`).

**Reset stays a command.** `{instrument}_livedata_commands` carries only `JobCommand`
with action `reset` (per `JobId` from the dashboard, per `WorkflowId` from NICOS, ADR
0006), delivered at most once. A replayed reset would wipe data. A reset missed during
a restart does not matter, because the restarted job starts empty.

**Unknown entries are removed only by the user.** A table entry for a workflow outside
the dashboard's registry (removed workflow, version bump) is listed in the UI with a
remove action that writes its tombstone. The dashboard never removes entries by itself:
a dashboard whose registry differs from the workers' (a development checkout on the
production broker, a rollback) would otherwise stop running jobs just by starting.

### Relation to ADR 0008

This ADR supersedes these parts of ADR 0008:

- **Heartbeat adoption establishes the current generation.** The desired table says
  which job number is current. Adoption and jobs "running with unknown config" are
  removed. A job not in the table is stopped by its worker.
- **Starts are never re-issued, and the dashboard re-issues stops.** The worker builds
  what the table wants and stops what it does not want. The dashboard re-issues
  nothing.
- **The single command writer is load-bearing.** With a keyed table the last writer
  wins per key and every reader converges on the same table.

ADR 0008 rejected commands-topic replay because "replay reconstructs intent, not
reality". That holds for the dashboard's question (what runs?), so heartbeats keep
their role as observed state. The worker's question (what should run?) is about
intent, which is what the topic stores.

## Alternatives considered

- **One record per (workflow, source).** A commit would be several records and not
  atomic: a dashboard crash between the writes leaves two job numbers for one
  workflow. Nothing addresses a single source.
- **Dashboard re-sends starts (#856, approach C).** Needs an idempotent start, a
  re-issue ledger for starts, and an amendment to ADR 0008. Recovery still depends on
  the dashboard running.
- **Stable consumer groups with committed offsets (#856, approach A).** Delivers
  commands a worker missed while it was down. It does not restore jobs lost by the
  restart itself.
- **Full event sourcing.** An unlimited-retention log that every reader folds. It adds
  a history of parameter changes, worth having only if provenance of live results
  becomes a requirement.
- **Reset as a value in the record** (a reset counter, as `kubectl rollout restart`
  does with a timestamp). This would retire the commands topic. But NICOS resets by
  `WorkflowId` with an acknowledgement, and a reset has no meaning across a restart.

## Consequences

- **A worker restart restores its jobs.** Accumulated data is still lost: a rebuilt job
  keeps its `JobId` and starts empty, which consumers see as a reset. A job gated on a
  context stream produced by another job (the wavelength lookup table, ADR 0010/0011)
  stays `pending_context` until that context is published again.
- **A job that kills the process can make the worker crash-loop**, taking the other
  jobs of that service down with it. The worker logs each `JobId` before building it
  (`job_building`), which names the job. Stopping the workflow in the dashboard writes
  a tombstone, which the worker reads on its next start, so the user ends the loop
  without operator access. There is no per-job restart budget.
- **Desired state lives until someone changes it.** After a long shutdown the previous
  jobs come back.
- **Topic configuration is a correctness requirement**, not tuning:

  ```
  cleanup.policy=compact       # compact only: with delete or compact,delete a
                               # long-running workflow loses its record silently
  partitions=1                 # never changed: the partition of a key depends on
                               # the partition count
  replication.factor=3
  min.insync.replicas=2        # the topic is the only durable copy
  ```

  The reader refuses a topic with more than one partition.
- **A worker never acts on a partial table.** If the reader cannot reach the end of
  the topic within 30 s at startup, or hits a fatal consumer error, the worker exits
  with an error. A non-fatal consumer error after startup keeps the last table. The
  dashboard likewise fails to start if it cannot read the topic to its end.
- **Undecodable records are logged and skipped**, at every start until their key is
  overwritten (`hostile_wire.malformed_desired_records`).
- Backend and dashboard deploy together (wire change). The table starts empty.
