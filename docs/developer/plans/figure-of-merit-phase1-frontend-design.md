# FOM Phase 1 — Frontend Design

Companion to `figure-of-merit-concept.md` (operational picture) and
`figure-of-merit-phase1-impl.md` (backend, shipped).

## Why FOM state lives outside `JobOrchestrator`

`JobOrchestrator` is a 1-workflow-1-jobset state machine.
`WorkflowState[workflow_id]` carries at most one `current` `JobSet`,
replaced atomically by `commit_workflow` (`job_orchestrator.py:354`).
`stop_workflow` / `reset_workflow` operate on `state.current.job_ids()`
— workflow-level, not job-level (`:756`, `:831`). The dashboard exposes
no per-job controls.

Three reasons FOM cannot share this state:

1. **Coexistence.** The same workflow must be runnable as a FOM and as
   a regular configuration concurrently — the instrument scientist may
   want to sweep parameters on workflow X while NICOS uses X as the
   FOM.
2. **The Stop button on the workflow card is the FOM stop.** Workflow
   cards expose Stop/Reset/Reconfigure prominently; these are daily
   actions for the scientist. Confirmation modals do not gate this
   safely; the risk is wasted beam time mid-scan.
3. **Keying mismatch.** FOM state's natural key is the slot
   (`fom-0`..`fom-N`), not the workflow. Two slots can legitimately use
   the same workflow with different parameters. A slot may also bind the
   same workflow output across N sources at once: the alias then carries
   N substreams that the consumer (NICOS) aggregates trivially.

## Structure: `FOMOrchestrator` parallel to `JobOrchestrator`

A sibling class instantiated once in `DashboardServices` alongside
`JobOrchestrator`, sharing dependencies (`command_service`,
`workflow_registry`, `active_job_registry`, `job_service`,
`notification_queue`).

```python
class FOMOrchestrator:
    def __init__(
        self,
        *,
        command_service: CommandService,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        active_job_registry: ActiveJobRegistry,
        job_service: JobService,
        notification_queue: NotificationQueue | None = None,
        n_slots: int = 2,
    ) -> None: ...

    _slots: dict[FOMSlot, FOMSlotState]  # FOMSlot = "fom-0".."fom-{n-1}"

    def commit_slot(
        self,
        slot: FOMSlot,
        *,
        workflow_id: WorkflowId,
        source_names: Sequence[str],
        output_name: str,
        params: dict,
        aux_source_names: dict,
    ) -> tuple[JobId, ...]: ...

    def release_slot(self, slot: FOMSlot) -> None: ...
    def reset_slot(self, slot: FOMSlot) -> None: ...

    def get_slot_state_version(self, slot: FOMSlot) -> int: ...
    def process_acknowledgement(
        self, message_id: str, response: str, error_message: str | None = None
    ) -> None: ...
    def on_job_status_updated(self, job_status: JobStatus) -> None: ...
```

`FOMSlotState` carries `(workflow_id, source_names, output_name, params,
aux_source_names, job_number)`. All source-jobs share one `job_number`
so they form one logical slot.

`n_slots` defaults to 2 (operational expectation: one or two parallel
FOMs). The slot rows are fixed: `fom-0`..`fom-{n_slots-1}`.

The FOM job has its own `JobNumber`. `JobOrchestrator._workflows` is
unaware of it — the regular workflow card cannot reach it. This is the
load-bearing isolation property.

### Command composition

The shipped backend allows multiple bindings per alias and idempotent
unbind (no auto-stop). The dashboard composes lifecycle explicitly,
sending a single batched Kafka flush per operation. ``N`` denotes the
number of source-jobs in the previous binding, ``M`` the number in the
new binding.

| Operation | Slot state before | Batch contents |
|---|---|---|
| `commit_slot` | unbound | `[WorkflowConfig(new) × M, Bind(new) × M]` |
| `commit_slot` | bound | `[Stop(prev) × N, Unbind, WorkflowConfig(new) × M, Bind(new) × M]` |
| `release_slot` | bound | `[Stop(prev) × N, Unbind]` |
| `reset_slot` | bound | `[JobCommand(reset, prev) × N]` |

`commit_slot` always uses a fresh `JobNumber` shared across all M new
source-jobs so the new logical slot is distinct from the one being
stopped. A single `Unbind` clears all old bindings under the alias.

### Acknowledgement routing

`FOMOrchestrator` owns its own `PendingCommandTracker`. Each batch
shares one `message_id` across all four messages with
`expected_count = len(batch)`; the existing tracker counts ACKs and
fires success/error notifications.

`Orchestrator._process_response` (`orchestrator.py:113`) fans the
`CommandAcknowledgement` out to *both* orchestrators; unknown
`message_id`s are silently ignored by each tracker.

### Job-status routing

`JobService.on_status_updated` becomes a fan-out (list of listeners or
composed dispatch). `FOMOrchestrator.on_job_status_updated` registers
alongside `JobOrchestrator.on_job_status_updated` so FOM reacts to its
own jobs reporting `stopped`.

## Placement

A dedicated FOM tab inserted between Workflows and System Status in
`PlotGridTabs`. The tab hosts a panel with one `FOMSlotWidget` per
slot, laid out side-by-side (optimised for `n_slots=2`).

A separate Panel app remains a future option; the orchestrator
boundary makes that move cheap. Not in scope here.

## Slot-row widget

`FOMSlotWidget` is a fresh widget — not a subclass or adapter of
`WorkflowStatusWidget`. The shared elements are small and concrete:

- `WorkflowWidgetStyles` (status colours, dimensions).
- Status-and-timing aggregation, dots renderer — lifted from
  `workflow_status_widget.py` to module scope as free functions so
  both widgets can use them without subclassing.
- Refresh-by-version pattern: `FOMOrchestrator.get_slot_state_version`
  parallels `JobOrchestrator.get_workflow_state_version`.

A `FOMSlotWidget` shows the slot name, status badge, one status dot
per bound source, timing, **live per-source numeric readout** (one row
per substream), and Configure / Reset / Release buttons. Per-source
status, timing aggregation, and dot rendering go through the shared
`derive_aggregate_status` helper in `workflow_status_widget.py`, which
is also designed to be adoptable by `WorkflowStatusWidget` in a later
clean-up.

The live readout subscribes to
`data_service[ResultKey(workflow_id, job_id, output_name)]` per
substream, like a regular plot. The `LIVEDATA_FOM` mirror is for NICOS
only — the dashboard never consumes it.

## Configuration UI: two-step wizard, slot from launch context

The slot is determined by which slot row's Configure button was
clicked; it is not a wizard step.

- **Step 1**: workflow + output selection. The existing
  `WorkflowAndOutputSelectionStep` (`plot_config_modal.py:279`) is
  generalised: its `initial_config` parameter accepts a small
  `Protocol` exposing `workflow_id` and `output_name` (both
  `PlotConfig` and a new `FOMSlotPrefill` satisfy it). The "Static
  Overlay" namespace is suppressed via a constructor flag.
- **Step 2**: workflow parameter configuration — the existing
  parameter widget wrapped as a `WizardStep`. The source selector is
  the default `MultiChoice`; selecting multiple sources binds them all
  under the same alias and the consumer aggregates the substreams.
  Final commit calls `FOMOrchestrator.commit_slot(slot, ...)`.

When the slot is unbound, Configure opens an empty wizard. When the
slot is bound, the wizard prefills from the slot's current state
(workflow, output, sources, params); cancelling without committing
gives the operator a way to inspect the binding. A
replace-confirmation modal fires before the final commit on a bound
slot, naming the alias, the workflow, and the source list being
replaced.

## State, persistence, multi-session

`FOMOrchestrator` is a process-singleton in `DashboardServices`,
shared across all browser sessions in the same process. All sessions
see the same slot state. Each session's `FOMSlotWidget` polls
`get_slot_state_version` and rebuilds on change — the standard
version-based refresh pattern.

`_slots` is in-memory only. v1 ships without persistence and without
restart recovery: dashboard restart with active FOM jobs orphans the
backend jobs, cleared by restarting backend services. The footgun is
acceptable for local testing and early operational use.

## What does not change

- Backend stream-alias schema and plumbing (already shipped).
- `JobOrchestrator`, `WorkflowConfigurationAdapter`, the wizard
  framework, the parameter widget, or any plot widget. They acquire
  one new consumer (`FOMOrchestrator`) without internal changes — the
  exception is `JobService.on_status_updated` becoming a fan-out, and
  `Orchestrator._process_response` fanning ACKs to both orchestrators.
- `LIVEDATA_DATA` copy semantics: regular subscribers (including any
  plot pointed at the FOM workflow when configured normally via
  `JobOrchestrator`) are unaffected.

## Implementation sketch

1. **Dashboard**: lift status-and-timing aggregation and dots renderer
   from `workflow_status_widget.py` to module scope. Add
   `derive_aggregate_status(job_service, job_ids)` so widgets that
   surface a logical group of jobs (workflow card, FOM slot row) drive
   per-source dots, timing, and error aggregation through one path.
   No behaviour change to `WorkflowStatusWidget`.
2. **Dashboard**: generalise `WorkflowAndOutputSelectionStep`'s prefill
   parameter to a `Protocol`; add a flag to suppress the
   "Static Overlay" namespace.
3. **Dashboard**: `FOMOrchestrator` with in-memory `_slots`,
   `PendingCommandTracker`, slot-state version, `commit_slot` /
   `release_slot` / `reset_slot` composing the explicit batches above.
4. **Dashboard**: convert `JobService.on_status_updated` to fan-out
   and register `FOMOrchestrator.on_job_status_updated`.
5. **Dashboard**: fan ACKs out to both orchestrators in
   `Orchestrator._process_response`.
6. **Dashboard**: wrap the parameter widget as a `WizardStep`. Wire
   together with the generalised step 1 into a two-step FOM wizard
   parameterised by the launching slot. Prefill on bound-slot opens;
   replace-confirmation modal before final commit on bound slot.
7. **Dashboard**: `FOMSlotWidget` — slot row showing title, status,
   one dot per bound source, timing, per-source numeric readout, and
   Configure / Reset / Release buttons.
8. **Dashboard**: FOM panel hosting `n_slots` `FOMSlotWidget`s
   side-by-side, mounted as a new top-level tab in `PlotGridTabs`
   between Workflows and System Status.
9. **Dashboard**: tests for `FOMOrchestrator` covering commit, release,
   reset, multi-slot, multi-source, and ACK handling.

Steps 1–2 are independent refactors and can land first. Steps 3–5 are
orchestrator wiring. Steps 6–8 are UI.

## Known v1 limitations (accepted)

- Dashboard restart with active FOM jobs orphans them; cleared by
  restarting backend services.
- A failed `Bind` (e.g., transient stale-alias state) leaves the slot
  inconsistent; operator releases and retries.
- FOM jobs do not appear in the regular workflow status list (no
  badging). With the structural separation, the regular workflow card
  cannot reach the FOM job, so this is a clarity gap, not a safety
  gap.
- No cross-restart slot reconstruction; sessions in the same process
  see consistent state, but a fresh process starts with empty slots.

## Open questions

- **Embedded plot in FOM panel**. Show a small live plot of the bound
  output? Probably yes long-term, no in v1. The bound output's
  `LIVEDATA_DATA` plot remains available in the regular dashboard
  *only* when the workflow is also configured normally via
  `JobOrchestrator` (FOM-only configuration leaves
  `JobOrchestrator`/`PlotOrchestrator` unaware of the job).

## Out of scope

- Backend changes (auto-stop on rebind, dedicated `_livedata_fom`
  topic, scheduled reset/start). Revisit once v1 operational evidence
  exists.
- NICOS-side interface — owned by the NICOS team.
- Separate frontend app — defer until operational evidence justifies
  it; the structural design here makes that future move cheap.
- Slot-state persistence and reconstruction from heartbeats.
