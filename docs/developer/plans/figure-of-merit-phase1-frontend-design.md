# FOM Phase 1 — Frontend Design

Companion to `figure-of-merit-concept.md` (operational picture),
`figure-of-merit-phase1-impl.md` (backend, shipped), and
`figure-of-merit-phase1-dashboard.md` (which presented Options 1/2/3
without recommendation).

This document supersedes the option-choice section of the dashboard
options doc by reframing the decision as **structural**, with placement
following.

## Reframing: structure first, placement second

The original Options 1/2/3 conflated two orthogonal questions:

- **Structural**: where does FOM lifecycle state live in the dashboard?
  Inside `JobOrchestrator`'s existing state, as a sub-structure within
  it, or in a parallel orchestrator.
- **Placement**: where does the configuration UI live? Affordance on the
  workflow widget, dedicated panel, or separate app.

Once the structural answer is fixed, placement falls out. The original
Option 1 implied embedding state inside `JobOrchestrator`; Options 2 and
3 are compatible with a parallel structure but did not name it as the
load-bearing choice.

## Why FOM state cannot live inside `JobOrchestrator`

`JobOrchestrator` is a 1-workflow-1-jobset state machine.
`WorkflowState[workflow_id]` carries at most one `current` `JobSet`,
replaced atomically by `commit_workflow` (`job_orchestrator.py:354`).
`stop_workflow` / `reset_workflow` operate on `state.current.job_ids()`
— workflow-level, not job-level (`:756`, `:831`). The dashboard exposes
no per-job controls.

If FOM inhabits this state machine:

1. **Coexistence is blocked.** The same workflow can run either as FOM
   or as a regular configuration, not both. The instrument scientist
   cannot run an interactive parameter sweep of workflow X while NICOS
   is using X as a FOM. Today's plan calls this "duplicate computation"
   in Option 2's downsides — inverted, it is the *correctness property*
   that isolates FOM from data exploration.
2. **The Stop button on the workflow card is the FOM stop.** Workflow
   cards expose Stop/Reset/Reconfigure prominently; these are daily
   actions for the scientist. Confirmation modals do not gate this
   safely. The risk is wasted beam time mid-scan.
3. **Keying mismatch.** FOM state's natural key is the slot
   (`fom-0`..`fom-N`), not the workflow. Two slots can legitimately use
   the same workflow with different parameters.

## The design

### Structure: `FOMOrchestrator` parallel to `JobOrchestrator`

A sibling class taking the same dependencies, with slot-keyed state:

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
    ) -> None: ...

    _slots: dict[FOMSlot, FOMSlotState]  # FOMSlot = "fom-0".."fom-N"

    def commit_slot(
        self,
        slot: FOMSlot,
        *,
        workflow_id: WorkflowId,
        source_name: str,
        output_name: str,
        params: dict,
    ) -> JobId: ...

    def release_slot(self, slot: FOMSlot) -> None: ...
    def reset_slot(self, slot: FOMSlot) -> None: ...
    def reconfigure_slot(self, slot: FOMSlot, **kw) -> None: ...
```

`FOMSlotState` carries the workflow + source + output + params + the
`JobNumber` that `commit_slot` generated.

Consequences:

- The FOM job has its own `JobNumber`. `JobOrchestrator._workflows` is
  unaware of it. The regular workflow card cannot reach it.
- The regular workflow status list still *displays* the FOM-owned job
  (via `JobService` heartbeats), rendered read-only with a "managed by
  FOM panel" indicator.
- `commit_slot` sends `WorkflowConfig` + `BindStreamAlias` as one batch
  via `CommandService.send_batch` (one Kafka flush, atomic to the
  dashboard).

### Placement: panel inside the dashboard

A dedicated FOM panel, listing slots `fom-0`..`fom-N` with per-slot
state and actions (Configure / Reset / Reconfigure / Release). This is
the cheap default; with `FOMOrchestrator` as the boundary, moving to a
separate app later is a packaging change (a thin Panel app importing
`FOMOrchestrator` and the slot widgets). Do not pre-build that.

### Configuration UI: reuse the wizard, slot from launch context

The slot is determined by which button the operator clicked on the FOM
panel, not by a wizard step. The wizard reduces to two steps, both
reused from the existing plot-config modal machinery
(`widgets/wizard.py` is generic; `widgets/plot_config_modal.py:279`
defines `WorkflowAndOutputSelectionStep`):

- **Step 1**: `WorkflowAndOutputSelectionStep`, reused as-is. Add an
  optional output predicate so non-scalar outputs can be filtered out.
- **Step 2**: workflow parameter configuration — wrap the existing
  parameter widget (driven by `WorkflowConfigurationAdapter`, 78 lines)
  as a `WizardStep`. Final commit calls
  `FOMOrchestrator.commit_slot(slot=<launch-context-slot>, ...)`.

Each slot row in the FOM panel has a single Configure button. When the
slot is unbound, it opens an empty wizard. When the slot is bound, it
opens the wizard prefilled from the slot's current state (workflow,
output, params). This subsumes both "configure" and "reconfigure" into
one entry point and gives the operator a way to *inspect* the current
binding by opening and closing the wizard without committing.

The replace-and-stop confirmation guard for an already-bound slot
fires as a modal before the final wizard commit (not as a wizard
step), naming the alias and the workflow being replaced.

The wizard skeleton, the workflow+output selector, and the parameter
widget are all reused unchanged. New code is roughly: `FOMOrchestrator`,
a thin shim wrapping the parameter widget as a `WizardStep`, and the
FOM panel.

## State and persistence — v1 minimum

v1 deliberately ships without dashboard-side persistence, without
multi-session reconstruction, and without restart correctness. We do
not yet have operational experience to justify those costs; revisit
once practice tells us where the pain is.

### What v1 does

- `FOMOrchestrator._slots` is in-memory, single-session. Configured
  slots survive within a session but not across dashboard restarts.
- The originating session is the only session that sees the rich slot
  state (workflow + params + output) on the FOM panel. Other sessions
  see slots as "configured elsewhere — value live, full state on
  originating session" (or simply blank rows; see open questions).
- Restart-during-experiment is recovered by hand: the previous FOM job
  is visible in the regular workflow status list as an ordinary
  running job, and can be stopped from there if needed. Auto-stop in
  the backend (recommended below) means a fresh `Bind` to the same
  slot from a new session cleans up the previous job automatically.

### Backend change: auto-stop on alias rebind/unbind

Without it, reconfigure requires the dashboard to track the previous
`(JobId, source)` so it can compose unbind + stop + config + bind. With
it, reconfigure is `WorkflowConfig + Bind` and the backend handles the
cascade.

Proposal:

- **`UnbindStreamAlias`**: holder service stops the bound job, then
  removes the binding, then ACKs.
- **`BindStreamAlias` for an alias already bound elsewhere**: current
  holder stops its job and clears its binding silently; the new actor
  binds and ACKs. Today's no-replace rule (concept-doc D1) becomes
  replace-and-stop.

This makes the FOM slot the owner of its job's lifecycle. The mechanism
remains generic in code, but the lifecycle coupling becomes a property
of the binding: holding an alias means owning the bound job. If a later
use case wants the no-coupling variant, it can be added as a flag on
the bind command — non-breaking, deferred until needed.

Adopting auto-stop is the load-bearing simplification that lets v1
ignore dashboard persistence entirely. Without it, in-memory-only slot
state would orphan jobs on every dashboard restart.

### Deferred to follow-ups (post-v1)

These are real concerns, but premature to solve before we run v1:

- **Heartbeat extension** (`bound_aliases: list[str]` on `JobStatus`).
  Enables: multi-session slot visibility, badging FOM-owned jobs in the
  regular status list, restart-aware slot reconstruction.
- **Slot-state reconstruction from heartbeats** in `FOMOrchestrator`.
  Builds on the heartbeat extension.
- **Read-only badging of bound jobs in the regular workflow status
  list**. Builds on the heartbeat extension. Note that with the
  Case-C structural separation, the regular workflow card cannot reach
  the FOM job in the first place — `stop_workflow` operates on
  `state.current.job_ids()` which excludes the FOM `JobNumber`. So
  v1's lack of badging is a clarity gap, not a safety gap. The Stop
  button silently does nothing for the FOM job; not ideal, but not
  dangerous.

## What does not change

- Backend stream-alias schema and plumbing (already shipped).
- `JobOrchestrator`, `WorkflowConfigurationAdapter`, the wizard
  framework, the workflow+output selector, or any plot widget. They
  acquire one new consumer (`FOMOrchestrator`) without internal
  changes.
- `LIVEDATA_DATA` copy semantics. Verification by viewing the FOM
  workflow's regular plot still works: subscribe via `PlotOrchestrator`
  to the FOM job's `JobNumber` like any other plot.

## Implementation sketch (v1)

1. **Backend**: switch alias semantics to auto-stop on rebind/unbind.
   Update `figure-of-merit-phase1-impl.md` D1 and concept-doc D1.
2. **Dashboard**: `FOMOrchestrator` with in-memory `_slots` (no
   reconstruction, no persistence).
3. **Dashboard**: `commit_slot` sends `WorkflowConfig + BindStreamAlias`
   in one batch; `release_slot` sends `UnbindStreamAlias` (which
   auto-stops the job in the backend); `reconfigure_slot` is just
   `commit_slot` to a slot that may or may not be currently bound.
4. **Dashboard**: wrap the existing parameter widget as a `WizardStep`.
   Wire it together with `WorkflowAndOutputSelectionStep` into a
   two-step FOM wizard, parameterized by the launching slot. Terminal
   action is `FOMOrchestrator.commit_slot(slot, ...)`. When opened on
   a bound slot, prefill from current state and gate the commit behind
   a replace-confirmation modal.
5. **Dashboard**: FOM panel widget listing slots with per-slot status,
   live value, and a Configure button. Button launches the wizard
   scoped to that slot.
6. **Dashboard**: tests for `FOMOrchestrator` covering commit, release,
   reset, reconfigure, multi-slot.

Step 1 is backend; the rest are dashboard. Steps 2 and 4 can proceed
in parallel.

## Open questions

- **Slot count and naming**. Plan defaults to `fom-0`..`fom-7`. Fixed
  list (simpler, matches the documented convention) vs arbitrary names
  (preserves the alias mechanism's genericity in the UI). Recommend
  fixed list rendered as the panel's slot rows; the alias schema stays
  string-typed so other consumers can use any naming.
- **Output eligibility predicate**. The wizard's Step 1 should filter
  workflow outputs to those producing a scalar. Concrete predicate
  TBD — likely "output type is `sc.DataArray` with scalar shape", but
  may need workflow-author annotation if the runtime type isn't known
  ahead of time.
- **Embedded plot in FOM panel**. Show a small live plot of the bound
  output? Probably yes long-term, no in the initial cut. The bound
  output's `LIVEDATA_DATA` plot is available in the regular dashboard.
- **What does the FOM panel show in a session that did not configure
  the slot?** With in-memory v1 state, the originating session has
  full info; other sessions and post-restart sessions have none. Two
  options: (a) panel shows empty rows, operator reconfigures from
  scratch when needed; (b) panel subscribes to `LIVEDATA_FOM` and
  shows alias + last value + `(JobId, output)` from the mirror, but no
  workflow/params metadata. (b) is cheap and useful for sanity-checking
  "is fom-0 actually producing?" without solving full reconstruction.

## Out of scope

- Phase 2 backend work (dedicated `_livedata_fom` topic).
- Phase 3 backend work (scheduled reset/start).
- NICOS-side interface — owned by the NICOS team.
- Separate frontend app — defer until operational evidence justifies
  it; the structural design here makes that future move cheap.
