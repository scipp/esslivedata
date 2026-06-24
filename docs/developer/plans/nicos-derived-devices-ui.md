# NICOS derived devices: dashboard UI/UX

Status: design / decision record. Nothing here is implemented.

This plan is independent of, and composes with, the backend/identity design in the
companion plan `docs/developer/plans/nicos-derived-devices.md` (the device-identity,
topic-projection, and static-contract work). It does not exist yet at the time of
writing; this document only relies on the converged decisions summarised below and
makes no further backend claims. Where the two overlap (the activation flag and its
persisted shape), this plan owns the dashboard surface and defers the wire/contract
shape to the backend plan.

## Scope

How does the dashboard let an operator expose a scalar workflow output as a NICOS
*derived device*, and protect the underlying job from accidental reset / stop /
reconfigure while NICOS may be using it -- without locking unrelated workflows.

Out of scope: device naming / identity, topic projection, the static
`(workflow, source, output) -> device` contract. Those are backend concerns (companion
plan).

## Decided context (do not re-open)

- One UNIFIED job per `(workflow, source)`. No dedicated NICOS jobs, no
  `NicosOrchestrator`, no fixed `job_number`. The prototype below is being removed.
- A derived device is a designated scalar output of a running job, projected onto a
  dedicated low-volume topic under a STABLE identity =
  `WorkflowId(instrument, name, version)` + `source_name` + `output_name`. `job_number`
  is dropped from the external identity.
- Protection against accidental disruption is via UI GATING, not job isolation.

## The prototype being replaced

The current branch carries a parallel, self-contained NICOS UI that the unified-job
decision supersedes. It is the source of reusable ideas, not the target design:

- `src/ess/livedata/dashboard/nicos_orchestrator.py` -- `NicosOrchestrator` drives a
  curated list under fixed `job_number`s, with its own `commit`/`stop`/`reset`
  (`nicos_orchestrator.py:165`, `:212`, `:231`) entirely separate from `JobOrchestrator`.
- `src/ess/livedata/dashboard/widgets/nicos_panel.py` -- a sidebar launch button opens a
  `pn.Modal` listing curated entries (`NicosWorkflowListWidget`,
  `nicos_panel.py:163`); each entry (`NicosWorkflowWidget`, `:36`) shows a
  RUNNING/STOPPED badge and, while running, hides configure and offers only Stop/Reset
  (`_create_buttons`, `nicos_panel.py:99`) -- a hard lockdown.
- `src/ess/livedata/config/nicos_workflows.py` -- the curated YAML list and fixed
  `job_number`s.

Reusable: the badge/"this is exposed" affordance and the running-state lockdown of
controls. Discardable: the separate orchestrator, the fixed `job_number`, the curated
list as the *only* entry point, and the separate modal as the *primary* surface.

## Where workflow control lives today

All start/stop/reset for ordinary workflows flows through one component,
`JobOrchestrator`:

- `commit_workflow` (`job_orchestrator.py:352`) -- start / restart (stops the prior
  `JobSet`, sends new configs).
- `stop_workflow` (`job_orchestrator.py:754`).
- `reset_workflow` (`job_orchestrator.py:836`).

These are the three control paths a gate must intercept. They are driven from
`WorkflowStatusWidget` in `widgets/workflow_status_widget.py`: the per-workflow card's
header buttons (`_create_header_buttons`, `:404`) call `_on_stop_click` /
`_on_reset_click` / `_on_commit_click` (`:1023`-`:1038`); configuration is staged via a
`ConfigurationModal` opened from the gear (`_on_gear_click`, `:977`). The card is the
existing, discoverable home of every control we must gate. The outputs of a workflow are
already rendered here as chips (`_create_outputs_section`, `:663`), iterating
`spec.get_output_views()` -- the natural anchor for a per-output activation control.

Persistence: `JobOrchestrator` already round-trips per-workflow state through a
`ConfigStore` (`config_store.py`), loading on init (`_load_configs_from_store`,
`job_orchestrator.py:175`) and writing on commit/stop (`_persist_state_to_store`, `:481`).
The store is a `MutableMapping[str, dict]`, file-backed YAML per instrument, shared
across sessions via one cached instance (`ConfigStoreManager.get_store`,
`config_store.py:306`). Activation state has a home here with no new infrastructure.

## Options

### Option A -- global dashboard lock + per-change confirmation

A single dashboard-wide lock/unlock toggle. Independently, any control acting on a
workflow that has a derived-device output prompts a warning/confirmation.

- (+) One visible mode; small surface.
- (-) Coarse. The lock conflates "I am being careful" with "this output is a device".
  Operators reconfigure unrelated workflows constantly; a global lock is either left off
  (no protection) or fights routine work (gets disabled, then forgotten off).
- (-) "Has a derived-device output" is not a dashboard-observable fact unless we already
  track activation -- so A still needs per-output activation state to know *which*
  changes to confirm. A is then Option B plus a redundant global toggle.
- (-) Confirmation-only protection trains click-through. Without a persisted "this is
  live" state, the warning cannot say *what* depends on the job.

### Option B -- per-output "expose as derived device" activation

The workflow card gains, per scalar output, an "expose to NICOS" toggle. Activating an
output:

1. signals the backend to project that `(workflow, source, output)` onto the device
   topic (mechanism owned by the companion plan), and
2. marks the owning workflow as device-bearing, which gates subsequent reset / stop /
   reconfigure of that workflow behind explicit confirmation.

Deactivation requires confirmation and stops the projection. "Device is live" and
"controls are gated" become one explicit, visible, persisted state.

- (+) Gating is scoped to exactly the workflows that bear a device. Unrelated workflows
  stay friction-free.
- (+) The activation toggle *is* the discoverability surface: exposure is a deliberate,
  visible act, not a hidden side effect of starting a job.
- (+) The persisted activation set lets every gate/warning name the affected device(s),
  and survives restart because it mirrors backend reality.
- (-) More state to persist and reconcile with backend truth.
- (-) Per-source granularity question (below) must be answered.

### Option C (proposed) -- B, with activation pinned to the running job, surfaced both
in the card and in a read-only "Derived devices" overview

Same activation model as B. Additions:

- Activation is only meaningful for a *running* `(workflow, source)`; the toggle is
  enabled only when that source has an active job. Activating a not-yet-running output
  is rejected with guidance ("start the workflow first"), avoiding a "device exists but
  nothing publishes it" state.
- A read-only overview panel (reusing the sidebar-modal shape of the prototype's
  `NicosWorkflowListWidget`) lists all currently active devices across workflows: device
  identity, source, owning workflow, run state. This is the operator's single answer to
  "what is exposed right now," replacing the prototype's curated modal with a *derived*
  view over real activation state. It carries no controls beyond a deep-link to the
  owning workflow card.

Recommendation: **Option C** (an extension of B). Option A is rejected: it cannot avoid
tracking activation anyway, and a global lock is the wrong granularity for a multi-day,
mixed-use dashboard.

## Recommendation rationale

- The requester leans B; pressure-testing confirms B's core (explicit per-output
  activation tying "live" to "gated") is right, and that A reduces to B-plus-noise.
- C closes B's two soft spots: it forbids the incoherent "activated but not running"
  state, and it gives operators a positive, restart-surviving answer to "what is exposed
  now" instead of inferring it from scattered card badges.
- Single source of truth: one activation set, owned by `JobOrchestrator` (the component
  that already owns job lifecycle and persistence), consumed by both the card gate and
  the overview. No second orchestrator.

## Concrete UI changes

All within the existing Workflows surface; no parallel control plane.

1. **Per-output activation control** in `WorkflowStatusWidget._create_outputs_section`
   (`workflow_status_widget.py:663`). Each scalar output chip gains a toggle (a
   `create_tool_button` with `lt-tool`/`lt-wf-{name}` hooks per the automation contract).
   Enabled only when the corresponding source has an active job. State reflects the
   persisted activation set.

2. **Device-bearing badge** on the card header (next to the status badge,
   `_create_header`, `:332`) when any output is active. Reuse the prototype's badge
   styling. This makes the gated state visible at a glance without expanding the card.

3. **Confirmation dialog** intercepting `_on_stop_click` / `_on_reset_click` /
   `_on_commit_click` and output-deactivation, only when the workflow is device-bearing.
   The dialog names the affected device(s) (possible because activation is tracked). The
   gate lives in the widget event handlers but consults shared
   `JobOrchestrator`-owned state so all sessions agree.

4. **Derived-devices overview** -- a read-only sidebar modal (shape borrowed from
   `NicosWorkflowListWidget`, `nicos_panel.py:163`) driven by the activation set, not a
   curated YAML list. Deep-links to owning workflow cards.

The prototype files (`nicos_orchestrator.py`, `nicos_panel.py`, `config/nicos_workflows.py`)
and their wiring in `dashboard_services.py` / `reduction.py` are removed.

## State model, persistence, restart

- Activation is keyed by the *stable external identity* `(WorkflowId, source_name,
  output_name)` -- the same key NICOS uses -- not by `job_number` (which is internal and
  changes across restarts/restarts-of-the-job). A `job_number` change must NOT drop a
  device.
- Owned by `JobOrchestrator`, persisted in the existing per-workflow `ConfigStore` entry
  (extend the persisted dict written by `_persist_state_to_store`,
  `job_orchestrator.py:481`, with an `active_devices` set; load it in
  `_load_configs_from_store`, `:175`). No new store, no new file.
- Restart: activation reflects backend reality and MUST survive dashboard restart. On
  load, the dashboard re-derives the gated state from the persisted activation set. Open
  question: whether the dashboard re-asserts the projection to the backend on startup, or
  the backend persists projection independently and the dashboard only mirrors it -- this
  is the seam with the companion plan and must be settled there.

## Multi-user / multi-browser

Orchestrators and the `ConfigStore` are singletons shared across sessions; widgets are
per-session and poll a version counter to rebuild (see
`.claude/rules/dashboard-widgets.md` and `WorkflowStatusWidget.refresh`,
`workflow_status_widget.py:1040`). Activation must follow the same discipline:

- Toggling activation mutates shared `JobOrchestrator` state and bumps the workflow's
  state version; every session's card rebuilds on its next refresh tick. No direct
  cross-session push.
- The confirmation dialog reads live shared state at click time, so operator B's gate
  reflects operator A's activation even if B's card has not yet visually refreshed.
- Open question: two operators acting within one refresh interval (e.g. A deactivates
  while B stops). The mutation is serialised in the shared orchestrator; the loser sees a
  rebuilt card. Acceptable, but confirm no torn intermediate state (e.g. a stop that
  races a deactivate) -- likely needs the gate check to re-read state inside the
  orchestrator method, not only in the widget.

## Knowing whether NICOS is actively scanning

We likely CANNOT know directly whether a device is currently feeding a live NICOS scan
-- NICOS is a separate system and the device topic is a one-way publish. Consequences:

- The gate cannot be "block iff a scan is running." It must be "this output is *exposed
  as a device*; changing it is disruptive iff NICOS happens to be using it." The
  confirmation must state this uncertainty plainly rather than imply certainty.
- Activation is therefore a *standing declaration of intent to expose*, not a live-usage
  signal. This is the honest model and motivates making activation explicit and visible
  (so the human, who may know the scan schedule, carries the judgement).
- Open question: is any back-channel from NICOS available (even a coarse "scan active"
  signal) that could upgrade the warning from "may be in use" to "in use now"? If not,
  do not fake it.

## Open questions

- Per-source granularity: a workflow runs per source; a device is per
  `(workflow, source, output)`. Does the card toggle activate one output across *all*
  active sources, or per source? Per-source is more precise but multiplies UI; per-output
  across-sources is simpler but couples sources. Leaning per-output-per-source, grouped
  in the card by source (mirroring the existing staging grouping). Needs a decision.
- Which outputs are eligible? Only scalar cumulative outputs are valid devices. The card
  must show the toggle only on eligible outputs (filter `get_output_views()` /
  `outputs.model_fields`). The eligibility predicate is shared with the backend contract.
- Re-assertion of projection on dashboard startup vs. backend-owned persistence (above).
- Does deactivation while a job keeps running need a grace period, or is an immediate
  stop-projecting acceptable? Likely immediate; confirm with NICOS expectations.
- Confirmation fatigue: should repeated routine commits within a session be batchable, or
  is per-action confirmation always required for device-bearing workflows?

## Phased implementation outline

1. Remove the prototype (`nicos_orchestrator.py`, `nicos_panel.py`,
   `config/nicos_workflows.py`, wiring). Land alongside the unified-job backend change so
   nothing references fixed `job_number`s.
2. Add the activation set to `JobOrchestrator` state + `ConfigStore` persistence; expose
   query/mutate methods and version bumping. No UI yet; unit-test load/persist/restart of
   activation keyed by stable identity.
3. Per-output activation toggle + device-bearing header badge in `WorkflowStatusWidget`,
   enabled only for running sources and eligible outputs.
4. Confirmation gate on stop/reset/reconfigure/deactivate for device-bearing workflows,
   reading live shared state, naming affected devices.
5. Read-only derived-devices overview modal driven by the activation set.
6. Reconcile dashboard activation with backend projection (seam with companion plan):
   startup re-assertion or mirror, per its decision.

Each phase is independently reviewable; phases 2-5 are dashboard-only and do not block on
the backend beyond the eligibility predicate and the persisted-shape agreement.
