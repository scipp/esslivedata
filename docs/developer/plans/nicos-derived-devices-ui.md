# NICOS derived devices: dashboard UI/UX

Composes with the backend/identity design in `nicos-derived-devices.md`. This plan
owns the dashboard surface: how an operator exposes a scalar workflow output as a
NICOS *derived device*, and how the owning job is protected from accidental reset /
stop / reconfigure while a device is exposed -- without locking unrelated
workflows.

Out of scope: device naming / identity, topic projection, the static
`(workflow, source, output) -> device` contract (companion plan).

## Model: per-output activation, gated where control already lives

A derived device is a designated scalar output of a running job, projected under a
stable identity (`WorkflowId + source_name + output_name`). The dashboard exposes
one control: a per-output **"expose to NICOS" activation toggle** on the workflow
card. Activating an output:

1. signals the backend to project that `(workflow, source, output)` (mechanism in
   the companion plan), and
2. marks the owning workflow as **device-bearing**, gating subsequent reset / stop /
   reconfigure of that workflow behind explicit confirmation.

Activation is a **standing declaration of intent to expose** -- persisted and
visible. "Device is live" and "controls are gated" are one explicit state.

**Gating, not isolation.** Even a dedicated job must be reconfigurable, so a UI gate
scoped to device-bearing workflows protects exactly what needs protecting while
leaving unrelated workflows friction-free. A dashboard-wide lock is the wrong
granularity for a multi-day, mixed-use dashboard: it conflates "I am being careful"
with "this output is a device", and to know which changes to confirm it would need
the same per-output activation state anyway.

## Where it attaches

All workflow control already flows through one component, `JobOrchestrator`
(`commit_workflow`, `stop_workflow`, `reset_workflow`), driven from the per-workflow
card in `WorkflowStatusWidget`. The card already renders a workflow's outputs as
chips (`_create_outputs_section`, iterating `spec.get_output_views()`) -- the
natural anchor for a per-output activation control -- and round-trips per-workflow
state through the per-instrument `ConfigStore`. Activation has a home here with no
new infrastructure and no second orchestrator.

The existing prototype (a separate `NicosOrchestrator`, a sidebar modal over a
curated `config/nicos_workflows.py` list, fixed `job_number`s) is **removed**: the
unified-job model supersedes it. Its reusable ideas are the device-bearing badge
and the running-state lockdown of controls.

## State, persistence, restart

- Activation is keyed by the **stable external identity** `(WorkflowId,
  source_name, output_name)` -- the key NICOS uses -- never by `job_number`. A
  `job_number` change (job restart) must not drop a device.
- Owned by `JobOrchestrator`, persisted in the existing per-workflow `ConfigStore`
  entry as an `active_devices` set, loaded on init. No new store, no new file.
- On dashboard startup the activation set is the **dashboard-owned source of
  truth**: the dashboard re-asserts the projection to the backend from it and
  re-derives the gated state. Activation survives restart because it mirrors a
  deliberate operator declaration, not transient job state.

## Multi-user / multi-browser

Orchestrator and `ConfigStore` are singletons shared across sessions; widgets are
per-session and rebuild on a polled version counter (see
`.claude/rules/dashboard-widgets.md`). Toggling activation mutates shared
`JobOrchestrator` state and bumps the workflow's version; every session's card
rebuilds on its next tick. The confirmation gate re-reads live shared state **inside
the orchestrator method**, not only in the widget, so a stop that races a deactivate
cannot act on stale state.

## Concrete UI changes

All within the existing Workflows surface; no parallel control plane.

1. **Per-output activation toggle** in
   `WorkflowStatusWidget._create_outputs_section`. Shown only on eligible outputs
   (scalar cumulative; the eligibility predicate is shared with the backend
   contract) and enabled only when that source has an active job. Activating a
   not-yet-running output is rejected with guidance ("start the workflow first") --
   there is no "device exists but nothing publishes it" state. Granularity is per
   `(output, source)`, grouped in the card by source (mirroring existing staging
   grouping). Uses `create_tool_button` with `lt-tool` / `lt-wf-{name}` hooks per
   the automation contract.
2. **Device-bearing badge** on the card header when any output is active (reusing
   the prototype's badge styling), visible without expanding the card.
3. **Confirmation dialog** intercepting stop / reset / reconfigure and
   output-deactivation, only for device-bearing workflows. It names the affected
   device(s). Deactivation stops the projection immediately (no grace period). The
   dashboard cannot know whether NICOS is actively scanning a device (one-way
   publish, separate system), so the dialog states plainly that the change is
   disruptive *iff* NICOS happens to be using the device -- it does not imply
   certainty. Each control action on a device-bearing workflow is confirmed
   individually; no batching.
4. **Read-only "Derived devices" overview** -- a sidebar modal (shape borrowed from
   the prototype's list widget) driven by the live activation set, not a curated
   yaml: device identity, source, owning workflow, run state, and a deep-link to the
   owning card. No controls.

## Phased implementation

1. Remove the prototype (`nicos_orchestrator.py`, `nicos_panel.py`,
   `config/nicos_workflows.py`, and their wiring in `dashboard_services.py` /
   `reduction.py`), landed alongside the unified-job backend change so nothing
   references fixed `job_number`s.
2. Add the activation set to `JobOrchestrator` state + `ConfigStore` persistence;
   query/mutate methods, version bumping. Unit-test load/persist/restart keyed by
   stable identity.
3. Per-output activation toggle + device-bearing header badge, enabled only for
   running sources and eligible outputs.
4. Confirmation gate on stop / reset / reconfigure / deactivate, reading live shared
   state, naming affected devices.
5. Read-only derived-devices overview modal.
6. Re-assert dashboard activation to the backend projection on startup (per State,
   persistence, restart).

Phases 2-5 are dashboard-only; they depend on the backend only for the eligibility
predicate and the persisted-shape agreement.
