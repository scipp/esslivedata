# NICOS derived devices: dashboard UI/UX

Composes with the backend/identity design in `nicos-derived-devices.md`. This plan
owns the dashboard surface: how an operator sees which running workflows are exposed
to NICOS as derived devices, and how those jobs are protected from accidental reset /
stop / reconfigure -- without locking unrelated workflows.

Out of scope: device naming / identity, topic projection, the static
`(workflow, source, output) -> device` contract (companion plan).

## Model: derived from the contract, no runtime activation

A derived device is a scalar output listed in the shared yaml contract. The backend
projects every yaml-listed output of a running `(workflow, source)` job, so a device
is **live iff its owning job is running**. There is no per-output "expose" toggle, no
persisted activation state, and no dashboard->backend control channel -- exposure is
declared once, in the yaml, and the dashboard only *reflects* and *protects* it.

The dashboard's role is read-only plus protection:

- show which running workflows bear a device (static lookup against the contract);
- gate reset / stop / reconfigure on those workflows behind explicit confirmation;
- offer a read-only overview of what is currently exposed.

**Why no activation toggle.** Runtime on/off would add a control channel and
persisted state the backend design does not have, and it fights NICOS's static-list
premise -- liveness would depend on an operator's UI state rather than on the
contract plus job state. A device cannot be exposed without its job running anyway,
and the projection is low-volume and harmless when NICOS is not scanning it, so
suppressing it buys nothing. If selective per-output suppression is ever genuinely
needed, it is a clean later addition.

**Lifecycle is workflow-level.** A job produces all its outputs together; start /
stop / reset / reconfigure act on the whole `(workflow, source)` job, never on a
single output. The per-output grain lives entirely in the yaml (which outputs are
devices); the gate acts on the owning job, the only granularity at which lifecycle
operations exist.

## Where it attaches

All workflow control already flows through `JobOrchestrator` (`commit_workflow`,
`stop_workflow`, `reset_workflow`), driven from the per-workflow card in
`WorkflowStatusWidget`. "Is this a device-bearing workflow?" is answered by a static
lookup: does a running job's `(workflow, source, output)` appear in the contract. No
new orchestrator, no new persisted state, no new store.

A self-contained prototype (a separate `NicosOrchestrator`, a sidebar modal over a
curated `config/nicos_workflows.py` list, fixed `job_number`s) lives on the unmerged
`nicos-workflows` branch. The unified-job model supersedes it, so that branch is
abandoned rather than merged -- there is nothing to delete here. Lift its reusable
ideas as inspiration: the device-bearing badge styling and the running-state
lockdown of controls.

## Concrete UI changes

All within the existing Workflows surface; no parallel control plane.

1. **Device-bearing badge** on the card header when a running workflow has at least
   one output in the contract (reusing the `nicos-workflows` prototype's badge
   styling). Marks the gated state at a glance.
2. **Per-output device marker** on the output chips in `_create_outputs_section`,
   indicating which outputs are exposed as devices (read-only; membership predicate
   shared with the backend contract).
3. **Confirmation dialog** intercepting stop / reset / reconfigure, only for
   device-bearing workflows. It names the affected device(s). The dashboard cannot
   know whether NICOS is actively scanning a device (one-way publish, separate
   system), so the dialog states plainly that the change is disruptive *iff* NICOS
   happens to be using the device -- it does not imply certainty. Each action is
   confirmed individually; no batching. Desired warning/confirmation popups/modals
   can be pulled from the (otherwise discarded) alternative approach in branch
   `figure-of-merit`. Note that *staging* new config should be allowed (user preparing
   for next scan), but *commit* must be gated.
4. **Read-only "Derived devices" overview** -- a sidebar modal (shape borrowed from
   the `nicos-workflows` prototype's list widget) listing currently exposed devices
   (contract
   intersect running jobs): device identity, source, owning workflow, run state, and
   a deep-link to the owning card. No controls.

## Multi-user / multi-browser

Nothing new to persist or reconcile: device-bearing status is derived each refresh
from shared `JobOrchestrator` job state and the static contract. Widgets are
per-session and rebuild on the polled version counter (see
`.claude/rules/dashboard-widgets.md`). The confirmation gate re-reads live job state
**inside the orchestrator method**, not only in the widget, so a stop that races a
job restart cannot act on stale state.

## Phased implementation

1. Device-bearing predicate (running job intersect contract) + header badge +
   per-output device marker in `WorkflowStatusWidget`.
2. Confirmation gate on stop / reset / reconfigure for device-bearing workflows,
   reading live job state, naming affected devices.
3. Read-only derived-devices overview modal.

All phases are dashboard-only; they depend on the backend only for the shared
contract / membership predicate.
