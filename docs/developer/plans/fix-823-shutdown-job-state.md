# Fix #823: Backend shutdown leaves jobs in "active" state

## PR: #854, Branch: `fix-823-shutdown-job-state`

## Problem Statement

When a backend worker shuts down (gracefully or crashes), the dashboard should reflect this clearly and allow the user to recover (stop the stale workflow and restart).

## Iteration History

### Iteration 1: Backend sends final job statuses (commit `54e4bfdd`)

**What was done:**
- Added `JobState.stopped` to the enum.
- Added `JobManager.mark_all_stopped()` — sets all jobs to `stopped` without removing tracking.
- Changed `_send_final_heartbeat()` to send both job and service statuses (previously only sent service status).
- Called `mark_all_stopped()` in `OrchestratingProcessor.shutdown()` before the final heartbeat.
- On the dashboard side in `_get_status_and_timing()`: added `stopped` to the worst-state priority, and distinguished stale-with-prior-heartbeats ("LOST", red) from never-seen ("PENDING", blue).

**Problem:** The dashboard showed STOPPED briefly, then flipped back to ACTIVE. Root cause: `shutdown()` is called while the worker thread is still running. The worker loop's `process_jobs()` overwrites job states back to `active` after `mark_all_stopped()` sets them to `stopped`. Then `_report_status()` in the next `process()` cycle sends the overwritten `active` states.

### Iteration 2: Move job marking to `report_stopped()` (commit `1d9fe95c`)

**What was done:**
- Moved `mark_all_stopped()` and job-status heartbeat from `shutdown()` to `report_stopped()`, which runs after the worker thread has joined.
- `shutdown()` now only sends a service-level "stopping" heartbeat via new `_send_service_heartbeat()` (no job statuses, avoiding the race).
- `report_error()` also marks jobs stopped (called from the exception handler after the loop exits).
- Added `tests/services/shutdown_test.py` with three tests enforcing the ordering contract.

**Result:** Backend side now works correctly. The race is eliminated.

### Iteration 3: Widget rebuild on backend stop (commit `597a23fc`)

**What was done:**
- Added `_backend_stopped` flag to `WorkflowStatusWidget`.
- In `refresh()`, detected when status transitions to STOPPED/LOST and triggered `_build_widget()`.
- In `_create_header_buttons()`, hid stop/reset buttons when `_backend_stopped` was True.
- Added early return in `_get_status_and_timing()` for `worst_state == stopped` → returns `'Backend shut down'` timing text instead of continuing to calculate elapsed time.

**Problem:** Once `_backend_stopped` was set and buttons were hidden, the user had no way to clear the stale `active_job_number` and recover. No stop button to click, no play button either (play requires uncommitted changes). The widget was stuck in a dead state with no interactive controls.

### Iteration 4: Keep buttons visible (commit `9af78b9c`)

**What was done:**
- Removed the `not self._backend_stopped` guard from `_create_header_buttons()`, so stop/reset always show when `active_job_number is not None`.
- Removed the `_backend_stopped` flag entirely since buttons no longer depend on it, and the incremental refresh path already updates badge/dots/timing.

**Problem:** Now the status says STOPPED but buttons still show stop/reset (the active-workflow buttons), which is confusing and wrong. The buttons should show the stopped-workflow controls (play button) instead.

## Root Cause Analysis

The widget has two independent code paths that determine what the user sees:

1. **`_get_status_and_timing()`** — computes the status badge, timing text, and per-source dots. Updated every refresh cycle via the incremental path in `refresh()`.
2. **`_create_header_buttons()`** — computes which buttons to show (play, stop, reset). Only runs during `_build_widget()` (full rebuild).

A full rebuild is triggered only when the orchestrator's `workflow_state_version` changes (staging, commit, user-initiated stop). It is NOT triggered when job statuses change.

The widget determines "is this workflow running?" from two independent sources:
- **Orchestrator** (`active_job_number`): Reflects dashboard-side state. Set by `commit_workflow()`, cleared by `stop_workflow()`.
- **Job service** (job heartbeats): Reflects backend-side state. Updated by Kafka heartbeats.

When the backend shuts down, these two sources disagree: the orchestrator still thinks the workflow is running (`active_job_number` is set), but the job service says all jobs are stopped. No amount of widget-level patching can cleanly resolve this — the shared state must be reconciled.

## Suggested Approach for Next Session

### Option A: Auto-stop on backend shutdown detection

When the dashboard detects that all jobs for a workflow are `stopped` (from heartbeats), automatically call `stop_workflow()` to clear `active_job_number`. This would trigger the version bump and full rebuild through the existing path. The widget would naturally show the "stopped" state with play button.

Risk: this changes user-visible state without user action. But arguably, the backend already stopped the workflow — the dashboard is just acknowledging reality.

### Option B: Reconcile in the orchestrator

Add a method on `JobOrchestrator` that checks the job service for stopped/stale statuses and updates its internal state accordingly. Called from the periodic refresh cycle. This keeps the reconciliation logic out of the widget.

### Option C: Full rebuild on status transition (widget-level)

When `_get_status_and_timing()` returns STOPPED/LOST, trigger a full `_build_widget()` and have `_create_header_buttons()` check the effective status. The tricky part is letting the user clear the stale state while showing the right buttons.

### Architecture Constraint (from `.claude/rules/dashboard-widgets.md`)

Widgets detect changes via version counters on shared state, not via push callbacks. All widget updates run inside `SessionUpdater._batched_update()`. Widget event handlers should only call methods on shared components, never rebuild directly. This makes Option A or B more aligned with the architecture than Option C.

## Key Files

| File | Role |
|------|------|
| `src/ess/livedata/dashboard/widgets/workflow_status_widget.py` | Widget: `_get_status_and_timing()`, `_create_header_buttons()`, `refresh()`, `_build_widget()` |
| `src/ess/livedata/dashboard/job_orchestrator.py` | `active_job_number`, `stop_workflow()`, version tracking |
| `src/ess/livedata/dashboard/job_service.py` | Job heartbeat storage, `is_status_stale()` |
| `src/ess/livedata/core/orchestrating_processor.py` | Backend shutdown lifecycle: `shutdown()`, `report_stopped()`, `report_error()` |
| `src/ess/livedata/core/job_manager.py` | `mark_all_stopped()`, `get_all_job_statuses()` |
| `src/ess/livedata/core/job.py` | `JobState` enum (including `stopped`), `ServiceState` |
| `tests/services/shutdown_test.py` | Tests for backend shutdown lifecycle ordering |
| `tests/dashboard/widgets/workflow_status_widget_test.py` | Tests for widget status display |

## Current Branch State

Commits on branch (oldest first):
1. `54e4bfdd` — Initial backend + dashboard changes (has stale-detection and worst-state fixes that work)
2. `597a23fc` — Widget rebuild on backend stop (`_backend_stopped` flag, early return for stopped timing)
3. `1d9fe95c` — Race fix: moved `mark_all_stopped()` to `report_stopped()`
4. `5b4da630` — Shutdown lifecycle tests
5. `9af78b9c` — Removed `_backend_stopped` from button logic (current HEAD, buttons still wrong)

### Iteration 5: Orchestrator-level reconciliation (current)

**What was done:**
- Moved the responsibility for detecting backend shutdown from the widget to the orchestrator.
- Added `reconcile_stopped_jobs()` on `JobOrchestrator`: checks all active workflows, and if all jobs report `stopped` state (fresh, non-stale heartbeats), calls `_deactivate_workflow()` to clear `active_job_number`, bump version, and notify subscribers.
- Factored `stop_workflow()`: extracted shared cleanup into `_deactivate_workflow()`. `stop_workflow()` sends stop commands then calls it; `reconcile_stopped_jobs()` calls it without sending commands (backend is already dead).
- Injected `JobService` into `JobOrchestrator` for status queries.
- Called `reconcile_stopped_jobs()` from `DashboardServices._update_loop()` (background thread, alongside existing `orchestrator.update()`).
- Cleaned up widget: removed LOST state (never observed in practice, independent concern), removed `stopped` early return in `_get_status_and_timing()`, removed `'lost'` color entry. The widget now relies entirely on `active_job_number is None` for the STOPPED state.
- Removed widget-level tests for LOST/stopped detection; added 8 orchestrator-level reconciliation tests.

**Result:** The version bump from `_deactivate_workflow()` triggers widget rebuild through the existing version-based polling path. The widget naturally shows STOPPED with a play button — no widget-level hacks needed.
