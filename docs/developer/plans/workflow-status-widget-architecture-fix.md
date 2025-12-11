# WorkflowStatusWidget Architecture Fix

## Status

**Phase 1 COMPLETE** - Cleanup and revert done. Widget now uses orchestrator directly.

**Phase 2 TODO** - Add lifecycle subscriptions to JobOrchestrator for cross-session sync.

## Problem Summary

The `WorkflowStatusWidget` implementation in the `workflow-control-widget` branch has architectural issues that break cross-session synchronization and mix concerns incorrectly.

### Key Issues Identified

1. **Widget updates itself directly** instead of reacting to orchestrator events
   - Calls `_build_widget()` after operations (stop, commit, remove)
   - Other sessions' widgets don't get notified of state changes

2. ~~**Mixed dependencies** - widget uses both `WorkflowController` and `JobOrchestrator` inconsistently~~ **FIXED**
   - ~~Stop goes through controller~~
   - ~~Commit goes directly to orchestrator~~
   - ~~Config manipulation goes directly to orchestrator~~

3. ~~**Wrong changes to WorkflowController**~~ **FIXED (reverted)**
   - ~~Added `stage_workflow()` with "clear all + stage all" behavior (legacy pattern)~~
   - ~~New widget needs per-source staging without clearing others~~
   - ~~Added `stop_workflow()` as pointless pass-through~~

4. **Missing notification mechanism** in JobOrchestrator (TODO)
   - `commit_workflow()` notifies via `_notify_workflow_available()` (but for PlotOrchestrator, not widgets)
   - `stop_workflow()` doesn't notify anyone
   - No notification for staging changes

## Architecture Decisions

### Shared vs Per-Session Components

**Shared (singleton across all browser sessions):**
- JobOrchestrator
- WorkflowController
- JobService
- CommandService

**Per-session (one instance per browser tab):**
- WorkflowStatusWidget
- WorkflowStatusListWidget
- Other UI widgets

### Dependency Structure

**Legacy widget path (unchanged):**
```
LegacyWidget → WorkflowController → JobOrchestrator → CommandService
```

**New WorkflowStatusWidget path:**
```
WorkflowStatusWidget → JobOrchestrator → CommandService
```

WorkflowController is NOT used by the new widget. It remains solely for legacy widget support.

### Config Modal Flow

The `on_configure` callback is passed into the widget from outside. Wherever this callback is implemented:
- Creates the WorkflowConfigurationAdapter
- Provides a callback that does Pydantic→dict conversion
- Calls `orchestrator.stage_config()` for selected sources only (no clearing)

## Required Changes

### 1. Revert workflow_controller.py ✅ DONE

Reverted all changes from this branch to restore legacy-only behavior:
- Removed `stage_workflow()` method
- Removed `stop_workflow()` method
- Reverted `create_workflow_adapter()` to use `start_callback` → `start_workflow()`

### 2. Add Lifecycle Subscriptions to JobOrchestrator (TODO)

Add a widget-facing subscription API:
- `subscribe_to_widget_lifecycle()` - returns SubscriptionId
- `unsubscribe_from_widget_lifecycle()` - cleanup

Callbacks needed:
- `on_staged_changed(workflow_id)` - staging area modified
- `on_workflow_committed(workflow_id)` - workflow started
- `on_workflow_stopped(workflow_id)` - workflow stopped

Methods that must notify:
- `stage_config()` → notify staged_changed
- `clear_staged_configs()` → notify staged_changed
- `commit_workflow()` → notify workflow_committed
- `stop_workflow()` → notify workflow_stopped

### 3. Update WorkflowStatusWidget ✅ DONE (partial)

**Remove dependencies:** ✅
- Removed `WorkflowController` parameter and usage

**Use JobOrchestrator directly:** ✅
- `stop_workflow()` → `orchestrator.stop_workflow()`
- `commit_workflow()` → already using orchestrator
- Config queries → already using orchestrator

**Subscribe to lifecycle events:** (TODO - waiting for step 2)
- Subscribe in `__init__`
- Rebuild widget on relevant callbacks
- Unsubscribe on cleanup/shutdown

**Remove explicit rebuilds:** (TODO - waiting for step 2)
- Currently `_build_widget()` calls remain with TODO comments
- Will be removed once subscription callbacks are implemented

### 4. Update WorkflowStatusListWidget ✅ DONE (partial)

- Removed `WorkflowController` parameter ✅
- Pass orchestrator to child widgets ✅
- May need to subscribe to lifecycle events to rebuild specific widgets (TODO)

### 5. Update Config Modal Integration (TODO)

Wherever `on_configure` callback is implemented (likely in reduction.py or dashboard setup):
- Create adapter with callback that stages without clearing
- Callback does Pydantic→dict conversion inline
- Callback calls `orchestrator.stage_config()` for each selected source

### 6. Update Widget Creation Sites ✅ DONE

Updated wherever WorkflowStatusWidget/ListWidget are instantiated:
- Removed WorkflowController from constructor calls in `reduction.py`
- Updated test fixtures in `workflow_status_widget_test.py` and `plot_grid_tabs_test.py`

## Cross-Session Synchronization

With the shared JobOrchestrator and subscription mechanism:
1. Session A clicks Stop → shared orchestrator.stop_workflow()
2. Orchestrator notifies all subscribers (all sessions' widgets)
3. All widgets rebuild and show updated state

For backend-driven updates (JobStatus via Kafka):
- Already handled via JobService subscription
- Widget's `_on_status_update()` updates badge/timing
- Full rebuild happens via orchestrator lifecycle subscriptions

## Testing Considerations

- Test that multiple widgets (simulating multiple sessions) stay synchronized
- Test that staging one source group doesn't affect others
- Test lifecycle subscription/unsubscription
- Test that legacy widget still works via WorkflowController
