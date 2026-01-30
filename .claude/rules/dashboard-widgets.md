---
paths: src/ess/livedata/dashboard/widgets/**/*.py
---

# Dashboard Widget Patterns

## Cross-Session Synchronization

**Problem**: Widgets that update themselves directly after user actions break multi-session synchronization.

**Wrong pattern** (breaks cross-session sync):
```python
def _on_stop_clicked(self, event):
    self.controller.stop_workflow(workflow_id)
    self._build_widget()  # BAD: Only updates THIS session's widget
```

**Correct pattern** (all sessions stay synchronized):
```python
def __init__(self, controller):
    # Subscribe to lifecycle events from the shared controller
    controller.subscribe(on_workflow_stopped=self._on_workflow_stopped)

def _on_stop_clicked(self, event):
    self.controller.stop_workflow(workflow_id)
    # Don't rebuild here - let the subscription callback handle it

def _on_workflow_stopped(self, workflow_id):
    self._build_widget()  # GOOD: All subscribed widgets rebuild
```

**Why this matters**: Controllers, orchestrators, and services are shared across all browser sessions (singletons), but each session has its own widget instances. When Session A triggers an action, Session B's widgets won't know about it unless all widgets subscribe to events from the shared component.

**Key principles**:
- Widgets must react to events from shared components (controllers/orchestrators/services), not update themselves after triggering actions
- Shared components notify all subscribers when state changes
- Widget event handlers should only call methods on shared components, never rebuild directly

## Icons

Do not use Unicode characters for button icons. Use embedded SVG icons from `dashboard/icons.py` via `get_icon()`. Use the `create_tool_button()` helper from `dashboard/buttons.py` for consistent styling.

## Avoiding flicker

Make sure all widget-updates that touch more than a single widget (or a single widget multiple times) use `pn.io.hold()`.