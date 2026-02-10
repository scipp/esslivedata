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

**Correct pattern** (version-based polling — all sessions detect changes):
```python
def __init__(self, orchestrator):
    self._last_state_version: int | None = None

def _on_stop_clicked(self, event):
    self.orchestrator.stop_workflow(workflow_id)
    # Don't rebuild here - refresh() will detect the version change

def refresh(self):
    # Called from SessionUpdater periodic callback in batched context
    version = self.orchestrator.get_workflow_state_version(self._workflow_id)
    if version != self._last_state_version:
        self._last_state_version = version
        self._build_widget()  # GOOD: Each session detects & rebuilds independently
```

**Why this matters**: Controllers, orchestrators, and services are shared across all browser sessions (singletons), but each session has its own widget instances. Shared components increment a version counter on state changes. Each session's periodic callback polls the version and rebuilds when it changes, ensuring updates run in the correct session context with batched recomputation.

**Key principles**:
- Widgets detect changes via version counters on shared state, not via push callbacks
- All widget updates run inside `SessionUpdater._batched_update()` for efficient recomputation
- Widget event handlers should only call methods on shared components, never rebuild directly

## Icons

Do not use Unicode characters for button icons. Use embedded SVG icons from `dashboard/icons.py` via `get_icon()`. Use the `create_tool_button()` helper from `dashboard/buttons.py` for consistent styling.

## Avoiding flicker

Make sure all widget-updates that touch more than a single widget (or a single widget multiple times) use `pn.io.hold()`.