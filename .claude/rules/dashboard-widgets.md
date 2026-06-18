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

Do not use Unicode characters for button icons. Use embedded SVG icons from `dashboard/widgets/icons.py` via `get_icon()`. Use the `create_tool_button()` helper from `dashboard/widgets/buttons.py` for consistent styling.

## Stable CSS hooks for automation

Tool buttons render as label-less icons inside per-widget shadow DOM, so they carry
no text, `title`, or `aria-label` — leaving nothing semantic for browser automation
(Playwright) or screenshot tooling to target. To avoid brittle coordinate-clicking,
`create_tool_button()` tags every button with **committed, visually-inert CSS classes**:

- `lt-tool` on all tool buttons, plus `lt-tool-{icon_name}` (e.g. `lt-tool-settings`,
  `lt-tool-player-play`, `lt-tool-x`).
- Callers pass `css_classes=[...]` for context. Workflow rows add `lt-wf-{workflow_id.name}`
  (the WorkflowId *name* slug, not the display title), so a workflow's gear is
  `.lt-wf-monitor_histogram.lt-tool-settings`.

These classes have no associated style rules — adding/removing them is visually inert.
Treat them as a stable contract: do not drop them in refactors (a test in
`buttons_test.py` guards the helper). When adding tool buttons to a view that repeats
the same icon (e.g. per-grid/per-cell controls in `plot_grid_manager.py`), pass a
context `css_classes` entry so each instance is uniquely addressable.

## Model creation and visibility

`pn.Tabs(dynamic=True)` prevents Bokeh model creation for hidden tabs — only the active
tab's models exist in the document. This is the preferred mechanism for deferring cost.

`visible=False` on a Panel component only hides it via CSS. All Bokeh models are still
created and registered in the document. Do not use `visible=False` as a performance
optimization to defer widget cost — instead, avoid creating the component until it is
needed (lazy creation) or use `dynamic=True` containers.

Note that `dynamic=True` only gates Bokeh model creation. Python-side periodic callbacks
(e.g., `SessionUpdater` custom handlers) still run for all registered widgets regardless
of which tab is visible. Use an `is_visible` predicate to skip refresh work for hidden tabs.

## Colors and styling

All colors must come from `dashboard/widgets/styles.py`. Do not hard-code hex color values
or rgba strings in widget files. The shared module provides:

- `StatusColors` — semantic status indicators (ERROR, SUCCESS, WARNING, etc.)
- `HoverColors` — translucent hover backgrounds derived from StatusColors
- `Colors` — neutral palette (BORDER, BG_LIGHT, TEXT, TEXT_MUTED, etc.)
- `ErrorBox` / `WarningBox` — alert box color sets (BG, BORDER, TEXT)

`ButtonStyles` in `buttons.py` re-exports commonly used color+hover pairs
(e.g., `DANGER_RED`/`DANGER_HOVER`, `PRIMARY_BLUE`/`PRIMARY_HOVER`).

Widget-specific decorative colors (e.g., output chip colors, grid preview cell colors)
that are not shared across widgets may stay local.

Panel does not support CSS custom properties (`var()`) in `styles=` dicts or inline
HTML `style=` attributes — only in `stylesheets=` parameters. This is why we use
Python constants rather than CSS variables for centralized color management.

## Avoiding flicker

Make sure all widget-updates that touch more than a single widget (or a single widget multiple times) use `pn.io.hold()`.

### Native tooltips on re-rendering panes

A native HTML `title=` tooltip baked into a `pn.pane.HTML` string is torn down
every time `pane.object` is reassigned — Panel replaces the pane's inner DOM, so
the browser drops any open hover tooltip. This makes hover tooltips unworkable on
any element whose content updates frequently (e.g. a live freshness/lag readout
updating per data frame at ~1 Hz): the tooltip flickers once per update.

Guarding the write (`if pane.object != html`) only helps if the rendered string is
genuinely piecewise-constant. Live values (timestamps, sub-second lag) defeat it.

For detail that must accompany a live-updating element, put it in a separate
*visible* label (e.g. a toolbar row) that can redraw freely, not a hover tooltip.
Encode continuously-changing signals as discrete bands (color/border) so the HTML
stays constant between threshold crossings.