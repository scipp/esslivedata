---
paths: src/ess/livedata/dashboard/widgets/**/*.py, src/ess/livedata/dashboard/reduction.py, scripts/drive_dashboard.py
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

### Driving the dashboard with Playwright

`scripts/drive_dashboard.py` is the committed driving kit (`Dashboard` library class +
a `--map` / `--launch` / `--screenshot` CLI). Reach for it before hand-rolling
navigation; run `--map` first to inventory the live tabs and `lt-*` hooks rather than
screenshotting to rediscover the layout.

**Launching & seeding.** `--launch` spawns a fake backend (no Kafka) seeded from the
committed dummy fixture, drives it, and tears it down. To run a server yourself instead
— e.g. to click the play buttons by hand rather than auto-starting — copy the fixture to
a scratch dir first (the dashboard writes to its config dir) and point `--config-dir` at
it, on a port other than 5009 (interactive dev uses 5009):

```sh
cp -r tests/dashboard/ui_config_fixtures/dummy "$TMP/cfg/dummy"
python -m ess.livedata.dashboard.reduction --instrument dummy --transport fake \
    --port 5011 --config-dir "$TMP/cfg" --no-fetch-announcements
```

Add `--auto-start` (requires `--transport fake`) to commit every staged workflow on
launch so plots render with no interaction. Regenerate a fixture by configuring via the
UI, then copying the persisted `workflow_configs.yaml` (strip the runtime
`current_job`/`previous_job` keys, keep `jobs`) and `plot_configs.yaml` from the config
dir back into the fixture; `ui_config_fixtures_test.py` guards against drift.

**Shadow DOM selectors.** Tool buttons and rows live in per-widget *open* shadow roots.
Plain Playwright CSS locators pierce these, so `page.locator(".lt-tool-settings")` works
— **but descendant combinators do not cross shadow boundaries.** Target a workflow's
button with a *compound* selector on one element:

- ✅ `.lt-wf-total_counts.lt-tool-player-stop` (both classes on the same button)
- ❌ `.lt-wf-total_counts .lt-tool-player-stop` (matches nothing — the descendant
  crosses a shadow boundary)

**Tabs.** The top-level tabs are Bokeh-owned `.bk-tab` divs with no `lt-*` hooks, so
navigate by visible text (`page.get_by_text("Detectors", exact=True)`). Static tab
titles are code constants: **Workflows**, **System Status**, **Manage Plots**; further
tabs are user/fixture plot-grid titles (the dummy fixture adds **Detectors**). With
`dynamic=True` only the active tab's models exist, so a DOM/`lt-*` inventory reflects the
*current* tab only — switch tabs before querying that tab's hooks.

**Modals.** Settings (gear), edit (pencil), and grid/workflow config open a `pn.Modal`
rendered as `[role=dialog]` — use that as the open/visible signal (`Dashboard.open_modal`
waits on it). Footer buttons are reachable by text (`Cancel`, `Update Plot`, `Back`). To
dismiss, press **Escape** (a `ModalEscapeCloser` widget makes this work from initial
focus) or click `.pnx-dialog-close`. Per-grid rows in **Manage Plots** carry
`lt-grid-{title-slug}` (e.g. `.lt-grid-detectors.lt-tool-pencil`).

### Driving workflow config flows

A `WorkflowStatusWidget` rebuilds its row (`_build_widget`) only when that workflow's
state *version* changes — staging, commit, or stop (`job_orchestrator.py`). Steady-state
status refresh just reassigns badge/dots/timing HTML in place, so it does *not* detach
elements. Two windows still detach the element under the cursor:
- **Cold start**: the first refresh tick after page load can fire one rebuild while
  Bokeh models are still settling. Wait a few seconds after load before the first click.
- **Multi-step flows**: each stage/commit rebuilds *that* row, so a click landing on a
  just-mutated row races the rebuild.

Wrap clicks in a small retry-on-detach helper (catch the Playwright timeout, re-locate,
retry) rather than assuming a single click lands. This is not a continuous re-render —
untouched rows stay stable.

### Keeping automation working

A UI change can silently break `scripts/drive_dashboard.py`, the `lt-*` contract, or the
seeded fixtures. When you touch the UI, keep these in sync:

- **New tool button** → build it with `create_tool_button()`; it auto-tags `lt-tool` +
  `lt-tool-{icon_name}`. If you must hand-roll one (toggling icon, `MenuButton`, etc.),
  add `css_classes=['lt-tool', 'lt-tool-{semantic}']` by hand **and** a guard test —
  `buttons_test.py` only covers the helper, so hand-rolled buttons drift unnoticed
  (see `plot_widgets.py`, `plot_grid_manager.py` for examples + their tests).
- **Repeated-instance view** (per-row/-cell/-grid controls) → pass a context class so
  each instance is uniquely addressable (`lt-wf-{name}`, `lt-grid-{slug}`).
- **New top-level tab** → tabs are Bokeh-owned `.bk-tab` with no hook, so the kit
  navigates by visible text; add the new title to the tab inventory above so callers
  aren't searching blind.
- **New modal** → it opens as `[role=dialog]` and is closed on Escape by
  `ModalEscapeCloser` automatically; nothing to wire, but verify it appears in the
  inventory.
- **Renamed/added workflow or output** → regenerate the affected `ui_config_fixtures`
  (the drift-guard in `ui_config_fixtures_test.py` fails loudly to remind you).
- **New instrument you want `--launch` to support** → add a
  `tests/dashboard/ui_config_fixtures/<instrument>/` fixture (only `dummy` exists today).

After a non-trivial UI change, sanity-check with
`python scripts/drive_dashboard.py --launch --map` (and `--screenshot`).

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