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
- All widget updates run inside `batched_update()` for efficient recomputation
- Widget event handlers should only call methods on shared components, never rebuild directly

### Plot-grid reconciler: policy vs mechanism

The plot-grid session pass (`plot_grid_tabs.py`) is split into a pure policy
function and a fixed differ/applier:

- **Policy** — *which cells should have a widget, built from what* — lives in
  `desired_cells` (`dashboard/cell_plan.py`), a pure function unit-tested with
  plain data (`cell_plan_test.py`). Any new visibility/materialization rule is
  a diff to this function, never a gate threaded through the pass.
- **Mechanism** — the differ rules (build when materialized inputs differ,
  dispose when the cell left topology, defer otherwise) and the apply ordering
  are fixed; do not add policy conditions there. `cell_plan.py` must not
  import Panel/Bokeh/HoloViews (guarded by a test).
- The wake predicate compares input stamps recorded at the last completed
  pass; do not add hand-written per-gate terms to it. A new pass input joins
  `_input_stamps` and the end-of-pass recording, nothing else.

The tests follow the same seam, and a new test belongs on the side its
subject does:

- A materialization or rebuild *rule* is a case in `cell_plan_test.py` —
  plain data, no fixtures, no Panel. Do not re-assert it through the widget
  stack; `plot_grid_tabs_test.py` only checks that the pass carries the rules
  out (`TestCellDiffer`) and that real state reaches the decision
  (`TestMaterializationWiring`).
- The wake gate is exercised *only* in `TestWakeGateContract`; everywhere
  else tests drive `_poll_for_plot_updates` unconditionally. A new stamped
  input therefore needs its own re-arm test there, and a deliberately
  unstamped one an asserted hole — nothing else will catch it.

Pop-out windows (`plot_popout.py`) ride entirely on these three seams: a cell
behind a showing window enters `SessionView.live_cell_ids` (policy), takes a
viewer token like a visible cell (tokens), and adds its grid to the per-grid
frame generations the stamps carry (`_live_generations`). Nothing else in the
pass knows pop-outs exist.

### A cell's views are torn down together

A `CellWidget` owns every pane rendering its plot — the grid cell's and one per
pop-out — because `Plot.cleanup` severs *all* weakly-held plot-refresh
subscribers on the streams it touches, not just its own
([holoviews#6988](https://github.com/holoviz/holoviews/issues/6988)). Removing
any one view therefore stops every other view of that cell updating, silently:
the plot stays on screen showing data that looks current.

Two consequences to preserve:

- **Rebuild after any removal that leaves a survivor.** Closing a pop-out runs
  Panel's pane cleanup, so `PlotGridTabs._close_popout` rebuilds the cell
  behind it. Adding a view is free; removing one is not.
- **Sever before the replacement renders.** `_build_cell` closes the window,
  builds the new widget, disposes the old one, and only then reopens the
  window — the reopened pane must subscribe *after* the disposal has cleared
  the pipes.

## Icons

Do not use Unicode characters for button icons. Use embedded SVG icons from `dashboard/widgets/icons.py` via `get_icon()`. Use the `create_tool_button()` helper from `dashboard/widgets/buttons.py` for consistent styling.

Where the icon cannot be a widget — a Bokeh tab label is plain text, for instance —
paint it on a `::before` pseudo-element with `mask-image: url(get_icon_data_uri(name,
color=None))` and `background-color: currentColor`. A mask reads only the alpha channel,
so the icon inherits the element's text color and follows its active/hover states without
a second, recolored copy (see `_tab_stylesheet` in `plot_grid_tabs.py`).

## Stable CSS hooks for automation

Tool buttons render as label-less icons inside per-widget shadow DOM, so they carry
no text, `title`, or `aria-label` — leaving nothing semantic for browser automation
(Playwright) or screenshot tooling to target. To avoid brittle coordinate-clicking,
`create_tool_button()` tags every button with **committed, visually-inert CSS classes**:

- `lt-tool` on all tool buttons, plus `lt-tool-{icon_name}` (e.g. `lt-tool-settings`,
  `lt-tool-player-play`, `lt-tool-x`). `create_download_button()` follows the same
  shape with a fixed `lt-tool-download`.
- Callers pass `css_classes=[...]` for context. Workflow rows add `lt-wf-{workflow_id.name}`
  (the WorkflowId *name* slug, not the display title), so a workflow's gear is
  `.lt-wf-monitor_histogram.lt-tool-settings`.

The empty cells of a plot grid are not tool buttons but are click targets — the
two-click place gesture that opens the plot wizard — so they carry
`lt-empty-cell` plus `lt-empty-cell-r{row}c{col}` (`plot_grid.py`). Empty and
occupied cells are hooked apart on purpose: a selector placing a plot must be
able to demand a cell that is still free, so an occupied cell's `lt-cell-*` hook
never answers for one.

`lt-wf-*` and `lt-grid-*` slug different things on purpose: a workflow has a stable
*name* identity to slug (`workflow_status_widget.py`'s `_tool_css_class`), but a grid has
none, so `lt-grid-*` slugs the grid *title* instead (`plot_grid_manager.py`) — the same
title that also drives the grid's download filename. Plot cells have neither a name nor
a stable title, so every button in their titlebar — pop-out, gear (or layer menu),
pencil, and layer-details toggle — carries the grid position as `lt-cell-r{row}c{col}`
(`cell.py`), unique per grid, and only the active tab's grid is rendered. Address one
with a compound selector: `.lt-cell-r0c1.lt-tool-settings`. Do not rely on DOM order for
cells: a rebuilt cell (e.g. after a rename) moves to the end of the document.

A cell's pop-out window (`plot_popout.py`) is slugged by the same grid position:
`lt-popout` + `lt-popout-r{row}c{col}`, one per cell at most.

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
`current_job` key, keep `jobs`) and `plot_configs.yaml` from the config
dir back into the fixture; `ui_config_fixtures_test.py` guards against drift.

**Diagnosing a failure.** When a block driving a session raises, everything is printed
to **stdout** — browser console tail, page state (active tab, dialog count and how many
of them are visible, `lt-*` hook inventory), a base64 JPEG screenshot, and the
dashboard's server log tail. Nothing is uploaded as a CI artifact, deliberately: the
artifact endpoint is firewalled in the devcontainer, whereas stdout lands in the pytest
failure report and in the run-level log zip, which is allowlisted and readable mid-run
(`gh api repos/scipp/esslivedata/actions/runs/<id>/logs > run.zip`). Recover the
screenshot with the `base64 -d` pipeline the log prints next to it.

`dialogs` vs `dialogs_visible` is the first thing to read on a modal timeout: a dialog
present but not visible means it rendered and was hidden, none at all means the click
never reached its handler.

The server log tail is what distinguishes *why* it never reached the handler. Static
assets are served off the same IOLoop as everything else, so their `tornado.access`
timings measure how long that loop was blocked: a `GET /static/...` taking seconds
means the click's round-trip was queued behind the session's periodic pass, not lost
(#1185). Sub-millisecond serves with no post-click activity at all point at a dropped
click instead.

One console line is usually a red herring: `pageerror: Cannot read properties of
undefined (reading 'parent_style')`. It is
[bokeh#15274](https://github.com/bokeh/bokeh/issues/15274) — our plot grid is a
`pn.GridSpec`, i.e. a Bokeh `GridBox`, and Panel emits a `children` change and the
recomputed sizing props in one patch, which that view indexes inconsistently. Real bug
(it drops the rest of the patch), but it is not the cause of any intermittent failure.

What triggers it is patching a *materialized* grid's children. Revealing a plot-grid
tab used to do that on every switch — the tab materialized the empty-cell
placeholders, then the cells replaced them — so the line fired on every run. Cells
are now built before the tab materializes (`_prebuild_revealed_grid`), and a reveal
no longer produces it. Treat it as expected only where children really do change
under a rendered grid (adding or removing a cell on the visible tab); on a plain tab
switch it now means something changed, so investigate rather than dismiss it.

**Ports.** `fake_dashboard(...)` without a port takes one the OS reports free — how the
browser tests launch, so two checkouts (or a suite next to an interactive dashboard) can
run at once. Do not hand a test a port literal; they collide silently across branches.
Pass an explicit port only when the URL must be known up front, e.g. to open it by hand.

**Shadow DOM selectors.** Tool buttons and rows live in per-widget *open* shadow roots.
Plain Playwright CSS locators pierce these, so `page.locator(".lt-tool-settings")` works
— **but descendant combinators do not cross shadow boundaries.** Target a workflow's
button with a *compound* selector on one element:

- ✅ `.lt-wf-total_counts.lt-tool-player-stop` (both classes on the same button)
- ❌ `.lt-wf-total_counts .lt-tool-player-stop` (matches nothing — the descendant
  crosses a shadow boundary)

**Sidebar.** The drawer starts collapsed by default, so the main content gets the full
window width and automation never has to click the hamburger; it survives
`page.reload()`, since it sets `MaterialTemplate.collapsed_sidebar` rather than toggling
the DOM. `--no-collapsed-sidebar` opens it, which for a driven session only narrows
the plots under test and their screenshots.

**Tabs.** The top-level tabs are Bokeh-owned `.bk-tab` divs with no `lt-*` hooks, so
navigate by visible text (`page.get_by_text("Detectors", exact=True)`). Static tab
titles are code constants: **Workflows**, **System Status**, **Manage Plots**; further
tabs are user/fixture plot-grid titles (the dummy fixture adds **Detectors** and
**Diagnostics**). Only the
static tabs carry a leading icon, keyed to their position, so a bare label identifies a
plot-grid tab visually — but the icon is a CSS pseudo-element, invisible to text
locators. With
`dynamic=True` only the active tab's models exist, so a DOM/`lt-*` inventory reflects the
*current* tab only — switch tabs before querying that tab's hooks.

**Pop-out windows.** The cell titlebar's pop-out tool
(`.lt-cell-r0c0.lt-tool-arrows-maximize`) opens a jsPanel `FloatPanel`, *not* a dialog —
`Dashboard.open_modal` times out on it. Wait on `.lt-popout-r0c0` instead, and close it
with `.jsPanel-btn-close` (or by clicking the tool again, which replaces the window).
Two windows opened back to back cascade by a few pixels; without that offset the second
would cover the first's close button and intercept the click. A window keeps rendering
while another tab is shown, so with `dynamic=True` tearing down the hidden grid's models
it is then the only plot in the document — which is what makes `assert_updating` on
another tab a pop-out liveness check.

Minimizing parks the panel *off-screen* (x ≈ −9000) and leaves a replacement strip of
small buttons at the bottom of the viewport. So after a minimize, `.jsPanel-btn-close`
still matches the parked panel and any click on it times out as "outside of the
viewport" — drive the strip instead (`.jsPanel-btn-sm.jsPanel-btn-normalize`, or the
matching `-close`). A minimized pop-out is deliberately *not* live, so
`assert_stops_updating` is the check there.

The window hangs from the top of the viewport and is sized in `vh`, because its title
bar holds the only close/minimize controls: centring a fixed pixel height puts them
above `y=0` on any viewport shorter than the window, and the window then cannot be
dismissed at all. Test viewports are 1000 px tall by default, which is *above* the
threshold — regressions here only show at laptop heights, so the geometry test
parametrizes over 700 px as well. Note `maxSize` is not a fix: jsPanel applies it to
interactive resizing only, not to the size the panel opens at; override `contentSize`
via `config` instead, which takes precedence over the size Panel derives from
`width`/`height`.

**Getting a plot to fill a FloatPanel** takes two things, both handled once per session
by `PopoutWindowFitter` (`plot_popout.py`) — expect neither to work by default:

- The height chain `.jsPanel-content` → `#float` (Panel's template root) → `#flex-item`
  is broken: those wrappers have no height, so nothing carries the window's size inward.
  A `stretch_both` child then collapses to a ~66 px sliver, and a child with a fixed
  height survives but can never follow the window. `stylesheets=` on the `FloatPanel`
  does **not** reach these wrappers — they are light DOM, so it takes a document-level
  rule (`.jsPanel-content > .bk-root`).
- Panel re-lays out only on `jspanelresizestop`, which a drag fires but maximize,
  normalize and smallify do not. Re-dispatch that event on `jspanelstatuschange` with
  `event.panel` copied across (Panel's handler matches on it) rather than reimplementing
  the layout call. A synthetic `window` resize does *not* work.

Any invisible `ReactiveHTML` helper needs a **public** class name: a leading underscore
yields `could not resolve type '_Foo1'` in the browser and the whole session fails to
render.

**Clicks that silently do nothing.** A rebuild racing a click detaches the target
between locating and pressing it; Playwright reports success and nothing happens. The
cold-start tick after load triggers this often enough to make a single click on a
freshly loaded tab roughly a 3-in-4 proposition. `click_until(dash, selector, condition,
label=...)` retries until the effect is observable — use it for the first click of a
session rather than `dash.click` plus a `wait_until`, which cannot recover.

**Modals.** Settings (gear), cell edit (pencil), workflow config, and the plot wizard
an empty grid cell opens all render a `pn.Modal` as `[role=dialog]` — use that as the
open/visible signal (`Dashboard.open_modal` waits on it). Footer buttons are reachable
by text (`Cancel`, `Back`, `Next`, `Add Plot`, `Update Plot`). To
dismiss, press **Escape** (a `ModalEscapeCloser` widget makes this work from initial
focus) or click `.pnx-dialog-close`. Per-grid rows in **Manage Plots** carry
`lt-grid-{title-slug}` (e.g. `.lt-grid-detectors.lt-tool-pencil`) — that pencil is the
exception: it opens edit mode *inline* in the row, not a dialog, so `Dashboard.open_modal`
times out on it. Wait on its fields instead (`input[placeholder="Enter grid title"]`) and
commit with the `Save Changes` button.

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

### Markup panes built after page load

Panel keeps a markup pane (`pn.pane.HTML`, `Markdown`, `Str`) behind
`visibility: hidden` until every `<link>` stylesheet in it has fired `load`, and arms
that reveal exactly once, while rendering. A pane built after page load first renders
against `cdn.holoviz.org` URLs — its model is not in a document yet, so Panel falls back
to the CDN — and Panel then swaps those links for the locally served copies. The load
events the reveal waits on belong to the discarded links, so the pane can stay invisible
for the rest of the session while its model, text and layout are all correct and
live-updating (#1154, holoviz/panel#8696). `dashboard/design.py` overrides the latch for
the whole app; a template built without `LivedataDesign` brings the bug back.

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

Chrome that a theme owns — the header background and the main tab strip — is the one
exception: those colors live in `dashboard/theme.py`, next to the `Theme` that selects
them. `styles.py` stays theme-independent, so a widget must never read `theme.py`.

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