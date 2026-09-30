---
paths: src/ess/livedata/dashboard/widgets/**/*.py, src/ess/livedata/dashboard/reduction.py, scripts/drive_dashboard.py
---

# Dashboard Widget Patterns

## Shared state vs per-session widgets

Controllers, orchestrators, services and computed plot elements are shared by all
browser sessions. Widgets are per session. Two rules follow.

**Widgets poll version counters; event handlers never rebuild.** A handler that
rebuilds its own widget updates only the clicking session:

```python
def _on_stop_clicked(self, event):
    self.orchestrator.stop_workflow(workflow_id)  # no rebuild here

def refresh(self):  # SessionUpdater periodic callback, inside batched_update()
    version = self.orchestrator.get_workflow_state_version(self._workflow_id)
    if version != self._last_state_version:
        self._last_state_version = version
        self._build_widget()
```

**No Bokeh model instances on the shared compute path.** A model (`TeeHead`,
`HoverTool`, ...) placed in HoloViews opts by a plotter ends up in every session's
document. With a plot open in two sessions Bokeh raises `Models must be owned by
only a single document`, and tab switches and buttons stop working. Only per-session
render code (hooks) may create Bokeh models, built from plain data inside the hook.
`plotter_registry_test.py` renders every plotter into two documents; a new plotter
needs a `_DATA` entry there.

## Plot grid

### Reconciler: policy vs mechanism

The session pass in `plot_grid_tabs.py` is a pure policy function plus a fixed
differ/applier.

- **Policy** (which cells get a widget, built from what) lives in `desired_cells`
  (`dashboard/cell_plan.py`). A new visibility or materialization rule is a change
  to this function and a plain-data case in `cell_plan_test.py`, never a gate
  threaded through the pass. `cell_plan.py` must not import Panel/Bokeh/HoloViews.
- **Mechanism** (differ rules, apply ordering) is fixed; add no policy conditions.
  `plot_grid_tabs_test.py` only checks that the pass carries the rules out
  (`TestCellDiffer`) and that real state reaches the decision
  (`TestMaterializationWiring`).
- **Wake predicate**: compares input stamps recorded at the last completed pass. A
  new pass input joins `_input_stamps` and the end-of-pass recording, nothing else,
  and needs a re-arm test in `TestWakeGateContract`, the only place the gate is
  exercised.

Pop-outs (`plot_popout.py`) and the phone layout's Plots tab (`plot_overview.py`)
ride on these seams (live cell set, viewer tokens, frame generations); see their
module docstrings.

### Cell teardown and the session DynamicMap

- `Plot.cleanup` severs *all* plot-refresh subscribers on a stream, not just its own
  (holoviews#6988). Removing any view of a cell (e.g. closing its pop-out) silently
  freezes the others, so rebuild the cell after any removal that leaves a survivor,
  and dispose the old widget before a replacement view subscribes. Details:
  `plot_popout.py` module docstring, `CellWidget.dispose`.
- `DynamicMap.opts()` wraps the map in place and returns the same object. The
  session DynamicMap outlives cell rebuilds, so anything decorating it at build time
  passes `clone=True`; otherwise wraps stack per rebuild and a map with kdims breaks
  from the second wrap on. Guarded by
  `test_rebuild_leaves_the_session_dynamicmap_unwrapped`.

## Model creation and visibility

- `pn.Tabs(dynamic=True)` prevents Bokeh model creation for hidden tabs. It is the
  preferred way to defer cost.
- `visible=False` only hides via CSS; all models are still created. Do not use it as
  a performance optimization; create the component lazily instead.
- `dynamic=True` does not stop Python-side periodic callbacks. Use an `is_visible`
  predicate to skip refresh work for hidden tabs.
- Markup panes built after page load can stay invisible forever (Panel's stylesheet
  reveal latch, #1154). `dashboard/design.py` overrides the latch app-wide; a
  template built without `LivedataDesign` brings the bug back.
- An invisible `ReactiveHTML` helper needs a public class name. A leading underscore
  gives `could not resolve type '_Foo1'` and the session fails to render.

## Icons, colors, flicker

- **Icons**: no Unicode characters. Use `get_icon()` (`widgets/icons.py`) and
  `create_tool_button()` (`widgets/buttons.py`). Where the icon cannot be a widget
  (e.g. a Bokeh tab label), paint a `::before` with
  `mask-image: url(get_icon_data_uri(name, color=None))` and
  `background-color: currentColor` (see `_tab_stylesheet` in `plot_grid_tabs.py`).
- **Colors** come from `widgets/styles.py` (`StatusColors`, `HoverColors`, `Colors`,
  `ErrorBox`/`WarningBox`; `ButtonStyles` in `buttons.py` re-exports pairs). No
  hard-coded hex/rgba in widget files, except decorative colors local to one widget.
  Theme-owned chrome (header, main tab strip) lives in `dashboard/theme.py`; widgets
  never read `theme.py`. Panel accepts `var()` only in `stylesheets=`, not in
  `styles=` or inline `style=`, hence Python constants.
- **Flicker**: wrap updates touching more than one widget (or one widget several
  times) in `pn.io.hold()`.
- **Tooltips**: reassigning `pane.object` replaces the pane's DOM and drops any open
  native `title=` tooltip. On live-updating elements, put detail in a separate
  visible label, and encode continuous signals as discrete bands so the HTML stays
  constant between threshold crossings.

## Automation contract: `lt-*` hooks

Tool buttons are label-less icons in per-widget shadow DOM. `create_tool_button()`
tags them with visually inert classes, a stable contract that refactors must keep:

| Hook | Where |
|---|---|
| `lt-tool`, `lt-tool-{icon_name}` | every tool button; download button is `lt-tool-download` |
| `lt-wf-{workflow_id.name}` | workflow rows (name slug, not display title) |
| `lt-grid-{title-slug}` | per-grid rows in Manage Plots (grids have no stable name) |
| `lt-cell-r{row}c{col}` | every button in a plot cell's titlebar |
| `lt-empty-cell`, `lt-empty-cell-r{row}c{col}` | empty grid cells (never answered by `lt-cell-*`) |
| `lt-popout`, `lt-popout-r{row}c{col}` | a cell's pop-out window |

Address a button with a compound selector, e.g. `.lt-cell-r0c1.lt-tool-settings`.
Do not rely on DOM order: a rebuilt cell moves to the end of the document.

When you change the UI:

- **New tool button** → use `create_tool_button()`. A hand-rolled one (toggle icon,
  `MenuButton`) needs `css_classes=['lt-tool', 'lt-tool-{semantic}']` **and** its own
  guard test; `buttons_test.py` covers only the helper.
- **Repeated-instance control** → add a context class so each instance is unique.
- **New top-level tab** → add its title to the tab list below.
- **Renamed/added workflow or output** → regenerate the affected
  `tests/dashboard/ui_config_fixtures` (`ui_config_fixtures_test.py` fails on drift).

Then run `python scripts/drive_dashboard.py --launch --map` (and `--screenshot`).

## Driving the dashboard with Playwright

Use `scripts/drive_dashboard.py` (library + CLI; see its module docstring, which also
covers shadow-DOM selectors). Run `--map` first rather than screenshotting to
rediscover the layout.

**Running a server by hand**: copy the fixture first (the dashboard writes to its
config dir) and avoid port 5009 (interactive dev):

```sh
cp -r tests/dashboard/ui_config_fixtures/dummy "$TMP/cfg/dummy"
python -m ess.livedata.dashboard.reduction --instrument dummy --transport fake \
    --port 5011 --config-dir "$TMP/cfg" --no-fetch-announcements [--auto-start]
```

Regenerate a fixture by configuring via the UI and copying `workflow_configs.yaml`
(strip `current_job`, keep `jobs`) and `plot_configs.yaml` back. Browser tests
launch via `_fake_dashboard(...)` without a port, so the OS picks a free one; never
hand a test a port literal.

**Navigation**

- Tabs are Bokeh `.bk-tab` divs with no hooks: navigate by text. Static tabs:
  **Workflows**, **System Status**, **Manage Plots**; the dummy fixture adds
  **Detectors** and **Diagnostics**. With `dynamic=True` only the active tab's hooks
  exist, so switch tabs before querying.
- The sidebar starts collapsed; `--no-collapsed-sidebar` opens it.
- Modals (gear, pencil, workflow config, plot wizard) render as `[role=dialog]`;
  dismiss with Escape or `.pnx-dialog-close`. Exception: the Manage Plots grid pencil
  edits inline; wait on `input[placeholder="Enter grid title"]`, commit with
  `Save Changes`.
- A click can land on an element a rebuild just detached; Playwright reports success
  and nothing happens. Likely on the first click after load and on a row that just
  staged/committed. Use `click_until(dash, selector, condition, label=...)` for such
  clicks.

**Pop-out windows** are jsPanel `FloatPanel`s, not dialogs:

- Wait on `.lt-popout-r0c0`, close with `.jsPanel-btn-close`.
- Minimizing parks the panel off-screen (x ≈ −9000); drive the bottom strip
  (`.jsPanel-btn-sm.jsPanel-btn-normalize`) instead. A minimized pop-out is not live.
- Resize (`.jsPanel-resizeit-{n,e,s,w,ne,se,sw,nw}`) before moving: a drag whose grip
  is off-screen silently does nothing. Geometry is client-only; assert it with
  `bounding_box()` on `.jsPanel`.
- Assert page scrolling via `overflowY`, never `getBoundingClientRect` (clipping does
  not shrink the rect). Geometry regressions show only at laptop heights; test at
  700 px as well as the 1000 px default.

**Diagnosing a failure**: a raising `Dashboard` block prints console tail, page
state, a base64 screenshot and the server log tail to stdout, so it lands in the
pytest report and the CI run-level log zip (readable only once the run finishes).

- `dialogs` present but not `dialogs_visible`: the modal rendered and was hidden.
  None at all: the click never reached its handler.
- Multi-second `tornado.access` times for `GET /static/...` mean the IOLoop was
  blocked and the click was queued (#1185); sub-millisecond serves with no activity
  mean a dropped click.
- `pageerror: ... reading 'parent_style'` is bokeh#15274, triggered by changing
  children of a rendered `GridSpec`. Expected when adding/removing a cell on the
  visible tab; on a plain tab switch it signals a regression.
