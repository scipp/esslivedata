# Plot axis autoscale — implementation plan

Companion to `plot-axis-autoscale-strategy.md` (the design discussion). That
document settled the *direction*; this one breaks the work into concrete,
shippable phases with file-level changes and named open questions.

Related issues: #910, #912.

## 0. Anchor: what we are building

Replace the `Autoscaler` + `framewise` toggle model with:

1. **Per-axis, per-session, per-cell autoscale toggles** (`x`, `y`, `c`).
   Toggle on (default) -> the corresponding Bokeh range follows current data
   extent on every frame. Toggle off -> the user's manual zoom on that axis
   survives indefinitely.
2. **Per-session "Fit" button** that snaps all axes to current data extent
   once, regardless of toggle state. Does not change toggle state.
3. **Follow-latest mode for time-series** (LinePlotter via
   `FullHistoryExtractor`) that replaces the x toggle with a windowed
   "last N seconds" mode.
4. **No more `framewise=True` in the normal path.** `framewise=False`
   globally; range writes go through Bokeh `hooks=[...]` callbacks that
   mutate `Range1d.start/end` and `color_mapper.low/high` directly.
5. **`Autoscaler` and its test deleted.** Thresholded debounce is gone.

Validated mechanics (§4.6 of the strategy doc) — repeated here because the
implementation depends on them:

- `xlim`/`ylim` opts do **not** propagate to Bokeh on Pipe re-renders; only
  `clim` does. Both x/y/clim flow through Bokeh hooks in the implementation
  to keep the path uniform and per-session (see §1 below).
- `RangesUpdate` events on Bokeh figures fire only on user interaction, not
  on programmatic range writes. Held in reserve in case the manual-toggle
  UX turns out wrong (strategy §4.3).

## 1. Architectural decisions made before any code lands

These were unresolved at the strategy-doc level and need to be settled
first; subsequent phases assume them.

### 1.1 Toggle scope: per-session

Each browser session has its own toggle and Fit state. Two users viewing
the same plot can zoom independently. (User confirmation: 2026-05-28.)

Consequence: **`clim` does not stay on the HoloViews opt surface.** The
strategy doc §4.2 proposed keeping `clim` as an opt because the shared
plotter sets it; but per-session toggles need per-session control over
whether `clim` is applied. We route `clim` through a Bokeh hook that
mutates `plot.handles['color_mapper'].low/high` directly, exactly as for
x/y `Range1d.start/end`. This empirically works for Pipe-streamed images
(it's how the Bokeh color-mapper handle is updated under the hood by the
`clim` opt; we just take the wheel directly).

### 1.2 Toggle owner: cell, not layer

A `PlotCell` can hold multiple overlay layers (e.g., image + ROI
rectangles), each a separate `Plotter`. The Bokeh figure has *one*
`Range1d` per axis shared across all layers. So:

- Toggles + Fit live at the **cell** level, not the layer level.
- "Current data extent" for the cell is the union of each layer's
  per-axis target range.
- Hook attachment happens during cell composition, alongside the existing
  `save_filename_hook` and `frame_aspect_hook` in
  `_get_session_composed_plot` (`widgets/plot_grid_tabs.py`).

### 1.3 UI placement: Bokeh toolbar (primary), Panel widget row as fallback

Primary: toggles and Fit are added as `bokeh.models.CustomAction`
entries on each cell's Bokeh figure toolbar, alongside the existing
pan/zoom/save tools. Attached via a HoloViews hook, like the existing
`save_filename_hook` and the `_make_hover_hook` in `flatten_plotter.py`.

Why Bokeh toolbar:

- Zero extra vertical space per cell — important for dense grids of
  small plots, which is a real usage shape on this dashboard.
- Lives with the plot. Each Bokeh figure gets its own toolbar, so
  per-cell, per-session state is the natural granularity.
- Precedent already exists: `frame_aspect.py`, `save_filename.py`, and
  `flatten_plotter.py` all reach into `plot.handles['plot'].toolbar`
  via hooks. We are extending an existing pattern, not inventing one.

Bokeh API surface we will use (validated against Bokeh 3.8, which the
project already depends on):

- `CustomAction(icon=..., description=..., active_callback="auto")` —
  a stateful toolbar button whose `active: bool` flips on click.
  Hooks read `tool.active` synchronously on each render; no event
  subscription needed for the toggles. Listed under the toolbar's
  "action" group with the existing reset/save tools.
- `CustomAction` for Fit (stateless from the user's perspective): use
  `active_callback="auto"` and a server-side `tool.on_change("active",
  handler)` that runs the fit and resets `active` back to False. Brief
  visual highlight on click, then back to neutral.
- Tooltips come for free via `description=`.
- Icons via the SVG strings in `dashboard/widgets/icons.py` (CustomAction
  accepts SVG data URLs or built-in names).

**Crowding mitigation: tool grouping.** Three toggles (X, Y, C) + Fit
on top of the existing toolbar can get dense. Two paths if it does:

1. **Long-press / dropdown grouping.** Bokeh's toolbar already groups
   tools of the same kind into a flyout (see how `ZoomIn`/`ZoomOut`
   collapse). The mechanism is `ToolProxy` plus `Toolbar.group_types`;
   we can stack the autoscale toggles under one icon with a flyout
   listing X/Y/C. Bokeh 3.8 supports this; we'll prototype before
   committing to flat vs grouped layout.
2. **Combine the three toggles into a single tri-state pill** if
   grouping is fiddly. The icon shows which axes are locked
   (e.g., `XY` greyed out vs lit). Less discoverable but more compact.

We'll start with flat (4 buttons), measure, and switch to grouped if
the toolbar feels cluttered. The hook layer is unaffected either way.

Panel widget row above the plot remains the fallback if Bokeh-side
state turns out to misbehave under multi-session — the controller's
hook factory is UI-agnostic; only the widget construction differs.

### 1.4 Per-plotter autoscale axes

Each plotter declares which axes have autoscale toggles. This avoids
useless toggles (e.g., x toggle on a Bars plot, c toggle on a 1D Line):

| Plotter             | Autoscalable axes  | Notes |
|---------------------|--------------------|-------|
| `LinePlotter`       | `x`, `y`           | x optional via Follow-latest for time-series |
| `ImagePlotter`      | `x`, `y`, `c`      | x/y often static for detector pixels; toggle still exposed for symmetry |
| `Overlay1DPlotter`  | `x`, `y`           | same as Line; union across slices |
| `FlattenPlotter`    | `x`, `y`, `c`      | inherits ImagePlotter |
| `SlicerPlotter`     | `c` only           | x/y change per slice; current pre-computed global `clim` becomes the `c` target |
| `BarsPlotter`       | (none)             | 0-D, categorical |
| `CorrelationHist1d` | `x`, `y`           | delegates to LinePlotter |
| `CorrelationHist2d` | `x`, `y`, `c`      | delegates to ImagePlotter |
| `*Readback`, `*Request` (ROI) | (none) | inherit parent layer's axes |

Static overlays (rectangles, vlines, hlines) have no autoscale axes.

## 2. Phasing

Phases are mostly independent. Phase 1 is the only true prerequisite. Each
phase ends in a runnable, mergeable state.

### Phase 1 — Range hook infrastructure (foundation)

Goal: introduce the hook mechanism and the data structure that conveys
"current target range" from plotter to hook. **No behavior change yet** —
the autoscaler keeps running, framewise keeps flipping. We're laying
plumbing.

New file: `src/ess/livedata/dashboard/range_hook.py`.

- Module-level types:
  - `Axis = Literal['x', 'y', 'c']`
  - `RangeTargets = dict[Axis, tuple[float, float]]`
- `class RangeHandles`: captures `x_range`, `y_range`, `color_mapper`
  handles from `plot.handles` on the first hook invocation per session.
  Stored on the cell-level controller so Fit can mutate handles directly
  without a pipe round-trip.
- `def make_range_hook(axis, get_target_callable, get_toggle_value_callable, handles) -> Callable`:
  returns a HoloViews-compatible `hook(plot, element)` that
  1. Captures handles into `handles` on first call.
  2. If toggle off, returns early.
  3. Reads `(lo, hi)` from `get_target_callable()`; if `None`, returns.
  4. For `x`/`y`: writes to `plot.handles[f'{axis}_range'].start/.end`.
     For `c`: writes to `plot.handles['color_mapper'].low/.high`.

Why callables instead of bare values: the hook is created once per session
at compose time, but the target changes every frame as new data arrives.
Closures over the *plotter's accessor method* let the hook always read the
current value without re-attaching.

Add unit tests using a synthetic plot/figure stub:
`tests/dashboard/range_hook_test.py`.

### Phase 2 — Plotter range-target API (no UI yet)

Goal: each plotter starts computing per-axis ranges from data on every
`compute()` and exposes them via a stable API. Still no UI; framewise
still rules behavior.

Changes in `src/ess/livedata/dashboard/plots.py`:

- Add to `Plotter` base:
  ```python
  AUTOSCALE_AXES: frozenset[Axis] = frozenset()  # override per subclass

  def get_range_targets(self) -> RangeTargets | None:
      """Latest per-axis (lo, hi) computed at last compute(), or None."""
  ```
- Cache extension: `_cached_state` keeps the element (unchanged); add a
  parallel `_range_targets: RangeTargets | None`. Update
  `_set_cached_state` to accept both. (Or wrap into a `PlotComputation`
  dataclass; either is fine — the strategy doc allows either.)

For each plotter, add a `_compute_range_targets(data) -> RangeTargets`
method called from `plot()` / `compute()`:

- `LinePlotter`: `{'x': (coord.min, coord.max), 'y': (data.nanmin, data.nanmax)}`
  with pad (§4.8: linear ±5%; log ÷×1.1).
- `ImagePlotter`: x/y from `_compute_image_bounds_from_edges` (or
  midpoint-based fallback — both already in `scipp_to_holoviews.py`),
  c from `data.nanmin/nanmax` after the log-scale mask
  (`_prepare_2d_image_data`).
- `Overlay1DPlotter`: union of all slices' x/y. Single shared computation
  per `compute()` call.
- `FlattenPlotter`: inherits `ImagePlotter`, computes on the flattened
  result.
- `SlicerPlotter`: keep `_compute_global_clim` (already pre-computes a
  global `clim`); expose as `c`. x/y aren't computed here because they
  change per slice — the per-slice render writes them via hook too, but
  with current-slice extent.
- `CorrelationHistogramPlotter`: delegates to the inner renderer; the
  renderer already does the work.

Add tests:
`tests/dashboard/plotter_range_targets_test.py` — for each plotter,
assert that after `compute()` the targets match the data extent (+ pad).

### Phase 3 — Cell-level controller and toggle widgets

Goal: replace the framewise behaviour with hook-driven writes plus
per-cell UI toggles. **This is the visible behaviour change.**

New file: `src/ess/livedata/dashboard/cell_autoscale.py`.

- `class CellAutoscaleController`: per-cell, per-session.
  - Constructor takes the list of session `Plotter`s for the cell and the
    union of their `AUTOSCALE_AXES`.
  - Owns one `bokeh.models.CustomAction` per axis (toggle) plus one for
    Fit. Each is created Python-side; their state is read/written by the
    hook on each render.
  - Owns a `RangeHandles` instance (populated by the hook on first
    render).
  - Methods:
    - `get_target(axis) -> tuple[float, float] | None`: returns the
      union of targets across all layers' plotters for that axis (skip
      layers whose plotter doesn't expose that axis).
    - `make_hook() -> Callable[[plot, element], None]`: a single hook
      that (a) installs the four `CustomAction` tools on the figure
      toolbar on first call (idempotent guard via `plot.handles`, same
      pattern as `_make_hover_hook` in `flatten_plotter.py`), (b)
      captures `x_range`/`y_range`/`color_mapper` into `RangeHandles`,
      (c) writes per-axis range based on each axis tool's `.active`.
    - `_on_fit_active_change(attr, old, new)`: server-side handler
      attached to the Fit tool's `active` property. On click (active
      flips True), reads current targets, writes to handles regardless
      of toggle state, resets `active = False`.

Wiring changes in `widgets/plot_grid_tabs.py`:

- `_get_session_composed_plot`:
  - Build a `CellAutoscaleController` for the cell from its layers
    (skip for Layouts, same as the existing `save_filename_hook` /
    `frame_aspect_hook` skip).
  - Append the controller's single hook to the existing `hooks` list.
  - Keep a session-local reference to the controller alongside the
    existing `_session_layers` so it isn't garbage collected (the
    `CustomAction` callbacks need it alive).
- `_create_cell_widget`: **no change.** All UI now lives in the Bokeh
  toolbar inside the plot — the cell `pn.Column` doesn't grow.

Same-cell multi-layer: the controller iterates all layers, taking the
**union** of each layer's per-axis targets. Layers without a given axis
contribute nothing.

Skip the controller entirely for Layouts (multiple sub-figures with
their own toolbars; the strategy doc keeps Layout out of scope, same as
the existing `save_filename_hook` skip).

### Phase 4 — Disable framewise; drop the Autoscaler

Goal: flip the default and delete the old machinery.

- In every plotter `plot()`: remove the `framewise=...` value and the
  `_update_autoscaler_and_get_framewise` call. Default `framewise` (i.e.,
  unset) on Bokeh+Pipe streaming means Bokeh keeps the previous range —
  exactly the behavior we want, since our hook is the only thing now that
  legitimately moves the range.
- `SlicerPlotter`: drop the `framewise=True` it sets explicitly in
  `render_slice`. The slice change is driven by kdims and is handled by
  the same hook-write-on-render path. Verify this with the existing slicer
  test; expect `test_render_slice_framewise_always_true` (plots_test.py:793)
  to be deleted alongside.
- Remove the `Autoscaler` import and field from `Plotter`.
- Delete `src/ess/livedata/dashboard/autoscaler.py`.
- Delete `tests/dashboard/autoscaler_test.py`.
- Remove `grow_threshold` from every `from_params` / `from_display_params`
  in `plots.py`, `slicer_plotter.py`, `flatten_plotter.py` (and any other
  hits — six total based on `rg grow_threshold`).
- Remove the `autoscaler_kwargs` / `autoscalers` attributes from
  `Plotter`.

The `_get_log_scale_clim` fallback in `Plotter._get_log_scale_clim`
**stays**: it's a separate workaround for HoloViews' `LogColorMapper`
crash when all data is NaN, not an autoscaler concern.

## 3. File-level change summary

New:
- `src/ess/livedata/dashboard/range_hook.py`
- `src/ess/livedata/dashboard/cell_autoscale.py`
- `tests/dashboard/range_hook_test.py`
- `tests/dashboard/cell_autoscale_test.py`
- `tests/dashboard/plotter_range_targets_test.py`

Modified:
- `src/ess/livedata/dashboard/plots.py` — add `AUTOSCALE_AXES`,
  `_range_targets`, `_compute_range_targets`; drop autoscaler/framewise.
- `src/ess/livedata/dashboard/slicer_plotter.py` — drop framewise,
  expose c target via existing `_compute_global_clim`.
- `src/ess/livedata/dashboard/flatten_plotter.py` — drop framewise.
- `src/ess/livedata/dashboard/correlation_plotter.py` — pass-through
  (renderer handles it).
- `src/ess/livedata/dashboard/widgets/plot_grid_tabs.py` —
  `_get_session_composed_plot` and `_create_cell_widget` attach the cell
  controller's hooks + toolbar.
- `tests/dashboard/plots_test.py` — remove framewise-asserting tests
  (`test_render_slice_framewise_always_true`, etc.).

Deleted:
- `src/ess/livedata/dashboard/autoscaler.py`
- `tests/dashboard/autoscaler_test.py`

## 4. Test strategy

Three layers, in order of investment:

1. **Range-target computation** (Phase 2): unit tests per plotter, real
   scipp data, assert exact (lo, hi) including pad. Easy, fast, high
   coverage of the math.
2. **Hook write behavior** (Phase 1): synthetic plot/figure with stubbed
   `plot.handles`. Assert: toggle on -> writes; toggle off -> skips; Fit
   writes regardless. No HoloViews needed.
3. **Cell controller integration** (Phase 3): construct a fake cell with
   2 layers, instantiate the controller, drive it through a render
   cycle. Heavier — uses real HoloViews stubs.

Skip end-to-end browser testing for the first cut. The strategy doc's
prototype (`.scratch/clim_zoom_prototype.py`) already validated the
HoloViews/Bokeh layer; the code we ship mirrors it.

Manual verification per phase (we can't automate browser interaction
cheaply):

- Phase 1: confirm prototype-equivalent on the dashboard. Open a plot,
  data update, axis follows.
- Phase 3: confirm toggle off freezes the axis under data updates.
  Confirm Fit recovers.
- Phase 5: confirm timeseries x follows the window, not the cumulative
  history.

## 5. Open questions (carry-over from strategy doc + new)

- **Follow-latest window value source.** Plan defaults to hardcoded 60 s.
  Promote to per-plot user input or per-plotter YAML when Phase 5 lands.
- **Single Fit vs split Fit ranges / Fit value.** Plan implements single
  Fit. Split is mentioned as a possible v2 once we see real usage.
- **Toolbar layout: flat vs grouped.** Start with 4 flat `CustomAction`
  buttons (X, Y, C toggles + Fit). If too dense, group the three toggles
  under one icon via `ToolProxy` + `Toolbar.group_types` (Bokeh's
  long-press flyout, the same mechanism used for ZoomIn/ZoomOut). Decide
  during Phase 3 prototyping; the hook layer is unaffected.
- **Follow-latest UI shape in the toolbar.** Two adjacent `CustomAction`s
  with mutual exclusion vs a single tri-state cycling button — pick
  during Phase 5. Window value source is also open (see below).
- **`CustomAction.on_change('active', ...)` server-side callbacks under
  multi-session.** Bokeh `Tool` instances created Python-side live in
  the document of the session that created them. Verify in Phase 3
  prototyping that each session's controller correctly receives its
  own tool's events without bleed between sessions. If we see bleed,
  fall back to the Panel widget row above the plot (hook layer
  unchanged).
- **Slicer x/y range writes.** The slicer's kdims change the data
  per-slice. Plan: handle x/y via the same per-axis hook; verify in
  Phase 3 that this doesn't fight the kdim-driven re-render. If it does,
  the slicer can keep its current `framewise=True` path as a localized
  exception.
- **Initial state when the pipe has dummy data.** Toggles default on (per
  strategy §4.8), but the first `compute()` may produce dummy bounds.
  Verify the first real-data tick correctly snaps the range; if not, the
  controller's first hook call needs a one-shot "fit" semantic.
- **Padding on log-scale c.** Linear pad of 5 % is fine; the strategy doc
  picks ÷×1.1 for log. Confirm that ÷1.1 on a very small positive value
  doesn't produce a non-positive lower bound (would crash
  `LogColorMapper`).
