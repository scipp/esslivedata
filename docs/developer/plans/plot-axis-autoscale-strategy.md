# Plot axis range auto-scaling — strategy

Related issues: #910 (zoom reset on rescale), #912 (colorbar not updating, intermittent).

This document is the design discussion that precedes a concrete plan. Section 1
maps the current implementation; section 2 enumerates limitations and observed
problems; section 3 catalogues options for moving forward.

## 1. Current implementation

### 1.1 What we want, in tension

Two requirements pull against each other:

- **Stream growth must be visible.** Counts accumulate, time-series append to
  the right, ROI sums grow. If the displayed range never widens, new data
  silently leaves the frame and the plot stops being useful. With HoloViews
  on a `Pipe` stream this requires opting into `framewise=True` somewhere,
  because Bokeh's default behaviour with a swapped data source is to keep the
  previous axis range.
- **Interactive zoom/pan must survive data updates.** Updates arrive ~1 Hz; a
  user who zooms in cannot have the range reset under their cursor every
  second. Naively setting `framewise=True` on every frame resets the range to
  data extents every frame, destroying pan/zoom.

The current implementation reconciles these by toggling `framewise` per
frame — `True` only when "the bounds changed enough" to justify resetting,
`False` the rest of the time.

### 1.2 Data flow

```
Kafka → DataService → on_data → plotter.compute(data)
                                ├── Autoscaler.update_bounds(data)
                                │   → returns `changed: bool`
                                ├── element = converter.to_holoviews(data)
                                ├── opts['framewise'] = changed
                                └── _set_cached_state(element.opts(...))
                                └── mark_presenters_dirty()
                                                ↓
                  SessionLayer.update_pipe (polled by Panel)
                                                ↓
                            pipe.send(cached_element)
                                                ↓
                       DynamicMap re-renders via passthrough
```

The opt is baked into the cached element, so every session receives the same
element with the same `framewise` value for that frame.

### 1.3 The `Autoscaler` class

`src/ess/livedata/dashboard/autoscaler.py` keeps per-plotter, per-`ResultKey`
state:

- `coord_bounds: dict[dim_name, (low, high)]` — last accepted axis range per
  dim;
- `value_bounds: (low, high)` — last accepted value/color range.

On every `update_bounds(data)`:

- For each 1-D dim coord, compare current data extent against tracked extent.
  Bounds update (and `changed = True`) if either edge moves outside its
  threshold band:
  - `grow_threshold` (default 0.01; instantiated as 0.1 by all plotters) —
    fraction of current extent by which the new edge must exceed the old
    before growth is accepted;
  - `shrink_threshold` (default 0.1) — fraction by which the new edge must lie
    inside the old before shrinkage is accepted.
- Multi-dim or missing coords fall back to size-based bounds (`(0, size)`).
- Value bounds use the same bidirectional threshold logic against
  `data.nanmin()/nanmax()`.

`update_bounds` returns `True` if any bound changed. The plotter passes that
into `opts(framewise=changed, ...)`.

### 1.4 Crucial observation: the autoscaler does not set ranges

The `coord_bounds`/`value_bounds` it accumulates are **never read out** —
nothing pushes them into `xlim`, `ylim`, `clim`, or `Range1d`. Their only role
is to debounce the `framewise` flag.

That means the "margin" people associate with the autoscaler is *implicit
lag*, not explicit padding:

- Between two `framewise=True` frames Bokeh keeps the range it computed at the
  last reset, which equals the data extent at that moment.
- Data continues to drift until it exceeds the tracked bounds by
  `grow_threshold` (10%), at which point `framewise=True` snaps the range to
  the *current* data extent (no overshoot).
- So the range lags the data by up to 10%; it never leads it.

The autoscaler's docstring describes intended behaviour ("update with
framewise=True only when bounds change significantly"), which the
implementation does match. The mental model some of us have ("autoscaler sets
a wider range so we don't have to update as often") is **not** what the code
does — and would require pushing explicit `xlim`/`ylim`/`clim` to Bokeh, which
nothing currently does.

### 1.5 Per-plotter use

| Plotter             | Uses autoscaler  | Framewise behaviour                                  |
|---------------------|------------------|------------------------------------------------------|
| `LinePlotter`       | yes (1 per key)  | `framewise = changed` for curve + error element      |
| `ImagePlotter`      | yes (1 per key)  | `framewise = changed`; explicit `clim` only as fallback when all-NaN on log scale |
| `Overlay1DPlotter`  | yes (1 per key)  | One `changed` from full 2D, same flag on all slices  |
| `FlattenPlotter`    | yes (via Image)  | `grow_threshold=0.1`; flattens then delegates        |
| `BarsPlotter`       | no               | 0-D, axis bounded by category labels                 |
| `SlicerPlotter`     | **no**           | `framewise=True` always; pre-computes global `clim` over the full 3-D buffer for cross-slice consistency |
| `CorrelationPlotter`| no               | No autoscaler; HoloViews default                     |

The split is interesting: every "live-growing" plotter goes through the
autoscaler; the slicer (where the user drives the dimension being viewed)
opts out and uses a hard `framewise=True` plus a pre-computed global `clim`.

### 1.6 Other range-affecting code

- `Plotter._get_log_scale_clim` — falls back to `(1, 10)` when all data is NaN
  on log scale, to dodge a HoloViews `LogColorMapper` crash.
- `SlicerPlotter._compute_global_clim` — scans the entire 3-D buffer once per
  `compute()` to pick a single `clim` for all slices.
- `scipp_to_holoviews._compute_coord_bounds` — element-level bounds derivation
  for QuadMesh; not a control surface.

There are **no** explicit `xlim` / `ylim` / `Range1d` / `DataRange1d` calls
anywhere in the dashboard. Range control happens entirely through HoloViews'
`framewise` opt.

## 2. Limitations and observed problems

### 2.1 Zoom resets on every threshold crossing (#910, root cause)

The user zooms in. Data continues to accumulate. Once vdim extent grows by
more than 10 % of the *current bounds*, `framewise=True` fires and Bokeh
resets the range to the new data extent — discarding the user's zoom.

The interval between resets shrinks geometrically for sub-linear growth
(counts accumulating in a histogram). 10 % of 100 is 10; once you're at 1000,
10 % is 100. So the resets *do* slow down as the experiment progresses, which
is presumably why the threshold approach was tried in the first place. But:

- Early in a run the resets are frequent enough to be unusable for
  interactive inspection.
- The user has no way to opt out, or to widen the threshold per plot.
- There is no signal to the user that a reset is about to happen.

### 2.2 The autoscaler description does not match what it does

The docstring and the explanation in the GitHub comment on #910 describe a
mechanism that "sets a range that exceeds what would be needed by a certain
margin". The code does not do that — it only debounces. This matters because
the obvious fix ("tune the margin") may not give the expected behaviour, and
any new contributor reading the code will form the wrong model.

### 2.3 Apparent "auto-adjust to keep the curve centered" (#910 report)

The reporter describes the y-axis scale "auto-adjusting to keep the curve
centered" *between* resets. Per our reading of the code, that should not
happen — `framewise=False` should freeze the range. Possible explanations
worth investigating before designing any fix:

- Bokeh's `DataRange1d` (the default range type) auto-follows the data
  source when its content changes, even without HoloViews re-issuing a
  `framewise` recalc. HoloViews uses `DataRange1d` for the figure axes by
  default. This may mean we have *less* control than we think — Bokeh is
  silently re-ranging on every Pipe send.
- HoloViews may itself recompute ranges on every `Pipe.send` and we have been
  misreading what `framewise=False` actually means in the streaming case.

Either way: a robust fix probably requires going below the
`framewise` abstraction to a level where we control the range objects
directly (see 3.4).

### 2.4 Time-series plots that never extend (#910 second symptom)

The user mentioned cases where the right edge of a time-series plot did not
extend with new data. Plausible causes:

- The autoscaler debounces growth, so for fine-grained additions to the right
  edge the threshold may not trigger and the rightmost data falls off the
  visible frame for many seconds.
- For dims without 1-D coords the autoscaler falls back to `(0, size)`. A
  time-series whose dim coord is timestamps would trigger growth correctly;
  one whose dim has no 1-D coord would be range-locked to integer indices.

We have not pinned down which of these is happening in the reported case; a
targeted reproduction is needed.

### 2.5 Intermittent colorbar freeze (#912)

The reporter sees a 2-D plot whose underlying data continues to update
(white tiles fill in) but whose colorbar `clim` does not follow, eventually
washing the image to a single colour. We do not have a confirmed root cause.
Hypotheses ordered by suspicion:

- **Value extent stays inside the threshold band.** For an image whose tiles
  fill in (rather than scale up), `value_bounds.high` may not change much
  even as the image visibly changes. `framewise=True` never fires, the
  colorbar stays fixed, and any tile that exceeds the early `clim` saturates.
  This is a *behaviour bug* of the threshold model, not a race.
- **A `framewise` flip lands on a frame where Bokeh has already cached the
  color-mapper for the previous extent.** This would be a HoloViews/Bokeh
  state-sync issue; reload fixes it (which matches the report).
- **`shrink_threshold` triggers a shrink that locks in too-low limits.** If
  early data has one large value that later goes away in the visible window,
  the colorbar may snap to a too-tight range.

These three need distinct fixes. The first is the most consequential because
it would make the autoscaler subtly wrong for the very case it is supposed to
serve (image streaming).

### 2.6 No "reset / zoom-to-fit" affordance

Bokeh's reset tool snaps back to whatever range HoloViews considers the
"home" range, which on a Pipe stream is the range at the most recent
framewise reset. There is no way for the user to say "fit the current data
extent now" without waiting for the threshold to fire. Several users have
asked for this in person.

### 2.7 No per-plot configurability

Thresholds are baked into `from_params`/`from_display_params`. There is no UI
control and no per-plot YAML override. A counts-accumulating instrument and a
slow-drifting time-series have very different needs but get the same 10 %.

### 2.8 Different plot types, same mechanism

All non-slicer streaming plotters use the same autoscaler, but the actual
"shape" of the range update problem differs:

- **Images (`ImagePlotter`)**: kdims are usually static (detector pixel
  layout, x/y bins). Only the colorbar (`clim`) really needs to grow. The
  axis ranges almost never need to grow.
- **1-D streaming line / histogram (`LinePlotter`)**: kdim is static
  (wavelength bins, ROI binning); vdim (counts) grows. Same shape as images
  but for y instead of color.
- **Time series (`LinePlotter` via `FullHistoryExtractor`)**: kdim grows to
  the right (timestamps) and vdim drifts. Both axes need to grow, often
  asymmetrically.
- **Overlay 1-D**: same as line but with N curves sharing one autoscaler.
- **Slicer**: kdim and vdim are user-driven (which slice). We already opt out
  of the autoscaler here.
- **Bars**: 0-D, mostly irrelevant.
- **Correlation histograms**: 1-D / 2-D, but axes are determined by chosen
  variables — kdim extent grows as new (x, value) pairs arrive.

The point: the autoscaler is uniform; the problem space is not. Treating
"axis range can grow" and "colorbar can grow" identically — and ignoring
which axis is naturally static — leaves a lot of leverage on the table.

### 2.9 Per-`ResultKey` autoscaler, but shared overlay

`Overlay1DPlotter` uses one autoscaler keyed by the *primary* `ResultKey` and
fans the same `framewise` out to every slice element. Correct for overlay
semantics, but in `LinePlotter` with multiple datasets each dataset has its
own autoscaler — and they don't coordinate. Bokeh sees an Overlay where
individual curves toggle `framewise` independently, which interacts with
shared-axes range resolution in ways we have not characterised.

## 3. Options going forward

Alternatives considered. §4 selects the combination we will pursue; this
section is kept for context on which approaches were weighed and why.

Each subsection is a self-contained option; they compose.

### 3.1 Remove the autoscaler, push explicit ranges

Replace the per-frame `framewise` flip with explicit `xlim` / `ylim` / `clim`
opts computed from accumulated bounds, plus `framewise=False` everywhere. The
range only ever changes by *us* setting new explicit limits, not by Bokeh
auto-snapping to data.

Pros:
- We control exactly when the range moves. Zoom preservation becomes a
  user-driven concern: if we don't change the limits, the range stays.
- Symmetric handling of axis range and colorbar.
- Removes the mismatch between docstring and code.

Cons:
- Explicit `xlim`/`ylim` on a HoloViews element interacts with user pan/zoom:
  setting limits *also* resets the user's zoom. We would need to detect
  "user has zoomed" and avoid overriding their range — likely via a Bokeh
  `hook` that reads the current `Range1d.start/end` before setting new ones.
- More state to manage; more places to get it wrong.
- Initial range needs a sensible empty-data default.

This is the "honest" version of what the docstring already promises and is
the foundation for several of the other options.

### 3.2 Add explicit padding (the "real" margin)

Within either the current `framewise` debouncing model or a fully explicit
range model, set the range *wider* than the current data extent — e.g.,
`high = data.max() * 1.1` for log axes or `+ 10% of extent` for linear. The
range only needs to move when data falls outside the *padded* range, not the
data extent.

Pros:
- Reset frequency drops by an order of magnitude with reasonable padding.
- Combines well with 3.1; the padding becomes the explicit margin.

Cons:
- Aesthetically: empty padding looks like wasted space.
- Has to be tuned per axis / per scale type.

### 3.3 User controls

Three obvious controls, each useful independently:

1. **"Fit data" button / keyboard shortcut.** Re-fit the range to current data
   extent once, then resume zoom-preserving updates. This is the single most
   requested feature in conversations.
2. **Autoscale on/off toggle, per axis (x / y / c).** When off, the axis is
   completely frozen until the user pans. When on, current heuristic applies.
   Could be a small dropdown next to Bokeh's reset tool, or a per-axis lock
   icon.
3. **"Follow latest" mode for time-series.** Axis tracks the right edge of
   the data with a fixed window (e.g., last 60 s), regardless of accumulated
   history. Pairs naturally with `FullHistoryExtractor`.

Adding (1) alone would dramatically improve the situation for #910 — users
can zoom freely and snap back manually when they want to see new extent.

### 3.4 Bokeh-level hook for live range tracking

Instead of opting into `framewise` (which throws away pan/zoom state), use a
`hooks=[...]` callback on the element to attach a Bokeh callback that:

- Reads the current `Range1d.start/end`;
- Compares against the new data extent;
- Updates `Range1d.start/end` *only if* the user has not interacted with the
  axis since the last frame (Bokeh exposes `_initial_start`/`_initial_end`
  and the active `RangeChange` events to detect this).

Pros:
- The "growing axis preserves zoom" problem becomes solvable in principle.
- We can implement asymmetric updates: grow the right edge without touching
  the left, exactly the time-series case.

Cons:
- Bokeh internals; brittle across versions.
- Requires shipping JS-side state about user interaction, which may not be
  available to a Pipe-driven render.

This is potentially the cleanest solution to #910 but is also the highest
risk in implementation.

### 3.5 Per-plot-type strategies

Different plot types have different "what's growing" profiles (2.8). A
per-type policy:

- **Image:** never auto-grow axis range (kdims are static); auto-grow only
  `clim`, with explicit padding (3.2). This alone likely fixes #912 as a
  side effect.
- **Streaming line/histogram (LinePlotter, non-time-series):** static kdim;
  auto-grow y with padding.
- **Time series:** asymmetric — pin left edge or follow window (user
  choice); grow right edge to current latest timestamp; auto-grow y as for
  line.
- **Overlay 1-D:** as line, but with a single shared autoscaler over all
  slices (already the case).
- **Slicer:** keep current behaviour (always `framewise=True`, global `clim`
  pre-computed).
- **Correlation:** TBD — needs separate review.

This requires labelling each plotter with its "growth profile" and letting
the range-update logic dispatch on that label.

### 3.6 Keep the autoscaler but fix its actual semantics

Minimum-risk option: stop pretending the autoscaler does what it doesn't.
Either:

- (a) Make the docstring honest: it is purely a debounce on `framewise`. No
  padding, no overshoot. Then expose the threshold per plot and add a "fit
  data" button (3.3.1) on top.
- (b) Make the *implementation* match the docstring: compute padded
  `xlim`/`ylim`/`clim` from the tracked bounds and apply them (3.1 + 3.2).

### 3.7 Investigation prerequisites

Before committing to (3.4) or rewiring `framewise` we should:

- Reproduce #910's "axis auto-adjusts to keep curve centered" — confirm
  whether Bokeh's `DataRange1d` is silently auto-following the data source
  independently of `framewise`. This changes which of (3.1)–(3.4) is even
  effective.
- Reproduce #912 with a controlled scenario (image tiles fill in, max
  unchanged) to confirm the threshold-band hypothesis (2.5).
- Decide whether `DataRange1d` or `Range1d` should be our baseline range
  type — `Range1d` gives us full control but disables Bokeh's reset.

## 4. Chosen direction

Converged on: **remove the autoscaler entirely and replace it with
per-axis manual toggles plus a Fit button**, plus a Follow-latest mode
for time-series x. Per-axis range updates go through a Bokeh
`hooks=[...]` callback that mutates `Range1d.start/end` directly --
HoloViews `xlim`/`ylim` opts empirically do not propagate to Bokeh on
subsequent renders of a Pipe-streamed element (see §4.6). Colorbar
(`clim`) stays on the HoloViews opt surface, which does propagate.

### 4.1 Why not "autoscale toggle = framewise passthrough"

An earlier shape of this section proposed wiring the autoscale toggle
straight to HoloViews' `framewise` opt. Reading the HoloViews source
(`plotting/plot.py::_compute_group_range`, `core/options.py` "norm" group)
shows that does not give us enough granularity:

- The "norm" option group on every element type exposes only two booleans:
  `framewise` and `axiswise`. There is no per-axis variant.
- `_compute_group_range` applies the framewise check uniformly across
  `el.dimensions("ranges")`, i.e., across kdims and vdims together. For an
  `hv.Image` this means x, y, and the colorbar are reset as one — the
  central case where we want vdim/color to autoscale while x/y stay zoomed
  *cannot* be expressed by a single `framewise` flag.

The decoupling mechanism is per-axis explicit limits. In an ideal world
those would be the HoloViews `xlim` / `ylim` / `clim` opts, but §4.6
shows that for live Pipe-streamed updates only `clim` actually
propagates -- so we use `clim` via opt and reach Bokeh's `Range1d`
directly via a `hooks=[...]` callback for x/y.

### 4.2 Shape of the solution

- **Per-axis autoscale toggles (one per autoscalable axis on a plot).**
  Each toggle defaults to "on" and gates whether we push a new limit
  for that axis on each frame. Toggle on -> push limit each tick.
  Toggle off -> skip the push, so the axis stays where the user left
  it. `framewise=False` is the global default; we never use
  `framewise=True` in the normal path.
- **Per-axis range updates go via Bokeh hook, not HoloViews opt.**
  Empirically (§4.6), HoloViews only honours `xlim`/`ylim` at initial
  figure construction; subsequent renders via `Pipe.send` don't
  propagate new values, and neither does `element.redim.range(...)`. A
  `hooks=[...]` callback that mutates `plot.handles['x_range'].start /
  .end` (similarly `y_range`) on each render works. We use this for
  axis ranges.
- **`clim` stays on the opt surface.** The color mapper handle gets
  mutated each render through a separate HoloViews code path, so
  `.opts(clim=(lo, hi))` per frame works as expected. Cleaner than
  reaching for `plot.handles['color_mapper'].low/high` ourselves.
  To investigate: Do 1D elements like `hv.Line` have a similar `vlim` option that
  works like this, or do we need to use the `ydim` mechanism via a hook?
- **Fit button: one-shot fit-to-data, regardless of toggle state.**
  Mutates the figure's `Range1d` / `color_mapper` handles directly to
  current data extent. Toggle state is unchanged afterwards. Single
  button covering all axes by default; see §4.9 for the open question
  of splitting into "Fit ranges" (kdims) and "Fit value" (vdim/clim).
- **Autoscaler class, `grow_threshold` / `shrink_threshold` kwargs,
  and the autoscaler test suite are deleted.**

### 4.3 What we rejected: auto-disable on interaction

An earlier draft of this section proposed that a `RangesUpdate`
listener auto-flip the corresponding axis toggle to "off" on first
user pan/zoom, so the natural workflow ("see something interesting ->
zoom") would work with autoscale on by default. We rejected this
after prototyping. Three reasons:

- **Granularity mismatch.** Bokeh's `RangesUpdate` reports both axes
  together, and most zoom tools (wheel zoom, box zoom) move x and y
  simultaneously. Auto-disable would freeze both even when the user
  only meant to inspect one. The "watch live, zoom x to inspect a
  feature, keep y/clim following" workflow is hard to express.
- **Hidden state changes.** "Why did Y autoscale stop?" is a real UX
  cost when an earlier zoom invisibly flipped a switch. Explicit
  toggles cost a click but never confuse.
- **Browser-scroll accidents.** Wheel zoom triggers on plain mouse
  wheel events when the cursor happens to be over a plot. Pairing
  this with auto-disable would silently freeze axes whenever the user
  scrolls past a plot. Fit-button recovery on an explicit-toggles
  model is cleaner.

The cost is one extra click before deliberate zoom (toggle the
relevant axis off first), which we judge acceptable for scientific
users. Fit recovers from accidental zoom under default (toggle-on)
state.

§4.6 validates that `RangesUpdate` would have been usable as a signal
(it fires only on user interaction, not on our hook writes), so the
option remains technically open if the manual-toggle UX turns out to
be wrong in practice.

### 4.4 What this fixes (and what stays open)

Mapping the problems in §2 onto the chosen design:

| §2 problem                                            | §4 resolution                                                          |
|-------------------------------------------------------|------------------------------------------------------------------------|
| 2.1 Zoom resets on threshold crossing (#910 root)     | Manual per-axis toggle off before zoom; Fit to recover.                |
| 2.2 Docstring/code mismatch in `Autoscaler`           | `Autoscaler` deleted.                                                  |
| 2.3 "Auto-adjust between resets" mystery              | Sidestepped: when a toggle is on we mutate `Range1d.start/end` via hook each frame, overriding whatever Bokeh's `DataRange1d` would have done. |
| 2.4 Time-series right edge not extending              | Hook writes `x_range.start/end` every frame (or via Follow-latest), so the right edge tracks current latest by construction. |
| 2.5 Intermittent colorbar freeze (#912)               | Fixed: explicit `clim` opt every frame (when toggle on) removes the threshold-band trap. |
| 2.6 No "fit data" affordance                          | Fit button (§4.2).                                                     |
| 2.7 No per-plot configurability                       | Per-plot toggles (§4.2); padding and follow window become user-facing. |
| 2.8 Different plot types, same mechanism              | Per-plotter policy decides which toggles exist (§4.2).                 |
| 2.9 Per-`ResultKey` autoscaler in overlays            | Single shared toggle state per plot; per-axis limits are computed over all overlaid datasets together. |

### 4.5 Smaller design points

- **Initial state.** Pipe starts with dummy data, so toggle defaults
  must be "on" — otherwise the plot is stuck on dummy bounds. Toggles
  only change on explicit user click; until then, all autoscalable
  axes follow data.
- **Follow + x-autoscale interaction.** Follow-latest is a specific
  form of x autoscale, so the two never coexist — time-series plots
  get Follow instead of an x toggle, other plot types do not get
  Follow.
- **User interaction during a running autoscale.** With a toggle on, a
  user pan/zoom is overwritten by the next tick's hook write. This is
  the documented behaviour: toggle off first to inspect; recover with
  Fit if the zoom was accidental.
- **No more threshold debounce.** Computing and pushing limits every
  frame is cheap; we no longer need to skip frames to "preserve zoom"
  because the toggle does that explicitly. The cost is one min/max
  scan per axis per frame.
- **Fit needs live handles, not a Pipe round-trip.** Fit mutates the
  current figure's `Range1d` / `color_mapper` directly, so it must
  capture those handles when the plot first renders (e.g., during the
  hook's first invocation) and keep the reference around.
- **Log-scale all-NaN workaround stays.** `Plotter._get_log_scale_clim`
  (the `(1, 10)` fallback that dodges a HoloViews `LogColorMapper`
  crash when all data is NaN on log scale) is independent of the
  autoscaler and is preserved.

### 4.6 Validation findings

Tested against HoloViews 1.23 + Bokeh 3.8 via
`.scratch/clim_zoom_prototype.py`. The non-obvious results:

- **`xlim`/`ylim` opts do NOT propagate to Bokeh on subsequent
  renders.** Setting these per frame via `element.opts(xlim=...)` has
  no effect after initial figure construction.
  `element.redim.range(...)` shares the same fate. `axiswise=True`
  does not change this. The working surface for live axis-range
  updates is a Bokeh `hooks=[...]` callback that mutates
  `plot.handles['x_range'].start / .end` directly. This drives §4.2's
  decision to use a hook for x/y.
- **`clim` opt DOES propagate on each render.** The color mapper
  handle is updated through a separate HoloViews code path, so
  `.opts(clim=(lo, hi))` per frame works as expected. We use it for
  the colorbar.
- **`RangesUpdate` fires only on user interaction.** Programmatic
  `Range1d.start/end` mutation via our hook does not trigger it. This
  would have made it a clean signal for the auto-disable mechanism
  proposed in the earlier §4.3, but §4.3 has been rejected on UX
  grounds; the validated event remains available if we revisit.

### 4.7 Implementation outline (not yet a plan)

- Delete `src/ess/livedata/dashboard/autoscaler.py` and its test.
- Drop `grow_threshold` / `autoscaler_kwargs` from `Plotter` and every
  subclass `from_params` factory.
- Add a per-axis "autoscale state" to each plotter's per-plot UI
  state (small bag of booleans: `y_autoscale`, `c_autoscale`, ...).
- Add a `_range_hook` helper that returns a `hooks=[...]`-compatible
  callback. The callback (a) captures `plot.handles['x_range']` /
  `y_range` / `color_mapper` into plotter state on first invocation so
  Fit can mutate them directly, and (b) reads its toggle's current
  value and, if on, mutates `plot.handles['<axis>_range'].start /
  .end` to the new limits.
- In each plotter's `plot()`, build opts with `framewise=False`,
  attach per-axis range hooks for the axes covered by the plotter's
  policy, and include `clim=(lo, hi)` in opts when the colour toggle
  (if any) is on.
- Add toggle UI and a Fit button per plot, per plotter policy. Fit
  reads the captured handles and writes current data extent to all
  axes regardless of toggle state.
- Add Follow-latest mode for time-series (`FullHistoryExtractor`-based
  `LinePlotter`); same hook mechanism, with `(latest - window,
  latest)` as the range.

Net code change is expected to be near zero or slightly negative: the
removed threshold/autoscaler machinery offsets the added hook factory,
toggles, and Fit plumbing.

### 4.8 Resolved defaults

- **Padding on explicit limits.** Yes, conservative defaults: linear axes
  use a small symmetric pad (e.g., 5 % of extent on each side); log axes
  use a multiplicative factor (`high *= 1.1`, `low /= 1.1`). Avoids data
  sitting flush against the frame edge and absorbs small frame-to-frame
  fluctuations without visible jitter.
- **Toggle persistence.** Reset each session. Matches how other plot UI
  state is handled today, avoids cross-session surprises, and the cost
  (re-toggling after reload) is negligible.

### 4.9 Open questions

- **UI placement.** Where do the toggles and Fit button live — Bokeh
  toolbar buttons (added via a hook; isolate well per-plot, but Bokeh
  internals), Panel widgets above each plot (easier to wire, state lives
  in Panel, more vertical space), or in a per-plot settings popover
  (compact, but extra clicks)? Needs a call from someone familiar with
  the existing dashboard widget conventions.
- **Follow-latest window: where does the value come from?** Options: a
  per-plot user input (slider or numeric field), a per-plotter default
  in YAML config, or a hard-coded default. The window also needs a
  unit/format decision (seconds? human-readable like `"60s"`/`"5min"`?).
- **Single Fit vs split "Fit ranges" / "Fit value".** A single Fit
  button refits every autoscalable axis at once. A two-button split
  would distinguish kdim refit ("Fit ranges": `x_range`/`y_range`)
  from vdim refit ("Fit value": `clim` for images, `y_range` for
  curves). Useful when the user has frozen x/y to inspect a feature
  but wants the colorbar to re-fit, or vice versa. The single-button
  prototype is fine for now; revisit during implementation once we
  see how often users actually want partial refit.
