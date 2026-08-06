# HoloViews live-update fast path — handoff (esslivedata #1200)

Started 2026-08-05, rewritten the same day after re-baselining against the fork,
extended 2026-08-06 with the app-side change (§2.3). Everything below was
measured in this devcontainer. The question this hands off: **can the remaining
per-layer repaint cost be removed in HoloViews itself, in a shape upstream would
accept, instead of as a private-API workaround in esslivedata?** Two
upstream-shaped fixes have landed on a fork branch, and the app-side half of §5
is merged; what is left is the upstream decision (§5).

## 1. Why this exists

esslivedata #1198 measured the dashboard's capacity ceiling: **~27 ms of shared
Tornado IOLoop time per visible plot cell per data frame**, i.e. ~35 cell-frames
per second per dashboard process, shared across all browser sessions. A 6x6 grid
(legal in the UI) at 1 Hz exceeds that on its own; so do three users on 12-cell
grids. Past the ceiling the loop never goes idle and every session on that server
is degraded for as long as the grid is open.

~85 % of a pass is the per-layer repaint: `SessionLayer.update_pipe` ->
`hv.streams.Pipe.send(element)` -> DynamicMap -> HoloViews plot update -> Bokeh
models. Every other item on the #1198 list (fewer models per cell, chunked
materialization, skipping no-op freeze-exits) trims constants; only the repaint
moves the ceiling.

Two architectural facts still shape every option below:

- **esslivedata computes the element once, shared across sessions**
  (`PlotDataService` / plotters, on the ingestion thread), then each session
  pushes that finished element through its own DynamicMap. The per-session cost
  is pure rendering, not computation.
- **We already own axis ranges.** `CellAutoscaleController`
  (`dashboard/cell_autoscale.py`) writes per-axis ranges through `RangeHandles`
  (`dashboard/range_hook.py`) on every render, via a HoloViews hook, so
  HoloViews' own `compute_ranges` / `_update_ranges` is largely redundant *for
  us* — we overwrite the result. Upstream cannot assume that.

## 2. Current state

### 2.1 Where the work lives

| | |
|---|---|
| esslivedata worktree | `/workspace/esslivedata/.worktrees/1200-holoviews-fast-path`, branch `1200-holoviews-fast-path` (no source changes, harness in `.scratch/1200/`) |
| dedicated venv | `<that worktree>/.venv`, created with `--system-site-packages`; editable installs of esslivedata **and** of the HoloViews worktree; `pandas` pinned `<3` (2.3.3) because HoloViews 1.23 does not work with the environment's pandas 3.0 |
| HoloViews worktree | `/workspace/hv-fastpath`, branch `perf/data-only-update-fast-path`, based on fork `main` `4b7532fbd` |
| harness | `.scratch/1200/` in the esslivedata worktree, copied to `/workspace/esslivedata/.scratch/1200/` so it survives the worktree |
| app-side work (§5 option 2) | `/workspace/esslivedata/.worktrees/1199-hoist-plot-styling`, branch `1199-hoist-plot-styling`, PR #1206; own `.venv` on **released** HoloViews 1.23.1, deliberately not the fork |

Switching the HoloViews worktree between `4b7532fbd` and the branch is how every
A/B below was taken — the editable install follows the checkout, no reinstall
needed. `/workspace/holoviews` (the main fork checkout) holds `main`, so the
worktree must use `--detach 4b7532fbd` rather than `checkout main`.

### 2.2 What landed (HoloViews branch, 2 commits)

- `cbe23523a` **Skip dataset linking when an operation returns its input.** An
  operation returning the element it was handed (`apply_nodata` with `nodata`
  unset — which is what the registered `Image`/`Raster`/`QuadMesh` compositors do
  on every frame) linked the element's dataset to itself, materializing a fresh
  `Dataset` from the element each time. ~0.8 ms per Image frame.
- `afa683827` **Reuse the resolved palette across frames.**
  `_get_colormapper` re-ran `process_cmap` on every frame, sampling the colormap
  and hex-formatting 256 colors, then found the result identical to the mapper's
  existing palette. ~1.0 ms per Image frame — 60 % of `_update_glyphs`.

Both are behaviour-preserving by construction and carry regression tests
(`test_rasterplot.py::test_streaming_with_static_cmap`,
`::test_streaming_cmap_cache_invalidation`,
`test_datasetproperty.py::test_apply_identity_dataset[_dynamic]`).

All three HoloViews branches are pushed to the fork (`SimonHeybrock/holoviews`):
`perf/skip-identity-dataset-link` and `perf/reuse-resolved-palette` carry one
commit each off `main` and are what upstream PRs should be filed from;
`perf/data-only-update-fast-path` carries both and is the state every A/B in §3
was measured against.

Neither is the change §4 of the first draft proposed; see §3.2 for why.

### 2.3 What landed in esslivedata (PR #1206, §5 option 2)

`DefaultPresenter.present` no longer styles the session's DynamicMap; `Plotter.compute`
applies `style_opts()` to the finished frame instead. `Opts._dynamicmap_opts` wraps the
map in a `Dynamic` operation that re-applies the options to every frame — verified
directly: `dmap.opts(...)` mints 99 custom option ids over 100 frames, `element.opts(...)`
on the shared frame mints one per frame for all sessions, and a `Store`-registered group
spec mints none.

Measured with `.scratch/1198/grid_stall.py --mode render --cells 6` (12 layers, 1 Hz,
single session), three alternating runs per side, pooled over settled full passes:
**11.7 / 12.4 / 11.7 -> 9.5 / 9.8 / 10.6 ms** per layer per frame. The saving is per
session, so it scales with session count; a single-session harness understates it.

Rendering equivalence was checked with a bokeh-level model signature over 51 plotter
configurations (`.scratch/1199/signature.py` in that worktree) — identical before and
after — and `plot_sizing_invariant_test.py` gained a test that fails on every
parametrisation if styling moves back onto the DynamicMap.

The cell's `.opts(hooks=...)` wrapper (`widgets/cell.py`) remains and is irreducible
under this approach: hooks are session state, so they can be neither hoisted to the
shared compute path nor registered in the option store. Removing it means attaching
hooks to the plot instance after render — private API, not attempted.

### 2.4 Test status

| suite | result |
|---|---|
| `holoviews/tests/plotting/bokeh` | 1243 passed, 21 skipped |
| `holoviews/tests/core` + `operation` | 259 failed / 1824 passed / 253 errors — **identical before and after**; pre-existing environment breakage (numpy/pandas deprecation-as-error), not caused by the branch |
| esslivedata `tests/dashboard` (against the branch) | 2062 passed, 4 xfailed |

## 3. What is measured now

Environment: esslivedata devcontainer, HoloViews fork `4b7532fbd`
(v1.23.2a1+11, already containing the merged perf PRs listed in §6), bokeh 3.9.2,
panel 1.9.3, Python 3.12, idle 24-core machine.

### 3.1 Isolated bench, one layer (`.scratch/1200/bench.py`)

One harness, elements built *before* the timed loop (as in the app, where the
element is shared compute from another thread). **Run one case per process**
(`--case`): within a process, later cases are inflated by accumulated
custom-option ids and GC — that is what made an earlier all-cases run read
7.7 ms for `Image+clim` where an isolated run reads 5.7.

ms/frame, `4b7532fbd` -> branch:

| case | plain passthrough | as shipped (2 opts wrappers) | fast path (`_update_glyphs`) |
|---|---|---|---|
| Image | 6.94 -> **5.22** | 8.72 -> **6.79** | 3.28 -> 2.13 |
| Image + per-frame `clim` | 8.04 -> **5.66** | 10.10 -> **8.27** | 3.17 -> 2.11 |
| Overlay of 2 Curves | 7.61 -> 7.15 | 8.35 -> 8.29 | 2.82 -> 2.85 |

Overlays barely move, as expected: no Image compositor, no color mapper. Both
landed fixes are 2D-only.

### 3.2 Phase breakdown, and what the first draft got wrong

`bench.py --phases` attributes time exclusively (nesting is not double-counted).
On `4b7532fbd`, plain passthrough, per frame:

| phase | Image | Overlay(2 Curves) | data-dependent? |
|---|---|---|---|
| DynamicMap re-evaluation (`dmap callback`, exclusive) | 2.3 | 0.1 | no |
| `compute_ranges` + `_update_ranges` | 0.9 | 2.3 | partly — and we overwrite it |
| `_update_plot` | 0.5 | 0.5 | no |
| `lookup_options` (11 / 34 calls) | 0.2 | 0.8 | no |
| **`_update_glyphs`** | **1.8** | **0.9** | **yes — the actual data push** |
| container `update_frame` own time | — | 0.8 | no |
| (`pipe.send` total) | 6.3 | 6.0 | |
| (wall, including `doc.models.freeze()` exit) | 7.3 | 7.5 | |

Corrections to the first draft, so they are not re-derived:

- **The DynamicMap re-evaluation happens *inside* `update_frame`**, via
  `_get_frame` -> `get_plot_frame` -> `dmap[key]`. The draft placed it before
  `update_frame` and inferred its cost by subtraction, which is why the two rows
  did not add up.
- **"~0.9 ms of ~8 ms is the data push" was wrong.** `_update_glyphs` was 1.8 ms
  for an Image, and ~1.0 ms of that was the colormap re-resolution now fixed. The
  remaining push is ~0.65 ms (Image) / ~0.9 ms (Overlay). That plus the ~1.4 ms
  `doc.models.freeze()` exit is exactly the 2.1–2.9 ms the local fast path
  measures, i.e. the floor for anything that still goes through HoloViews.
- The draft's §2.5 health warning (numbers not comparable across three scripts)
  is obsolete: there is one harness now. The new caveat is one case per process.

### 3.3 End to end in the app (`.scratch/1198/grid_stall.py --mode render`)

Median of the settled passes in the last 20 s, parsed by `.scratch/1200/passes.py`:

| | fork main | branch |
|---|---|---|
| 6 cells (12 layers), pipe per layer | 11.2 ms | **8.9 ms** (−21 %) |
| 6 cells, pass (handler + freeze exit) | 154 ms | 141 ms |
| 24 cells (48 layers), pass | 943 ms | 1009 ms |

**The 24-cell number is not a measurement of the fix.** At 24 cells the loop is
past 100 % duty, so the pass time is dominated by queueing and GC (individual
passes range 202–1483 ms in both runs). The ceiling moved; that configuration
cannot see it. Use 6 or 12 cells for A/B, and keep 24 cells only for the #1198
acceptance check once the per-layer cost is low enough to leave the loop idle.

## 4. Where the remaining time goes

Of the **8.9 ms per layer** left in the app, **~6.7 ms is DynamicMap callback
evaluation** (`dmap_cb` 80 ms / 60 calls per pass at 6 cells, i.e. ~5 callbacks
per layer per frame). That is the two chained `.opts()` wrappers the app stacked
(`plots.py` `DefaultPresenter.present`, `widgets/cell.py` `_compose_plot`) plus
the compositor's own `Dynamic`. PR #1206 removed the first of the two; the
figures in this section predate it.

The per-frame cost of a wrapper is the cost of re-applying options to a new
element:

- `element.opts(typed Options)` — **0.47 ms**; `element.opts(**kwargs)` — 0.87 ms.
- Each call **mints a new custom-option id**: 250 ids in `Store._custom_options`
  after 200 calls, one per element. They are reclaimed by GC, which is visible in
  the app as `gc=` spikes up to 300 ms in a saturated pass.

So the ranking of what is left, per layer per frame, is now (the first row is
half taken: PR #1206 removed the style wrapper, §2.3):

| target | worth | risk |
|---|---|---|
| ~~per-frame `.opts()` re-application (2 wrappers)~~ → the hooks wrapper only | ~1.0–1.3 ms + GC relief | needs hooks attached outside the option system |
| ranges (`compute_ranges` + `_update_ranges`) | 0.9 ms (2D) / 2.3 ms (1D) | high upstream: the color mapper's clim comes from these ranges |
| `_update_plot` + `lookup_options` + `match_spec` | ~0.7–1.3 ms | low |
| `_update_glyphs` | 0.65–0.9 ms | this is the floor |
| `doc.models.freeze()` exit | ~1.4 ms | esslivedata #1202, not HoloViews |

The prototype local fast path (push a structurally identical frame straight into
the existing subplots via `_update_glyphs`, hooks re-run by hand) still measures
**2.8–3.1x** against `pipe.send` and is exercised by `bench.py`. It skips
everything in the table except the glyph write, which is why it beats any single
upstream change.

## 5. Decisions needed before continuing

The original plan's candidate 1 (a structural-identity gate in
`ElementPlot.update_frame`) is no longer the biggest lever, and its most valuable
part — skipping ranges — is the part upstream cannot take unconditionally. Pick
a direction:

1. **Upstream: make repeated identical `.opts()` cheap.** Reuse the custom-option
   id when the same static options are applied to an equivalent element instead
   of minting a tree per element. Biggest lever (~2.6 ms/layer plus GC relief),
   benefits every streaming HoloViews app — but it touches `Store`, the most
   delicate part of HoloViews, and interacts with the already-merged
   holoviz/holoviews#6904.
2. **~~esslivedata: drop the two per-frame `.opts()` wrappers.~~ Done, PR #1206**
   — see §2.3. Styling now rides on the computed frame; the hooks wrapper stays.
   Dropping it too needs the hooks attached to the plot instance after render,
   since anything routed through the option system is re-applied per frame.
3. **Candidate 1, scoped to the safe parts.** Gate on structural identity (same
   element type, kdims/vdims, group/label, ordered overlay keys) and skip
   `lookup_options` / `match_spec` / `_update_plot` / `_set_active_tools` /
   `_setup_data_callbacks`, but leave ranges alone. ~1 ms/layer, low risk, easy
   upstream sell, and it composes with (1).
4. **Fallback: take the local `_update_glyphs` workaround in esslivedata**
   (2.8x, three private names, guarded by a test that fails loudly on rename), or
   own the CDS writes outright (~1.7 ms floor, but the conversion correctness
   surface — hover columns, error bars, datetime axes, log-scale masking — moves
   to us permanently).

Recommendation from the measurements was **2, then 1**. With 2 done, the open
question is 1: an upstream `.opts()` fix now has to preserve what #1206 relies
on, i.e. a single typed application to a finished frame must keep costing one
lookup and one id, and elements must stay shareable across sessions after it.
(3) remains the low-risk upstream sell and still composes with (1); (4) is
untouched as the fallback.

Open questions that survive whichever way this goes:

- **Structural equality is still the crux** for (3) and (4). Too loose and stale
  titles/axes/hover tooltips ship; too strict and the fast path never fires. Our
  own frames change `clim` per frame via `element.opts(clim=...)` in
  `ImagePlotter`, which changes the custom-option id, so any style cache must key
  on it.
- **Hover**: `_update_hover` only matters when dims change, which the gate
  excludes — confirm with a test rather than by reading.
- **Overlays**: subplot ordering must be stable for the zip in the local fast
  path to be sound; compare ordered overlay keys, not counts.
- **Operations/datashade**: the gate lives after the element is obtained, so it
  should be orthogonal — check that `Dynamic` chains that *do* compute per frame
  are unaffected.

## 6. Harness inventory

`.scratch/1200/` (new, this session):

| file | what it does |
|---|---|
| `bench.py` | the one harness. Variants (2 wrappers / 1 / plain / local fast path) and `--phases` for the exclusive-time breakdown. `--case` to isolate a case in its own process. |
| `passes.py` | median settled pass cost from a `grid_stall --mode render` log |
| `compositor_probe.py` | proves the per-frame `Compositor` cost by emptying `Compositor.definitions` |
| `compositor_phases.py` | exclusive timing inside `Compositor.map` |
| `prof.py`, `prof2.py` | cProfile of one `pipe.send` loop, flat and holoviews-filtered |
| `*_hvmain.txt`, `*_hvfix.txt`, `variants_ab.txt` | the raw A/B captures behind §3 |

`.scratch/1198/` (previous session, still current):

| file | what it does |
|---|---|
| `grid_stall.py` | end-to-end driver. `--mode render --cells N --wait MS` (server log is the report), `--mode two --sessions N`, `--mode observer`, `--mode tour`. Needs `PLAYWRIGHT_CHROMIUM_EXECUTABLE=/usr/bin/chromium`, run from a repo root. |
| `sitecustomize.py` | injected into the dashboard via PYTHONPATH: per-pass timing, IOLoop stall monitor, GC accounting, per-phase timers, model census, cProfile arming |
| `repaint_bench.py`, `opts_bench.py`, `fastpath_probe.py` | superseded by `bench.py`; keep for the styling-declaration variants. Their numbers suffer the cross-case inflation `bench.py --case` fixes, which is why #1199's original table read "compute-time styling does not help" and the re-measurement inverted it |

`.scratch/1199/` in the `1199-hoist-plot-styling` worktree:

| file | what it does |
|---|---|
| `signature.py` | bokeh model signature over 51 plotter configurations; run on both sides of a change, diff the JSON |
| `agg.py` | pooled pipe ms per layer over the settled full passes of several render logs — the A/B metric, less noisy than `passes.py`'s per-pass median |
| `optsid_probe.py`, `group_probe*.py` | in `.scratch/1200/`: per-frame cost and custom-option-id growth for `dmap.opts` vs `element.opts` vs a `Store`-registered group, and that group's inheritance/title/key-path behaviour |

Fork-side prior art in `/workspace/holoviews`: `max-range-fast-path-proposal.md`,
`holoviews-custom-option-id-blowup.md`, and `.scratch/` with `benchmark_e2e.py`,
`bench_max_range.py`, `profile_suite.py`, `streaming_app.py`, plus PR write-ups.
**Read `benchmark_e2e.py` before writing a benchmark for an upstream PR** — it is
their measuring stick.

## 7. Environment gotchas

- **cProfile is process-wide on Python 3.12** (`sys.monitoring` is global), so a
  profile taken on the IOLoop thread also contains the ingestion thread's work.
  Use `time.thread_time` for attribution.
- **HoloViews' pre-commit `prettier` hook cannot install here**: npm refuses
  git-sourced packages (`EALLOWGIT`). Commit with `SKIP=prettier`; every other
  hook (ruff check/format, typos, whitespace) runs and passes.
- When that hook failed mid-run it **left the worktree's git index containing the
  hook repo's own files** (`package.json`, `update.py`, `.pre-commit-hooks.yaml`,
  `.github/workflows/update.yaml`), so `git status` failed with `unable to read
  <blob>` and `git fsck` reported missing blobs. Nothing was actually lost:
  `git read-tree HEAD` in the worktree repairs it, working tree untouched.
- Numbers are only comparable on an otherwise idle machine; do not A/B while a
  test suite is running.

## 8. Pointers

esslivedata issues: **#1198** tracker (measurement baseline + checklist),
**#1199** (per-frame `.opts()` wrappers — closed by PR #1206 via §5 option 2
rather than the merge-the-wrappers proposal in its body; the issue carries a
comment correcting its stale measurements), **#1200** (this work), **#1201**
(models per cell), **#1202**
(freeze-exit), **#1203** (chunked materialization), **#1204** done (PR #1205,
merged: lazy cell toolbars, 98 -> 75 models/cell).

esslivedata code: `dashboard/session_layer.py` (`SessionComponents.update_pipe`,
where a local fast path would live), `dashboard/plots.py`
(`DefaultPresenter.present`, `Plotter.compute`, `Plotter.style_opts`),
`dashboard/widgets/cell.py` (`_compose_plot`, the remaining `.opts()` wrapper),
`dashboard/cell_autoscale.py`
and `dashboard/range_hook.py` (we own ranges), `dashboard/session_updater.py`
(the pass, hold+freeze).

Upstream appetite is good: holoviz/holoviews **#6835** (skip Bokeh property
validation in `_update_datasource`), **#6837** (skip `param.update` when plot
options unchanged), **#6839** (cache static layout properties in `_update_plot`),
**#6904** (custom-option id growth) are all merged, same author; **#6982** is
open. The two commits in §2.2 are in the same mould and are pushed to the fork as
PR-ready branches, but no PR has been opened yet.
