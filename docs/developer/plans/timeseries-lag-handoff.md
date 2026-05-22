# Timeseries-plot lag investigation — handoff

Status: open. GitHub issue: [scipp/esslivedata#940](https://github.com/scipp/esslivedata/issues/940). Branch: `timeseries-lag-investigation`. Worktree: `/workspace/esslivedata-timeseries-lag`.

This doc is the load-bearing context for the next session. It does **not** re-describe code that's already in `src/`; it captures what's been tried, what was ruled out, where to look, and the open questions.

---

## TL;DR

- The user observes "entire dashboard freezes" with a handful of timeseries plots, even with no browser connected for a while and especially when switching to/from a plot-heavy tab.
- We built a test harness (server-side metric monkey-patches) and a real-dashboard repro (production `ReductionApp` with `NullTransport` + synthetic feeder).
- Server-side measurements **rule out** `DocumentModelManager.recompute` amplification, `Document.add_root`/`remove_root` mutation, WebSocket payload bursts, and `Plotter.compute` bursts as the dominant cause of the **tab-switch** freeze.
- Server-side measurements **confirm** that steady-state `pipe.send` against a live `Document` is ~100–130 ms/call (vs. ~9 ms headless), and that BACK > TO in cost (progressive degradation within a session).
- The **next thing to instrument is the browser**. Server side has done what it can.

---

## File map

Worktree root: `/workspace/esslivedata-timeseries-lag` (branch `timeseries-lag-investigation`). Venv: `.venv` (activate with `source .venv/bin/activate`).

### Production code (read-only — do not modify for measurement; use monkey-patches)

| Path | Why it matters |
|---|---|
| `src/ess/livedata/dashboard/plots.py` | `LinePlotter`, `Plotter.compute` (the rebuild-from-full-buffer hot path) |
| `src/ess/livedata/dashboard/extractors.py` | `FullHistoryExtractor` — per-tick O(N) datetime coord rebuild |
| `src/ess/livedata/dashboard/temporal_buffers.py` | `TemporalBuffer` (`DEFAULT_MAX_MEMORY = 20 MB`, ring buffer once full) |
| `src/ess/livedata/dashboard/session_layer.py:54` | `pipe.send(presenter.consume_update())` — the per-session push |
| `src/ess/livedata/dashboard/session_updater.py:146-164` | `_batched_update` with `pn.io.hold() + doc.models.freeze()` — the batching envelope around per-session pipe pumps |
| `src/ess/livedata/dashboard/widgets/plot_grid_tabs.py:1014-1060` | `_BatchedTabs` and the **active-grid streaming gate** (`update_pipe()` only on active grid) |
| `src/ess/livedata/dashboard/orchestrator.py:54-77` | `Orchestrator.update` — the bg ingestion thread that calls `_run_compute` synchronously inside `DataService.transaction()` |
| `src/ess/livedata/dashboard/dashboard_services.py:98` | `_update_interval = 0.2 s` — bg-thread poll cadence |
| `src/ess/livedata/dashboard/transport.py:113` | `NullTransport` — already-existing seam we use to avoid Kafka |
| `src/ess/livedata/dashboard/active_job_registry.py:65-72` | `ingestion_guard()` — `RLock` held during the whole bg-thread `Orchestrator.update` |

### Measurement tooling (under `.scratch/` — git-ignored)

| Path | Purpose |
|---|---|
| `.scratch/harness/metrics.py` | `Metrics`, `JsonlSink`, `install_patches`. The single import that wires up all hooks. **Read first.** |
| `.scratch/harness/server.py` | Minimal-app harness server (NOT the real dashboard). Used for the H2 scan. |
| `.scratch/harness/run_h2_scan.py` | Sweeps K / N / decoy / hz / freeze; produces `.scratch/harness/runs/h2/summary.json`. |
| `.scratch/harness/bokeh_client.py` | Headless `bokeh.client.pull_session` smoke test. **Use instead of Playwright in this devcontainer.** |
| `.scratch/harness/playwright_driver.py` | Skeleton, **not runnable here** (Chromium download is firewalled). Left for when env unblocks. |
| `.scratch/harness/h2_findings.md` | H2 report. Includes the "1000-decoy" experiment that we later showed doesn't apply to production. |
| `.scratch/harness/sensitivity.md` | Harness sensitivity proof — needed before trusting any number below. |
| `.scratch/repro/live_dashboard.py` | **The real `ReductionApp` repro.** Boots full production UI with pre-loaded fake timeseries. See "Reproduction recipes" below. |
| `.scratch/repro/metrics.jsonl` | Per-second windowed metrics written by the live-dashboard run. Truncated on each run. |
| `.scratch/repro/markers.txt` | User appends labels here; a watcher in `live_dashboard.py` emits them as `marker` events in the JSONL so we can correlate. |
| `.scratch/bench_timeseries.py`, `.scratch/bench_pipe_send.py` | The original **headless** microbenchmarks. **Misleading** — see "Pitfalls". Kept for historical comparison only. |
| `.scratch/timeseries_lag_findings.md` | The original write-up from before the live-document harness existed. Several numbers in it are wrong; this handoff supersedes. |

### Curated docs (committed)

| Path | Purpose |
|---|---|
| `docs/developer/plans/timeseries-lag-handoff.md` | This file. |

---

## What's been measured — by hypothesis

The original investigation framed six hypotheses (H1–H6). Status:

| | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | bg-thread `compute()` GIL load scales with K | **Partially confirmed**, but small absolute size at K≤4 | H2 scan, K1→K16: `compute_ms/s` grew 11→82, linear in K. Not enough alone to explain freeze at K=3–5. |
| H2 | `DocumentModelManager.recompute` BFS dominates | **Overstated.** Real but bounded in production. | H2 scan: recompute is **invariant** to K, N, tick rate; scales with total document model count (decoy scan, 0→200→1000 = 85→309→681 ms/s). But production uses `pn.Tabs(dynamic=True)` + active-grid gate, so total model count is bounded to active-tab content. |
| H3 | live-document `pipe.send` ≫ headless | **Strongly confirmed.** ~100–130 ms/call vs ~9 ms headless. | Live repro (`.scratch/repro/metrics.jsonl`): `ps_max ≈ 100–130 ms`, `ps_ms/s ≈ 400–600` with K=4 N=200k. |
| H4 | HV `Stream._subscribers` leak inflates GC pause time over a session | **Not yet directly tested.** Indirect evidence supports it. | One single-session run showed BACK > TO degradation (`ps_ms` 395→520, `cp_max` 25→93, a 136 ms gen-2 GC pause appeared). Multi-cycle connect/disconnect test not run. |
| H5 | browser-side render cost dominates | **Strongly supported by exclusion.** Open and worth instrumenting next. | All server-side hooks ruled out the switch cost. The "3-second delay" before pipe.send resumes after switch-TO shows the server idle during it. |
| H6 | first-paint connect-time cost | **Partially observed** in every run as a ~200 ms `cp_max` + 70 ms GC spike at t=0. Doesn't account for multi-second freezes. | Every harness run shows this; steady-state max is 15–25 ms. |

### Concrete numbers worth remembering

From the most recent live-dashboard run (K=4 plots, N=200k pre-fill, 1 Hz feed, single browser session, four tab switches):

| Quantity | Value | Where |
|---|---|---|
| Steady-state `Plotter.compute` (any phase) | ~30–40 ms/s, P50 ~10 ms, max ~15 ms | Fires for all 4 plots whether tab visible or not |
| Steady-state `Pipe.send` when plot tab visible | **~400–520 ms/s, max 100–280 ms per call** | This is the biggest steady-state cost |
| Steady-state `DocumentModelManager.recompute` | ~12–25 ms/s | 10 Hz Bokeh document tick, invariant to K/N |
| `Document.add_root` / `remove_root` during tab switches | **0 calls, all switches** | Confirms switching is visibility-toggle, not graph mutation |
| `patch_doc` rate during switches | ~1/s × ~200 bytes — flat | WS payload is not the cost |
| RSS during run | 226 MB → 570 MB then plateau | Mostly buffer pre-fill + Bokeh document state |
| `gc.get_objects()` count | 283 k → 451 k over ~70 s | Steady linear growth — leak signature, needs longer test |
| Worst single-window server burst | 340 ms (cp_max 93 + gc 136 + ps_max 167) | During switch-BACK transition (t=40) |

---

## Reproduction recipes

### Run the harness sensitivity proof

```sh
cd /workspace/esslivedata-timeseries-lag/.scratch
source ../.venv/bin/activate
python -m harness.run_sensitivity --dwell 30 --plots 4
```

Outputs `.scratch/harness/sensitivity.md` and `runs/sensitivity-*/`. Confirms the three hooks (compute, recompute, leak) respond to planted faults. **Always re-run this first if you change `metrics.py`.**

### Run the H2 scan (recompute amplification scan)

```sh
cd /workspace/esslivedata-timeseries-lag/.scratch
source ../.venv/bin/activate
python -m harness.run_h2_scan --scan all --dwell 35
```

~25 minutes for 16 runs (5 K, 3 N, 3 decoy, 4 hz, 1 nofreeze). Outputs `runs/h2/summary.json`. The `nofreeze` variant typically stalls the metrics sampler — known limitation, not a crash.

### Run the live-dashboard repro

```sh
cd /workspace/esslivedata-timeseries-lag/.scratch
source ../.venv/bin/activate
python -m repro.live_dashboard --plots 4 --warm-fill 200000 --tick-hz 1.0 --port 5009
```

Then open `http://127.0.0.1:5009/` in a browser. Metrics stream to `.scratch/repro/metrics.jsonl`. To annotate the timeline, from a second terminal:

```sh
M=/workspace/esslivedata-timeseries-lag/.scratch/repro/markers.txt
echo "before-switch-to-plots" >> $M     # do the action you described
# ... wait ~5s between markers ...
echo "stop" >> $M
```

The `marker` events get the same `ts` clock as the `window` events; correlation is straightforward.

Knobs: `--plots N`, `--warm-fill N`, `--tick-hz F`, `--port P`, `--metrics-file PATH`, `--marker-file PATH`.

### Analysing a run

```python
import json
from pathlib import Path

records = [json.loads(l) for l in Path('.scratch/repro/metrics.jsonl').read_text().splitlines() if l.strip()]
markers = [r for r in records if r['event'] == 'marker']
windows = [r for r in records if r['event'] == 'window']
# Each window has compute, pipe_send, recompute, add_root, remove_root,
# doc_change, patch_doc, gc, proc. See metrics.py for the full schema.
```

Useful per-window fields:

- `compute / pipe_send / add_root / remove_root / doc_change`: `count`, `p50_ms`, `p99_ms`, `sum_ms`, `max_ms`.
- `patch_doc`: same plus `bytes`.
- `recompute`: per-thread `{thread_name: {count, total_ms, max_ms}}`.
- `gc`: `gen2_count`, `gen2_ms`.
- `proc`: `rss_mib`, `gc_objects`, `thread_cpu_s` (per-thread CPU seconds in this window).

---

## Pitfalls and gotchas

1. **Headless `pipe.send` numbers (`.scratch/bench_pipe_send.py`) are misleading.** They show ~9 ms at N=100k. The real cost in a live `Document` is **~100 ms**. The headless rig has no `curdoc`, so the Bokeh `DocumentModelManager.recompute` BFS and the CDS-patch dispatch never run. Always measure against the live-dashboard repro for `pipe.send` cost.
2. **The original `.scratch/timeseries_lag_findings.md`** uses those wrong numbers in its extrapolations. Don't propagate them.
3. **`pn.Tabs(dynamic=True)` does NOT mutate the model graph on tab switch.** No `add_root`/`remove_root` calls. The "tab switch tears down models" hypothesis is wrong. Switching is visibility-toggling on already-attached models.
4. **`Plotter.compute` fires for every layer every tick, regardless of session state.** The active-grid gate is on `pipe.send` only (`plot_grid_tabs.py:1059`). This is the "even without a browser" baseline.
5. **`pn.Tabs(dynamic=True)` semantics**: `_BatchedTabs._update_active` (`plot_grid_tabs.py:54-72`) wraps the switch in `pn.io.hold() + doc.models.freeze()`. **Do not** remove this envelope to "see what it saves" — the H2 scan tried and the metrics sampler stalled within 1 s. Replicate the envelope verbatim if you stand up a new test rig.
6. **The harness's "nofreeze-K4" run logs 0 windows.** It is a known stall, not a crash. To quantify the envelope's benefit you would need a metrics sampler that doesn't share the document IOLoop — out of scope so far.
7. **Workflow registration must happen before `ReductionApp` construction**, because `JobOrchestrator._load_configs_from_store` reads the registry at init. `live_dashboard.py:_register_synth_workflow` shows the pattern.
8. **`ConfigStoreManager` is patched to in-memory in `live_dashboard.py`** so the repro boots clean every time. The patch is monkey-patched at the class level — do not rely on a fresh `ConfigStoreManager(store_type='file')` in the same process; it will also become memory-backed.
9. **`ResultKey` JSON round-trip is bypassed in the repro.** We write `data_service[result_key] = value` directly. Production does the same after `Orchestrator.forward` parses the `StreamId`. If you add a test that exercises the JSON round-trip, do it explicitly.
10. **`pre-existing`: `dummy/detector_overview.yaml` grid template has 4-segment `workflow_id` strings** which `WorkflowId.from_string` rejects. The dummy instrument log shows a warning per template. Cosmetic — not on the lag path. Worth a separate bug.
11. **`pre-existing`: "Layer X in READY state but showing placeholder" warning** on first session render. First-render race in `PlotGridTabs._get_composed_plot`. Resolves on next poll. Cosmetic.
12. **Worktree branch `timeseries-lag-investigation` is unrelated to** the more recent (`main`-side) `jobmanager-context-gate` work. Branch off `main` for any new investigation worktree.

---

## Open questions and where to look next

### Highest leverage: confirm browser-side cost

The "couple of seconds" tab-switch freeze the user reports is **server-idle** for most of its duration in our measurements. The cost is in the browser. To quantify:

- **Option A — manual DevTools capture.** Ask the user to open Chrome DevTools → Performance, hit Record, click the heavy tab, stop. Save the trace and read the flame chart for layout/paint/script bottlenecks. This is the cheapest move and should be the next thing tried.
- **Option B — unblock Playwright.** `.scratch/harness/playwright_driver.py` is ready; needs Chromium binary access. Once unblocked, can capture WS `Network.dataReceived` + `PerformanceObserver` longtask entries automatically, and drive synthetic switch sequences.
- **Option C — knob test.** Re-run the live repro with `--warm-fill 1000` instead of 200000 and ask the user if the switch freeze persists. If it vanishes, browser-side render cost is N-proportional and that's the fix axis. **This is the single cheapest experiment** and probably the right first step.

### Second priority: `pipe.send` at ~130 ms/call

The largest single per-call server cost in any of our data. Inner `HvConverter1d.curve()` is ~2 ms; the remaining ~125 ms is in Bokeh CDS replace + protocol patch + IOLoop dispatch. Worth profiling the hot path with `py-spy --on-cpu` for ~10 s while the plot tab is visible and seeing what dominates. If the cost is N-bound (full data on every send), server-side downsampling or `ColumnDataSource.stream()` becomes a clear fix axis.

### Third priority: confirm or refute the HV #6875 leak's contribution

Drive N connect-disconnect cycles via `bokeh.client.pull_session` against the live repro, sample RSS + `gc.get_objects()` + `gc.collect()` time. Look for monotonic growth. If positive: backport or pin the PR before further fix work.

### Lower priority

- **`Plotter.compute` on hidden layers.** Today every layer recomputes every tick whether visible or not. Cost is small per-layer (~10 ms) but scales with K and is a wasted floor. Would also remove the per-tick allocation in `FullHistoryExtractor._to_local_datetime` for hidden layers.
- **`Autoscaler.update_bounds`** does `nanmin`/`nanmax` over full buffer each tick. Could be incremental. Small win on top of the others.
- **`DEFAULT_MAX_MEMORY = 20 MB` per buffer** means a ~625k-point cap for scalar f144 (~7 days at 1 Hz). Buffer growth is bounded, not infinite — not the lag source, just worth knowing.

---

## What NOT to redo

These cost real time and produced clear answers:

- The headless microbenchmarks (`.scratch/bench_*.py`) — keep for historical reference, do not extend.
- The K/N/hz axes of the H2 scan — recompute is invariant to all three within the production range.
- The decoy-bloat experiment — established the mechanism, but production model count is bounded by `dynamic=True` so it does not extrapolate to real dashboards.
- The freeze-removal experiment — known to stall the metrics sampler. Requires re-architecting the sampler off the IOLoop to retry meaningfully.

---

## Conversation context the next session will not have

The user pushed back twice during this investigation:

1. After the initial issue framing: *"I'm not convinced by the '1000 extra ColumnDataSources' explanation: We already use hv.Tabs with dynamic=True AND disable streaming updates into invisible tabs."* — and they were right. That pushback drove the live-document instrumentation that produced this handoff.
2. After H2-style steady-state numbers came in low: *"We do not have enough info. Can you setup the real dashboard with some fake data so we can test without having to wait a day or more?"* — that drove the live-dashboard repro.

The pattern to repeat: when our metrics don't add up to the user's symptom, **measure in the production code path**, not a minimal rig. The headless harness was foundational but its numbers do not generalise to live `Document` cost.

The user is Simon (SimonHeybrock on GitHub), domain owner. He prefers terse responses, real data over speculation, and will tell you when an extrapolation is too aggressive. Issue #940 is his to triage and prioritise; the comment we've left there is the latest state.
