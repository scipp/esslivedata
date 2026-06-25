# Reduce dashboard plot-update latency (#1011)

Worktree: `.claude/worktrees/investigate-1011-lag-diff`.

## Problem

The titlebar freshness pill (`age = now − data_end`) reads substantially higher
than the per-layer detail lag (`lag = created_at − data_end`): e.g. pill `2.0s`
vs lag `1.1s` on a detector Tube View. Both read the *same* `TimeBounds` object,
so `min_end` is identical and the gap is exactly:

```
age − lag = now_poll − created_at
```

i.e. how long the freshly-computed frame sat in the presenter before the
dashboard pushed it to the browser. This is not clock skew and not inter-frame
data starvation — the data is computed and waiting; the dashboard just isn't
flushing it.

## Diagnosis: serial polling stages on top of the backend's 1 Hz

Two threads, three polling stages (all unsynchronized with each other and with
the backend's ~1 Hz publish):

| Stage | Thread | Mechanism | Added latency |
|---|---|---|---|
| Kafka poll | `orchestrator-update` | `consumer.poll(timeout=0.05)` (`kafka/source.py:44`) | 0–50 ms |
| Drain + `compute()` | `orchestrator-update` | loop sleeps **200 ms** when idle (`dashboard_services.py:98,161`) | 0–200 ms (folds into `lag`) |
| `compute()` → `pipe.send()` | **→ Bokeh main** | **1000 ms** periodic poll (`dashboard.py:167`) | 0–1000 ms (**this is the ~0.9 s gap**) |

`compute()` runs on the orchestrator thread the instant a burst is drained
(`created_at` set), then the result sits until the next 1 Hz session poll calls
`SessionLayer.update_pipe()` → `pipe.send()` (`plot_grid_tabs.py:652` →
`session_layer.py:54`). The measured ~0.9 s ≈ the 1000 ms session-poll wait.

## Hard constraint: session-bound updates must run on the session thread

`pipe.send`, Bokeh model mutations, and `pn.io.hold()` **must** run in the
target session's own document context (`pn.state.curdoc` = that session's doc,
on Tornado's IOLoop). Off-IOLoop mutation corrupts the document and misroutes
updates to the wrong tab (Panel #5488; historical design doc
`git show 2b349005a:docs/developer/plans/multi-session-architecture.md`).
`pn.io.hold()` no-ops without a session doc (`session_updater.py:181-183`).

Consequences:

- The orchestrator thread **cannot** touch any session-bound object. A naive
  event-driven push (orchestrator thread → `pipe.send`) reintroduces exactly
  the bug class the polling architecture was built to avoid.
- `pn.state.execute` does **not** help from the orchestrator thread — it
  schedules onto the *current* session context, so it only works when already
  on the session thread.
- All session periodic callbacks share one Tornado loop and run **serially,
  never concurrently** — keeping flushes on the periodic callback preserves
  that guarantee (and is why `PlotDataService`'s lock is "defensive").

The 1 Hz poll also provides **visual synchronization**: `periodic_update()`
wraps the whole pass in `_batched_update()` = `pn.io.hold()` +
`doc.models.freeze()` (`session_updater.py:153,166-184`), collapsing every
layer's `pipe.send` in a pass into **one** WebSocket flush. Naively lowering the
period keeps each pass batched but scatters computes that finished in different
sub-windows across different passes → distracting cross-plot stagger.

## Design: `frame_generation` counter + gated fast poll (pull, not push)

Keep every session-bound mutation on the periodic callback. Change two cheap
things:

1. **Orchestrator thread publishes a `frame_generation` counter** — a plain int
   (the one safe cross-thread primitive; sessions read, orchestrator writes).
   Bump it **once on the work→idle transition** in `_update_loop`
   (`dashboard_services.py:159-161`), i.e. after a burst is fully drained *and*
   all dirty visible layers computed. This is the data-defined frame boundary.

2. **Run the session periodic callback faster (~100 ms), but gate the data-flush
   pass.** Each tick:
   - Tick the freshness pill (stall aging; see Tunables).
   - Run the lightweight lifecycle/version scan + rebuilds (cheap int compares;
     keep every tick so job start/stop/error stays prompt).
   - **Data-flush gate**: run the `update_pipe` flush pass (under
     `hold()`+`freeze()`) over active layers only when the gate is open;
     otherwise skip. Record the generation flushed.

Why this satisfies all three requirements:

- **Thread-safe**: zero new cross-thread session access; `pipe.send` stays in
  the periodic callback. The orchestrator only increments an int.
- **Synced (no stagger)**: the gate opens only *after the whole burst is
  computed*, so all layers that arrived together flush in one `hold()`+`freeze()`
  pass = one WebSocket flush.
- **Low latency**: the wait drops from 0–1000 ms (fixed phase) to 0–~100 ms
  (fast poll picks up the bump). Residual floor is the compute-pass spread,
  which already exists today.

### Flush-gate open conditions

Open the data-flush pass when ANY of:

- `frame_generation > last_flushed_generation` (new visible data), OR
- active grid changed since last tick (tab switch), OR
- modal just closed (`_get_active_grid_id` None→GridId — treat as tab change), OR
- this pass rebuilt/activated a cell that now has a pending update (covers the
  tab-switch synchronous 0→1 gate compute, `plot_orchestrator.py:1002-1011`).

Per-presenter `has_pending_update()` still decides *which* layers actually send,
so an empty/irrelevant bump is a cheap no-op.

## Subtleties (verified against code)

**(a) Visibility gating is built in.** `compute()` runs only for layers with a
held interest token = the active tab (`activate_layer(is_active)` →
`set_active`/`stash_pending`, `plot_orchestrator.py:999`,
`plot_data_service.py:248,316`). Hidden tabs only stash, no compute → no frames.
So `frame_generation` (bumped on orchestrator work→idle) inherently tracks only
visible recomputes. Note `active` is per-layer across all sessions (any session
viewing holds a token); per-session dirty flags localize the actual flush.

**(b) Tab switches.** On switch, the next poll's `activate_layer(…, True)` 0→1
transition runs the stashed (hidden-accumulated) compute **synchronously on the
polling thread**, then the same pass rebuilds (`plot_orchestrator.py:1002-1011`,
`plot_grid_tabs.py:662`). With a ~100 ms poll and the active-grid-changed gate
condition, newly-visible cells render within one tick regardless of
`frame_generation`. Deactivated (old-tab) layers release tokens as today
(`plot_grid_tabs.py:669-673`). Modal open/close is the None↔GridId case of the
same active-grid-changed condition.

## Implementation sketch

- `dashboard_services.py`: add a thread-safe `frame_generation` (int + lock, or
  `itertools.count`/atomic). In `_update_loop`, track `did_work` from
  `orchestrator.update()` (needs `update()` to report whether any compute ran);
  bump generation on the work→idle transition. Expose a getter for sessions.
  - `orchestrator.update()` / `_run_compute` should signal "a compute ran" so
    idle-after-work is distinguishable from idle-after-stash-only.
- `dashboard.py:167`: lower default `period` (e.g. 1000 → ~100 ms). Consider
  keeping 1 Hz default with fast cadence opt-in until pill cost is measured.
- `session_updater.py` / `plot_grid_tabs.py`: thread `last_flushed_generation`
  and `last_active_grid_id` through `_poll_for_plot_updates`; split the cheap
  per-tick work (pill aging, version scan, rebuilds) from the gated data-flush
  (`update_pipe` loop under the existing `_batched_update`).
- `dashboard_services.py:98`: lower orchestrator idle sleep (200 → ~50 ms) so
  burst *detection* granularity isn't the new floor.

## Risks / tunables

- **Pill refresh churn**: at 100 ms, sub-10 s ages (formatted to 0.1 s,
  `cell.py:_format_age_short`) re-render the pill HTML every tick × N cells.
  Decouple pill refresh onto a slower sub-cadence (~250–500 ms) or only
  re-render when the formatted string changes. Measure with many cells.
- **Frame split across the idle gap**: a backend frame whose messages straddle
  the orchestrator idle transition → two bumps → two flushes → slight stagger.
  Rare; a few-ms debounce before bumping on idle coalesces it if observed.
- **Fast-poll scan cost**: the lifecycle/version scan runs 10×/s. It's int
  compares + dict lookups; verify it stays cheap with many layers, else gate it
  too (with a separate structure-generation bumped on lifecycle events).
- `update()` reporting "did work" must not count hidden-tab stash-only passes as
  frames (would cause empty no-op flush ticks — cheap, but avoidable).

## Verification

- Measure `age − lag` before/after via the UI (issue screenshot scenario) and
  with `scripts/drive_dashboard.py` (fake backend); expect the pill to sit near
  `lag` right after each frame instead of climbing to ~2× `lag`.
- Multi-cell active tab: confirm all plots repaint in one frame (no stagger) —
  visually and by checking a single WebSocket flush per burst.
- Tab switch and modal close: newly-visible cells render within ~1 fast tick.
- Stalled stream: pill still climbs (age aging preserved, #967).
- Many cells: profile poll cost (`.scratch/recompute_profiler.py`) to confirm
  the faster cadence doesn't regress recompute load.

## Out of scope

- Backend publish cadence (`batch_length_s = 1.0`) — the inter-frame floor; a
  separate question and a system-wide change.
- Parallelizing `compute()` to shrink the compute-pass stagger floor.
