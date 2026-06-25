# ADR 0005: Frame-gated per-session plot flush

- Status: accepted
- Deciders: Simon
- Date: 2026-06-25

## Context

Reduced data reaches the browser through two dashboard threads. A shared
ingestion thread (`DashboardServices._update_loop`) drains a Kafka batch,
forwards it through `DataService`, and runs `Plotter.compute` synchronously for
every visible layer -- the compute records `created_at` and marks the layer's
presenter dirty. A *separate* per-session callback
(`pn.state.add_periodic_callback` → `SessionUpdater.periodic_update` →
`PlotGridTabs._poll_for_plot_updates`) then pushes the dirty presenter's data to
that session's HoloViews `Pipe` via `pipe.send`.

The push could only happen on the session callback because session-bound objects
(`hv.streams.Pipe`, `DynamicMap`, `pn.io.hold`) **must** be mutated in their own
Bokeh document context, on that session's IOLoop. Mutating them from the
ingestion thread corrupts the document and misroutes updates to the wrong tab
(Panel #5488). This is a hard constraint, not a preference.

The session callback ran at a fixed 1000 ms period, unrelated to data arrival.
Consequences (issue #1011):

- A freshly computed frame sat undisplayed for up to one poll period. The
  freshness pill (data age, `now − data_end`) and the per-layer pipeline lag
  (`created_at − data_end`) read the *same* `TimeBounds`, so the observed gap
  was exactly `now_poll − created_at` -- the undisplayed wait. A ~1.1 s pipeline
  lag showed as a ~2.0 s pill.
- The fixed period doubled as a frame synchronizer: `periodic_update` wraps the
  whole pass in `pn.io.hold()` + `doc.models.freeze()`, collapsing every layer's
  `pipe.send` into one WebSocket flush. Naively shortening the period would
  scatter a burst's layers (whose computes finish at staggered times) across
  ticks, producing distracting cross-plot stagger.

The design problem: cut the latency without losing per-frame synchronization and
without pushing from the ingestion thread.

## Decision

Decouple the flush *trigger* from the clock while keeping the flush *itself* on
the session thread, coordinated by a frame counter.

A `FrameClock` (one per `DashboardServices`, shared with `PlotOrchestrator`)
carries a `generation` counter keyed by grid (tab). The ingestion thread calls
`mark(grid_id)` whenever a visible layer is recomputed
(`PlotOrchestrator._run_compute`) and `commit()` once a drained burst is done
(after each `Orchestrator.update()` in the loop); `commit()` advances the
generation of every grid marked since the last commit.

The per-session poll period drops to 100 ms, but `_poll_for_plot_updates` gates
the data flush (`update_pipe` → `pipe.send`, plus the per-layer time/lag row) on
the *active grid's* generation having advanced since this session last flushed,
or the active tab having changed. The cheap per-tick work -- lifecycle/version
scan, layer activation, freshness-pill aging -- still runs every tick. All
session-bound mutation stays inside the periodic callback, so the threading
constraint holds.

The freshness pill refreshes in step with the data flush (reading the true lag
of the frame just shown) and otherwise on a slow stall cadence
(`_FRESHNESS_STALL_INTERVAL_S = 2 s`) so a stalled stream still visibly ages.

## Alternatives considered

- **Push from the ingestion thread** (event-driven `pipe.send`, or
  `doc.add_next_tick_callback` onto each captured session document). Rejected: it
  reintroduces the cross-thread session mutation the polling architecture exists
  to avoid. `pn.state.execute` does not help -- it targets the *current* session
  context, useless from the ingestion thread.
- **Naively lower the fixed poll period.** Rejected: each pass stays batched, but
  computes finishing in different sub-windows flush on different ticks, so
  multi-plot updates stagger.
- **User-selectable poll period.** Rejected: it exposes the latency/stagger
  trade-off to the user instead of resolving it; the low-latency setting still
  staggers.

## Key design choices

- **Generation gate, not push.** Synchronization needs a flush *per data burst*,
  not a slow timer. The burst boundary is data-defined (`commit()` on drain), so
  gating a fast pull on it gives both low latency and one-frame-per-burst
  batching.
- **Visibility falls out of the existing compute gate.** `compute` runs only for
  layers holding a viewer interest token (the active tab), so `mark()` -- placed
  in `_run_compute` -- inherently tracks only visible recomputes. Hidden tabs
  stash without computing and never advance the generation.
- **Generation keyed per grid, not global.** With multiple sessions each showing
  a different tab, a single global counter would wake every session on any tab's
  data: harmless for the data push (dirty-gated, so it shows nothing wrong) but
  it would re-tick each session's freshness pill on other tabs' frames, undoing
  the per-frame pill cadence. Keying the generation by grid means a session is
  woken only by bursts in the tab it is displaying.
- **`pipe.send` stays dirty-gated** by `has_pending_update`, so it fires at most
  at the data rate regardless of poll frequency. The faster poll therefore adds
  only cheap no-op scans, not extra Bokeh model work.
- **Flush after `activate_layer`**, so a tab-switch 0→1 synchronous build is
  pushed on the same tick.
- **Stall cadence above backend cadence.** With the stall interval equal to the
  ~1 s publish cadence, the stall tick beat against the flush and double-updated
  the age; keeping it at 2 s lets each flush reset the timer first, so a healthy
  stream updates the pill once per frame. Slower-than-2 s streams age the pill,
  which is informative rather than distracting.

## Consequences

- Display latency drops from up to one 1000 ms poll period to roughly one 100 ms
  tick; the pill reads near the pipeline lag right after each frame.
- Multi-plot updates remain synchronized (one `hold`+`freeze` flush per burst).
- `PlotOrchestrator` gains a `frame_generation(grid_id)` accessor;
  `DashboardServices` owns the `FrameClock` and commits it; the ingestion idle
  sleep dropped to 50 ms so burst detection is not the new floor.
- The pill's stall cadence is coupled to the backend publish cadence. This is
  accepted: at slower cadences ticking the age every 2 s is reasonable.
- A rare burst split across the ingestion idle gap yields two generations, hence
  a slight stagger; acceptable and self-correcting.
