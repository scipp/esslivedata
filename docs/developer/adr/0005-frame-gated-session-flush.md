# ADR 0005: Frame-gated per-session plot flush

- Status: accepted (amended 2026-07-02 and 2026-07-06, see bottom)
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
Bokeh document context, on that session's IOLoop. Panel cannot even resolve the
right session context from a background thread: `pn.state` reflects the *current*
session, which is absent off the session IOLoop (Panel #5488). Mutating
session-bound objects from the ingestion thread therefore corrupts the document
and misroutes updates to the wrong tab -- the empirical failure that drove the
per-session polling architecture (original root-cause analysis: `git show
2b349005a:docs/developer/plans/multi-session-architecture.md`). This is a hard
constraint, not a preference.

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
carries a `generation` counter keyed by grid (tab). As a burst drains, the
ingestion thread buckets each visible layer's due recompute by grid
(`PlotOrchestrator._enqueue_compute`) rather than computing inline. Once the
burst is drained (after each `MessagePump.update()` in the loop),
`flush_frames` runs the buckets grid by grid and `commit(grid_id)`s each grid
the moment its own layers finish -- so a session showing one tab sees its frame
without waiting on any other tab's compute.

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
  `doc.add_next_tick_callback` carrying the payload onto each captured session
  document). Rejected: it reintroduces the cross-thread session mutation the
  polling architecture exists to avoid. `pn.state.execute` does not help -- it
  targets the *current* session context, useless from the ingestion thread.
  This rejection covers *data-carrying* push; see the 2026-07-06 amendment for
  the data-free wake-up case.
- **Naively lower the fixed poll period.** Rejected: each pass stays batched, but
  computes finishing in different sub-windows flush on different ticks, so
  multi-plot updates stagger.
- **User-selectable poll period.** Rejected: it exposes the latency/stagger
  trade-off to the user instead of resolving it; the low-latency setting still
  staggers.

## Key design choices

- **Generation gate, not push.** Synchronization needs a flush *per data burst*,
  not a slow timer. The burst boundary is data-defined (`commit(grid_id)` on
  drain), so gating a fast pull on it gives both low latency and
  one-flush-per-grid-per-burst batching. A burst only *approximates* a backend
  frame -- see *Scope and limits of synchronization*.
- **Visibility falls out of the existing compute gate.** A build is due only for
  layers holding a viewer interest token (the active tab), so bucketing the due
  builds inherently tracks only visible recomputes. Hidden tabs stash without
  computing and never bucket, so their generation never advances.
- **Generation keyed per grid, not global.** With multiple sessions each showing
  a different tab, a single global counter would wake every session on any tab's
  data: harmless for the data push (dirty-gated, so it shows nothing wrong) but
  it would re-tick each session's freshness pill on other tabs' frames, undoing
  the per-frame pill cadence. Keying the generation by grid means a session is
  woken only by bursts in the tab it is displaying.
- **Commit per grid, not once per burst.** Bucketing the burst's computes by
  grid and committing each grid as its layers finish means a session waits only
  on its own tab's compute, never the union of every visible tab's compute
  across all sessions. Deferring dispatch out of the inline `DataService._notify`
  to the post-drain `flush_frames` also stamps `created_at` closer to display
  time.
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

## Scope and limits of synchronization

The guarantee is *intra-grid within a burst*: one `commit(grid_id)` collapses
that grid's layers computed from one drained batch into a single `hold`+`freeze`
flush on each session showing it. Because `get_messages` drains the whole
consumer queue, a burst coalesces however many Kafka batches accumulated since
the last `update()`. What that burst aligns with on the backend rests on the
following assumptions, which are reasoned, not measured:

- **No backend frame exists.** Compute is partitioned into independent workers
  (detector-view, monitor-histograms, data-reduction, device-metadata), each
  publishing ~1 Hz at its own phase, some slower (e.g. motor positions). They are
  not mutually synchronized and carry no cross-worker frame marker -- the backend
  never declares "these results are one moment." Per-frame sync is therefore an
  approximation made at the dashboard, not a property of the data.
- **Sync only matters within a tab**, since the gate is per grid and a session
  views one tab. Tabs are *typically* fed by a single worker, whose outputs are
  computed per tick and published together. The working assumption is that such a
  worker's co-published outputs arrive close enough to drain into one burst, so
  the cells sharing a moment flush together. We have *no hard proof* that a
  worker's output set always lands in one consumer batch / one `update()`.
- **Cross-worker data is not synchronized** (different phases, sub-Hz sources).
  Accepted: those sources share no backend frame and mostly live on separate tabs
  that the per-grid gate isolates. A custom tab mixing workers gets no
  cross-worker alignment -- but neither did the old fixed period, which only
  aligned sources by the luck of its 1 s bucket phase, never true coherence, and
  at up to 1 s latency.

These expectations are unverified. We also lack a current measurement of how real
~1 Hz traffic (with ~200 ms compute per batch) partitions into bursts -- one
burst per second plus empty ticks, or several. The design degrades gracefully if
an assumption fails: a torn frame staggers by one ~100 ms tick and self-corrects.
Partial sync is the accepted fallback; real-world use will confirm or refute the
above, and this section should be revisited if intra-tab stagger proves visible.

## Consequences

- The *polling* component of display latency drops from up to one 1000 ms period
  to roughly one 100 ms tick. For a single visible layer that is the entire
  wait. Committing per grid also removes *cross-tab* interference: a session no
  longer waits on other sessions' tabs' computes before its own frame commits.
  The residual wait is the *within-tab* serial compute -- a grid's layers still
  run sequentially on the ingestion thread before that grid commits -- plus the
  100 ms tick. Parallelizing that per-grid compute is a separate concern, out of scope here.
- The per-layer pill lag (`created_at − min_end`) and the headline age
  (`now_flush − min_end`) differ by exactly the display latency, so they
  coincide only for the last-computed layer of a grid (and any single-plot
  tab); the first-computed layer's age sits a full grid-compute above its lag.
  The end-to-end age is identical for every layer in a grid, since one flush
  shows them all.
- A grid's layers from one burst flush together (one `hold`+`freeze` flush per
  grid generation). This is intra-grid only; cross-grid, cross-burst, and
  cross-worker alignment are not guaranteed -- see *Scope and limits of
  synchronization*.
- `PlotOrchestrator` gains a `frame_generation(grid_id)` accessor and a
  `flush_frames` that runs the per-grid compute buckets and commits each grid;
  `DashboardServices` owns the `FrameClock` and calls `flush_frames` after each
  drain; the ingestion idle sleep dropped to 50 ms so burst detection is not the
  new floor.
- The pill's stall cadence is coupled to the backend publish cadence. This is
  accepted: at slower cadences ticking the age every 2 s is reasonable.
- A logical frame can split across bursts when its arrival straddles an
  `update()` cycle boundary -- made marginally likelier by the ~200 ms serial
  compute per batch (observed in practice) holding the loop. The result is a
  one-tick (~100 ms) stagger: acceptable and self-correcting. How often this
  happens in real traffic is unmeasured.

## Amendment (2026-07-02): pull-on-frame input side

The input side of the mechanism changed from push to pull (issue #1044). As a
burst drains, `DataService` no longer extracts data and delivers it to each
layer's subscriber; it notifies keys-only, and the orchestrator merely marks
the affected layers dirty (per grid). `flush_frames` then pulls each dirty
*viewed* layer's input through its extractors from the current buffer state
(`DataService.snapshot`), rebuilds, and commits the grid as before. Dirty
flags of unviewed layers are dropped; the 0→1 viewer transition rebuilds from
current buffer state unconditionally.

Consequences for the guarantees above:

- The frame a grid's layers repaint from is now *the buffer state at flush
  time* rather than *the drained burst's payload*. Layers of one grid are
  pulled back-to-back on the ingestion thread with no writes in between, so
  intra-grid coherence is unchanged. The `DataService` lock is
  transaction-scoped, so a pull from any thread (including the 0→1
  activation rebuild on the polling thread, and flushes racing a UI-thread
  cleanup eviction) observes either none or all of a transaction's writes.
- A logical frame that splits across bursts now heals at the next flush by
  construction (the pull sees whatever has arrived), rather than by a
  one-tick stagger of stashed payloads.
- Deferring or coalescing flushes is lossless: a pull can never observe older
  data than a delivery would have carried. This is what makes a future
  min-frame-interval throttle in `flush_frames` a policy choice rather than a
  correctness question.

## Amendment (2026-07-06): scope of the `add_next_tick_callback` rejection

The *Alternatives considered* rejection above reads as a blanket ban on
`doc.add_next_tick_callback` from the ingestion thread. It is not. What that
rejection encodes is **data-carrying push and direct cross-thread mutation of
session-bound objects**, plus `pn.state.execute`, which genuinely cannot
resolve a session context off the session IOLoop (Panel #5488).

A **data-free wake-up** is permitted: the ingestion thread may, after
`flush_frames` commits a grid, schedule a no-argument tick on a registered
session document. The invariant this ADR exists to protect is untouched --
every session-bound mutation still happens inside the tick, on that session's
IOLoop, in document context. Only the *trigger* moves from clock to event.

Verified on panel 1.9.3 / bokeh 3.9.1: a background-thread
`doc.add_next_tick_callback(tick)` runs the tick in the correct session
context (`pn.state.curdoc` resolves; `pn.io.hold()` + `doc.models.freeze()`
behave normally) across concurrent sessions, and scheduling into a destroyed
document raises catchably, so stale registrations unregister lazily.

Because the tick body is the existing generation-gated pass -- idempotent, and
a no-op when nothing advanced -- a lost or duplicated wake costs nothing. That
makes the periodic callback a safety net rather than the detection clock, so
it can drop to ~1--2 s for the things that genuinely need a clock (heartbeat,
notifications, freshness-pill stall aging), retiring the 100 ms tick's
per-tick `hold`+`freeze` recompute and layer scan. Sequencing and the
`WakeupHub` design live in #1046; the delivery model this follows from is
ADR 0007.
