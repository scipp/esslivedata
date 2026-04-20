# Rate-Aware Batcher: Clarity & Compactness Assessment

Living document. Captures structural smells in
`src/ess/livedata/core/rate_aware_batcher.py` and tracks cleanup
progress.

## Assessment

### 1. `RateAwareMessageBatcher` mixes 3–4 distinct concerns

State splits cleanly along these lines:

- **Stream registry**: `_estimators`, `_grids`, `_absent_batches`
  (+ eviction in `_close_batch`, rate→grid in `_update_grid`).
- **Logical clock**: `_high_water_mark` (+ `_clamped_hwm`).
- **Active batch**: `_active_batch`, `_batch_trackers` (+ routing).
- **Overflow buffer**: `_overflow`, `_overflow_needs_rerouting`.

Extracting these would let `batch()` read as a short script. Today it
is ~50 lines wrestling with all four.

### 2. Gap-recovery block (lines 329–346) is the clearest design-driven edge case

It exists because non-gated messages go directly into
`_active_batch.messages` (line 399) while gated messages live in
`_batch_trackers` — two parallel data structures that must be kept in
sync. When overflow forces a window jump, the code has to detach the
batch's message list, clear trackers, advance the window, re-route.
The comments call out each hazard because the invariant is fragile.

If the active window held only bounds + per-stream buckets (gated *and*
non-gated), and `MessageBatch.messages` was flattened only at
`_close_batch`, gap recovery would be: clear buckets, advance window,
re-route. No detach/stash/clear choreography.

### 3. `_BatchStreamTracker` earns its keep only marginally

It is `list + max_slot`. `max_slot = -1` is a sentinel set on the
`grid is None` path (line 408); the `tracker.max_slot < 0` branch in
`_is_batch_complete` (line 436) is unreachable in practice because
`_is_batch_complete` iterates `self._grids`, and a stream only gets a
grid after `_close_batch` (which clears trackers). Drop the branch or
re-verify with a test.

### 4. `_overflow_needs_rerouting` encodes "has the active window moved since overflow was populated?"

Correct but imperative. Alternatives: stamp each overflow entry with
the window start it was rejected from and drain lazily, or unify the
re-route into `_close_batch`'s new-window setup so the flag disappears.

### 5. Two separate "hostile timestamp" guards

`_MAX_HWM_PAST_WINDOW_BATCHES` (clamp HWM) and
`_MAX_ORIGIN_OFFSET_BATCHES` (reject grid origin) both defend against
malformed-future timestamps, each with extensive justifying comments.
A single input-stage filter — "reject timestamps implausibly far from
HWM/active window" — would subsume both and remove the need to justify
the guard at two call sites.

### 6. `_update_grid` mixes four cases in one function

Rate-not-ready, sub-rate demotion, existing-grid refresh (origin
preserved), new-grid creation. The `existing is not None` branch
mutates control flow in ways that make the function's shape hard to
hold in your head. Proposed split:

- `_resolve_grid_params(int_rate)` → `(period_ns, slots)` or `None`.
- `_resolve_origin(sid, tracker, batch_start, existing)` → origin or
  `None`.
- Assemble, then store or drop.

### 7. `_route_message`'s two-path routing leaks the dual-data-structure problem

Non-gated → batch list; gated → tracker + maybe batch list + maybe
overflow. If buckets are unified, this function shrinks to:
observe → decide slot → drop in bucket or overflow.

## Proposed Cleanup Order

Orthogonality matters: items that do not conflict with the
`_ActiveWindow` refactor can go first without rework.

### Phase A — Orthogonal small items (touch-once)

These live in code paths the big refactor does not rewrite:

- **6. Split `_update_grid`.** Stream-registry concern, unaffected by
  active-window shape.
- **5. Unify the two hostile-timestamp guards** into one input-stage
  filter. Affects `_clamped_hwm` and the origin check, both outside
  the active-window refactor's scope.

### Phase B — `_ActiveWindow` refactor (main payoff)

Introduce an `_ActiveWindow` type owning `(start, end,
per_stream_buckets)` where buckets cover gated and non-gated streams.
Build `MessageBatch` only at close. This subsumes:

- **2.** Gap-recovery choreography → clear buckets, advance, re-route.
- **3.** `_BatchStreamTracker`'s `max_slot = -1` sentinel (tracker
  becomes the bucket, or is absorbed).
- **4.** `_overflow_needs_rerouting` flag (re-routing becomes
  unconditional at close).
- **7.** `_route_message` two-path routing.

### Phase C — Structural extraction (optional, quality of life)

- **1.** Extract `StreamRegistry` (estimators + grids + absence +
  eviction) as a named collaborator once Phase B has thinned
  `RateAwareMessageBatcher`. Does not conflict with Phase B but
  benefits from starting after it.

### Items to *not* address before Phase B

- **3.** Dead `max_slot < 0` branch: tracker is about to change shape;
  fixing now is rework.
- **4.** `_overflow_needs_rerouting` flag: same — disappears in B.

## Status

- [x] Phase A.6 — split `_update_grid` (commit `c03c5eb7`)
- [~] Phase A.5 — **skipped**.  On closer inspection the HWM clamp (3
  batches, tight) and the grid-origin guard (1000 batches, loose) serve
  different purposes with different thresholds; a single input-stage
  filter would either compromise one protection or silently drop
  messages that currently reach downstream consumers.  Reassessed
  mid-stream and skipped rather than forcing a shoehorn.
- [x] Phase B — `_ActiveWindow` refactor (commit `3ec0c7e2`).
  Subsumed the gap-recovery choreography, removed
  `_overflow_needs_rerouting`, dropped the dead `max_slot < 0` branch,
  unified routing into a single bucket-lookup entry point.
- [~] Phase C — **skipped**.  After Phase B the grid cluster
  (`_update_grid`, `_choose_origin`, `_origin_too_far`,
  `_pick_grid_origin`, `_rebuild_grids`) sits as a self-contained
  ~100-line section at the bottom of the file.  Extraction would still
  need to thread `batch_start`, `bucket`, `sid` through every registry
  call, tests would need to reach through a passthrough (they currently
  inject pathological grids via `batcher._grids[X] = ...`), and the
  net line reduction is roughly zero.  Independent senior-engineer
  review concurred.  Replaced by a smaller in-class
  `_refresh_stream_registry` helper that consolidates the
  grid-update/evict/apply-pending-length block inside `_close_batch`.

## Outcome

Top-level flow (`batch()`) is now a short linear script; gap recovery
is three steps (flatten, advance, re-route); overflow drains directly
into the next window at close time; routing has one path instead of
two.  The file grew slightly (611 → 633 lines) due to new dataclass
docstrings, but the conceptual surface area shrank.
