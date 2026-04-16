# RateAwareMessageBatcher — Next Steps

## What exists (commit bebad3a3)

Module: `src/ess/livedata/core/rate_aware_batcher.py`
Tests: `tests/core/rate_aware_batcher_test.py`

### Core algorithm

**Slot-based completion.** Each message is assigned a pulse slot index:

```
slot = round((msg.timestamp - batch_start_ns) / pulse_period_ns)
```

A batch is complete for a gated stream when `max_slot >= expected_count - 1`,
i.e., when a message has been seen whose timestamp places it in the last
expected pulse slot. This is *not* message counting — missing pulses (empty
ev44 never published) don't block completion, and split messages (two ev44s
with the same timestamp) don't produce false positives.

**Overflow.** Messages with `slot >= expected_count` are held for the next
batch.

**Rate estimation.** Per-stream EMA of message count per batch. Streams must
pass `MIN_BATCHES_FOR_GATE` (3) observations before participating in the
slot-based gate. Unconverged streams include all messages; timeout drives
batch closure.

**Timeout fallback.** Wall-clock timeout (default 1.5 * batch_length) closes
batches when slot completion isn't reached.

### Tests (12 passing)

- Empty input, initial batch
- Single-stream: last-slot completion, incomplete batch returns None
- Missing pulse: slot 7 absent, slot 14 still triggers completion
- Split messages: early-slot duplicate doesn't false-trigger; last-slot
  duplicate still completes
- Timeout closes incomplete batches
- Multi-stream: waits for all gated converged streams
- Overflow: excess messages appear in next batch

### Test helper

`make_converged_batcher(rate_hz, batch_length_s, streams, timeout_s, clock)`
returns `(batcher, next_batch_start)`. Feeds `MIN_BATCHES_FOR_GATE` batches
via timeout to converge the rate estimates. The `next_batch_start` float lets
tests generate messages aligned to the active batch window.


## What's missing — roughly in priority order

### A. Phase offset between streams and batch grid

**Problem.** The slot index uses `batch_start` as reference, but streams have
a phase offset relative to the batch grid. A 14 Hz stream might produce its
first pulse 0.02s after batch_start. Currently, `_slot_index` uses
`batch_start` directly, which works when messages start near batch_start but
breaks when the offset is significant (> 0.5 / rate).

**Example.** Batch [0, 1s), 14 Hz stream with phase offset 0.04s. Messages
arrive at 0.04, 0.111, ..., 0.968. The last message's slot is
`round(0.968 * 14) = round(13.55) = 14`, which equals `expected_count` and
would be classified as overflow — even though it's the legitimate 14th pulse
of this batch.

**Proposed fix.** Track a per-stream phase offset, estimated from the first
few messages. Use `batch_start + offset` as the reference for slot
computation. Alternatively, use the timestamp of the first message from that
stream in the batch as slot 0's reference. This needs thought: if the first
message is late, it shifts the whole slot grid.

**Tests to write:**
- Stream with constant phase offset (e.g., 0.04s at 14 Hz): all 14 messages
  should land in one batch, not 13 + 1 overflow.
- Two streams with different phase offsets: both should complete independently.
- Phase offset near half a pulse period (worst case for rounding).

### B. Jitter resilience

**Problem.** Real message timestamps have jitter from network delays, Kafka
batching, and imprecise source clocks. The slot assignment is `round(dt /
period)`, which can flip slot assignment when jitter pushes a timestamp past
the midpoint between two slots.

**Tests to write:**
- 14 Hz stream with +/- 10ms Gaussian jitter on each timestamp.
- 14 Hz stream with +/- 30ms uniform jitter (worst case).
- Verify: batch always gets 14 messages (or 13/15 in rare jitter extremes,
  but never 10 or 18).
- Verify: over 100 batches, the *average* message count per batch is 14.0.

**Open question.** Is `round()` sufficient, or do we need hysteresis / a
dead zone at slot boundaries? If jitter consistently pushes messages past the
boundary, we might systematically lose or gain one message per batch.

### C. 1 Hz stream with 1s batch (single-slot edge case)

**Problem.** `expected_count = round(1.0 * 1.0) = 1`. Last slot = slot 0.
Any message from this stream fills the last slot immediately.

**Tests to write:**
- 1 Hz monitor with 1s batch: exactly 1 message per batch.
- Message arrives slightly before batch_start (negative slot): should this
  be included or go to previous batch?
- Two messages from the same pulse (one early, one late): both go in one
  batch?
- Verify overflow works: message with slot >= 1 goes to next batch.

### D. Time gaps (stream pauses and resumes)

**Problem.** If a stream pauses (e.g., source restart, Kafka partition
rebalance) and resumes minutes later, the active batch window is stale.
Messages at t=1000 arrive but the batch window is at t=5. Currently, the
first message gets slot ~14000, which is >= expected_count, so everything
overflows and the batch never closes (except by timeout).

**Proposed fix.** Detect "stale batch" when all incoming messages have
slots >> expected_count. Options:
- Auto-advance the batch window (skip empty batches) when a gap is detected.
- Reset the batch to start from the new messages' timestamps.
- Keep the SimpleMessageBatcher's behavior of emitting empty batches one
  at a time to maintain the timeline.

**Tests to write:**
- Stream pauses for 5 batch lengths, then resumes with normal-rate messages.
- Stream pauses for 100 batch lengths (large gap).
- Verify batch continuity: batch start/end times should cover the gap
  (possibly with empty batches).

### E. Drift correction

**Problem.** If the source rate drifts slightly (e.g., 13.98 Hz instead of
14 Hz), the per-stream phase offset slowly drifts. After enough batches,
the last slot may shift outside the batch window, causing systematic
overflow of one message per batch and under-filling.

**Proposed fix.** The EMA rate estimate should track the drift, so
`expected_count` adjusts. But the slot index computation uses `batch_start`
as reference, which doesn't account for accumulated offset. Options:
- Periodically re-anchor the slot grid to the most recent messages.
- Use a running offset that adjusts each batch based on where the first
  message actually arrived vs. where it was expected.
- Rely on the catch-up/margin mechanism from the earlier plan: if a stream
  falls behind by more than a tolerance, include its "overflow" messages
  as catch-up instead.

**Tests to write:**
- Source at 13.98 Hz over 100 batches: verify no systematic message loss.
- Source rate changes abruptly (14 Hz -> 7 Hz): verify convergence to new
  rate after ~MIN_BATCHES_FOR_GATE batches.

### F. Stream lifecycle (appearance, disappearance, eviction)

**Problem.** Streams can appear mid-run (new source comes online) or
disappear (source turned off, detector disconnected). A disappeared stream
that's still in the gated set blocks batch completion forever.

**Proposed fix.** Evict streams absent for N consecutive batches (e.g., 5).
Evicted streams are removed from `_streams`, so they no longer participate
in the gate.

**Tests to write:**
- New stream appears after 10 batches: converges and joins the gate.
- Stream disappears: batches close via timeout for N batches, then stream
  is evicted.
- Evicted stream reappears: re-enters convergence phase.

### G. `set_batch_length()` for AdaptiveMessageBatcher integration

**Problem.** The `AdaptiveMessageBatcher` calls `set_batch_length()` on the
inner batcher when escalating/de-escalating. `RateAwareMessageBatcher`
doesn't implement this yet.

**Proposed fix.** Update `_batch_length`. The next `_close_batch` uses the
new length for the next batch boundary. `expected_count` automatically
rescales via `round(rate * new_batch_length_s)`.

**Tests to write:**
- Batch length changes from 1s to 2s: next batch gets ~28 messages at 14 Hz.
- Batch length changes from 2s to 1s: next batch gets ~14 messages.
- Overflow from old batch length is correctly handled in new batch.

### H. Rate estimation accuracy with missing/split messages

**Problem.** `update_rate` uses `tracker.count / batch_length_s` as the
observed rate. If pulses are missing (count=13 instead of 14) or split
(count=15), the rate estimate gets noisy observations. With `alpha=0.05`
this is fine for rare occurrences, but systematic drops (e.g., a source
that consistently skips every 100th pulse) could bias the estimate.

**Open question.** Should rate estimation use the slot-based count
(number of distinct slots filled) rather than raw message count? This would
be robust to splits (two messages in one slot count as one) but not to
missing pulses (a skipped slot still counts as missing).

### I. Non-gated stream handling

Currently non-gated streams go to whatever batch is active. This is the
right behavior but has no tests.

**Tests to write:**
- Log messages included in current batch regardless of timestamp.
- Non-gated messages don't affect completion gate.


## Suggested implementation order

The dependencies between these items are loose. Suggested order based on
criticality for real-world use:

- **A (phase offset)** — without this, streams that don't happen to align
  with the batch grid will systematically overflow one message per batch.
  This is the most important correctness issue.
- **C (1 Hz edge case)** — small, can be done alongside A. Important because
  we have 1 Hz monitors in production.
- **D (time gaps)** — without this, any stream interruption requires timeout
  to recover and leaves the batch window stuck.
- **F (stream lifecycle)** — without this, a disappeared stream blocks all
  batches indefinitely after timeout.
- **G (set_batch_length)** — needed to replace SimpleMessageBatcher as the
  inner batcher for AdaptiveMessageBatcher.
- **B (jitter)** — may be solved already by the slot rounding; tests will
  confirm.
- **E (drift)** — lower priority since the EMA handles gradual drift; only
  matters for long runs with non-trivial drift rates.
- **H (rate estimation accuracy)** — can be deferred; EMA smoothing is
  sufficient for typical noise levels.
- **I (non-gated streams)** — trivial test-only item.
