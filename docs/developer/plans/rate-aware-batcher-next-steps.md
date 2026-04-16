# RateAwareMessageBatcher — Status and Next Steps

## What exists (commit 8bc027f2)

Module: `src/ess/livedata/core/rate_aware_batcher.py`
Tests: `tests/core/rate_aware_batcher_test.py` (30 tests)

### Core algorithm

**Slot-based completion.** Each message is assigned a pulse slot index:

```
slot = round((msg.timestamp - batch_start - phase_offset) / pulse_period)
```

A batch is complete for a gated stream when `max_slot >= expected_count - 1`,
i.e., when a message has been seen whose timestamp places it in the last
expected pulse slot. This is *not* message counting — missing pulses (empty
ev44 never published) don't block completion, and split messages (two ev44s
with the same timestamp) don't produce false positives.

**Phase offset.** Per-stream phase offset estimated as
`(first_msg.timestamp - batch_start) % period` after each batch close.
Subtracted in slot computation so streams misaligned with the batch grid
don't systematically overflow. Works because `(t - B) % P` is invariant
to which message is chosen (they all share the same phase).

**Overflow.** Messages with `slot >= expected_count` are held for the next
batch.

**Rate estimation.** Per-stream EMA of message count per batch. Streams must
pass `MIN_BATCHES_FOR_GATE` (3) observations before participating in the
slot-based gate. Unconverged streams include all messages; timeout drives
batch closure.

**Gap recovery.** When all gated messages land in overflow (stream paused and
resumed), the batch window advances to the data in one step by jumping
forward in whole-batch-length increments. No timeout cycles needed.

**Stream eviction.** Streams absent for `ABSENT_BATCHES_FOR_EVICTION` (5)
consecutive batches are removed from the gate, preventing a disappeared
source from blocking batch completion indefinitely. Evicted streams that
reappear re-enter the convergence phase.

**`set_batch_length()`** Deferred via `_pending_batch_length`, applied in
`_close_batch` so the current batch finishes with its original length.
Required for `AdaptiveMessageBatcher` integration.

**Timeout fallback.** Wall-clock timeout (default 1.5 * batch_length) closes
batches when slot completion isn't reached.

### Test coverage (30 tests)

- Empty input, initial batch
- Single-stream: last-slot completion, incomplete batch returns None
- Missing pulse: slot 7 absent, slot 14 still triggers completion
- Split messages: early-slot duplicate doesn't false-trigger; last-slot
  duplicate still completes
- Timeout closes incomplete batches
- Multi-stream: waits for all gated converged streams
- Overflow: excess messages appear in next batch
- Phase offset: 56% of period, two streams with different offsets,
  near-half-period worst case
- Jitter: ±10ms at 14 Hz single batch, average over 50 batches
- 1 Hz edge case: single-slot completion, overflow at slot 1
- Time gaps: 5-batch gap recovers, batch start ≤ first resumed message
- Stream lifecycle: new stream joins gate, disappeared stream evicted,
  evicted stream reappears
- Drift: 13.98 Hz over 100 batches (<1% loss), abrupt 14→7 Hz converges
- `set_batch_length`: increase and decrease, deferred application
- Non-gated streams: included in batch, don't affect gate


## Tested envelope

### Confirmed working (43 tests)

- **Phase offsets:** 0–56% of period (tested at 0%, 49%, 50%, 51%, 56%)
- **Jitter:** ±80% of period with no message loss (avg always 14.0).
  Even ±150% jitter degrades gracefully (no crash or infinite loop).
- **Out-of-order messages:** messages before batch_start get negative
  slots, are included in the batch, and don't affect completion logic.
- **Sub-Hz streams:** expected_count rounds to 0 → no slot gating,
  don't block other streams. Passes through to timeout.
- **14 Hz with 0.1s batch:** expected_count=1, single-slot behavior.
  Works but second message in the period always overflows.
- **Non-integer rate×batch (14.5 Hz):** <2% message loss over 100 batches.
  The extra message alternately overflows and gets absorbed.
- **Burst delivery:** all 14 messages in one `batch()` call works fine
  (algorithm uses message timestamps, not arrival order).
- **Realistic EMA alpha (0.05):** after 14→7 Hz rate change, all 100
  batches close (mix of timeout and slot-gate). No message loss.
- **Overflow accumulation:** max overflow = 0–2 messages over 200 batches.
  No unbounded growth.
- **Drift:** 13.98 Hz over 100 batches (<1% loss), abrupt 14→7 Hz works.
- **Gaps:** 5–100 batch-length pauses, single-call recovery.
- **Batch length changes:** 0.5s to 2.0s via `set_batch_length()`.

### Known limitation: jitter-induced timeout fallback

With jitter, the phase offset estimate (computed from the first message
of each batch) drifts. When the estimate is off by more than ~0.5 pulse
periods, the last slot shifts to overflow and the batch must close via
timeout instead of slot gate.

Measured at 14 Hz over 200 batches:

| Jitter (% of period) | Slot-gate rate | Timeout rate | Avg count |
|-----------------------|----------------|--------------|-----------|
| 0%                    | 100%           | 0%           | 14.0      |
| 10–30%                | ~45%           | ~55%         | 14.0      |
| 40–80%                | ~50%           | ~50%         | 14.0      |

**No messages are ever lost** — the timeout fallback ensures every batch
closes and all messages are delivered. The only effect is added latency
(up to `timeout_s`) on the affected batches.

**Why this is acceptable for ESS:** pulse timestamps come from the timing
system and are nanosecond-precise. The "jitter" tested here (random
perturbations to timestamps) does not occur in production. Kafka batching
affects arrival time but not message timestamps, so the slot computation
remains accurate. The timeout fallback exists as a safety net, not as the
primary completion mechanism.

**Possible future improvements:**
- EMA smoothing of the phase offset (decouple from rate alpha)
- Circular-statistics-based phase estimation (median of residuals, with
  wrapping handled via the interquartile range). Attempted; works for zero
  offset with jitter but breaks for genuine large offsets (>50% period)
  due to the circular ambiguity. Needs more thought.


## Remaining work

### Integration with AdaptiveMessageBatcher

`set_batch_length()` is implemented. Next: wire `RateAwareMessageBatcher`
as the inner batcher for `AdaptiveMessageBatcher` and verify the
escalation/de-escalation cycle works correctly.

### Rate estimation accuracy (item H from original plan)

`update_rate` uses raw message count, not distinct-slot count. This means
split messages inflate the rate estimate and missing messages deflate it.
With `alpha=0.05` this is fine for occasional anomalies but could bias the
estimate under systematic patterns. Deferred — EMA smoothing is sufficient
for typical noise levels.
