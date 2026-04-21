# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unit tests for ``_GatedStream.rebuild_grid``.

Exercises the grid lifecycle directly against the class -- much tighter
than driving the same transitions through the batcher.  End-to-end flow
is covered by ``rate_aware_batcher_test.py``.
"""

from __future__ import annotations

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.core.rate_aware_batcher import (
    _MAX_ORIGIN_OFFSET_BATCHES,
    MIN_DIFFS_FOR_GATE,
    _GatedStream,
)
from ess.livedata.core.timestamp import Duration, Timestamp

STREAM = StreamId(kind=StreamKind.DETECTOR_EVENTS, name="det")
ONE_SECOND = Duration.from_seconds(1.0)


def ts(seconds: float) -> Timestamp:
    return Timestamp.from_ns(int(seconds * 1e9))


def msg(t: float) -> Message[str]:
    return Message(timestamp=ts(t), stream=STREAM, value="")


def converged(
    rate_hz: float, start: float = 0.0, n: int = MIN_DIFFS_FOR_GATE + 1
) -> _GatedStream:
    """Return a _GatedStream whose estimator has converged to ``rate_hz``."""
    stream = _GatedStream()
    period = 1.0 / rate_hz
    for i in range(n):
        stream.observe(msg(start + i * period))
    assert stream.estimator.integer_rate_hz == round(rate_hz)
    return stream


class TestRebuildGrid:
    def test_no_grid_before_convergence(self):
        stream = _GatedStream()
        stream.observe(msg(0.0))  # one sample -> no diffs
        stream.rebuild_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid is None

    def test_integer_rate_builds_grid(self):
        stream = converged(14.0)
        stream.rebuild_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        grid = stream.grid
        assert grid is not None
        assert grid.period_ns == round(1e9 / 14)
        assert grid.slots_per_batch == 14

    def test_non_integer_rate_leaves_grid_absent(self):
        """Non-integer rate (0.85 Hz rounds to 1 Hz but fails snap tolerance)."""
        stream = _GatedStream()
        period = 1.0 / 0.85
        for i in range(MIN_DIFFS_FOR_GATE + 1):
            stream.observe(msg(i * period))
        assert stream.estimator.integer_rate_hz is None
        stream.rebuild_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid is None

    def test_sub_rate_drops_existing_grid(self):
        """1 Hz stream gridded at 1 s: shrink batch to 0.6 s → grid dropped."""
        stream = converged(1.0)
        stream.rebuild_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid is not None
        stream.rebuild_grid(
            batch_start=ts(1.0), batch_length=Duration.from_seconds(0.6)
        )
        assert stream.grid is None

    def test_sub_rate_never_builds_grid(self):
        """First rebuild at sub-rate (1 Hz * 0.6 s < 1) creates no grid."""
        stream = converged(1.0)
        stream.rebuild_grid(
            batch_start=ts(1.0), batch_length=Duration.from_seconds(0.6)
        )
        assert stream.grid is None

    def test_grow_batch_length_regates_sub_rate_stream(self):
        """Sub-rate → grid absent.  Growing batch back to 1 s re-gates it."""
        stream = converged(1.0)
        stream.rebuild_grid(
            batch_start=ts(1.0), batch_length=Duration.from_seconds(0.6)
        )
        assert stream.grid is None
        stream.rebuild_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid is not None
        assert stream.grid.slots_per_batch == 1

    def test_origin_preserved_on_repeat_rebuild(self):
        stream = converged(14.0)
        stream.rebuild_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        origin = stream.grid.origin_ns
        stream.rebuild_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid.origin_ns == origin

    def test_origin_preserved_within_drift_bound(self):
        """Batch_start advancing within _MAX_ORIGIN_OFFSET_BATCHES keeps origin."""
        stream = converged(14.0)
        stream.rebuild_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        origin = stream.grid.origin_ns
        # 50 batches advance -- well within the 1000-batch bound.
        stream.rebuild_grid(batch_start=ts(51.0), batch_length=ONE_SECOND)
        assert stream.grid.origin_ns == origin

    def test_stale_origin_dropped_without_candidate(self):
        """Drift past bound with no fresh bucket message → grid dropped."""
        stream = converged(14.0)
        stream.rebuild_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid is not None
        # Batch_start jumps past _MAX_ORIGIN_OFFSET_BATCHES; estimator's
        # last_ts_ns is still in the old epoch → no viable replacement.
        far = float(_MAX_ORIGIN_OFFSET_BATCHES + 10)
        stream.rebuild_grid(batch_start=ts(far), batch_length=ONE_SECOND)
        assert stream.grid is None

    def test_stale_origin_replaced_from_bucket_message(self):
        """Drift past bound but bucket has a healthy message → grid refreshed."""
        stream = converged(14.0)
        stream.rebuild_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        stale = stream.grid.origin_ns
        far = float(_MAX_ORIGIN_OFFSET_BATCHES + 10)
        # Seed bucket with a message near the new batch_start (simulating
        # a healthy arrival routed into the active window).
        stream.messages.append(msg(far + 0.1))
        stream.rebuild_grid(batch_start=ts(far), batch_length=ONE_SECOND)
        assert stream.grid is not None
        assert stream.grid.origin_ns != stale

    def test_bucket_message_preferred_over_estimator_last_ts(self):
        """Origin picked from bucket at cold start, not estimator.last_ts_ns."""
        stream = converged(14.0)
        # Estimator's last_ts_ns is at the end of the convergence sequence;
        # add a fresher bucket message and verify it wins.
        fresh_ns = stream.estimator.last_ts_ns + 5_000_000_000
        stream.messages.append(
            Message(timestamp=Timestamp.from_ns(fresh_ns), stream=STREAM, value="")
        )
        stream.rebuild_grid(
            batch_start=Timestamp.from_ns(fresh_ns), batch_length=ONE_SECOND
        )
        assert stream.grid is not None
        assert stream.grid.origin_ns == fresh_ns

    def test_rate_change_rebuilds_grid(self):
        """Re-converging at a new rate updates period and slots_per_batch."""
        stream = converged(14.0)
        stream.rebuild_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid.slots_per_batch == 14
        # Reset estimator and seed at 10 Hz.
        stream.estimator.diffs.clear()
        stream.estimator.last_ts_ns = None
        period = 1.0 / 10.0
        for i in range(MIN_DIFFS_FOR_GATE + 1):
            stream.observe(msg(100.0 + i * period))
        stream.rebuild_grid(batch_start=ts(101.0), batch_length=ONE_SECOND)
        assert stream.grid is not None
        assert stream.grid.period_ns == round(1e9 / 10)
        assert stream.grid.slots_per_batch == 10
