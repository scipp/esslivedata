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
    PulseGrid,
    StreamPeriodEstimator,
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


class TestStreamPeriodEstimator:
    """Period estimator: unbiased under jitter, robust to missed pulses."""

    @staticmethod
    def _feed(estimator: StreamPeriodEstimator, timestamps_ns: list[int]) -> None:
        for t in timestamps_ns:
            estimator.observe(t)

    @staticmethod
    def _simulate_pulses(
        rate_hz: float,
        jitter_ms: float,
        p_missing: float,
        n_samples: int,
        seed: int,
    ) -> list[int]:
        """Realistic pulse timestamps: periodic + Gaussian jitter + dropouts.

        Generates enough nominal pulses to collect ``n_samples`` surviving
        timestamps after applying the missing-pulse fraction.
        """
        import random

        rng = random.Random(seed)
        period_ns = int(1e9 / rate_hz)
        jitter_ns = jitter_ms * 1e6
        timestamps: list[int] = []
        i = 0
        while len(timestamps) < n_samples:
            i += 1
            if rng.random() < p_missing:
                continue
            timestamps.append(i * period_ns + int(rng.gauss(0, jitter_ns)))
        return timestamps

    def test_clean_integer_period_recovered(self):
        est = StreamPeriodEstimator()
        period = int(1e9 / 14)
        self._feed(est, [i * period for i in range(32)])
        assert est.integer_rate_hz == 14

    def test_missed_pulses_handled_via_integer_multiples(self):
        """Outlier diffs that are k·P still contribute a per-pulse estimate."""
        est = StreamPeriodEstimator()
        period = int(1e9 / 14)
        ts_ns: list[int] = []
        t = 0
        for i in range(40):
            if i % 4 != 3:
                ts_ns.append(t)
            t += period
        self._feed(est, ts_ns)
        assert est.integer_rate_hz == 14

    def test_split_messages_filtered(self):
        """Zero diffs (split messages at same timestamp) don't break estimation."""
        est = StreamPeriodEstimator()
        period = int(1e9 / 14)
        ts_ns: list[int] = []
        for i in range(32):
            ts_ns.append(i * period)
            ts_ns.append(i * period)
        self._feed(est, ts_ns)
        assert est.integer_rate_hz == 14

    def test_heavy_jitter_at_14hz(self):
        """14 Hz with 6.5ms jitter: ~9% of period, regression for `min` bias."""
        successes = 0
        trials = 200
        for seed in range(trials):
            est = StreamPeriodEstimator()
            ts = self._simulate_pulses(
                rate_hz=14.0,
                jitter_ms=6.5,
                p_missing=0.0,
                n_samples=32,
                seed=seed,
            )
            self._feed(est, ts)
            if est.integer_rate_hz == 14:
                successes += 1
        assert successes >= trials * 0.9, (
            f"Only {successes}/{trials} trials recovered 14 Hz"
        )

    def test_tight_snap_at_high_rate_with_small_jitter(self):
        """100 Hz has ~0.5% snap tolerance — where `min` bias catastrophically fails."""
        successes = 0
        trials = 200
        for seed in range(trials):
            est = StreamPeriodEstimator()
            ts = self._simulate_pulses(
                rate_hz=100.0,
                jitter_ms=0.1,  # 1% of period
                p_missing=0.0,
                n_samples=32,
                seed=seed,
            )
            self._feed(est, ts)
            if est.integer_rate_hz == 100:
                successes += 1
        assert successes >= trials * 0.95, (
            f"Only {successes}/{trials} trials recovered 100 Hz"
        )

    def test_jitter_with_missing_pulses(self):
        """Combined stress: moderate jitter + 20% missing pulses at 14 Hz."""
        successes = 0
        trials = 200
        for seed in range(trials):
            est = StreamPeriodEstimator()
            ts = self._simulate_pulses(
                rate_hz=14.0,
                jitter_ms=3.0,
                p_missing=0.2,
                n_samples=32,
                seed=seed,
            )
            self._feed(est, ts)
            if est.integer_rate_hz == 14:
                successes += 1
        assert successes >= trials * 0.95, (
            f"Only {successes}/{trials} trials recovered 14 Hz"
        )

    def test_low_rate_1hz_with_jitter(self):
        """1 Hz stream (monitor): must converge despite slow diff accumulation."""
        est = StreamPeriodEstimator()
        ts = self._simulate_pulses(
            rate_hz=1.0, jitter_ms=5.0, p_missing=0.0, n_samples=32, seed=0
        )
        self._feed(est, ts)
        assert est.integer_rate_hz == 1

    def test_non_integer_sub_hz_rate_rejected(self):
        """True 0.7 Hz producer: raw rate snaps to 1 with ~30% error.

        Without a confidence guard, the estimator would falsely report
        this as a 1 Hz stream, and the batcher would gate-include it
        even though it delivers far fewer than one pulse per batch.
        """
        est = StreamPeriodEstimator()
        ts = self._simulate_pulses(
            rate_hz=0.7, jitter_ms=5.0, p_missing=0.0, n_samples=32, seed=0
        )
        self._feed(est, ts)
        assert est.integer_rate_hz is None

    def test_non_integer_rate_between_integers_rejected(self):
        """True 1.4 Hz producer: raw rate is ~40% away from either 1 or 2.

        Must not snap to either neighbour — return None so the caller
        keeps the stream out of the gated set.
        """
        est = StreamPeriodEstimator()
        ts = self._simulate_pulses(
            rate_hz=1.4, jitter_ms=5.0, p_missing=0.0, n_samples=32, seed=0
        )
        self._feed(est, ts)
        assert est.integer_rate_hz is None

    def test_near_1hz_non_integer_rate_rejected(self):
        """True 0.85 Hz producer: 15% off from 1 Hz.

        A pure-relative 20% tolerance accepts this as "1 Hz" and puts
        slot math 17% wrong — the stream then never gates cleanly.
        An absolute tolerance at low rates must reject it.
        """
        est = StreamPeriodEstimator()
        ts = self._simulate_pulses(
            rate_hz=0.85, jitter_ms=5.0, p_missing=0.0, n_samples=32, seed=0
        )
        self._feed(est, ts)
        assert est.integer_rate_hz is None

    def test_clean_1hz_snaps(self):
        """Pulse-aligned 1 Hz stream still snaps to 1 Hz under the tighter tolerance."""
        est = StreamPeriodEstimator()
        period = int(1e9)
        self._feed(est, [i * period for i in range(32)])
        assert est.integer_rate_hz == 1

    def test_14hz_with_realistic_jitter_still_snaps(self):
        """14 Hz with ±1 ms jitter must still snap — absolute tolerance floor
        must not starve high-rate streams whose estimator noise scales with rate.
        """
        est = StreamPeriodEstimator()
        ts = self._simulate_pulses(
            rate_hz=14.0, jitter_ms=1.0, p_missing=0.0, n_samples=32, seed=0
        )
        self._feed(est, ts)
        assert est.integer_rate_hz == 14


class TestPulseGrid:
    """Grid-level slot assignment: no pulse inside [batch_start, batch_end)
    may be classified as slot < 0 or slot >= slots_per_batch.
    """

    def test_origin_just_before_batch_start_in_window_pulse_at_slot_0(self):
        """Origin within 1% of period before batch_start: the in-window pulse
        (just before batch_end) is slot 0, not slot 1.

        At slots_per_batch=1 this is catastrophic: mis-classifying slot 0
        as slot 1 puts the only in-window pulse in overflow.
        """
        period_ns = 1_000_000_000
        grid = PulseGrid(
            origin_ns=-5_000_000,  # 5 ms before batch_start=0
            period_ns=period_ns,
            slots_per_batch=1,
        )
        # Pulse at 0.995 s lies inside batch window [0, 1 s).
        pulse = Timestamp.from_ns(995_000_000)
        batch_start = Timestamp.from_ns(0)
        assert grid.slot_in_batch(pulse, batch_start) == 0

    def test_batch_start_past_origin_pulse_by_phase_offset(self):
        """batch_start 5 ms past a pulse: the pre-window pulse is slot -1,
        and the next pulse becomes slot 0.
        """
        period_ns = 1_000_000_000
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=2)
        batch_start = Timestamp.from_ns(5_000_000)
        assert grid.slot_in_batch(Timestamp.from_ns(0), batch_start) == -1
        assert grid.slot_in_batch(Timestamp.from_ns(period_ns), batch_start) == 0
        assert grid.slot_in_batch(Timestamp.from_ns(2 * period_ns), batch_start) == 1

    def test_batch_start_just_after_origin_pulse_absorbs_drift(self):
        """Integer-Hz rounding drift (a few ns past a pulse) is absorbed:
        that pulse is still slot 0.
        """
        period_ns = 71_428_571  # 14 Hz integer period
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=14)
        # After N batches of 1 s, drift is ~6*N ns (well under 1 ms for
        # realistic N). 100 ns chosen to exercise the tolerance band.
        batch_start = Timestamp.from_ns(100)
        assert grid.slot_in_batch(Timestamp.from_ns(0), batch_start) == 0

    def test_batch_start_just_before_next_pulse_snaps_forward(self):
        """batch_start within 1% of the next pulse: that pulse is slot 0."""
        period_ns = 1_000_000_000
        tolerance = period_ns // 100  # 10 ms
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=1)
        # batch_start = 1 s - 5 ms: pulse at 1 s (inside tolerance ahead) is slot 0.
        batch_start = Timestamp.from_ns(period_ns - tolerance // 2)
        assert grid.slot_in_batch(Timestamp.from_ns(period_ns), batch_start) == 0

    def test_batch_start_exactly_on_pulse(self):
        period_ns = 1_000_000_000
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=2)
        batch_start = Timestamp.from_ns(period_ns)
        assert grid.slot_in_batch(Timestamp.from_ns(period_ns), batch_start) == 0
        assert grid.slot_in_batch(Timestamp.from_ns(2 * period_ns), batch_start) == 1


class TestGatedStream:
    def test_no_grid_before_convergence(self):
        stream = _GatedStream()
        stream.observe(msg(0.0))  # one sample -> no diffs
        stream.refresh_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid is None

    def test_integer_rate_builds_grid(self):
        stream = converged(14.0)
        stream.refresh_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
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
        stream.refresh_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid is None

    def test_sub_rate_drops_existing_grid(self):
        """1 Hz stream gridded at 1 s: shrink batch to 0.6 s → grid dropped."""
        stream = converged(1.0)
        stream.refresh_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid is not None
        stream.refresh_grid(
            batch_start=ts(1.0), batch_length=Duration.from_seconds(0.6)
        )
        assert stream.grid is None

    def test_sub_rate_never_builds_grid(self):
        """First rebuild at sub-rate (1 Hz * 0.6 s < 1) creates no grid."""
        stream = converged(1.0)
        stream.refresh_grid(
            batch_start=ts(1.0), batch_length=Duration.from_seconds(0.6)
        )
        assert stream.grid is None

    def test_grow_batch_length_regates_sub_rate_stream(self):
        """Sub-rate → grid absent.  Growing batch back to 1 s re-gates it."""
        stream = converged(1.0)
        stream.refresh_grid(
            batch_start=ts(1.0), batch_length=Duration.from_seconds(0.6)
        )
        assert stream.grid is None
        stream.refresh_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid is not None
        assert stream.grid.slots_per_batch == 1

    def test_origin_preserved_on_repeat_rebuild(self):
        stream = converged(14.0)
        stream.refresh_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        grid = stream.grid
        stream.refresh_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        # Identity check pins down the fast-path no-op branch: an identical
        # (origin, period, slots) triple must not allocate a new PulseGrid.
        assert stream.grid is grid

    def test_origin_preserved_within_drift_bound(self):
        """Batch_start advancing within _MAX_ORIGIN_OFFSET_BATCHES keeps origin."""
        stream = converged(14.0)
        stream.refresh_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        origin = stream.grid.origin_ns
        # 50 batches advance -- well within the 1000-batch bound.
        stream.refresh_grid(batch_start=ts(51.0), batch_length=ONE_SECOND)
        assert stream.grid.origin_ns == origin

    def test_stale_origin_dropped_without_candidate(self):
        """Drift past bound with no fresh bucket message → grid dropped."""
        stream = converged(14.0)
        stream.refresh_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid is not None
        # Batch_start jumps past _MAX_ORIGIN_OFFSET_BATCHES; estimator's
        # last_ts_ns is still in the old epoch → no viable replacement.
        far = float(_MAX_ORIGIN_OFFSET_BATCHES + 10)
        stream.refresh_grid(batch_start=ts(far), batch_length=ONE_SECOND)
        assert stream.grid is None

    def test_stale_origin_replaced_from_bucket_message(self):
        """Drift past bound but bucket has a healthy message → grid refreshed."""
        stream = converged(14.0)
        stream.refresh_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        stale = stream.grid.origin_ns
        far = float(_MAX_ORIGIN_OFFSET_BATCHES + 10)
        # Seed bucket with a message near the new batch_start (simulating
        # a healthy arrival routed into the active window).
        stream.messages.append(msg(far + 0.1))
        stream.refresh_grid(batch_start=ts(far), batch_length=ONE_SECOND)
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
        stream.refresh_grid(
            batch_start=Timestamp.from_ns(fresh_ns), batch_length=ONE_SECOND
        )
        assert stream.grid is not None
        assert stream.grid.origin_ns == fresh_ns

    def test_rate_change_rebuilds_grid(self):
        """Re-converging at a new rate updates period and slots_per_batch."""
        stream = converged(14.0)
        stream.refresh_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        assert stream.grid.slots_per_batch == 14
        original_origin = stream.grid.origin_ns
        # Reset estimator and seed at 10 Hz.
        stream.estimator.diffs.clear()
        stream.estimator.last_ts_ns = None
        period = 1.0 / 10.0
        for i in range(MIN_DIFFS_FOR_GATE + 1):
            stream.observe(msg(100.0 + i * period))
        stream.refresh_grid(batch_start=ts(101.0), batch_length=ONE_SECOND)
        assert stream.grid is not None
        assert stream.grid.period_ns == round(1e9 / 10)
        assert stream.grid.slots_per_batch == 10
        # Old origin is within the 1000-batch bound (100 s offset), so it
        # must be preserved even though the period changed.
        assert stream.grid.origin_ns == original_origin

    def test_origin_preserved_across_batch_length_shrink(self):
        """Shrinking batch above sub-rate keeps origin, updates slot count."""
        stream = converged(14.0)
        stream.refresh_grid(batch_start=ts(1.0), batch_length=ONE_SECOND)
        original_origin = stream.grid.origin_ns
        stream.refresh_grid(
            batch_start=ts(1.0), batch_length=Duration.from_seconds(0.5)
        )
        assert stream.grid is not None
        assert stream.grid.origin_ns == original_origin
        assert stream.grid.slots_per_batch == 7

    def test_bucket_entirely_before_batch_start_seeds_from_first_message(self):
        """Bucket with only pre-batch_start messages falls back to messages[0]."""
        stream = converged(14.0)
        # Clear bucket state left by the convergence sequence, then seed with
        # messages that are strictly before a later batch_start but still
        # within the drift bound.
        stream.messages.clear()
        first_ns = 10_000_000_000  # 10 s
        stream.messages.append(
            Message(timestamp=Timestamp.from_ns(first_ns), stream=STREAM, value="")
        )
        stream.messages.append(
            Message(
                timestamp=Timestamp.from_ns(first_ns + 500_000_000),
                stream=STREAM,
                value="",
            )
        )
        stream.refresh_grid(batch_start=ts(20.0), batch_length=ONE_SECOND)
        assert stream.grid is not None
        assert stream.grid.origin_ns == first_ns
