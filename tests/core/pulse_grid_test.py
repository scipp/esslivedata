# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for PulseGrid and StreamPeriodEstimator components."""

import pytest

from ess.livedata.core.rate_aware_batcher import (
    MIN_DIFFS_FOR_GATE,
    PulseGrid,
    StreamPeriodEstimator,
)
from ess.livedata.core.timestamp import Timestamp


def ts(seconds: float) -> Timestamp:
    return Timestamp.from_ns(int(seconds * 1e9))


def _observe(est: StreamPeriodEstimator, times_ns: list[int]) -> None:
    for t in times_ns:
        est.observe(t)


# --- StreamPeriodEstimator ---


class TestStreamPeriodEstimator:
    def test_initial_state(self):
        est = StreamPeriodEstimator()
        assert est.last_ts_ns is None
        assert len(est.diffs) == 0
        assert est.integer_rate_hz is None

    def test_first_observation_sets_last_ts(self):
        est = StreamPeriodEstimator()
        est.observe(1_000_000_000)
        assert est.last_ts_ns == 1_000_000_000
        assert len(est.diffs) == 0

    def test_second_observation_records_diff(self):
        est = StreamPeriodEstimator()
        est.observe(1_000_000_000)
        est.observe(1_071_428_571)  # +1/14 s
        assert list(est.diffs) == [71_428_571]

    def test_duplicate_timestamp_filtered(self):
        """Split messages (same timestamp) produce zero diffs, ignored."""
        est = StreamPeriodEstimator()
        est.observe(1_000_000_000)
        est.observe(1_000_000_000)
        est.observe(1_071_428_571)
        assert list(est.diffs) == [71_428_571]

    def test_retrograde_timestamp_does_not_advance_last_ts(self):
        """Out-of-order messages do not corrupt the period estimate."""
        est = StreamPeriodEstimator()
        est.observe(1_000_000_000)
        est.observe(1_100_000_000)
        est.observe(1_050_000_000)  # late arrival, should not reset last_ts
        est.observe(1_200_000_000)
        assert list(est.diffs) == [100_000_000, 100_000_000]

    def test_not_converged_below_min_diffs(self):
        period = 71_428_571
        est = StreamPeriodEstimator()
        times = [i * period for i in range(MIN_DIFFS_FOR_GATE)]  # MIN_DIFFS - 1 diffs
        _observe(est, times)
        assert est.integer_rate_hz is None

    def test_converged_at_min_diffs(self):
        period = 71_428_571
        est = StreamPeriodEstimator()
        times = [i * period for i in range(MIN_DIFFS_FOR_GATE + 1)]
        _observe(est, times)
        assert est.integer_rate_hz == 14

    def test_missing_pulse_tolerated(self):
        """One large-outlier diff is ignored by min()."""
        period = 71_428_571
        est = StreamPeriodEstimator()
        times = [0, period, 2 * period, 4 * period, 5 * period, 6 * period]
        _observe(est, times)
        assert est.integer_rate_hz == 14

    def test_integer_rate_snap(self):
        """Period slightly off integer Hz still snaps to nearest integer."""
        period = round(1e9 / 13.98)  # 13.98 Hz
        est = StreamPeriodEstimator()
        times = [i * period for i in range(MIN_DIFFS_FOR_GATE + 1)]
        _observe(est, times)
        assert est.integer_rate_hz == 14

    def test_sub_hz_rate_returns_none(self):
        """Rate rounding to 0 returns None so no grid is built."""
        period = 2_000_000_000  # 0.5 Hz
        est = StreamPeriodEstimator()
        times = [i * period for i in range(MIN_DIFFS_FOR_GATE + 1)]
        _observe(est, times)
        assert est.integer_rate_hz is None

    def test_diffs_buffer_bounded(self):
        """Buffer does not grow unbounded."""
        from ess.livedata.core.rate_aware_batcher import DIFF_BUFFER_SIZE

        est = StreamPeriodEstimator()
        for i in range(DIFF_BUFFER_SIZE * 3):
            est.observe(i * 71_428_571)
        assert len(est.diffs) == DIFF_BUFFER_SIZE


# --- PulseGrid ---


class TestPulseGrid:
    def test_pulse_index_at_origin(self):
        grid = PulseGrid(origin_ns=1000, period_ns=100, slots_per_batch=14)
        assert grid.pulse_index(Timestamp.from_ns(1000)) == 0

    def test_pulse_index_one_period_from_origin(self):
        grid = PulseGrid(origin_ns=1000, period_ns=100, slots_per_batch=14)
        assert grid.pulse_index(Timestamp.from_ns(1100)) == 1

    def test_pulse_index_negative(self):
        grid = PulseGrid(origin_ns=1000, period_ns=100, slots_per_batch=14)
        assert grid.pulse_index(Timestamp.from_ns(900)) == -1

    def test_pulse_index_with_jitter_rounds_correctly(self):
        period_ns = round(1e9 / 14)
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=14)
        jitter = period_ns // 4
        # Pulse 5 with positive jitter
        assert grid.pulse_index(Timestamp.from_ns(5 * period_ns + jitter)) == 5
        # Pulse 5 with negative jitter
        assert grid.pulse_index(Timestamp.from_ns(5 * period_ns - jitter)) == 5

    def test_slot_in_batch_first_pulse(self):
        period_ns = round(1e9 / 14)
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=14)
        batch_start = Timestamp.from_ns(100 * period_ns)
        assert grid.slot_in_batch(Timestamp.from_ns(100 * period_ns), batch_start) == 0

    def test_slot_in_batch_last_pulse_14hz(self):
        period_ns = round(1e9 / 14)
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=14)
        batch_start = Timestamp.from_ns(100 * period_ns)
        assert grid.slot_in_batch(Timestamp.from_ns(113 * period_ns), batch_start) == 13

    def test_slot_in_batch_negative_for_late_arrival(self):
        period_ns = round(1e9 / 14)
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=14)
        batch_start = Timestamp.from_ns(100 * period_ns)
        assert grid.slot_in_batch(Timestamp.from_ns(99 * period_ns), batch_start) == -1

    def test_split_messages_same_slot(self):
        period_ns = round(1e9 / 14)
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=14)
        batch_start = Timestamp.from_ns(0)
        t = Timestamp.from_ns(5 * period_ns)
        assert grid.slot_in_batch(t, batch_start) == grid.slot_in_batch(t, batch_start)

    def test_omitted_messages_do_not_affect_indices(self):
        """Pulse indices are absolute — gaps don't shift later indices."""
        period_ns = round(1e9 / 14)
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=14)
        batch_start = Timestamp.from_ns(0)
        # Even if pulses 3,4,5 are missing, pulse 6 is still slot 6
        assert grid.slot_in_batch(Timestamp.from_ns(6 * period_ns), batch_start) == 6

    def test_batch_base_index_at_origin(self):
        grid = PulseGrid(origin_ns=0, period_ns=100, slots_per_batch=14)
        assert grid.batch_base_index(Timestamp.from_ns(0)) == 0

    def test_batch_base_index_between_pulses(self):
        """Clearly between pulses: snaps to next pulse."""
        period_ns = round(1e9 / 14)
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=14)
        # 40% into the period — clearly past
        assert grid.batch_base_index(Timestamp.from_ns(period_ns * 4 // 10)) == 1

    def test_batch_base_index_absorbs_rounding_drift(self):
        """Tiny remainder from period*rate != batch_length stays at current pulse."""
        period_ns = round(1e9 / 14)
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=14)
        # 14 * period = 999999994, batch_length = 1e9 → 6ns past pulse 14
        batch_start = Timestamp.from_ns(14 * period_ns + 6)
        assert grid.batch_base_index(batch_start) == 14

    def test_consistent_across_batches(self):
        """Same grid produces stable slot assignments across different batches.

        This is the key property that eliminates per-batch phase drift.
        """
        period_ns = round(1e9 / 14)
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=14)

        for batch_idx in range(10):
            batch_start = Timestamp.from_ns(batch_idx * 14 * period_ns)
            for pulse in range(14):
                t = Timestamp.from_ns((batch_idx * 14 + pulse) * period_ns)
                assert grid.slot_in_batch(t, batch_start) == pulse

    def test_consistent_with_phase_offset(self):
        """Grid with non-zero origin still assigns slots consistently.

        batch_start is NOT at a pulse — it's at an arbitrary point. The
        grid's batch_base_index (ceiling) ensures slot 0 = first pulse
        at or after batch_start.
        """
        period_ns = round(1e9 / 14)
        offset_ns = period_ns * 4 // 10  # 40% of period
        grid = PulseGrid(origin_ns=offset_ns, period_ns=period_ns, slots_per_batch=14)

        for batch_idx in range(10):
            # batch_start is NOT aligned to pulses
            batch_start_ns = batch_idx * 14 * period_ns
            batch_start = Timestamp.from_ns(batch_start_ns)
            for pulse in range(14):
                # Messages are at the pulse grid positions offset from batch_start
                t = Timestamp.from_ns(offset_ns + (batch_idx * 14 + pulse) * period_ns)
                assert grid.slot_in_batch(t, batch_start) == pulse

    def test_jitter_tolerance_within_half_period(self):
        """Jitter up to period/2 - 1ns maps to the correct slot."""
        period_ns = round(1e9 / 14)
        grid = PulseGrid(origin_ns=0, period_ns=period_ns, slots_per_batch=14)
        batch_start = Timestamp.from_ns(0)
        max_jitter = period_ns // 2 - 1

        for pulse in range(14):
            base_ns = pulse * period_ns
            assert (
                grid.slot_in_batch(Timestamp.from_ns(base_ns + max_jitter), batch_start)
                == pulse
            )
            assert (
                grid.slot_in_batch(Timestamp.from_ns(base_ns - max_jitter), batch_start)
                == pulse
            )

    def test_frozen(self):
        grid = PulseGrid(origin_ns=0, period_ns=100, slots_per_batch=14)
        with pytest.raises(AttributeError):
            grid.origin_ns = 1  # type: ignore[misc]
