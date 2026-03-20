# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Scenario-based tests for adaptive message batching strategies.

These tests simulate realistic load patterns by running a processing loop that
feeds batch outcomes back into a ``MessageBatcher``.  They assert on observable
properties — escalation time, maximum backlog, oscillation — rather than on
implementation internals, so they remain valid as the strategy evolves.

The simulation model:
- Time advances discretely per processing cycle.
- Each cycle, the batcher's current ``batch_length_s`` determines how much
  wall-clock data is covered.
- A ``processing_cost`` function returns how long the batch *takes* to process,
  based on the batch window and a per-batch overhead.
- If processing takes longer than the batch window, backlog accumulates
  (the system falls behind real-time).
- Random jitter is optionally added to processing times.

All acceptance thresholds are collected in :data:`LIMITS` so that tuning the
strategy and its acceptable bounds can be done in one place.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol
from unittest.mock import patch

from ess.livedata.core.message_batcher import (
    AdaptiveMessageBatcher,
    MessageBatcher,
)

# ===========================================================================
# Acceptance limits — one place to view and adjust all thresholds
# ===========================================================================

# Each scenario test references a key from this dict.  When tuning the
# batching strategy or its parameters, start here: tighten the bounds,
# run the tests, and iterate.
#
# Convention:
#   max_*    — upper bound (test asserts  value <= limit)
#   min_*    — lower bound (test asserts  value >= limit)
#
# All time values are in seconds.

LIMITS: dict[str, dict[str, float]] = {
    # -- Step-function escalation (shutter open) --------------------------
    "step_function_escalation": {
        "max_time_to_first_escalation_s": 10.0,
    },
    "step_function_backlog": {
        "max_backlog_s": 5.0,
        "max_final_backlog_s": 1.0,
    },
    "severe_step_function": {
        "min_level_reached": 2,
    },
    # -- No escalation when not needed ------------------------------------
    "light_load": {
        "max_level": 0,
    },
    "gc_jitter": {
        "max_level": 0,
    },
    # -- No oscillation ---------------------------------------------------
    "steady_load_oscillation": {
        "max_oscillations": 0,
    },
    "boundary_oscillation": {
        "max_oscillations": 4,
    },
    # -- Creeping overload ------------------------------------------------
    "creeping_overload": {
        "min_level_reached": 1,
    },
    "creeping_overload_backlog": {
        "max_backlog_s": 5.0,
    },
    "mild_creeping_overload": {
        "max_level": 1,
    },
    # -- De-escalation ----------------------------------------------------
    "deescalation_to_idle": {
        "max_final_level": 0,
    },
    "deescalation_to_light_load": {
        "max_final_level": 0,
    },
    # -- Realistic shutter ------------------------------------------------
    "shutter_open_close": {
        "min_level_reached": 1,
        "max_final_level": 0,
        "max_backlog_s": 10.0,
    },
    "repeated_shutter_cycles": {
        "max_final_level": 0,
    },
    # -- Backlog draining -------------------------------------------------
    "backlog_drains": {
        "max_final_backlog_s": 1.0,
    },
    # -- Processing-time awareness ----------------------------------------
    "fast_escalation_clear_overload": {
        "max_time_to_first_escalation_s": 5.0,
    },
    "no_escalation_when_fits": {
        "max_level": 0,
    },
}


# ---------------------------------------------------------------------------
# Simulation infrastructure
# ---------------------------------------------------------------------------


class ProcessingCostFn(Protocol):
    """Returns the processing time (seconds) for a batch of given window."""

    def __call__(
        self, batch_window_s: float, wall_time_s: float
    ) -> float: ...


@dataclass
class CycleRecord:
    """A single processing-loop iteration."""

    wall_time_s: float
    batch_window_s: float
    processing_time_s: float
    backlog_s: float
    level: int


@dataclass
class SimulationResult:
    """Aggregate outcome of a simulation run."""

    cycles: list[CycleRecord] = field(default_factory=list)

    @property
    def max_backlog_s(self) -> float:
        if not self.cycles:
            return 0.0
        return max(c.backlog_s for c in self.cycles)

    @property
    def final_backlog_s(self) -> float:
        return self.cycles[-1].backlog_s if self.cycles else 0.0

    @property
    def final_level(self) -> int:
        return self.cycles[-1].level if self.cycles else 0

    @property
    def max_level(self) -> int:
        if not self.cycles:
            return 0
        return max(c.level for c in self.cycles)

    @property
    def total_wall_time_s(self) -> float:
        return self.cycles[-1].wall_time_s if self.cycles else 0.0

    def time_at_level(self, level: int) -> float:
        """Total wall time spent at a given level."""
        return sum(
            c.processing_time_s
            for c in self.cycles
            if c.level == level
        )

    def level_changes(self) -> list[tuple[float, int, int]]:
        """List of (wall_time, old_level, new_level) transitions."""
        return [
            (
                self.cycles[i].wall_time_s,
                self.cycles[i - 1].level,
                self.cycles[i].level,
            )
            for i in range(1, len(self.cycles))
            if self.cycles[i].level != self.cycles[i - 1].level
        ]

    def first_escalation_time_s(self) -> float | None:
        """Wall time of the first escalation, or None."""
        for t, old, new in self.level_changes():
            if new > old:
                return t
        return None

    def oscillation_count(self) -> int:
        """Number of direction changes (up->down or down->up)."""
        changes = self.level_changes()
        if len(changes) < 2:
            return 0
        directions = [
            1 if new > old else -1 for _, old, new in changes
        ]
        return sum(
            1
            for i in range(1, len(directions))
            if directions[i] != directions[i - 1]
        )


class FakeClock:
    """Deterministic monotonic clock for simulation."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def simulate(
    batcher: MessageBatcher,
    duration_s: float,
    cost_fn: ProcessingCostFn,
    clock: FakeClock,
    *,
    idle_poll_interval_s: float = 0.1,
) -> SimulationResult:
    """Run a simulated processing loop.

    The loop mimics ``OrchestratingProcessor.process()``:
    1. Read the batcher's current window size.
    2. Compute how long processing takes (via ``cost_fn``).
    3. If processing < window, the remaining time is idle cycles.
    4. Advance the clock and report the batch outcome.
    5. If processing > window, backlog accumulates.
    """
    result = SimulationResult()
    backlog_s = 0.0

    while clock.now < duration_s:
        window = batcher.batch_length_s
        processing_time = cost_fn(window, clock.now)

        if processing_time <= 0:
            clock.advance(idle_poll_interval_s)
            batcher.report_batch(None, processing_time_s=0.0)
            level = _get_level(batcher)
            result.cycles.append(CycleRecord(
                wall_time_s=clock.now,
                batch_window_s=window,
                processing_time_s=0.0,
                backlog_s=backlog_s,
                level=level,
            ))
            continue

        clock.advance(processing_time)

        if processing_time > window:
            backlog_s += processing_time - window
        else:
            spare = window - processing_time
            drained = min(spare, backlog_s)
            backlog_s -= drained
            remaining_idle = spare - drained
            if remaining_idle > 0:
                n_idle = int(remaining_idle / idle_poll_interval_s)
                for _ in range(n_idle):
                    clock.advance(idle_poll_interval_s)
                    batcher.report_batch(
                        None, processing_time_s=0.0
                    )

        batcher.report_batch(
            100, processing_time_s=processing_time
        )

        level = _get_level(batcher)
        result.cycles.append(CycleRecord(
            wall_time_s=clock.now,
            batch_window_s=window,
            processing_time_s=processing_time,
            backlog_s=backlog_s,
            level=level,
        ))

    return result


def _get_level(batcher: MessageBatcher) -> int:
    if hasattr(batcher, 'state'):
        return batcher.state.level
    return 0


# ---------------------------------------------------------------------------
# Processing cost models
# ---------------------------------------------------------------------------


def constant_overhead_cost(
    overhead_s: float,
    per_second_cost: float,
    *,
    jitter_fraction: float = 0.0,
    rng: random.Random | None = None,
) -> ProcessingCostFn:
    """Fixed overhead + linear data cost, with optional jitter.

    ``processing_time = overhead_s + per_second_cost * window + jitter``

    The system keeps up when ``overhead_s < window * (1 - per_second_cost)``.
    """
    _rng = rng or random.Random(42)

    def cost(batch_window_s: float, wall_time_s: float) -> float:
        base = overhead_s + per_second_cost * batch_window_s
        if jitter_fraction > 0:
            jitter = _rng.gauss(0, jitter_fraction * base)
            base = max(0.01, base + jitter)
        return base

    return cost


def step_function_cost(
    step_time_s: float,
    before: ProcessingCostFn,
    after: ProcessingCostFn,
) -> ProcessingCostFn:
    """Switch cost functions at a given wall-clock time."""

    def cost(batch_window_s: float, wall_time_s: float) -> float:
        if wall_time_s < step_time_s:
            return before(batch_window_s, wall_time_s)
        return after(batch_window_s, wall_time_s)

    return cost


def idle_cost() -> ProcessingCostFn:
    """No data to process."""

    def cost(batch_window_s: float, wall_time_s: float) -> float:
        return 0.0

    return cost


def creeping_cost(
    overhead_s: float,
    per_second_cost_start: float,
    per_second_cost_end: float,
    ramp_duration_s: float,
    ramp_start_s: float = 0.0,
    *,
    jitter_fraction: float = 0.0,
    rng: random.Random | None = None,
) -> ProcessingCostFn:
    """Processing cost that linearly ramps up over time."""
    _rng = rng or random.Random(42)

    def cost(batch_window_s: float, wall_time_s: float) -> float:
        elapsed = max(0.0, wall_time_s - ramp_start_s)
        frac = (
            min(1.0, elapsed / ramp_duration_s)
            if ramp_duration_s > 0
            else 1.0
        )
        rate_range = per_second_cost_end - per_second_cost_start
        per_s = per_second_cost_start + frac * rate_range
        base = overhead_s + per_s * batch_window_s
        if jitter_fraction > 0:
            jitter = _rng.gauss(0, jitter_fraction * base)
            base = max(0.01, base + jitter)
        return base

    return cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_scenario(
    batcher: MessageBatcher,
    duration_s: float,
    cost_fn: ProcessingCostFn,
) -> SimulationResult:
    clock = FakeClock()
    with patch(
        'ess.livedata.core.message_batcher.time.monotonic', clock
    ):
        return simulate(batcher, duration_s, cost_fn, clock)


def make_default_batcher(**kwargs) -> AdaptiveMessageBatcher:
    defaults = {"base_batch_length_s": 1.0, "max_level": 3}
    defaults.update(kwargs)
    return AdaptiveMessageBatcher(**defaults)


# ===========================================================================
# Scenario tests
# ===========================================================================


class TestStepFunctionEscalation:
    """Shutter-open scenario: sudden jump from idle to high load."""

    def test_escalates_within_bounded_time(self):
        """After a step increase in load, the batcher must escalate quickly."""
        lim = LIMITS["step_function_escalation"]
        batcher = make_default_batcher()

        # 10s idle, then overhead-dominated load with jitter
        # At 1s window: 0.8 + 0.3 = 1.1s → overloaded
        cost = step_function_cost(
            step_time_s=10.0,
            before=idle_cost(),
            after=constant_overhead_cost(
                overhead_s=0.8,
                per_second_cost=0.3,
                jitter_fraction=0.1,
                rng=random.Random(123),
            ),
        )

        result = run_scenario(batcher, 120.0, cost)

        first_esc = result.first_escalation_time_s()
        assert first_esc is not None, "Batcher never escalated"
        time_to_escalate = first_esc - 10.0
        assert time_to_escalate < lim["max_time_to_first_escalation_s"], (
            f"Took {time_to_escalate:.1f}s to first escalate after step "
            f"(limit: {lim['max_time_to_first_escalation_s']}s)"
        )

    def test_limits_backlog(self):
        """Backlog during escalation must remain bounded.

        At 1s window: 0.6 + 0.6 = 1.2s (20% over budget).
        At 2s window: 0.6 + 1.2 = 1.8s (OK).
        """
        lim = LIMITS["step_function_backlog"]
        batcher = make_default_batcher()

        cost = step_function_cost(
            step_time_s=5.0,
            before=idle_cost(),
            after=constant_overhead_cost(
                overhead_s=0.6, per_second_cost=0.6
            ),
        )

        result = run_scenario(batcher, 120.0, cost)

        assert result.max_backlog_s < lim["max_backlog_s"], (
            f"Backlog reached {result.max_backlog_s:.1f}s "
            f"(limit: {lim['max_backlog_s']}s)"
        )
        assert result.final_backlog_s < lim["max_final_backlog_s"], (
            f"Residual backlog {result.final_backlog_s:.1f}s "
            f"(limit: {lim['max_final_backlog_s']}s)"
        )

    def test_severe_overload_reaches_adequate_level(self):
        """Under severe overload, the batcher must reach a high enough level.

        At 1s: 1.8 + 0.2 = 2.0s (2x overloaded).
        At 2s: 1.8 + 0.4 = 2.2s (1.1x overloaded).
        At 4s: 1.8 + 0.8 = 2.6s < 4s (OK).
        """
        lim = LIMITS["severe_step_function"]
        batcher = make_default_batcher()

        cost = step_function_cost(
            step_time_s=5.0,
            before=idle_cost(),
            after=constant_overhead_cost(
                overhead_s=1.8, per_second_cost=0.2
            ),
        )

        result = run_scenario(batcher, 120.0, cost)
        assert result.max_level >= lim["min_level_reached"], (
            f"Only reached level {result.max_level} "
            f"(need >= {lim['min_level_reached']})"
        )


class TestNoEscalationWhenNotNeeded:
    """The batcher must not escalate when the system keeps up."""

    def test_no_escalation_under_light_load(self):
        """Fast processing should never trigger escalation."""
        lim = LIMITS["light_load"]
        batcher = make_default_batcher()
        cost = constant_overhead_cost(
            overhead_s=0.1, per_second_cost=0.1
        )

        result = run_scenario(batcher, 60.0, cost)
        assert result.max_level <= lim["max_level"], (
            f"Escalated to level {result.max_level} under light load "
            f"(limit: {lim['max_level']})"
        )

    def test_no_escalation_with_gc_jitter(self):
        """Occasional GC/scheduling spikes should not cause escalation.

        Processing is fast on average (0.3s) but with significant jitter
        that occasionally exceeds the 1s window.
        """
        lim = LIMITS["gc_jitter"]
        batcher = make_default_batcher()
        cost = constant_overhead_cost(
            overhead_s=0.2,
            per_second_cost=0.1,
            jitter_fraction=0.5,
            rng=random.Random(999),
        )

        result = run_scenario(batcher, 120.0, cost)
        assert result.max_level <= lim["max_level"], (
            f"Escalated to level {result.max_level} from jitter alone "
            f"(limit: {lim['max_level']})"
        )


class TestNoOscillation:
    """The batcher must not oscillate between levels."""

    def test_no_oscillation_at_steady_load(self):
        """Constant load near the threshold should stabilize."""
        lim = LIMITS["steady_load_oscillation"]
        batcher = make_default_batcher()

        # Processing at ~90% of 1s window
        cost = constant_overhead_cost(
            overhead_s=0.5, per_second_cost=0.4
        )

        result = run_scenario(batcher, 120.0, cost)
        assert result.oscillation_count() <= lim["max_oscillations"], (
            f"Oscillated {result.oscillation_count()} times "
            f"(limit: {lim['max_oscillations']})"
        )

    def test_limited_oscillation_at_boundary(self):
        """Processing right at the window with jitter: bounded oscillation."""
        lim = LIMITS["boundary_oscillation"]
        batcher = make_default_batcher()

        # Mean processing = 1.0s = window, jitter +-10%
        cost = constant_overhead_cost(
            overhead_s=0.5,
            per_second_cost=0.5,
            jitter_fraction=0.1,
            rng=random.Random(42),
        )

        result = run_scenario(batcher, 180.0, cost)
        assert result.oscillation_count() <= lim["max_oscillations"], (
            f"Oscillated {result.oscillation_count()} times "
            f"(limit: {lim['max_oscillations']})"
        )


class TestCreepingOverload:
    """Load that gradually increases past processing capacity."""

    def test_eventually_escalates(self):
        """As cost ramps up, the batcher must escalate."""
        lim = LIMITS["creeping_overload"]
        batcher = make_default_batcher()

        # Ramp from 0.5s to 1.3s at 1s window over 60s
        cost = creeping_cost(
            overhead_s=0.3,
            per_second_cost_start=0.2,
            per_second_cost_end=1.0,
            ramp_duration_s=60.0,
        )

        result = run_scenario(batcher, 120.0, cost)
        assert result.max_level >= lim["min_level_reached"], (
            f"Only reached level {result.max_level} "
            f"(need >= {lim['min_level_reached']})"
        )

    def test_limits_backlog(self):
        """Backlog from creeping overload should remain bounded."""
        lim = LIMITS["creeping_overload_backlog"]
        batcher = make_default_batcher()

        cost = creeping_cost(
            overhead_s=0.3,
            per_second_cost_start=0.2,
            per_second_cost_end=1.0,
            ramp_duration_s=60.0,
        )

        result = run_scenario(batcher, 120.0, cost)
        assert result.max_backlog_s < lim["max_backlog_s"], (
            f"Backlog reached {result.max_backlog_s:.1f}s "
            f"(limit: {lim['max_backlog_s']}s)"
        )

    def test_mild_overload_does_not_over_escalate(self):
        """A slow creep to barely over 1x should not jump to max level.

        overhead=0.3, per_s ramps 0.5 -> 0.8 over 60s.
        At 1s window: 0.3 + 0.8 = 1.1s -> needs escalation.
        At 2s window: 0.3 + 0.8*2 = 1.9s < 2s -> stable at level 1.
        """
        lim = LIMITS["mild_creeping_overload"]
        batcher = make_default_batcher()

        cost = creeping_cost(
            overhead_s=0.3,
            per_second_cost_start=0.5,
            per_second_cost_end=0.8,
            ramp_duration_s=60.0,
        )

        result = run_scenario(batcher, 180.0, cost)
        assert result.max_level <= lim["max_level"], (
            f"Over-escalated to level {result.max_level} "
            f"(limit: {lim['max_level']})"
        )


class TestDeescalation:
    """The batcher must de-escalate when load subsides."""

    def test_deescalates_after_load_drops_to_idle(self):
        """After high load followed by idle, must return to level 0."""
        lim = LIMITS["deescalation_to_idle"]
        batcher = make_default_batcher()

        # 30s high load, then idle
        cost = step_function_cost(
            step_time_s=0.0,
            before=idle_cost(),
            after=step_function_cost(
                step_time_s=30.0,
                before=constant_overhead_cost(
                    overhead_s=0.8, per_second_cost=0.3
                ),
                after=idle_cost(),
            ),
        )

        result = run_scenario(batcher, 120.0, cost)
        assert result.final_level <= lim["max_final_level"], (
            f"Final level {result.final_level} "
            f"(limit: {lim['max_final_level']})"
        )

    def test_deescalates_after_step_down_to_light_load(self):
        """When load decreases to light (but non-zero), must de-escalate.

        This requires the batcher to de-escalate even when data is flowing
        continuously, not just when the system goes fully idle.
        """
        lim = LIMITS["deescalation_to_light_load"]
        batcher = make_default_batcher()

        # Heavy load for 40s, then light load
        cost = step_function_cost(
            step_time_s=0.0,
            before=idle_cost(),
            after=step_function_cost(
                step_time_s=40.0,
                before=constant_overhead_cost(
                    overhead_s=0.9, per_second_cost=0.05
                ),
                after=constant_overhead_cost(
                    overhead_s=0.1, per_second_cost=0.1
                ),
            ),
        )

        result = run_scenario(batcher, 180.0, cost)
        assert result.final_level <= lim["max_final_level"], (
            f"Final level {result.final_level} "
            f"(limit: {lim['max_final_level']})"
        )


class TestRealisticShutterScenario:
    """End-to-end shutter open/close simulation with noise."""

    def test_shutter_open_close_cycle(self):
        """Idle -> shutter open (high load) -> shutter close (idle).

        Must handle the full cycle: escalation, stable operation,
        de-escalation back to base.
        """
        lim = LIMITS["shutter_open_close"]
        batcher = make_default_batcher()

        rng = random.Random(42)
        cost = step_function_cost(
            step_time_s=10.0,
            before=idle_cost(),
            after=step_function_cost(
                step_time_s=70.0,
                before=constant_overhead_cost(
                    overhead_s=0.7,
                    per_second_cost=0.4,
                    jitter_fraction=0.15,
                    rng=rng,
                ),
                after=idle_cost(),
            ),
        )

        result = run_scenario(batcher, 180.0, cost)

        assert result.max_level >= lim["min_level_reached"], (
            f"Only reached level {result.max_level} during shutter open "
            f"(need >= {lim['min_level_reached']})"
        )
        assert result.final_level <= lim["max_final_level"], (
            f"Final level {result.final_level} after shutter close "
            f"(limit: {lim['max_final_level']})"
        )
        assert result.max_backlog_s < lim["max_backlog_s"], (
            f"Backlog reached {result.max_backlog_s:.1f}s "
            f"(limit: {lim['max_backlog_s']}s)"
        )

    def test_repeated_shutter_cycles(self):
        """Multiple on/off cycles should not cause runaway escalation."""
        lim = LIMITS["repeated_shutter_cycles"]
        batcher = make_default_batcher()

        rng = random.Random(42)
        high = constant_overhead_cost(
            overhead_s=0.7,
            per_second_cost=0.4,
            jitter_fraction=0.1,
            rng=rng,
        )
        low = idle_cost()

        # 20s on / 20s off cycles
        def cost(
            batch_window_s: float, wall_time_s: float
        ) -> float:
            cycle_pos = wall_time_s % 40.0
            if cycle_pos < 20.0:
                return high(batch_window_s, wall_time_s)
            return low(batch_window_s, wall_time_s)

        result = run_scenario(batcher, 200.0, cost)

        assert result.final_level <= lim["max_final_level"], (
            f"Stuck at level {result.final_level} after repeated cycles "
            f"(limit: {lim['max_final_level']})"
        )


class TestBacklogDraining:
    """Once the batcher escalates, accumulated backlog should drain."""

    def test_backlog_drains_after_escalation(self):
        """Sustained load triggering escalation should drain the backlog.

        At 1s: 0.6 + 0.6 = 1.2s (overloaded).
        At 2s: 0.6 + 1.2 = 1.8s (OK, surplus drains backlog).
        """
        lim = LIMITS["backlog_drains"]
        batcher = make_default_batcher()

        cost = constant_overhead_cost(
            overhead_s=0.6, per_second_cost=0.6
        )

        result = run_scenario(batcher, 120.0, cost)

        assert result.final_backlog_s < lim["max_final_backlog_s"], (
            f"Backlog not drained: {result.final_backlog_s:.1f}s "
            f"(limit: {lim['max_final_backlog_s']}s)"
        )

    def test_backlog_does_not_grow_indefinitely(self):
        """Even under sustained load, the backlog must peak and decrease.

        At 1s: 0.8 + 0.3 = 1.1s (overloaded).
        At 2s: 0.8 + 0.6 = 1.4s (OK).
        """
        batcher = make_default_batcher()

        cost = constant_overhead_cost(
            overhead_s=0.8, per_second_cost=0.3
        )

        result = run_scenario(batcher, 120.0, cost)

        peak_idx = max(
            range(len(result.cycles)),
            key=lambda i: result.cycles[i].backlog_s,
        )
        assert peak_idx < len(result.cycles) - 1, (
            "Backlog was still at peak at end of simulation"
        )


class TestProcessingTimeAwareness:
    """The batcher should use processing_time_s for faster decisions."""

    def test_fast_escalation_on_clear_overload(self):
        """When processing demonstrably exceeds the batch window,
        escalation should be fast."""
        lim = LIMITS["fast_escalation_clear_overload"]
        batcher = make_default_batcher()

        # Clear overload: 1.5x the window at every level
        cost = constant_overhead_cost(
            overhead_s=0.0, per_second_cost=1.5
        )

        result = run_scenario(batcher, 60.0, cost)

        first_esc = result.first_escalation_time_s()
        assert first_esc is not None, "Never escalated under overload"
        assert first_esc < lim["max_time_to_first_escalation_s"], (
            f"First escalation at {first_esc:.1f}s "
            f"(limit: {lim['max_time_to_first_escalation_s']}s)"
        )

    def test_no_escalation_when_processing_fits(self):
        """No escalation if processing completes within the window."""
        lim = LIMITS["no_escalation_when_fits"]
        batcher = make_default_batcher()

        cost = constant_overhead_cost(
            overhead_s=0.1, per_second_cost=0.3
        )

        result = run_scenario(batcher, 60.0, cost)
        assert result.max_level <= lim["max_level"], (
            f"Escalated to {result.max_level} despite fitting "
            f"(limit: {lim['max_level']})"
        )
