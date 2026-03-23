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

import pytest

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
    # -- Escalation reaches appropriate level for given severity ----------
    # Levels are half-steps: window = base * sqrt(2)^level.
    # Escalation jumps +2 (x2), de-escalation drops -1 (x1/sqrt(2)).
    #   level 0: 1.0s   level 3: 2.83s   level 6: 8.0s
    #   level 1: 1.41s  level 4: 4.0s
    #   level 2: 2.0s   level 5: 5.66s
    #
    # overhead_s=0.6, per_s=0.6 -> at 1s: 1.2, at 2s: 1.8 (OK at level 2)
    # Merged: level bounds + backlog bounds (same simulation).
    "moderate_overload_step": {
        "min_level": 2,
        "max_level": 2,
        "max_backlog_s": 1.0,
        "max_final_backlog_s": 0.5,
    },
    # overhead_s=0.8, per_s=0.3 -> at 1s: 1.1, at 1.41s: 1.22 (OK at level 1)
    # Merged: level bounds + stabilization + backlog-peaks (same simulation).
    "overhead_dominated_step": {
        "min_level": 1,
        "max_level": 2,
        "max_oscillations": 1,
        "min_peak_backlog_s": 0.1,
    },
    # overhead_s=1.8, per_s=0.2 -> needs level 3+ (2.83s window: 2.37s OK)
    "severity_severe": {
        "min_level": 3,
        "max_level": 5,
    },
    # overhead_s=0.5, per_s=1.5 -> overloaded at every level, must reach max.
    "severity_extreme": {
        "min_level": 6,
        "max_level": 6,
    },
    # -- Non-default base batch length ------------------------------------
    # overhead_s=1.2, per_s=0.6 -> at 2s: 2.4, at 4s: 3.6 (OK at level 2)
    "non_default_base": {
        "min_level": 2,
        "max_level": 2,
    },
    # -- No escalation when not needed ------------------------------------
    # Parameterized across utilization levels.
    "light_load_20pct": {"max_level": 0},
    "light_load_60pct": {"max_level": 0},
    "light_load_80pct": {"max_level": 0},
    "light_load_85pct": {"max_level": 0},
    "gc_jitter": {
        "max_level": 0,
    },
    # -- Steady overload --------------------------------------------------
    # overhead_s=0.6, per_s=0.6 -> constant overload from t=0.
    # Merged: oscillation + backlog draining (same simulation).
    "steady_moderate_overload": {
        "max_oscillations": 0,
        "min_level_reached": 1,
        "min_peak_backlog_s": 0.1,
        "max_final_backlog_s": 0.5,
    },
    # overhead_s=0.5, per_s=0.5, jitter=10% -> mean = window exactly.
    # Merged: oscillation bounds + sticky escalation (same simulation).
    "boundary_jitter": {
        "max_oscillations": 5,
        "min_level": 1,
        "min_final_level": 1,
    },
    # -- Creeping overload ------------------------------------------------
    "creeping_overload": {
        "min_level_reached": 4,
        "max_backlog_s": 3.5,
    },
    "mild_creeping_overload": {
        "min_level_reached": 1,
        "max_level": 2,
    },
    # -- De-escalation ----------------------------------------------------
    "deescalation_to_idle": {
        "min_level_during_load": 1,
        "max_final_level": 0,
    },
    "deescalation_to_light_load": {
        "min_level_during_load": 1,
        "max_final_level": 0,
    },
    "deescalation_moderate_load": {
        "min_level_during_load": 1,
        "max_final_level": 0,
    },
    "multi_level_deescalation": {
        "min_level_during_load": 3,
        "max_final_level": 0,
    },
    "partial_deescalation": {
        "min_level_during_load": 3,
        "max_final_level": 2,
    },
    # -- Realistic shutter ------------------------------------------------
    "shutter_open_close": {
        "min_level_reached": 1,
        "max_final_level": 0,
        "max_backlog_s": 2.0,
    },
    "repeated_shutter_cycles": {
        "min_level_reached": 1,
        "max_final_level": 0,
        "min_escalation_events": 4,
    },
    "severe_to_cosmic_background": {
        "min_level_during_load": 3,
        "max_final_level": 0,
    },
    # -- Processing-time awareness ----------------------------------------
    "fast_escalation_clear_overload": {
        "max_time_to_first_escalation_s": 4.0,
    },
    # -- Dead zone (70-100% utilization at escalated level) ---------------
    # Documents limitation: batcher cannot de-escalate when processing
    # fills the dead zone, even if a lower level would suffice.
    "dead_zone_stuck": {
        "min_level_during_load": 4,
        "min_final_level": 3,
    },
    # -- Time-gap batches (message_count=0) -------------------------------
    "time_gaps_during_escalation": {
        "min_level_reached": 1,
    },
    "time_gaps_during_deescalation": {
        "max_final_level": 0,
    },
}


# ---------------------------------------------------------------------------
# Simulation infrastructure
# ---------------------------------------------------------------------------


class ProcessingCostFn(Protocol):
    """Returns the processing time (seconds) for a batch of given window."""

    def __call__(self, batch_window_s: float, wall_time_s: float) -> float: ...


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
        return sum(c.processing_time_s for c in self.cycles if c.level == level)

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
        directions = [1 if new > old else -1 for _, old, new in changes]
        return sum(
            1 for i in range(1, len(directions)) if directions[i] != directions[i - 1]
        )

    def cycles_after(self, wall_time_s: float) -> list[CycleRecord]:
        """All cycles with wall_time_s > the given time."""
        return [c for c in self.cycles if c.wall_time_s > wall_time_s]


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
            result.cycles.append(
                CycleRecord(
                    wall_time_s=clock.now,
                    batch_window_s=window,
                    processing_time_s=0.0,
                    backlog_s=backlog_s,
                    level=level,
                )
            )
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
                    batcher.report_batch(None, processing_time_s=0.0)

        batcher.report_batch(100, processing_time_s=processing_time)

        level = _get_level(batcher)
        result.cycles.append(
            CycleRecord(
                wall_time_s=clock.now,
                batch_window_s=window,
                processing_time_s=processing_time,
                backlog_s=backlog_s,
                level=level,
            )
        )

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
        frac = min(1.0, elapsed / ramp_duration_s) if ramp_duration_s > 0 else 1.0
        rate_range = per_second_cost_end - per_second_cost_start
        per_s = per_second_cost_start + frac * rate_range
        base = overhead_s + per_s * batch_window_s
        if jitter_fraction > 0:
            jitter = _rng.gauss(0, jitter_fraction * base)
            base = max(0.01, base + jitter)
        return base

    return cost


def cyclic_cost(
    on_duration_s: float,
    off_duration_s: float,
    on_cost: ProcessingCostFn,
    off_cost: ProcessingCostFn,
) -> ProcessingCostFn:
    """Alternating on/off cost function with configurable duty cycle."""
    period = on_duration_s + off_duration_s

    def cost(batch_window_s: float, wall_time_s: float) -> float:
        cycle_pos = wall_time_s % period
        if cycle_pos < on_duration_s:
            return on_cost(batch_window_s, wall_time_s)
        return off_cost(batch_window_s, wall_time_s)

    return cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_default_batcher(
    **kwargs,
) -> tuple[AdaptiveMessageBatcher, FakeClock]:
    clock = kwargs.pop("clock", None) or FakeClock()
    defaults = {"base_batch_length_s": 1.0, "max_level": 3, "clock": clock}
    defaults.update(kwargs)
    return AdaptiveMessageBatcher(**defaults), clock


def run_scenario(
    batcher: AdaptiveMessageBatcher,
    duration_s: float,
    cost_fn: ProcessingCostFn,
    clock: FakeClock,
) -> SimulationResult:
    return simulate(batcher, duration_s, cost_fn, clock)


# ===========================================================================
# Scenario tests
# ===========================================================================


class TestStepFunctionEscalation:
    """Shutter-open scenario: sudden jump from idle to high load."""

    def test_escalates_within_bounded_time(self):
        """After a step increase in load, the batcher must escalate quickly."""
        lim = LIMITS["step_function_escalation"]
        batcher, clock = make_default_batcher()

        # 10s idle, then overhead-dominated load with jitter
        # At 1s window: 0.8 + 0.3 = 1.1s -> overloaded
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

        result = run_scenario(batcher, 120.0, cost, clock)

        first_esc = result.first_escalation_time_s()
        assert first_esc is not None, "Batcher never escalated"
        time_to_escalate = first_esc - 10.0
        assert time_to_escalate < lim["max_time_to_first_escalation_s"], (
            f"Took {time_to_escalate:.1f}s to first escalate after step "
            f"(limit: {lim['max_time_to_first_escalation_s']}s)"
        )

    def test_moderate_overload(self):
        """Moderate overload after idle: correct level, bounded backlog.

        At 1s window: 0.6 + 0.6 = 1.2s (20% over budget, escalates).
        At 2s window: 0.6 + 1.2 = 1.8s (90%, dead zone — stable at level 2).
        """
        lim = LIMITS["moderate_overload_step"]
        batcher, clock = make_default_batcher()

        cost = step_function_cost(
            step_time_s=5.0,
            before=idle_cost(),
            after=constant_overhead_cost(overhead_s=0.6, per_second_cost=0.6),
        )

        result = run_scenario(batcher, 120.0, cost, clock)
        assert result.max_level >= lim["min_level"], (
            f"Only reached level {result.max_level} (need >= {lim['min_level']})"
        )
        assert result.max_level <= lim["max_level"], (
            f"Over-escalated to level {result.max_level} (limit: {lim['max_level']})"
        )
        assert result.max_backlog_s < lim["max_backlog_s"], (
            f"Backlog reached {result.max_backlog_s:.1f}s "
            f"(limit: {lim['max_backlog_s']}s)"
        )
        assert result.final_backlog_s < lim["max_final_backlog_s"], (
            f"Residual backlog {result.final_backlog_s:.1f}s "
            f"(limit: {lim['max_final_backlog_s']}s)"
        )

    def test_overhead_dominated_overload(self):
        """Overhead-dominated overload: correct level, stabilization, backlog peak.

        At 1s window: 0.8 + 0.3 = 1.1s (overloaded, escalates).
        At 1.41s window: 0.8 + 0.42 = 1.22s (87%, dead zone — stable).
        After escalation the backlog must peak and then decrease.
        """
        lim = LIMITS["overhead_dominated_step"]
        batcher, clock = make_default_batcher()

        cost = step_function_cost(
            step_time_s=5.0,
            before=idle_cost(),
            after=constant_overhead_cost(overhead_s=0.8, per_second_cost=0.3),
        )

        result = run_scenario(batcher, 120.0, cost, clock)
        assert result.max_level >= lim["min_level"], (
            f"Only reached level {result.max_level} (need >= {lim['min_level']})"
        )
        assert result.max_level <= lim["max_level"], (
            f"Over-escalated to level {result.max_level} (limit: {lim['max_level']})"
        )
        assert result.oscillation_count() <= lim["max_oscillations"], (
            f"Oscillated {result.oscillation_count()} times "
            f"(limit: {lim['max_oscillations']})"
        )
        # After the initial transient, the level should be stable.
        late_cycles = result.cycles_after(60.0)
        assert late_cycles, "Simulation too short for stabilization check"
        late_levels = {c.level for c in late_cycles}
        assert len(late_levels) == 1, (
            f"Not stabilized: levels {sorted(late_levels)} observed "
            f"in second half of simulation"
        )
        assert result.max_backlog_s >= lim["min_peak_backlog_s"], (
            f"Precondition: meaningful backlog must build up "
            f"(peak was {result.max_backlog_s:.2f}s)"
        )
        peak_idx = max(
            range(len(result.cycles)),
            key=lambda i: result.cycles[i].backlog_s,
        )
        assert peak_idx < len(result.cycles) - 1, (
            "Backlog was still at peak at end of simulation"
        )

    @pytest.mark.parametrize(
        ("overhead_s", "per_second_cost", "limits_key"),
        [
            pytest.param(
                1.8,
                0.2,
                "severity_severe",
                id="severe: overhead=1.8 per_s=0.2",
            ),
            pytest.param(
                0.5,
                1.5,
                "severity_extreme",
                id="extreme: overhead=0.5 per_s=1.5",
            ),
        ],
    )
    def test_reaches_appropriate_level_for_severity(
        self, overhead_s, per_second_cost, limits_key
    ):
        """The batcher must reach an appropriate level for the overload severity,
        without over-escalating.

        The limits table specifies both a minimum and maximum level for each
        severity, ensuring the response is proportional.
        """
        lim = LIMITS[limits_key]
        batcher, clock = make_default_batcher()

        cost = step_function_cost(
            step_time_s=5.0,
            before=idle_cost(),
            after=constant_overhead_cost(
                overhead_s=overhead_s, per_second_cost=per_second_cost
            ),
        )

        result = run_scenario(batcher, 120.0, cost, clock)
        assert result.max_level >= lim["min_level"], (
            f"Only reached level {result.max_level} (need >= {lim['min_level']})"
        )
        assert result.max_level <= lim["max_level"], (
            f"Over-escalated to level {result.max_level} (limit: {lim['max_level']})"
        )


class TestNonDefaultBaseBatchLength:
    """Verify scaling with a non-default base batch length."""

    def test_escalation_with_doubled_base(self):
        """With base=2.0, the level grid shifts: level 0 = 2s, level 2 = 4s.

        The batcher must scale correctly — a bug that hardcodes sqrt(2)^level
        without multiplying by the base would produce wrong batch windows.

        Level 0 (2.0s): 1.2 + 1.2 = 2.4s (overloaded).
        Level 2 (4.0s): 1.2 + 2.4 = 3.6s (90%, dead zone — stable).
        """
        lim = LIMITS["non_default_base"]
        batcher, clock = make_default_batcher(base_batch_length_s=2.0)

        cost = step_function_cost(
            step_time_s=5.0,
            before=idle_cost(),
            after=constant_overhead_cost(overhead_s=1.2, per_second_cost=0.6),
        )

        result = run_scenario(batcher, 120.0, cost, clock)
        assert result.max_level >= lim["min_level"], (
            f"Only reached level {result.max_level} (need >= {lim['min_level']})"
        )
        assert result.max_level <= lim["max_level"], (
            f"Over-escalated to level {result.max_level} (limit: {lim['max_level']})"
        )


class TestNoEscalationWhenNotNeeded:
    """The batcher must not escalate when the system keeps up."""

    @pytest.mark.parametrize(
        ("overhead_s", "per_second_cost", "limits_key"),
        [
            pytest.param(0.1, 0.1, "light_load_20pct", id="20% utilization"),
            pytest.param(0.3, 0.3, "light_load_60pct", id="60% utilization"),
            pytest.param(0.4, 0.4, "light_load_80pct", id="80% utilization"),
            pytest.param(0.3, 0.55, "light_load_85pct", id="85% utilization"),
        ],
    )
    def test_no_escalation_under_light_load(
        self, overhead_s, per_second_cost, limits_key
    ):
        """Processing that fits within the window should never trigger escalation,
        even at high utilization.
        """
        lim = LIMITS[limits_key]
        batcher, clock = make_default_batcher()
        cost = constant_overhead_cost(
            overhead_s=overhead_s, per_second_cost=per_second_cost
        )

        result = run_scenario(batcher, 60.0, cost, clock)
        assert result.max_level <= lim["max_level"], (
            f"Escalated to level {result.max_level} at "
            f"{overhead_s + per_second_cost:.0%} utilization "
            f"(limit: {lim['max_level']})"
        )

    @pytest.mark.parametrize(
        "seed",
        [pytest.param(s, id=f"seed={s}") for s in (42, 999, 12345)],
    )
    def test_no_escalation_with_gc_jitter(self, seed):
        """Occasional GC/scheduling spikes should not cause escalation.

        Processing is fast on average (0.3s) but with high jitter
        (std = 1.2 * mean = 0.36s) that regularly sends individual batches
        into the dead zone (75-100% of window) and occasionally past the
        window entirely (~4 overloaded cycles per 120s run).

        The batcher must tolerate these isolated spikes because its
        escalation heuristic requires *consecutive* overloaded batches.
        Tested with multiple RNG seeds to avoid seed-dependent false
        confidence.
        """
        lim = LIMITS["gc_jitter"]
        batcher, clock = make_default_batcher()
        cost = constant_overhead_cost(
            overhead_s=0.2,
            per_second_cost=0.1,
            jitter_fraction=1.2,
            rng=random.Random(seed),
        )

        result = run_scenario(batcher, 120.0, cost, clock)
        assert result.max_level <= lim["max_level"], (
            f"Escalated to level {result.max_level} from jitter alone "
            f"(seed={seed}, limit: {lim['max_level']})"
        )


class TestSteadyOverload:
    """Constant overload from t=0: escalation, stabilization, backlog draining."""

    def test_moderate_overload_stabilizes_and_drains(self):
        """Constant 20% overload: must escalate, not oscillate, and drain backlog.

        Level 0 (1.0s): 0.6 + 0.6 = 1.2s (overloaded, escalates).
        Level 2 (2.0s): 0.6 + 1.2 = 1.8s (90%, dead zone — stable).
        Surplus at level 2 drains the backlog accumulated during escalation.
        """
        lim = LIMITS["steady_moderate_overload"]
        batcher, clock = make_default_batcher()

        cost = constant_overhead_cost(overhead_s=0.6, per_second_cost=0.6)

        result = run_scenario(batcher, 120.0, cost, clock)
        assert result.max_level >= lim["min_level_reached"], (
            f"Precondition: load must trigger escalation "
            f"(reached level {result.max_level}, "
            f"need >= {lim['min_level_reached']})"
        )
        assert result.oscillation_count() <= lim["max_oscillations"], (
            f"Oscillated {result.oscillation_count()} times "
            f"(limit: {lim['max_oscillations']})"
        )
        assert result.max_backlog_s >= lim["min_peak_backlog_s"], (
            f"Precondition: meaningful backlog must build up "
            f"(peak was {result.max_backlog_s:.2f}s, "
            f"need >= {lim['min_peak_backlog_s']}s)"
        )
        assert result.final_backlog_s < lim["max_final_backlog_s"], (
            f"Backlog not drained: {result.final_backlog_s:.1f}s "
            f"(limit: {lim['max_final_backlog_s']}s)"
        )

    def test_boundary_jitter_escalates_and_sticks(self):
        """Mean processing = window with 10% jitter: bounded oscillation,
        but escalation becomes permanent due to the dead zone.

        At level 0 (1s window): 0.5 + 0.5 = 1.0s mean, jitter +/-10%.
            ~50% of cycles are overloaded (processing > 1.0).
            P(2 consecutive overloaded) ~ 25%, so escalation is very likely.

        At level 2 (2s window): 0.5 + 1.0 = 1.5s mean (75% utilization).
            At the dead-zone boundary (>= 75%), so de-escalation never triggers.
            Documents limitation: once escalated, stays stuck due to dead zone.
        """
        lim = LIMITS["boundary_jitter"]
        batcher, clock = make_default_batcher()

        cost = constant_overhead_cost(
            overhead_s=0.5,
            per_second_cost=0.5,
            jitter_fraction=0.1,
            rng=random.Random(42),
        )

        result = run_scenario(batcher, 180.0, cost, clock)
        assert result.oscillation_count() <= lim["max_oscillations"], (
            f"Oscillated {result.oscillation_count()} times "
            f"(limit: {lim['max_oscillations']})"
        )
        assert result.max_level >= lim["min_level"], (
            f"Expected escalation from boundary jitter "
            f"(reached level {result.max_level})"
        )
        assert result.final_level >= lim["min_final_level"], (
            f"Expected to stay at level {lim['min_final_level']}+ "
            f"(dead zone prevents de-escalation)"
        )


class TestCreepingOverload:
    """Load that gradually increases past processing capacity."""

    def test_eventually_escalates_and_limits_backlog(self):
        """As cost ramps up, the batcher must escalate and keep backlog bounded.

        Ramp from 0.5s to 1.3s at 1s window over 60s.
        """
        lim = LIMITS["creeping_overload"]
        batcher, clock = make_default_batcher()

        cost = creeping_cost(
            overhead_s=0.3,
            per_second_cost_start=0.2,
            per_second_cost_end=1.0,
            ramp_duration_s=60.0,
        )

        result = run_scenario(batcher, 120.0, cost, clock)
        assert result.max_level >= lim["min_level_reached"], (
            f"Only reached level {result.max_level} "
            f"(need >= {lim['min_level_reached']})"
        )
        assert result.max_backlog_s < lim["max_backlog_s"], (
            f"Backlog reached {result.max_backlog_s:.1f}s "
            f"(limit: {lim['max_backlog_s']}s)"
        )

    def test_mild_overload_does_not_over_escalate(self):
        """A slow creep to barely over 1x should escalate but not beyond level 2.

        overhead=0.3, per_s ramps 0.5 -> 0.8 over 60s.
        Level 0 (1.0s): 0.3 + 0.8 = 1.1s (overloaded).
        Level 2 (2.0s): 0.3 + 1.6 = 1.9s (95%, dead zone — stable).
        """
        lim = LIMITS["mild_creeping_overload"]
        batcher, clock = make_default_batcher()

        cost = creeping_cost(
            overhead_s=0.3,
            per_second_cost_start=0.5,
            per_second_cost_end=0.8,
            ramp_duration_s=60.0,
        )

        result = run_scenario(batcher, 180.0, cost, clock)
        assert result.max_level >= lim["min_level_reached"], (
            f"Only reached level {result.max_level} — mild overload should "
            f"still trigger escalation (need >= {lim['min_level_reached']})"
        )
        assert result.max_level <= lim["max_level"], (
            f"Over-escalated to level {result.max_level} (limit: {lim['max_level']})"
        )


class TestDeescalation:
    """The batcher must de-escalate when load subsides."""

    @pytest.mark.parametrize(
        (
            "heavy_duration_s",
            "after_overhead",
            "after_per_s",
            "duration_s",
            "limits_key",
        ),
        [
            pytest.param(
                30.0,
                None,
                None,
                120.0,
                "deescalation_to_idle",
                id="heavy→idle",
            ),
            pytest.param(
                40.0,
                0.1,
                0.1,
                180.0,
                "deescalation_to_light_load",
                id="heavy→light (0.2s at 1s window)",
            ),
            pytest.param(
                40.0,
                0.3,
                0.3,
                180.0,
                "deescalation_moderate_load",
                id="heavy→moderate (0.6s at 1s window)",
            ),
        ],
    )
    def test_deescalates_when_load_drops(
        self, heavy_duration_s, after_overhead, after_per_s, duration_s, limits_key
    ):
        """After overload (0.8 + 0.3 = 1.1s at 1s window), load drops.
        The batcher must de-escalate back to level 0 regardless of whether
        the lighter phase is idle, light, or moderate — as long as processing
        fits within the base window.
        """
        lim = LIMITS[limits_key]
        batcher, clock = make_default_batcher()

        if after_overhead is None:
            after = idle_cost()
        else:
            after = constant_overhead_cost(
                overhead_s=after_overhead, per_second_cost=after_per_s
            )

        cost = step_function_cost(
            step_time_s=0.0,
            before=idle_cost(),
            after=step_function_cost(
                step_time_s=heavy_duration_s,
                before=constant_overhead_cost(overhead_s=0.8, per_second_cost=0.3),
                after=after,
            ),
        )

        result = run_scenario(batcher, duration_s, cost, clock)
        assert result.max_level >= lim["min_level_during_load"], (
            f"Precondition: batcher must have escalated during heavy-load "
            f"phase (reached level {result.max_level}, "
            f"need >= {lim['min_level_during_load']})"
        )
        assert result.final_level <= lim["max_final_level"], (
            f"Final level {result.final_level} (limit: {lim['max_final_level']})"
        )

    def test_multi_level_deescalation(self):
        """After reaching level 3+, a drop to light load should step back
        through all levels to 0.

        Heavy phase:
            Level 0 (1.0s): 1.8 + 0.2 = 2.0s (overloaded).
            Level 2 (2.0s): 1.8 + 0.4 = 2.2s (overloaded).
            Level 4 (4.0s): 1.8 + 0.8 = 2.6s (65%, underloaded → settles).
        Light phase at any level: 0.1 + 0.1*w = well within any window.
        """
        lim = LIMITS["multi_level_deescalation"]
        batcher, clock = make_default_batcher()

        cost = step_function_cost(
            step_time_s=0.0,
            before=idle_cost(),
            after=step_function_cost(
                step_time_s=60.0,
                before=constant_overhead_cost(overhead_s=1.8, per_second_cost=0.2),
                after=constant_overhead_cost(overhead_s=0.1, per_second_cost=0.1),
            ),
        )

        result = run_scenario(batcher, 240.0, cost, clock)
        assert result.max_level >= lim["min_level_during_load"], (
            f"Precondition: must reach level {lim['min_level_during_load']}+ "
            f"during heavy phase (reached {result.max_level})"
        )
        assert result.final_level <= lim["max_final_level"], (
            f"Final level {result.final_level} after load dropped "
            f"(limit: {lim['max_final_level']})"
        )

    def test_partial_deescalation(self):
        """Load drops from severe to moderate: should partially de-escalate,
        not stay stuck at the peak level.

        Severe phase (escalates to level 4):
            Level 0 (1.0s): 1.8 + 0.2 = 2.0s (overloaded).
            Level 2 (2.0s): 1.8 + 0.4 = 2.2s (overloaded).
            Level 4 (4.0s): 1.8 + 0.8 = 2.6s (65%, underloaded).
            De-escalates to level 3 (2.83s): 1.8 + 0.57 = 2.37s (84%, dead zone).

        Moderate phase (de-escalates from level 3 to level 2):
            Level 3 (2.83s): 0.6 + 1.41 = 2.01s (71%, underloaded).
            Level 2 (2.0s): 0.6 + 1.0 = 1.6s (80%, dead zone — stuck).
        """
        lim = LIMITS["partial_deescalation"]
        batcher, clock = make_default_batcher()

        cost = step_function_cost(
            step_time_s=0.0,
            before=idle_cost(),
            after=step_function_cost(
                step_time_s=60.0,
                before=constant_overhead_cost(overhead_s=1.8, per_second_cost=0.2),
                after=constant_overhead_cost(overhead_s=0.6, per_second_cost=0.5),
            ),
        )

        result = run_scenario(batcher, 240.0, cost, clock)
        assert result.max_level >= lim["min_level_during_load"], (
            f"Precondition: must reach level {lim['min_level_during_load']}+ "
            f"during severe phase (reached {result.max_level})"
        )
        assert result.final_level <= lim["max_final_level"], (
            f"Final level {result.final_level} after load reduced "
            f"(limit: {lim['max_final_level']})"
        )


class TestRealisticShutterScenario:
    """End-to-end shutter open/close simulation with noise."""

    def test_shutter_open_close_cycle(self):
        """Cosmic background -> shutter open (high load) -> shutter close
        (cosmic background).

        Must handle the full cycle: escalation, stable operation,
        de-escalation back to base.  The shutter-closed phase is not idle:
        cosmic background produces a continuous stream of ev44 messages
        with very few events, resulting in overhead-dominated processing.
        """
        lim = LIMITS["shutter_open_close"]
        batcher, clock = make_default_batcher()

        rng = random.Random(42)
        cosmic = constant_overhead_cost(overhead_s=0.2, per_second_cost=0.01)
        cost = step_function_cost(
            step_time_s=10.0,
            before=cosmic,
            after=step_function_cost(
                step_time_s=70.0,
                before=constant_overhead_cost(
                    overhead_s=0.7,
                    per_second_cost=0.4,
                    jitter_fraction=0.15,
                    rng=rng,
                ),
                after=cosmic,
            ),
        )

        result = run_scenario(batcher, 180.0, cost, clock)

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
        """Multiple on/off cycles should not cause runaway escalation.

        Each on-phase must trigger escalation, and each off-phase (cosmic
        background) must allow de-escalation back to base.
        """
        lim = LIMITS["repeated_shutter_cycles"]
        batcher, clock = make_default_batcher()

        rng = random.Random(42)
        high = constant_overhead_cost(
            overhead_s=0.7,
            per_second_cost=0.4,
            jitter_fraction=0.1,
            rng=rng,
        )
        cosmic = constant_overhead_cost(overhead_s=0.2, per_second_cost=0.01)

        cost = cyclic_cost(
            on_duration_s=20.0,
            off_duration_s=20.0,
            on_cost=high,
            off_cost=cosmic,
        )

        result = run_scenario(batcher, 200.0, cost, clock)

        assert result.max_level >= lim["min_level_reached"], (
            f"Precondition: at least one on-phase must trigger escalation "
            f"(reached level {result.max_level}, "
            f"need >= {lim['min_level_reached']})"
        )
        assert result.final_level <= lim["max_final_level"], (
            f"Stuck at level {result.final_level} after repeated cycles "
            f"(limit: {lim['max_final_level']})"
        )
        escalation_events = sum(
            1 for _, old, new in result.level_changes() if new > old
        )
        assert escalation_events >= lim["min_escalation_events"], (
            f"Only {escalation_events} escalation event(s) — expected the batcher "
            f"to re-escalate during subsequent on-phases "
            f"(need >= {lim['min_escalation_events']})"
        )

    def test_severe_overload_to_cosmic_background(self):
        """After severe overload reaching level 3+, shutter close drops load
        to cosmic background.  Must de-escalate through all levels back to 0.

        This is the most operationally important de-escalation path: ev44
        messages keep flowing with very few events (cosmic rays), so the
        system is never truly idle.  Wall-clock idle de-escalation does not
        apply; the batcher must de-escalate via the underload counter.

        Severe phase (overhead-dominated):
            Level 0 (1.0s): 2.0 + 0.3 = 2.3s (overloaded).
            Level 2 (2.0s): 2.0 + 0.6 = 2.6s (overloaded).
            Level 4 (4.0s): 2.0 + 1.2 = 3.2s (80%, dead zone — stable).

        Cosmic background phase (overhead-dominated, near-zero data cost):
            Level 4 (4.0s): 0.2 + 0.04 = 0.24s (6% utilization).
            Level 2 (2.0s): 0.2 + 0.02 = 0.22s (11% utilization).
            Level 0 (1.0s): 0.2 + 0.01 = 0.21s (21% utilization).
            All levels are well below the 75% headroom threshold.
        """
        lim = LIMITS["severe_to_cosmic_background"]
        batcher, clock = make_default_batcher()

        cost = step_function_cost(
            step_time_s=0.0,
            before=idle_cost(),
            after=step_function_cost(
                step_time_s=60.0,
                before=constant_overhead_cost(overhead_s=2.0, per_second_cost=0.3),
                after=constant_overhead_cost(overhead_s=0.2, per_second_cost=0.01),
            ),
        )

        result = run_scenario(batcher, 240.0, cost, clock)
        assert result.max_level >= lim["min_level_during_load"], (
            f"Precondition: must reach level {lim['min_level_during_load']}+ "
            f"during severe phase (reached {result.max_level})"
        )
        assert result.final_level <= lim["max_final_level"], (
            f"Final level {result.final_level} after shutter close to cosmic "
            f"background (limit: {lim['max_final_level']})"
        )


class TestProcessingTimeAwareness:
    """The batcher should use processing_time_s for faster decisions."""

    def test_fast_escalation_on_clear_overload(self):
        """When processing demonstrably exceeds the batch window,
        escalation should be fast."""
        lim = LIMITS["fast_escalation_clear_overload"]
        batcher, clock = make_default_batcher()

        # Clear overload: 1.5x the window at every level
        cost = constant_overhead_cost(overhead_s=0.0, per_second_cost=1.5)

        result = run_scenario(batcher, 60.0, cost, clock)

        first_esc = result.first_escalation_time_s()
        assert first_esc is not None, "Never escalated under overload"
        assert first_esc < lim["max_time_to_first_escalation_s"], (
            f"First escalation at {first_esc:.1f}s "
            f"(limit: {lim['max_time_to_first_escalation_s']}s)"
        )


class TestDeescalationDeadZone:
    """The 75-100% utilization dead zone where de-escalation cannot trigger.

    When processing fills 75-100% of the escalated window, it falls in the
    "in between" zone: not overloaded (processing < window) and not
    underloaded (processing >= 0.75 * window).  Both consecutive counters
    are reset every cycle, so neither escalation nor de-escalation can
    trigger — even if a lower level would handle the load fine.
    """

    def test_stuck_in_dead_zone_after_load_drop(self):
        """After severe overload, a moderate load that lands in the dead zone
        at the escalated level keeps the batcher stuck, even though a lower
        level would work.

        Severe phase (reaches level 4):
            Level 0 (1.0s): 2.0 + 0.3 = 2.3s (overloaded).
            Level 2 (2.0s): 2.0 + 0.6 = 2.6s (overloaded).
            Level 4 (4.0s): 2.0 + 1.2 = 3.2s (80%, dead zone).

        Moderate phase (de-escalates from level 4 to level 3, then stuck):
            Level 4 (4.0s): 0.5 + 2.4 = 2.9s (72.5%, underloaded < 75%).
            Level 3 (2.83s): 0.5 + 1.7 = 2.2s (78%, dead zone — stuck).
            Level 2 (2.0s): 0.5 + 1.2 = 1.7s (would fit at 85%).
            Level 0 (1.0s): 0.5 + 0.6 = 1.1s (would be overloaded).
        """
        lim = LIMITS["dead_zone_stuck"]
        batcher, clock = make_default_batcher()

        cost = step_function_cost(
            step_time_s=0.0,
            before=idle_cost(),
            after=step_function_cost(
                step_time_s=60.0,
                before=constant_overhead_cost(overhead_s=2.0, per_second_cost=0.3),
                after=constant_overhead_cost(overhead_s=0.5, per_second_cost=0.6),
            ),
        )

        result = run_scenario(batcher, 240.0, cost, clock)

        assert result.max_level >= lim["min_level_during_load"], (
            f"Precondition: must reach level {lim['min_level_during_load']}+ "
            f"during severe phase (reached {result.max_level})"
        )
        # Documents the limitation: batcher stays at level 2 despite level 1
        # being sufficient.  If the strategy is improved to probe lower levels,
        # this assertion should change to max_final_level: 1.
        assert result.final_level >= lim["min_final_level"], (
            f"Final level {result.final_level} — expected to stay stuck "
            f"at level {lim['min_final_level']}+ (dead zone)"
        )


class TestTimeGapBatches:
    """Time-gap batches (message_count=0) should not disrupt adaptive behavior.

    The ``SimpleMessageBatcher`` can return empty batches when there is a
    time gap in the data stream.  The ``AdaptiveMessageBatcher`` treats
    these as a no-op, which means they should not interfere with ongoing
    escalation or de-escalation.
    """

    def test_time_gaps_do_not_disrupt_escalation(self):
        """Interleaving empty (time-gap) batches with overloaded batches
        should not prevent escalation.

        Uses a cost model that alternates between real overloaded batches
        and time gaps (processing_time=0 reported as message_count=0).
        """
        lim = LIMITS["time_gaps_during_escalation"]
        batcher, clock = make_default_batcher()

        for _ in range(20):
            # Overloaded real batch
            clock.advance(1.5)
            batcher.report_batch(100, processing_time_s=1.5)
            # Time-gap empty batch (should be a no-op)
            batcher.report_batch(0)

        assert batcher.state.level >= lim["min_level_reached"], (
            f"Time gaps prevented escalation: only reached level "
            f"{batcher.state.level} (need >= {lim['min_level_reached']})"
        )

    def test_time_gaps_do_not_disrupt_deescalation(self):
        """Interleaving empty (time-gap) batches with underloaded batches
        should not prevent de-escalation.
        """
        lim = LIMITS["time_gaps_during_deescalation"]
        batcher, clock = make_default_batcher()

        # Escalate to level 1
        for _ in range(3):
            window = batcher.batch_length_s
            clock.advance(window * 1.5)
            batcher.report_batch(100, processing_time_s=window * 1.5)
        assert batcher.state.level >= 1, "Precondition: must escalate"

        # Underloaded batches interleaved with time gaps
        for _ in range(20):
            window = batcher.batch_length_s
            processing = window * 0.3
            clock.advance(processing)
            batcher.report_batch(100, processing_time_s=processing)
            # Time-gap empty batch
            batcher.report_batch(0)

        assert batcher.state.level <= lim["max_final_level"], (
            f"Time gaps prevented de-escalation: stuck at level "
            f"{batcher.state.level} (limit: {lim['max_final_level']})"
        )
