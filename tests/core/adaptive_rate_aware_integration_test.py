# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration tests for AdaptiveMessageBatcher(inner=RateAwareMessageBatcher).

Covers the wiring between the adaptive wrapper and a rate-aware inner: that
escalation propagates through ``set_batch_length`` to the inner, that the next
batch window reflects the new length, and that messages round-trip through
escalate/de-escalate cycles without loss.
"""

from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.core.message_batcher import (
    DEESCALATION_UNDERLOAD_THRESHOLD,
    ESCALATION_OVERLOAD_THRESHOLD,
    AdaptiveMessageBatcher,
)
from ess.livedata.core.rate_aware_batcher import (
    MIN_DIFFS_FOR_GATE,
    RateAwareMessageBatcher,
)
from ess.livedata.core.timestamp import Timestamp

DETECTOR = StreamId(kind=StreamKind.DETECTOR_EVENTS, name="det")
_HWM = StreamId(kind=StreamKind.LOG, name="_hwm")

_CONVERGENCE_TIMEOUT_FACTOR = 0.8


def _ts(seconds: float) -> Timestamp:
    return Timestamp.from_ns(int(seconds * 1e9))


def _msg(t: float, stream: StreamId = DETECTOR) -> Message[str]:
    return Message(timestamp=_ts(t), stream=stream, value="")


def _hwm_trigger(t: float) -> Message[str]:
    return Message(timestamp=_ts(t), stream=_HWM, value="")


def _msgs_at(rate_hz: float, start: float, duration: float) -> list[Message[str]]:
    count = round(rate_hz * duration)
    period = 1.0 / rate_hz
    return [_msg(start + i * period) for i in range(count)]


def _rate_aware_inner_factory(batch_length_s: float) -> RateAwareMessageBatcher:
    """Inner factory with a short timeout so test warmup converges quickly."""
    return RateAwareMessageBatcher(
        batch_length_s=batch_length_s,
        timeout_s=batch_length_s * _CONVERGENCE_TIMEOUT_FACTOR,
    )


def _make_converged(
    rate_hz: float = 14.0,
    max_level: int = 2,
) -> tuple[AdaptiveMessageBatcher, float]:
    """Build an Adaptive/RateAware stack and converge the inner estimator.

    Returns ``(batcher, t_next)`` where ``t_next`` is the start time of the
    currently active (empty) batch.
    """
    batcher = AdaptiveMessageBatcher(
        base_batch_length_s=1.0,
        max_level=max_level,
        inner_factory=_rate_aware_inner_factory,
    )

    initial = _msgs_at(rate_hz, start=0.0, duration=1.0)
    batcher.batch(initial)
    batch_start = max(m.timestamp for m in initial).to_ns() / 1e9

    warmup = MIN_DIFFS_FOR_GATE + 1
    for i in range(warmup):
        t0 = batch_start + i * 1.0
        batch_msgs = _msgs_at(rate_hz, start=t0, duration=1.0)
        batch_msgs.append(_hwm_trigger(t0 + _CONVERGENCE_TIMEOUT_FACTOR + 0.01))
        result = batcher.batch(batch_msgs)
        assert result is not None, f"Warmup batch {i} should close"
        batcher.report_batch(len(result.messages), processing_time_s=0.01)

    return batcher, batch_start + warmup * 1.0


class TestAdaptiveWithRateAwareInner:
    def test_escalation_propagates_to_inner_batch_window(self):
        """Overload reports -> new batch window contains the expected count."""
        batcher, t = _make_converged(rate_hz=14.0)
        assert batcher.batch_length_s == 1.0

        for _ in range(ESCALATION_OVERLOAD_THRESHOLD):
            batcher.report_batch(100, processing_time_s=1.5)
        assert batcher.batch_length_s == 2.0

        # Close the currently active 1s window so the pending change applies.
        result = batcher.batch(
            [*_msgs_at(14.0, start=t, duration=1.0), _hwm_trigger(t + 0.81)]
        )
        assert result is not None
        t_next = t + 1.0

        # Next batch spans 2s -> 28 messages at 14 Hz land in one batch.
        result = batcher.batch(_msgs_at(14.0, start=t_next, duration=2.0))
        assert result is not None
        assert len(result.messages) == 28

    def test_oscillation_preserves_messages(self):
        """Escalate, run, de-escalate, run: total_in == total_out.

        Uses "in-between" processing times (>= 0.75 * batch_length, < 1.0 *
        batch_length) to hold each level during its run phase; then explicit
        overload/underload bursts move between levels.
        """
        rate_hz = 14.0
        batcher, t = _make_converged(rate_hz=rate_hz, max_level=3)

        total_in = 0
        total_out = 0

        def run(duration: float, hold_time_s: float) -> None:
            """Feed one batch window, assert close, report a hold-level time."""
            nonlocal t, total_in, total_out
            msgs = _msgs_at(rate_hz, start=t, duration=duration)
            total_in += len(msgs)
            result = batcher.batch(msgs)
            assert result is not None, (
                f"Batch at t={t} duration={duration} should close via slot gate"
            )
            total_out += len(result.messages)
            t += duration
            batcher.report_batch(len(result.messages), processing_time_s=hold_time_s)

        # Level 0 (1s): hold with processing_time = 0.9 (in-between)
        for _ in range(3):
            run(1.0, hold_time_s=0.9)

        # Escalate to level 2 (2s)
        for _ in range(ESCALATION_OVERLOAD_THRESHOLD):
            batcher.report_batch(14, processing_time_s=1.5)
        assert batcher.batch_length_s == 2.0

        # Close the lingering 1s batch (transition). Report in-between for 2s.
        run(1.0, hold_time_s=1.8)
        # Level 2 (2s): hold with processing_time = 1.8
        for _ in range(4):
            run(2.0, hold_time_s=1.8)

        # De-escalate to level 1 (~1.43s) via underload reports.
        for _ in range(DEESCALATION_UNDERLOAD_THRESHOLD):
            batcher.report_batch(28, processing_time_s=0.1)
        expected_level_1 = 20 / 14  # round(14 * sqrt(2)) / 14
        assert batcher.batch_length_s == expected_level_1

        # Close the lingering 2s batch (transition). Report in-between for ~1.43s.
        run(2.0, hold_time_s=1.3)
        # Level 1 (~1.43s): hold with processing_time = 1.3
        for _ in range(3):
            run(expected_level_1, hold_time_s=1.3)

        assert total_out == total_in
