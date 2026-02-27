# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest

from ess.livedata.core.load_shedder import (
    _ACTIVATION_THRESHOLD,
    _BUCKET_DURATION_S,
    _DEACTIVATION_THRESHOLD,
    _N_BUCKETS,
    DROPPABLE_KINDS,
    LoadShedder,
)
from ess.livedata.core.message import Message, StreamId, StreamKind


def _make_message(kind: StreamKind, name: str = "src") -> Message:
    return Message(timestamp=0, stream=StreamId(kind=kind, name=name), value=b"")


class FakeClock:
    """Deterministic clock for testing the rolling window."""

    def __init__(self, start: float = 0.0) -> None:
        self._time = start

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


def _make_shedder(clock: FakeClock | None = None) -> LoadShedder:
    if clock is None:
        clock = FakeClock()
    return LoadShedder(clock=clock)


def _activate(shedder: LoadShedder) -> None:
    for _ in range(_ACTIVATION_THRESHOLD):
        shedder.report_batch_result(batch_produced=True)
    assert shedder.state.is_shedding is True


class TestLoadShedderInitialState:
    def test_not_shedding_initially(self):
        shedder = _make_shedder()
        assert shedder.state.is_shedding is False

    def test_zero_dropped_initially(self):
        shedder = _make_shedder()
        assert shedder.state.messages_dropped == 0

    def test_zero_eligible_initially(self):
        shedder = _make_shedder()
        assert shedder.state.messages_eligible == 0


class TestLoadShedderActivation:
    def test_activates_after_consecutive_batches(self):
        shedder = _make_shedder()
        _activate(shedder)

    def test_does_not_activate_below_threshold(self):
        shedder = _make_shedder()
        for _ in range(_ACTIVATION_THRESHOLD - 1):
            shedder.report_batch_result(batch_produced=True)
        assert shedder.state.is_shedding is False

    def test_idle_cycle_resets_consecutive_count(self):
        shedder = _make_shedder()
        for _ in range(_ACTIVATION_THRESHOLD - 1):
            shedder.report_batch_result(batch_produced=True)
        shedder.report_batch_result(batch_produced=False)
        # Restart counting — should not activate after fewer than threshold
        for _ in range(_ACTIVATION_THRESHOLD - 1):
            shedder.report_batch_result(batch_produced=True)
        assert shedder.state.is_shedding is False


class TestLoadShedderDeactivation:
    @pytest.fixture
    def active_shedder(self):
        shedder = _make_shedder()
        _activate(shedder)
        return shedder

    def test_deactivates_after_consecutive_idle(self, active_shedder):
        for _ in range(_DEACTIVATION_THRESHOLD):
            active_shedder.report_batch_result(batch_produced=False)
        assert active_shedder.state.is_shedding is False

    def test_does_not_deactivate_below_threshold(self, active_shedder):
        for _ in range(_DEACTIVATION_THRESHOLD - 1):
            active_shedder.report_batch_result(batch_produced=False)
        assert active_shedder.state.is_shedding is True

    def test_batch_resets_idle_count(self, active_shedder):
        for _ in range(_DEACTIVATION_THRESHOLD - 1):
            active_shedder.report_batch_result(batch_produced=False)
        active_shedder.report_batch_result(batch_produced=True)
        # Restart idle counting
        for _ in range(_DEACTIVATION_THRESHOLD - 1):
            active_shedder.report_batch_result(batch_produced=False)
        assert active_shedder.state.is_shedding is True


class TestLoadShedderShed:
    def test_passes_everything_when_inactive(self):
        shedder = _make_shedder()
        messages = [
            _make_message(StreamKind.DETECTOR_EVENTS),
            _make_message(StreamKind.LOG),
            _make_message(StreamKind.MONITOR_EVENTS),
        ]
        result = shedder.shed(messages)
        assert result == messages

    def test_preserves_non_droppable_when_active(self):
        shedder = _make_shedder()
        _activate(shedder)

        non_droppable_kinds = [
            StreamKind.LOG,
            StreamKind.LIVEDATA_COMMANDS,
            StreamKind.LIVEDATA_RESPONSES,
            StreamKind.LIVEDATA_DATA,
            StreamKind.LIVEDATA_ROI,
            StreamKind.LIVEDATA_STATUS,
            StreamKind.UNKNOWN,
        ]
        messages = [_make_message(kind) for kind in non_droppable_kinds]
        result = shedder.shed(messages)
        assert result == messages

    def test_drops_roughly_half_of_droppable_when_active(self):
        shedder = _make_shedder()
        _activate(shedder)

        messages = [_make_message(StreamKind.DETECTOR_EVENTS) for _ in range(100)]
        result = shedder.shed(messages)
        assert len(result) == 50

    def test_all_droppable_kinds_are_shed(self):
        shedder = _make_shedder()
        _activate(shedder)

        for kind in DROPPABLE_KINDS:
            messages = [_make_message(kind) for _ in range(10)]
            before = shedder.state.messages_dropped
            result = shedder.shed(messages)
            assert len(result) < len(messages), f"{kind} was not shed"
            assert shedder.state.messages_dropped > before

    def test_mixed_messages_preserves_non_droppable(self):
        shedder = _make_shedder()
        _activate(shedder)

        log_msg = _make_message(StreamKind.LOG)
        cmd_msg = _make_message(StreamKind.LIVEDATA_COMMANDS)
        det_msgs = [_make_message(StreamKind.DETECTOR_EVENTS) for _ in range(10)]
        messages = [log_msg, *det_msgs, cmd_msg]

        result = shedder.shed(messages)
        assert log_msg in result
        assert cmd_msg in result


class TestRollingWindow:
    def test_dropped_count_within_window(self):
        clock = FakeClock()
        shedder = _make_shedder(clock)
        _activate(shedder)

        messages = [_make_message(StreamKind.DETECTOR_EVENTS) for _ in range(100)]
        shedder.shed(messages)
        assert shedder.state.messages_dropped == 50
        assert shedder.state.messages_eligible == 100

    def test_counts_accumulate_across_calls_in_same_bucket(self):
        clock = FakeClock()
        shedder = _make_shedder(clock)
        _activate(shedder)

        batch = [_make_message(StreamKind.DETECTOR_EVENTS) for _ in range(10)]
        shedder.shed(batch)
        shedder.shed(batch)
        assert shedder.state.messages_dropped == 10
        assert shedder.state.messages_eligible == 20

    def test_counts_decay_after_window_expires(self):
        clock = FakeClock()
        shedder = _make_shedder(clock)
        _activate(shedder)

        messages = [_make_message(StreamKind.DETECTOR_EVENTS) for _ in range(100)]
        shedder.shed(messages)
        assert shedder.state.messages_dropped == 50

        # Advance past the full window
        clock.advance(_N_BUCKETS * _BUCKET_DURATION_S + 1)
        assert shedder.state.messages_dropped == 0
        assert shedder.state.messages_eligible == 0

    def test_partial_window_decay(self):
        clock = FakeClock()
        shedder = _make_shedder(clock)
        _activate(shedder)

        # Record in bucket 0
        batch = [_make_message(StreamKind.DETECTOR_EVENTS) for _ in range(10)]
        shedder.shed(batch)
        dropped_first = shedder.state.messages_dropped

        # Advance to a new bucket and record more
        clock.advance(_BUCKET_DURATION_S)
        shedder.shed(batch)
        assert shedder.state.messages_dropped > dropped_first

        # Advance so the first bucket expires but not the second
        clock.advance((_N_BUCKETS - 1) * _BUCKET_DURATION_S)
        state = shedder.state
        # Only the second bucket's data should remain
        assert state.messages_dropped == 5
        assert state.messages_eligible == 10

    def test_eligible_tracked_when_not_shedding(self):
        """Even when not shedding, eligible messages are counted."""
        clock = FakeClock()
        shedder = _make_shedder(clock)
        # Not activated — no shedding
        messages = [_make_message(StreamKind.DETECTOR_EVENTS) for _ in range(20)]
        shedder.shed(messages)
        state = shedder.state
        assert state.messages_dropped == 0
        assert state.messages_eligible == 20


class TestLoadShedderState:
    def test_state_reflects_shedding(self):
        shedder = _make_shedder()
        assert shedder.state.is_shedding is False
        _activate(shedder)
        assert shedder.state.is_shedding is True

    def test_state_reports_shedding_level(self):
        shedder = _make_shedder()
        assert shedder.state.shedding_level == 0
        _activate(shedder)
        assert shedder.state.shedding_level == 1

    def test_state_is_snapshot(self):
        shedder = _make_shedder()
        state = shedder.state
        _activate(shedder)
        # Original snapshot unchanged (frozen dataclass)
        assert state.is_shedding is False
        assert shedder.state.is_shedding is True


def _escalate_to(shedder: LoadShedder, level: int) -> None:
    """Escalate the shedder to the given level."""
    for _ in range(level):
        for _ in range(_ACTIVATION_THRESHOLD):
            shedder.report_batch_result(batch_produced=True)
    assert shedder.state.shedding_level == level


def _deescalate_by(shedder: LoadShedder, steps: int) -> None:
    """De-escalate the shedder by the given number of steps."""
    for _ in range(steps):
        for _ in range(_DEACTIVATION_THRESHOLD):
            shedder.report_batch_result(batch_produced=False)


class TestMultiLevelEscalation:
    def test_escalates_to_level_2(self):
        shedder = _make_shedder()
        _escalate_to(shedder, 2)
        assert shedder.state.shedding_level == 2

    def test_escalates_to_level_3(self):
        shedder = _make_shedder()
        _escalate_to(shedder, 3)
        assert shedder.state.shedding_level == 3

    def test_escalation_requires_threshold_per_level(self):
        shedder = _make_shedder()
        _escalate_to(shedder, 1)
        # Not enough batches for next level
        for _ in range(_ACTIVATION_THRESHOLD - 1):
            shedder.report_batch_result(batch_produced=True)
        assert shedder.state.shedding_level == 1


class TestMultiLevelDeescalation:
    def test_deescalates_one_level_at_a_time(self):
        shedder = _make_shedder()
        _escalate_to(shedder, 3)
        _deescalate_by(shedder, 1)
        assert shedder.state.shedding_level == 2

    def test_deescalates_to_zero(self):
        shedder = _make_shedder()
        _escalate_to(shedder, 2)
        _deescalate_by(shedder, 2)
        assert shedder.state.shedding_level == 0
        assert shedder.state.is_shedding is False

    def test_deescalation_requires_threshold_per_level(self):
        shedder = _make_shedder()
        _escalate_to(shedder, 2)
        for _ in range(_DEACTIVATION_THRESHOLD - 1):
            shedder.report_batch_result(batch_produced=False)
        assert shedder.state.shedding_level == 2


class TestMultiLevelDropRates:
    @pytest.mark.parametrize(
        ("level", "expected_kept"),
        [
            (1, 128),  # keep 1/2 of 256
            (2, 64),  # keep 1/4 of 256
            (3, 32),  # keep 1/8 of 256
            (4, 16),  # keep 1/16 of 256
        ],
    )
    def test_drop_rate_at_level(self, level, expected_kept):
        shedder = _make_shedder()
        _escalate_to(shedder, level)
        messages = [_make_message(StreamKind.DETECTOR_EVENTS) for _ in range(256)]
        result = shedder.shed(messages)
        assert len(result) == expected_kept
