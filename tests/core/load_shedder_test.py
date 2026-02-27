# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest

from ess.livedata.core.load_shedder import (
    _ACTIVATION_THRESHOLD,
    _DEACTIVATION_THRESHOLD,
    DROPPABLE_KINDS,
    LoadShedder,
)
from ess.livedata.core.message import Message, StreamId, StreamKind


def _make_message(kind: StreamKind, name: str = "src") -> Message:
    return Message(timestamp=0, stream=StreamId(kind=kind, name=name), value=b"")


class TestLoadShedderInitialState:
    def test_not_shedding_initially(self):
        shedder = LoadShedder()
        assert shedder.state.is_shedding is False

    def test_zero_dropped_initially(self):
        shedder = LoadShedder()
        assert shedder.state.messages_dropped == 0


class TestLoadShedderActivation:
    def test_activates_after_consecutive_batches(self):
        shedder = LoadShedder()
        for _ in range(_ACTIVATION_THRESHOLD):
            shedder.report_batch_result(batch_produced=True)
        assert shedder.state.is_shedding is True

    def test_does_not_activate_below_threshold(self):
        shedder = LoadShedder()
        for _ in range(_ACTIVATION_THRESHOLD - 1):
            shedder.report_batch_result(batch_produced=True)
        assert shedder.state.is_shedding is False

    def test_idle_cycle_resets_consecutive_count(self):
        shedder = LoadShedder()
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
        shedder = LoadShedder()
        for _ in range(_ACTIVATION_THRESHOLD):
            shedder.report_batch_result(batch_produced=True)
        assert shedder.state.is_shedding is True
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
        shedder = LoadShedder()
        messages = [
            _make_message(StreamKind.DETECTOR_EVENTS),
            _make_message(StreamKind.LOG),
            _make_message(StreamKind.MONITOR_EVENTS),
        ]
        result = shedder.shed(messages)
        assert result == messages

    def test_preserves_non_droppable_when_active(self):
        shedder = LoadShedder()
        for _ in range(_ACTIVATION_THRESHOLD):
            shedder.report_batch_result(batch_produced=True)

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
        shedder = LoadShedder()
        for _ in range(_ACTIVATION_THRESHOLD):
            shedder.report_batch_result(batch_produced=True)

        messages = [_make_message(StreamKind.DETECTOR_EVENTS) for _ in range(100)]
        result = shedder.shed(messages)
        assert len(result) == 50

    def test_dropped_count_accuracy(self):
        shedder = LoadShedder()
        for _ in range(_ACTIVATION_THRESHOLD):
            shedder.report_batch_result(batch_produced=True)

        messages = [_make_message(StreamKind.DETECTOR_EVENTS) for _ in range(100)]
        shedder.shed(messages)
        assert shedder.state.messages_dropped == 50

    def test_dropped_count_is_cumulative(self):
        shedder = LoadShedder()
        for _ in range(_ACTIVATION_THRESHOLD):
            shedder.report_batch_result(batch_produced=True)

        batch = [_make_message(StreamKind.DETECTOR_EVENTS) for _ in range(10)]
        shedder.shed(batch)
        shedder.shed(batch)
        assert shedder.state.messages_dropped == 10

    def test_all_droppable_kinds_are_shed(self):
        shedder = LoadShedder()
        for _ in range(_ACTIVATION_THRESHOLD):
            shedder.report_batch_result(batch_produced=True)

        for kind in DROPPABLE_KINDS:
            messages = [_make_message(kind) for _ in range(10)]
            before = shedder.state.messages_dropped
            result = shedder.shed(messages)
            assert len(result) < len(messages), f"{kind} was not shed"
            assert shedder.state.messages_dropped > before

    def test_mixed_messages_preserves_non_droppable(self):
        shedder = LoadShedder()
        for _ in range(_ACTIVATION_THRESHOLD):
            shedder.report_batch_result(batch_produced=True)

        log_msg = _make_message(StreamKind.LOG)
        cmd_msg = _make_message(StreamKind.LIVEDATA_COMMANDS)
        det_msgs = [_make_message(StreamKind.DETECTOR_EVENTS) for _ in range(10)]
        messages = [log_msg, *det_msgs, cmd_msg]

        result = shedder.shed(messages)
        assert log_msg in result
        assert cmd_msg in result


class TestLoadShedderState:
    def test_state_reflects_shedding(self):
        shedder = LoadShedder()
        assert shedder.state.is_shedding is False
        for _ in range(_ACTIVATION_THRESHOLD):
            shedder.report_batch_result(batch_produced=True)
        assert shedder.state.is_shedding is True

    def test_state_is_snapshot(self):
        shedder = LoadShedder()
        state = shedder.state
        for _ in range(_ACTIVATION_THRESHOLD):
            shedder.report_batch_result(batch_produced=True)
        # Original snapshot unchanged (frozen dataclass)
        assert state.is_shedding is False
        assert shedder.state.is_shedding is True
