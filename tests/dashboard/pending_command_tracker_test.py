# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for PendingCommandTracker, focused on command expiry."""

import pytest

from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.pending_command_tracker import PendingCommandTracker


class MutableClock:
    """Manually advanceable clock for deterministic expiry tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def workflow_id() -> WorkflowId:
    return WorkflowId(instrument="test", name="wf", version=1)


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def tracker(clock: MutableClock) -> PendingCommandTracker:
    return PendingCommandTracker(clock=clock)


def test_expire_stale_removes_only_aged_entries(
    tracker: PendingCommandTracker, clock: MutableClock, workflow_id: WorkflowId
) -> None:
    tracker.register("old", workflow_id, "start")
    clock.advance(20.0)
    tracker.register("new", workflow_id, "stop")

    clock.advance(15.0)  # "old" is now 35s, "new" is 15s
    expired = tracker.expire_stale(max_age_seconds=30.0)

    assert [cmd.action for cmd in expired] == ["start"]
    assert len(tracker) == 1


def test_expire_stale_returns_command_details(
    tracker: PendingCommandTracker, clock: MutableClock, workflow_id: WorkflowId
) -> None:
    tracker.register("id", workflow_id, "reset", expected_count=3)
    clock.advance(31.0)

    (expired,) = tracker.expire_stale(max_age_seconds=30.0)

    assert expired.workflow_id == workflow_id
    assert expired.action == "reset"
    assert expired.expected_count == 3


def test_expire_stale_removes_expired_from_tracking(
    tracker: PendingCommandTracker, clock: MutableClock, workflow_id: WorkflowId
) -> None:
    tracker.register("id", workflow_id, "start")
    clock.advance(31.0)

    tracker.expire_stale(max_age_seconds=30.0)

    assert len(tracker) == 0
    assert tracker.expire_stale(max_age_seconds=30.0) == []


def test_response_after_expiry_does_not_crash(
    tracker: PendingCommandTracker, clock: MutableClock, workflow_id: WorkflowId
) -> None:
    tracker.register("id", workflow_id, "start")
    clock.advance(31.0)
    tracker.expire_stale(max_age_seconds=30.0)

    result = tracker.record_response("id", success=True)

    assert result is None


def test_non_expired_command_still_completes(
    tracker: PendingCommandTracker, clock: MutableClock, workflow_id: WorkflowId
) -> None:
    tracker.register("id", workflow_id, "start", expected_count=1)
    clock.advance(10.0)
    tracker.expire_stale(max_age_seconds=30.0)

    result = tracker.record_response("id", success=True)

    assert result is not None
    assert result.all_succeeded
    assert len(tracker) == 0
