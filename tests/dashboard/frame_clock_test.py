# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ess.livedata.dashboard.frame_clock import FrameClock


def test_starts_at_zero() -> None:
    assert FrameClock().generation == 0


def test_commit_without_mark_does_not_advance() -> None:
    clock = FrameClock()
    clock.commit()
    assert clock.generation == 0


def test_mark_then_commit_advances_once() -> None:
    clock = FrameClock()
    clock.mark()
    clock.commit()
    assert clock.generation == 1


def test_multiple_marks_coalesce_into_one_commit() -> None:
    """A burst of recomputes collapses to a single frame."""
    clock = FrameClock()
    clock.mark()
    clock.mark()
    clock.mark()
    clock.commit()
    assert clock.generation == 1


def test_commit_clears_pending() -> None:
    clock = FrameClock()
    clock.mark()
    clock.commit()
    clock.commit()  # nothing marked since the last commit
    assert clock.generation == 1


def test_successive_frames_advance_per_burst() -> None:
    clock = FrameClock()
    clock.mark()
    clock.commit()
    clock.mark()
    clock.commit()
    assert clock.generation == 2
