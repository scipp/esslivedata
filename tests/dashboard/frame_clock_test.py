# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ess.livedata.dashboard.frame_clock import FrameClock


def test_unknown_key_starts_at_zero() -> None:
    assert FrameClock().generation('a') == 0


def test_commit_without_mark_does_not_advance() -> None:
    clock = FrameClock()
    clock.commit()
    assert clock.generation('a') == 0


def test_mark_then_commit_advances_once() -> None:
    clock = FrameClock()
    clock.mark('a')
    clock.commit()
    assert clock.generation('a') == 1


def test_multiple_marks_of_same_key_coalesce_into_one_commit() -> None:
    """A burst of recomputes in one grid collapses to a single frame."""
    clock = FrameClock()
    clock.mark('a')
    clock.mark('a')
    clock.mark('a')
    clock.commit()
    assert clock.generation('a') == 1


def test_commit_clears_pending() -> None:
    clock = FrameClock()
    clock.mark('a')
    clock.commit()
    clock.commit()  # nothing marked since the last commit
    assert clock.generation('a') == 1


def test_successive_frames_advance_per_burst() -> None:
    clock = FrameClock()
    clock.mark('a')
    clock.commit()
    clock.mark('a')
    clock.commit()
    assert clock.generation('a') == 2


def test_keys_advance_independently() -> None:
    """A burst in one grid does not advance another grid's generation."""
    clock = FrameClock()
    clock.mark('a')
    clock.commit()
    assert clock.generation('a') == 1
    assert clock.generation('b') == 0


def test_one_commit_advances_all_marked_keys() -> None:
    clock = FrameClock()
    clock.mark('a')
    clock.mark('b')
    clock.commit()
    assert clock.generation('a') == 1
    assert clock.generation('b') == 1
