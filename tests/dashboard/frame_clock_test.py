# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ess.livedata.dashboard.frame_clock import FrameClock


def test_unknown_key_starts_at_zero() -> None:
    assert FrameClock().generation('a') == 0


def test_commit_advances_once() -> None:
    clock = FrameClock()
    clock.commit('a')
    assert clock.generation('a') == 1


def test_successive_frames_advance_per_burst() -> None:
    clock = FrameClock()
    clock.commit('a')
    clock.commit('a')
    assert clock.generation('a') == 2


def test_keys_advance_independently() -> None:
    """A burst in one grid does not advance another grid's generation."""
    clock = FrameClock()
    clock.commit('a')
    assert clock.generation('a') == 1
    assert clock.generation('b') == 0
