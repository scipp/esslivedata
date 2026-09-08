# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the dashboard shell's selectable themes."""

from __future__ import annotations

import colorsys

import pytest

from ess.livedata.dashboard.theme import THEMES, Theme


def _hls(color: str) -> tuple[float, float, float]:
    return colorsys.rgb_to_hls(*(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)))


@pytest.mark.parametrize('theme', THEMES.values(), ids=lambda theme: theme.name)
class TestFloatingHeaderBackground:
    """A floating window's title bar: the chrome color, but not exactly it.

    Matched exactly, the title bar disappears into the header and the rail
    wherever a window overlaps them -- which is most of the time, since a
    window opens against the top of the viewport.
    """

    def test_it_is_lighter_than_the_chrome_color(self, theme: Theme) -> None:
        _, lightness, _ = _hls(theme.floating_header_background)
        _, chrome_lightness, _ = _hls(theme.header_background)
        assert lightness > chrome_lightness

    def test_it_keeps_the_chrome_hue(self, theme: Theme) -> None:
        hue, _, saturation = _hls(theme.floating_header_background)
        chrome_hue, _, chrome_saturation = _hls(theme.header_background)
        assert hue == pytest.approx(chrome_hue, abs=0.01)
        assert saturation == pytest.approx(chrome_saturation, abs=0.01)
