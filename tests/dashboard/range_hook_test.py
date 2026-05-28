# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the per-axis range hook factory."""

from __future__ import annotations

from typing import Any

from ess.livedata.dashboard.range_hook import RangeHandles, make_range_hook


class _StubRange:
    def __init__(self, start: float | None = None, end: float | None = None) -> None:
        self.start = start
        self.end = end
        self.writes: list[tuple[str, float]] = []
        self.document = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {'start', 'end'} and 'writes' in self.__dict__:
            self.writes.append((name, value))
        object.__setattr__(self, name, value)


class _StubColorMapper:
    def __init__(self, low: float | None = None, high: float | None = None) -> None:
        self.low = low
        self.high = high
        self.writes: list[tuple[str, float]] = []
        self.document = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {'low', 'high'} and 'writes' in self.__dict__:
            self.writes.append((name, value))
        object.__setattr__(self, name, value)


class _StubSubPlot:
    def __init__(self, color_mapper: _StubColorMapper) -> None:
        self.handles: dict[str, Any] = {'color_mapper': color_mapper}


class _StubPlot:
    def __init__(
        self,
        *,
        x_range: _StubRange | None = None,
        y_range: _StubRange | None = None,
        color_mapper: _StubColorMapper | None = None,
        subplots: dict[str, _StubSubPlot] | None = None,
    ) -> None:
        self.handles: dict[str, object] = {}
        if x_range is not None:
            self.handles['x_range'] = x_range
        if y_range is not None:
            self.handles['y_range'] = y_range
        if color_mapper is not None:
            self.handles['color_mapper'] = color_mapper
        self.subplots = subplots


def _make_plot_xy() -> tuple[_StubPlot, _StubRange, _StubRange]:
    x_range = _StubRange()
    y_range = _StubRange()
    plot = _StubPlot(x_range=x_range, y_range=y_range)
    return plot, x_range, y_range


def test_hook_writes_x_range_start_and_end_when_enabled() -> None:
    plot, x_range, _ = _make_plot_xy()
    hook = make_range_hook(
        axis='x',
        get_target=lambda: (1.0, 5.0),
        get_enabled=lambda: True,
    )

    hook(plot, None)

    assert x_range.start == 1.0
    assert x_range.end == 5.0


def test_hook_writes_y_range_start_and_end_when_enabled() -> None:
    plot, _, y_range = _make_plot_xy()
    hook = make_range_hook(
        axis='y',
        get_target=lambda: (-2.0, 7.5),
        get_enabled=lambda: True,
    )

    hook(plot, None)

    assert y_range.start == -2.0
    assert y_range.end == 7.5


def test_hook_writes_color_mapper_low_and_high_for_c_axis() -> None:
    mapper = _StubColorMapper()
    plot = _StubPlot(color_mapper=mapper)
    hook = make_range_hook(
        axis='c',
        get_target=lambda: (0.1, 99.9),
        get_enabled=lambda: True,
    )

    hook(plot, None)

    assert mapper.low == 0.1
    assert mapper.high == 99.9


def test_hook_is_noop_when_disabled() -> None:
    plot, x_range, _ = _make_plot_xy()
    hook = make_range_hook(
        axis='x',
        get_target=lambda: (1.0, 5.0),
        get_enabled=lambda: False,
    )

    hook(plot, None)

    assert x_range.start is None
    assert x_range.end is None


def test_hook_is_noop_when_target_is_none() -> None:
    plot, x_range, _ = _make_plot_xy()
    hook = make_range_hook(
        axis='x',
        get_target=lambda: None,
        get_enabled=lambda: True,
    )

    hook(plot, None)

    assert x_range.start is None
    assert x_range.end is None


def test_hook_reads_target_each_call() -> None:
    plot, x_range, _ = _make_plot_xy()
    targets: list[tuple[float, float] | None] = [(0.0, 1.0), (2.0, 3.0)]
    hook = make_range_hook(
        axis='x',
        get_target=lambda: targets.pop(0),
        get_enabled=lambda: True,
    )

    hook(plot, None)
    assert (x_range.start, x_range.end) == (0.0, 1.0)

    hook(plot, None)
    assert (x_range.start, x_range.end) == (2.0, 3.0)


def test_c_hook_noop_when_color_mapper_missing() -> None:
    plot = _StubPlot()
    hook = make_range_hook(
        axis='c',
        get_target=lambda: (0.0, 1.0),
        get_enabled=lambda: True,
    )

    # Should not raise.
    hook(plot, None)


def test_xy_hook_noop_when_range_handle_missing() -> None:
    plot = _StubPlot()
    hook = make_range_hook(
        axis='x',
        get_target=lambda: (0.0, 1.0),
        get_enabled=lambda: True,
    )

    # Should not raise.
    hook(plot, None)


def test_hook_follows_handle_swap_per_render() -> None:
    """No handle caching: hook writes to whatever ``plot.handles`` exposes
    on each call. Guards against figure swaps leaving us writing to a
    detached Range1d."""
    x_first = _StubRange()
    plot = _StubPlot(x_range=x_first)
    hook = make_range_hook(
        axis='x',
        get_target=lambda: (1.0, 2.0),
        get_enabled=lambda: True,
    )

    hook(plot, None)
    assert (x_first.start, x_first.end) == (1.0, 2.0)

    # Figure swap: a new x_range handle replaces the old one.
    x_second = _StubRange()
    plot.handles['x_range'] = x_second
    hook(plot, None)
    assert (x_second.start, x_second.end) == (1.0, 2.0)


def test_color_mapper_resolved_from_subplot_when_top_level_missing() -> None:
    """Overlay case: top-level plot lacks color_mapper; image sub-plot has it."""
    mapper = _StubColorMapper()
    sub = _StubSubPlot(color_mapper=mapper)
    plot = _StubPlot(subplots={'Image': sub})

    assert RangeHandles.color_mapper(plot) is mapper
    assert RangeHandles.color_mapper_plot(plot) is sub


def test_write_avoids_inverted_range() -> None:
    """``start <= end`` must hold after each individual property write so
    no observer ever sees an inverted intermediate frame."""
    handle = _StubRange(start=0.0, end=1.0)
    # Moving range up: new start (10) > current end (1) -> write end first.
    RangeHandles.write(_StubPlot(x_range=handle), 'x', 10.0, 11.0)
    # After each write the invariant lo <= hi must hold.
    seen_start = 0.0
    seen_end = 1.0
    for attr, value in handle.writes:
        if attr == 'start':
            seen_start = value
        else:
            seen_end = value
        assert seen_start <= seen_end, f"inverted: start={seen_start} end={seen_end}"


def test_write_avoids_inverted_range_downward() -> None:
    handle = _StubRange(start=10.0, end=11.0)
    # Moving range down: new end (1) < current end (11) -> write start first.
    RangeHandles.write(_StubPlot(x_range=handle), 'x', 0.0, 1.0)
    seen_start = 10.0
    seen_end = 11.0
    for attr, value in handle.writes:
        if attr == 'start':
            seen_start = value
        else:
            seen_end = value
        assert seen_start <= seen_end, f"inverted: start={seen_start} end={seen_end}"


def test_write_color_mapper_avoids_inversion() -> None:
    mapper = _StubColorMapper(low=0.0, high=1.0)
    RangeHandles.write(_StubPlot(color_mapper=mapper), 'c', 10.0, 11.0)
    seen_lo = 0.0
    seen_hi = 1.0
    for attr, value in mapper.writes:
        if attr == 'low':
            seen_lo = value
        else:
            seen_hi = value
        assert seen_lo <= seen_hi, f"inverted: low={seen_lo} high={seen_hi}"
