# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for per-axis range and color-mapper handle access."""

from __future__ import annotations

from typing import Any

import numpy as np

from ess.livedata.dashboard.range_hook import RangeHandles


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


def test_write_x_range_sets_start_and_end() -> None:
    handle = _StubRange()
    assert RangeHandles.write(_StubPlot(x_range=handle), 'x', 1.0, 5.0)
    assert (handle.start, handle.end) == (1.0, 5.0)


def test_write_y_range_sets_start_and_end() -> None:
    handle = _StubRange()
    assert RangeHandles.write(_StubPlot(y_range=handle), 'y', -2.0, 7.5)
    assert (handle.start, handle.end) == (-2.0, 7.5)


def test_write_color_mapper_sets_low_and_high() -> None:
    mapper = _StubColorMapper()
    assert RangeHandles.write(_StubPlot(color_mapper=mapper), 'c', 0.1, 99.9)
    assert (mapper.low, mapper.high) == (0.1, 99.9)


def test_write_returns_false_when_color_mapper_missing() -> None:
    assert RangeHandles.write(_StubPlot(), 'c', 0.0, 1.0) is False


def test_write_returns_false_when_range_handle_missing() -> None:
    assert RangeHandles.write(_StubPlot(), 'x', 0.0, 1.0) is False


def test_handles_read_per_call_follow_figure_swap() -> None:
    """No handle caching: ``write`` targets whatever ``plot.handles`` exposes
    on each call. Guards against figure swaps leaving us writing to a
    detached Range1d."""
    x_first = _StubRange()
    plot = _StubPlot(x_range=x_first)
    RangeHandles.write(plot, 'x', 1.0, 2.0)
    assert (x_first.start, x_first.end) == (1.0, 2.0)

    # Figure swap: a new x_range handle replaces the old one.
    x_second = _StubRange()
    plot.handles['x_range'] = x_second
    RangeHandles.write(plot, 'x', 1.0, 2.0)
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


def test_write_datetime_axis_coerces_float_targets() -> None:
    """Bokeh stores datetime ranges as ``np.datetime64``; range targets are
    floats (epoch counts). The float must be cast back to a datetime64 of the
    handle's unit so neither the ordering comparison nor Bokeh's range patch
    mixes datetime64/float (which raises ``UFuncTypeError``)."""
    lo_dt = np.datetime64('2026-06-17T11:52:00', 'ns')
    hi_dt = np.datetime64('2026-06-17T11:52:04', 'ns')
    handle = _StubRange(start=lo_dt, end=hi_dt)
    target_lo = float(np.datetime64('2026-06-17T11:52:01', 'ns').astype('int64'))
    target_hi = float(np.datetime64('2026-06-17T11:52:03', 'ns').astype('int64'))

    assert RangeHandles.write(_StubPlot(x_range=handle), 'x', target_lo, target_hi)

    assert handle.start == np.datetime64('2026-06-17T11:52:01', 'ns')
    assert handle.end == np.datetime64('2026-06-17T11:52:03', 'ns')


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
