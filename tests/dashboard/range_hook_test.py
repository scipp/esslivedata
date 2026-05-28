# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the per-axis range hook factory."""

from __future__ import annotations

from ess.livedata.dashboard.range_hook import RangeHandles, make_range_hook


class _StubRange:
    def __init__(self) -> None:
        self.start: float | None = None
        self.end: float | None = None


class _StubColorMapper:
    def __init__(self) -> None:
        self.low: float | None = None
        self.high: float | None = None


class _StubPlot:
    def __init__(
        self,
        *,
        x_range: _StubRange | None = None,
        y_range: _StubRange | None = None,
        color_mapper: _StubColorMapper | None = None,
    ) -> None:
        self.handles: dict[str, object] = {}
        if x_range is not None:
            self.handles['x_range'] = x_range
        if y_range is not None:
            self.handles['y_range'] = y_range
        if color_mapper is not None:
            self.handles['color_mapper'] = color_mapper


def _make_plot_xy() -> tuple[_StubPlot, _StubRange, _StubRange]:
    x_range = _StubRange()
    y_range = _StubRange()
    plot = _StubPlot(x_range=x_range, y_range=y_range)
    return plot, x_range, y_range


def test_hook_writes_x_range_start_and_end_when_enabled() -> None:
    plot, x_range, _ = _make_plot_xy()
    handles = RangeHandles()
    hook = make_range_hook(
        axis='x',
        get_target=lambda: (1.0, 5.0),
        get_enabled=lambda: True,
        handles=handles,
    )

    hook(plot, None)

    assert x_range.start == 1.0
    assert x_range.end == 5.0


def test_hook_writes_y_range_start_and_end_when_enabled() -> None:
    plot, _, y_range = _make_plot_xy()
    handles = RangeHandles()
    hook = make_range_hook(
        axis='y',
        get_target=lambda: (-2.0, 7.5),
        get_enabled=lambda: True,
        handles=handles,
    )

    hook(plot, None)

    assert y_range.start == -2.0
    assert y_range.end == 7.5


def test_hook_writes_color_mapper_low_and_high_for_c_axis() -> None:
    mapper = _StubColorMapper()
    plot = _StubPlot(color_mapper=mapper)
    handles = RangeHandles()
    hook = make_range_hook(
        axis='c',
        get_target=lambda: (0.1, 99.9),
        get_enabled=lambda: True,
        handles=handles,
    )

    hook(plot, None)

    assert mapper.low == 0.1
    assert mapper.high == 99.9


def test_hook_is_noop_when_disabled() -> None:
    plot, x_range, _ = _make_plot_xy()
    handles = RangeHandles()
    hook = make_range_hook(
        axis='x',
        get_target=lambda: (1.0, 5.0),
        get_enabled=lambda: False,
        handles=handles,
    )

    hook(plot, None)

    assert x_range.start is None
    assert x_range.end is None


def test_hook_is_noop_when_target_is_none() -> None:
    plot, x_range, _ = _make_plot_xy()
    handles = RangeHandles()
    hook = make_range_hook(
        axis='x',
        get_target=lambda: None,
        get_enabled=lambda: True,
        handles=handles,
    )

    hook(plot, None)

    assert x_range.start is None
    assert x_range.end is None


def test_hook_reads_target_each_call() -> None:
    plot, x_range, _ = _make_plot_xy()
    handles = RangeHandles()
    targets: list[tuple[float, float] | None] = [(0.0, 1.0), (2.0, 3.0)]
    hook = make_range_hook(
        axis='x',
        get_target=lambda: targets.pop(0),
        get_enabled=lambda: True,
        handles=handles,
    )

    hook(plot, None)
    assert (x_range.start, x_range.end) == (0.0, 1.0)

    hook(plot, None)
    assert (x_range.start, x_range.end) == (2.0, 3.0)


def test_capture_is_idempotent() -> None:
    plot, x_range, y_range = _make_plot_xy()
    handles = RangeHandles()

    handles.capture(plot)
    assert handles.x_range is x_range
    assert handles.y_range is y_range

    other_x = _StubRange()
    other_y = _StubRange()
    plot.handles['x_range'] = other_x
    plot.handles['y_range'] = other_y

    handles.capture(plot)
    assert handles.x_range is x_range
    assert handles.y_range is y_range


def test_handles_accessible_after_first_hook_invocation() -> None:
    mapper = _StubColorMapper()
    x_range = _StubRange()
    y_range = _StubRange()
    plot = _StubPlot(x_range=x_range, y_range=y_range, color_mapper=mapper)
    handles = RangeHandles()
    hook = make_range_hook(
        axis='c',
        get_target=lambda: (0.0, 1.0),
        get_enabled=lambda: False,
        handles=handles,
    )

    hook(plot, None)

    assert handles.x_range is x_range
    assert handles.y_range is y_range
    assert handles.color_mapper is mapper


def test_c_hook_noop_when_color_mapper_missing() -> None:
    plot = _StubPlot()
    handles = RangeHandles()
    hook = make_range_hook(
        axis='c',
        get_target=lambda: (0.0, 1.0),
        get_enabled=lambda: True,
        handles=handles,
    )

    hook(plot, None)

    assert handles.color_mapper is None


def test_xy_hook_noop_when_range_handle_missing() -> None:
    plot = _StubPlot()
    handles = RangeHandles()
    hook = make_range_hook(
        axis='x',
        get_target=lambda: (0.0, 1.0),
        get_enabled=lambda: True,
        handles=handles,
    )

    hook(plot, None)

    assert handles.x_range is None
