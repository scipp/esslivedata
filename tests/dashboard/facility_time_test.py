# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import holoviews as hv
import numpy as np
import pytest
from bokeh.models import CustomJSHover, CustomJSTickFormatter, DatetimeAxis, HoverTool
from holoviews.streams import Pipe

from ess.livedata.core.timestamp import Timestamp
from ess.livedata.dashboard.facility_time import (
    facility_time_hook,
    format_facility_time,
)

hv.extension('bokeh')


def _utc(*args: int, microsecond: int = 0) -> Timestamp:
    dt = datetime(*args, microsecond=microsecond, tzinfo=UTC)
    return Timestamp.from_ns(int(dt.timestamp()) * 10**9 + microsecond * 1000)


class TestFormatFacilityTime:
    def test_winter_time_is_utc_plus_one(self):
        assert format_facility_time(_utc(2026, 1, 15, 12, 0, 0)) == '13:00:00.0'

    def test_summer_time_is_utc_plus_two(self):
        assert format_facility_time(_utc(2026, 7, 15, 12, 0, 0)) == '14:00:00.0'

    def test_offset_changes_at_spring_daylight_saving_change(self):
        # Clocks go from 02:00 CET to 03:00 CEST at 01:00 UTC on 2026-03-29.
        assert format_facility_time(_utc(2026, 3, 29, 0, 59, 59)) == '01:59:59.0'
        assert format_facility_time(_utc(2026, 3, 29, 1, 0, 0)) == '03:00:00.0'

    def test_hour_repeats_at_autumn_daylight_saving_change(self):
        # Clocks go from 03:00 CEST back to 02:00 CET at 01:00 UTC on 2026-10-25.
        assert format_facility_time(_utc(2026, 10, 25, 0, 30, 0)) == '02:30:00.0'
        assert format_facility_time(_utc(2026, 10, 25, 1, 30, 0)) == '02:30:00.0'

    def test_tenths_of_seconds_are_truncated(self):
        ts = _utc(2026, 1, 15, 12, 0, 5, microsecond=399_999)
        assert format_facility_time(ts) == '13:00:05.3'


def _ms(*args: int) -> float:
    return datetime(*args, tzinfo=UTC).timestamp() * 1000


def _curve(n: int, label: str = '') -> hv.Curve:
    times = np.datetime64('2026-03-29T00:30', 'ns') + np.arange(n) * np.timedelta64(
        10, 'm'
    )
    return hv.Curve((times, np.arange(n, dtype=float)), 'time', 'value', label=label)


def _render(callback: Callable[[int], hv.Element]) -> tuple[Any, Pipe]:
    """Render as a session does: a DynamicMap updated through a Pipe."""
    pipe = Pipe(data=3)
    dmap = hv.DynamicMap(callback, streams=[pipe]).opts(
        hv.opts.Curve(hooks=[facility_time_hook], tools=['hover'])
    )
    return hv.renderer('bokeh').get_plot(dmap).state, pipe


def _single(data: int) -> hv.Curve:
    return _curve(data)


def _overlay(data: int) -> hv.Overlay:
    return _curve(data, label='a') * _curve(data, label='b')


def _hover_formatters(fig: Any) -> list[Any]:
    return [
        hover.formatters['@{time}']
        for hover in fig.toolbar.tools
        if isinstance(hover, HoverTool)
    ]


class TestFacilityTimeHook:
    @pytest.mark.parametrize('callback', [_single, _overlay])
    def test_installs_facility_time_formatters(self, callback):
        fig, _ = _render(callback)

        assert isinstance(fig.xaxis[0], DatetimeAxis)
        assert isinstance(fig.xaxis[0].formatter, CustomJSTickFormatter)
        formatters = _hover_formatters(fig)
        assert formatters
        assert all(isinstance(f, CustomJSHover) for f in formatters)

    @pytest.mark.parametrize('callback', [_single, _overlay])
    def test_updates_keep_the_installed_models(self, callback):
        """HoloViews resets hover formatters on update; no new model per update.

        BokehJS never frees replaced models, so a new one per update would grow
        browser memory for as long as the session is open.
        """
        fig, pipe = _render(callback)
        tick = fig.xaxis[0].formatter
        hovers = _hover_formatters(fig)

        pipe.send(5)
        pipe.send(7)

        assert fig.xaxis[0].formatter is tick
        assert all(a is b for a, b in zip(_hover_formatters(fig), hovers, strict=True))

    def test_numeric_axis_and_hover_are_left_alone(self):
        dmap = hv.DynamicMap(
            lambda: hv.Curve(([0.0, 1.0], [1.0, 2.0]), 'x', 'value')
        ).opts(hv.opts.Curve(hooks=[facility_time_hook], tools=['hover']))
        fig = hv.renderer('bokeh').get_plot(dmap).state

        assert not isinstance(fig.xaxis[0].formatter, CustomJSTickFormatter)
        for hover in fig.toolbar.tools:
            if isinstance(hover, HoverTool):
                assert not any(
                    isinstance(f, CustomJSHover) for f in hover.formatters.values()
                )


@pytest.fixture(scope='module')
def js_page() -> Iterator[Any]:
    pytest.importorskip('playwright.sync_api')
    from tests.helpers.browser import open_browser

    with open_browser() as browser:
        yield browser.new_page()


@pytest.fixture(scope='module')
def shipped_formatters() -> tuple[Any, Any]:
    """The tick and hover formatter models as sent to the browser."""
    fig, _ = _render(_single)
    return fig.xaxis[0].formatter, _hover_formatters(fig)[0]


# Bokeh 3.9 compiles both formatters in strict mode, with their ``args`` as named
# parameters. It calls the tick formatter once per tick with ``this`` bound to a
# cache shared by one formatting pass, and the hover formatter unbound.
_CALL_TICK_JS = """([names, values, code, ticks]) => {
    const f = new Function('tick', 'index', 'ticks', ...names, '"use strict";' + code);
    const cache = {};
    return ticks.map((tick, i) => f.call(cache, tick, i, ticks, ...values));
}"""
_CALL_HOVER_JS = """([names, values, code, value]) => {
    const f = new Function(
        ...names, 'value', 'format', 'special_vars', '"use strict";' + code
    );
    return f(...values, value, '%F %T', {});
}"""


def _ticks(page: Any, formatter: Any, ticks: list[float]) -> list[str]:
    args = dict(formatter.args)
    return page.evaluate(
        _CALL_TICK_JS, [list(args), list(args.values()), formatter.code, ticks]
    )


def _hover(page: Any, formatter: Any, value: float) -> str:
    args = dict(formatter.args)
    return page.evaluate(
        _CALL_HOVER_JS, [list(args), list(args.values()), formatter.code, value]
    )


@pytest.mark.browser
class TestFacilityTimeJavaScript:
    def test_hover_shows_winter_and_summer_time_with_zone(
        self, js_page, shipped_formatters
    ):
        _, hover = shipped_formatters
        winter = _hover(js_page, hover, _ms(2026, 1, 15, 12, 0, 0))
        summer = _hover(js_page, hover, _ms(2026, 7, 15, 12, 0, 0))
        assert winter == '2026-01-15 13:00:00 CET'
        assert summer == '2026-07-15 14:00:00 CEST'

    def test_hover_tells_apart_the_repeated_autumn_hour(
        self, js_page, shipped_formatters
    ):
        _, hover = shipped_formatters
        first = _hover(js_page, hover, _ms(2026, 10, 25, 0, 30, 0))
        second = _hover(js_page, hover, _ms(2026, 10, 25, 1, 30, 0))
        assert first == '2026-10-25 02:30:00 CEST'
        assert second == '2026-10-25 02:30:00 CET'

    @pytest.mark.parametrize(
        ('step_ms', 'expected'),
        [
            (100, ['13:00:05.250\n2026-01-15', '13:00:05.350']),
            (1_000, ['13:00:05\n2026-01-15', '13:00:06']),
            (60_000, ['13:00\n2026-01-15', '13:01']),
            (86_400_000, ['2026-01-15', '2026-01-16']),
        ],
    )
    def test_tick_precision_follows_tick_spacing(
        self, js_page, shipped_formatters, step_ms, expected
    ):
        tick, _ = shipped_formatters
        start = _ms(2026, 1, 15, 12, 0, 5) + 250
        assert _ticks(js_page, tick, [start, start + step_ms]) == expected

    def test_ticks_across_spring_daylight_saving_change(
        self, js_page, shipped_formatters
    ):
        tick, _ = shipped_formatters
        start = _ms(2026, 3, 29, 0, 30, 0)
        ticks = [start + i * 1_800_000 for i in range(3)]
        assert _ticks(js_page, tick, ticks) == ['01:30\n2026-03-29', '03:00', '03:30']

    def test_ticks_across_autumn_daylight_saving_change(
        self, js_page, shipped_formatters
    ):
        tick, _ = shipped_formatters
        start = _ms(2026, 10, 25, 0, 30, 0)
        ticks = [start + i * 1_800_000 for i in range(3)]
        assert _ticks(js_page, tick, ticks) == ['02:30\n2026-10-25', '02:00', '02:30']

    def test_tick_where_facility_date_changes_shows_the_date(
        self, js_page, shipped_formatters
    ):
        tick, _ = shipped_formatters
        # 22:00 UTC is midnight CEST.
        start = _ms(2026, 7, 15, 20, 0, 0)
        ticks = [start + i * 3_600_000 for i in range(4)]
        assert _ticks(js_page, tick, ticks) == [
            '22:00\n2026-07-15',
            '23:00',
            '00:00\n2026-07-16',
            '01:00',
        ]

    def test_hover_on_nan_does_not_throw(self, js_page, shipped_formatters):
        _, hover = shipped_formatters
        assert _hover(js_page, hover, float('nan')) == 'NaN'
