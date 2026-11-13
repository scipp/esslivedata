# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Display of wall-clock times in the facility's timezone.

All timestamps in the dashboard stay in UTC (epoch nanoseconds, or ``datetime64``
on plot axes). They are converted to the facility timezone only when formatted
for display: in Python by :func:`format_facility_time` and ``FACILITY_TIMEZONE``,
and in the browser by the Bokeh formatters :func:`facility_time_hook` installs.
Converting each value, instead of shifting data by a fixed offset, keeps the
display correct across daylight-saving changes.

The zone is fixed rather than taken from the server or the browser, so every
user sees the same times as the control room, wherever the dashboard runs or is
viewed from.
"""

from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

from ess.livedata.core.timestamp import Timestamp

#: IANA name of the timezone wall-clock times are displayed in. ESS is in Lund.
#: The same string configures both Python (``zoneinfo``) and the browser
#: (``Intl.DateTimeFormat``).
FACILITY_TIMEZONE_NAME = 'Europe/Stockholm'
FACILITY_TIMEZONE = ZoneInfo(FACILITY_TIMEZONE_NAME)


def format_facility_time(ts: Timestamp) -> str:
    """
    Format a timestamp as HH:MM:SS.s in the facility timezone.

    Parameters
    ----------
    ts:
        Timestamp to format.

    Returns
    -------
    :
        Formatted time string with 0.1s precision, e.g., "14:32:05.3".
    """
    dt = ts.to_datetime(tz=FACILITY_TIMEZONE)
    return f"{dt.strftime('%H:%M:%S')}.{dt.microsecond // 100000}"


# Bokeh formats datetime values in UTC, and its ``DatetimeTickFormatter`` and
# ``'datetime'`` hover formatter have no timezone option. The JavaScript below
# replaces them, converting each value with ``Intl.DateTimeFormat`` in the zone
# passed as ``tz``.

# Arguments of ``Intl.DateTimeFormat``. Constructing one costs ~5x formatting
# with it, so the tick formatter keeps it on ``this``, the cache object Bokeh
# binds for one formatting pass over all ticks.
_INTL_FORMAT_JS = """new Intl.DateTimeFormat('en-GB', {
    timeZone: tz, hourCycle: 'h23', timeZoneName: 'short',
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    fractionalSecondDigits: 3,
})"""

# ``parts(ms)``: the fields of epoch milliseconds in the facility timezone, keyed
# by Intl part type (year, month, day, hour, minute, second, fractionalSecond,
# timeZoneName). ``formatToParts`` instead of a formatted string, so the result
# does not depend on the locale's field order.
_PARTS_JS = """
const parts = (ms) => Object.fromEntries(
    fmt.formatToParts(new Date(ms)).map((part) => [part.type, part.value])
);
const dateOf = (p) => `${p.year}-${p.month}-${p.day}`;
"""

# Precision follows the tick spacing, as for Bokeh's DatetimeTickFormatter. Below
# a day's spacing, the date goes on a second line of the first tick and of every
# tick where the facility date changes, so an operator can tell the day apart.
_TICK_JS = f"""
const fmt = (this.fmt ??= {_INTL_FORMAT_JS});
{_PARTS_JS}
const p = parts(tick);
const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 0;
if (step >= 86400000) {{ return dateOf(p); }}
const hms = `${{p.hour}}:${{p.minute}}:${{p.second}}`;
const time = step >= 60000 ? `${{p.hour}}:${{p.minute}}`
    : step >= 1000 || step === 0 ? hms
    : `${{hms}}.${{p.fractionalSecond}}`;
const newDay = index === 0 || dateOf(parts(ticks[index - 1])) !== dateOf(p);
return newDay ? `${{time}}\n${{dateOf(p)}}` : time;
"""

# The zone abbreviation tells apart the hour that repeats when daylight saving
# ends (02:30 CEST, then 02:30 CET). Bokeh calls hover formatters unbound, so
# there is no cache to keep ``fmt`` on.
_HOVER_JS = f"""
if (!Number.isFinite(value)) {{ return 'NaN'; }}
const fmt = {_INTL_FORMAT_JS};
{_PARTS_JS}
const p = parts(value);
return `${{dateOf(p)}} ${{p.hour}}:${{p.minute}}:${{p.second}} ${{p.timeZoneName}}`;
"""

_FORMATTER_ARGS = {'tz': FACILITY_TIMEZONE_NAME}

# Attribute holding a hover tool's formatter. HoloViews resets the formatters of
# datetime fields to ``'datetime'`` on every update, so the hook re-installs one
# on every update; a new model each time would pile up in the browser, where
# BokehJS never frees replaced models. Bokeh does not serialize ``_``-prefixed
# attributes. Kept on the tool rather than in a module-level cache, which would
# keep a session's Document alive after the session is dropped.
_HOVER_FORMATTER_ATTR = '_facility_time_formatter'


def facility_time_hook(plot: Any, element: Any) -> None:
    """
    HoloViews hook showing datetime values in the facility timezone.

    Replaces the tick formatter of every ``DatetimeAxis`` of the figure and the
    ``'datetime'`` formatters of its hover tools. A hover field's format spec
    (e.g. ``%F %T``) is ignored in favor of ``YYYY-MM-DD HH:MM:SS <zone>``. The
    hook is a no-op on figures without datetime axes or datetime hover fields,
    so it can be attached to any element.

    Tick positions are still chosen by Bokeh's ``DatetimeTicker`` in UTC: ticks
    an hour or less apart fall on round facility times, but wider ones do not
    (6 h ticks read 01:00, 07:00, ... in winter). The labels are correct.
    """
    del element
    from bokeh.models import (
        CustomJSHover,
        CustomJSTickFormatter,
        DatetimeAxis,
        HoverTool,
    )

    figure = plot.state
    for axis in (*figure.xaxis, *figure.yaxis):
        if isinstance(axis, DatetimeAxis) and not isinstance(
            axis.formatter, CustomJSTickFormatter
        ):
            axis.formatter = CustomJSTickFormatter(args=_FORMATTER_ARGS, code=_TICK_JS)
    # ``toolbar.tools`` rather than ``figure.select``: the hook runs once per
    # leaf element on every update, and ``select`` walks the whole model graph,
    # which costs O(leaves^2) per update.
    for hover in figure.toolbar.tools:
        if not isinstance(hover, HoverTool) or 'datetime' not in (
            hover.formatters.values()
        ):
            continue
        if (formatter := getattr(hover, _HOVER_FORMATTER_ATTR, None)) is None:
            formatter = CustomJSHover(args=_FORMATTER_ARGS, code=_HOVER_JS)
            setattr(hover, _HOVER_FORMATTER_ATTR, formatter)
        hover.formatters = {
            field: formatter if value == 'datetime' else value
            for field, value in hover.formatters.items()
        }
