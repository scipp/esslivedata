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
from weakref import WeakKeyDictionary
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

# ``p``: the fields of epoch milliseconds in the facility timezone, keyed by Intl
# part type (year, month, day, hour, minute, second, fractionalSecond,
# timeZoneName). ``formatToParts`` instead of a formatted string, so the result
# does not depend on the locale's field order.
_PARTS_JS = """
const p = Object.fromEntries(
    fmt.formatToParts(new Date(ms)).map((part) => [part.type, part.value])
);
"""

# Precision follows the tick spacing, as for Bokeh's DatetimeTickFormatter.
_TICK_JS = f"""
const fmt = (this.fmt ??= {_INTL_FORMAT_JS});
const ms = tick;
{_PARTS_JS}
const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 0;
if (step >= 86400000) {{ return `${{p.year}}-${{p.month}}-${{p.day}}`; }}
if (step >= 60000) {{ return `${{p.hour}}:${{p.minute}}`; }}
const hms = `${{p.hour}}:${{p.minute}}:${{p.second}}`;
return step >= 1000 || step === 0 ? hms : `${{hms}}.${{p.fractionalSecond}}`;
"""

# The zone abbreviation tells apart the hour that repeats when daylight saving
# ends (02:30 CEST, then 02:30 CET). Bokeh calls hover formatters unbound, so
# there is no cache to keep ``fmt`` on.
_HOVER_JS = f"""
const fmt = {_INTL_FORMAT_JS};
const ms = value;
{_PARTS_JS}
const date = `${{p.year}}-${{p.month}}-${{p.day}}`;
return `${{date}} ${{p.hour}}:${{p.minute}}:${{p.second}} ${{p.timeZoneName}}`;
"""

_FORMATTER_ARGS = {'tz': FACILITY_TIMEZONE_NAME}

# One hover formatter per hover tool. HoloViews resets the formatters of datetime
# fields to ``'datetime'`` on every update, so the hook re-installs one on every
# update; a new model each time would pile up in the browser, where BokehJS never
# frees replaced models.
_hover_formatters: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


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
        if (formatter := _hover_formatters.get(hover)) is None:
            formatter = CustomJSHover(args=_FORMATTER_ARGS, code=_HOVER_JS)
            _hover_formatters[hover] = formatter
        hover.formatters = {
            field: formatter if value == 'datetime' else value
            for field, value in hover.formatters.items()
        }
