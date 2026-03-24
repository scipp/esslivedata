# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Time utilities for local timezone handling in the dashboard.

This module provides utilities for converting UTC timestamps to local time
for display purposes. The main use case is making Bokeh/Holoviews display
time axes in local time rather than UTC.
"""

from datetime import UTC, datetime

from ess.livedata.core.timestamp import Duration, Timestamp


def get_local_timezone_offset_ns() -> Duration:
    """
    Get the current local timezone offset from UTC in nanoseconds.

    Returns
    -------
    :
        Timezone offset as a Duration. Positive values mean local time
        is ahead of UTC (e.g., +1h for CET), negative values mean local
        time is behind UTC (e.g., -5h for EST).

    Notes
    -----
    The offset is computed for the current time and may change with
    daylight saving time transitions. For live data visualization,
    this is acceptable since all displayed data is typically recent.
    """
    now_utc = datetime.now(tz=UTC)
    now_local = now_utc.astimezone()
    offset = now_local.utcoffset()
    if offset is None:
        return Duration.from_ns(0)
    return Duration.from_seconds(offset.total_seconds())


def format_time_ns_local(ns: Timestamp) -> str:
    """
    Format a timestamp as HH:MM:SS.s in local time.

    Parameters
    ----------
    ns:
        Timestamp to format.

    Returns
    -------
    :
        Formatted time string with 0.1s precision, e.g., "14:32:05.3".
    """
    dt = ns.to_datetime().astimezone()
    return f"{dt.strftime('%H:%M:%S')}.{dt.microsecond // 100000}"
