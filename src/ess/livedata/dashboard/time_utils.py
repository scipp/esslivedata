# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Time utilities for local timezone handling in the dashboard.

This module provides utilities for converting UTC timestamps to local time
for display purposes. The main use case is making Bokeh/Holoviews display
time axes in local time rather than UTC.
"""

from datetime import UTC, datetime


def get_local_timezone_offset_ns() -> int:
    """
    Get the current local timezone offset from UTC in nanoseconds.

    Returns
    -------
    :
        Timezone offset in nanoseconds. Positive values mean local time
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
        return 0
    return int(offset.total_seconds() * 1_000_000_000)


def format_time_ns_local(ns: int) -> str:
    """
    Format nanoseconds since epoch as HH:MM:SS.s in local time.

    Parameters
    ----------
    ns:
        Nanoseconds since Unix epoch (UTC).

    Returns
    -------
    :
        Formatted time string with 0.1s precision, e.g., "14:32:05.3".
    """
    dt = datetime.fromtimestamp(ns / 1e9, tz=UTC).astimezone()
    return f"{dt.strftime('%H:%M:%S')}.{dt.microsecond // 100000}"
