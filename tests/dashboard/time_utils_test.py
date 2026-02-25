# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import time
from datetime import UTC, datetime

from ess.livedata.dashboard.time_utils import (
    format_time_ns_for_filename,
    format_time_ns_local,
    get_local_timezone_offset_ns,
)


class TestGetLocalTimezoneOffsetNs:
    """Tests for get_local_timezone_offset_ns."""

    def test_returns_integer(self):
        """Timezone offset is returned as an integer."""
        offset = get_local_timezone_offset_ns()
        assert isinstance(offset, int)

    def test_offset_in_reasonable_range(self):
        """Timezone offset should be within +/- 14 hours of UTC."""
        offset = get_local_timezone_offset_ns()
        max_offset_ns = 14 * 60 * 60 * 1_000_000_000  # 14 hours in ns
        assert -max_offset_ns <= offset <= max_offset_ns

    def test_offset_matches_python_datetime(self):
        """Offset should match what Python's datetime returns."""
        offset_ns = get_local_timezone_offset_ns()
        offset_s = offset_ns / 1_000_000_000

        now_utc = datetime.now(tz=UTC)
        now_local = now_utc.astimezone()
        expected_offset = now_local.utcoffset()
        assert expected_offset is not None
        expected_offset_s = expected_offset.total_seconds()

        assert offset_s == expected_offset_s


class TestFormatTimeNsLocal:
    """Tests for format_time_ns_local."""

    def test_format_returns_string(self):
        """Format returns a string."""
        ns = int(1.733e18)
        result = format_time_ns_local(ns)
        assert isinstance(result, str)

    def test_format_includes_decimal(self):
        """Format includes a decimal point for subsecond precision."""
        ns = int(1.733e18)
        result = format_time_ns_local(ns)
        assert '.' in result

    def test_format_matches_expected_pattern(self):
        """Format matches HH:MM:SS.d pattern."""
        ns = int(1.733e18)
        result = format_time_ns_local(ns)
        # Pattern: "HH:MM:SS.d" where d is 0-9
        parts = result.split(':')
        assert len(parts) == 3
        assert len(parts[0]) == 2  # HH
        assert len(parts[1]) == 2  # MM
        assert len(parts[2]) == 4  # SS.d

    def test_format_uses_local_time(self):
        """Format uses local time, not UTC."""
        ns = time.time_ns()
        result = format_time_ns_local(ns)

        # Compare with what Python datetime returns
        dt = datetime.fromtimestamp(ns / 1e9, tz=UTC).astimezone()
        expected_hour = f"{dt.hour:02d}"
        expected_minute = f"{dt.minute:02d}"
        expected_second = f"{dt.second:02d}"
        expected_subsecond = str(dt.microsecond // 100000)

        expected = (
            f"{expected_hour}:{expected_minute}:{expected_second}.{expected_subsecond}"
        )
        assert result == expected

    def test_format_zero_subsecond(self):
        """Format handles zero subsecond precision correctly."""
        # Use a timestamp with exactly 0 microseconds
        # 1733000000.0 seconds since epoch
        ns = 1733000000 * 1_000_000_000
        result = format_time_ns_local(ns)
        assert result.endswith('.0')


class TestFormatTimeNsForFilename:
    """Tests for format_time_ns_for_filename."""

    def test_format_matches_expected_pattern(self):
        """Format matches YYYY-MM-DDTHH-MM pattern."""
        import re

        ns = int(1.733e18)
        result = format_time_ns_for_filename(ns)
        assert re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}', result)

    def test_format_uses_local_time(self):
        """Format uses local time, not UTC."""
        ns = time.time_ns()
        result = format_time_ns_for_filename(ns)

        dt = datetime.fromtimestamp(ns / 1e9, tz=UTC).astimezone()
        expected = dt.strftime('%Y-%m-%dT%H-%M')
        assert result == expected

    def test_format_is_filename_safe(self):
        """Format contains no characters unsafe for filenames."""
        ns = time.time_ns()
        result = format_time_ns_for_filename(ns)
        unsafe_chars = set(':/\\*?"<>| ')
        assert not any(c in unsafe_chars for c in result)
