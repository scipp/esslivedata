# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.dashboard.format_utils import extract_error_summary


@pytest.mark.parametrize(
    ("error_message", "expected"),
    [
        pytest.param(
            "ValueError: bad value",
            "ValueError: bad value",
            id="single_line",
        ),
        pytest.param(
            (
                "Traceback (most recent call last):\n"
                '  File "foo.py", line 10, in bar\n'
                "    result = baz()\n"
                "ValueError: bad value"
            ),
            "ValueError: bad value",
            id="traceback",
        ),
        pytest.param(
            (
                "Job failed to compute result.\n"
                "\n"
                "Traceback (most recent call last):\n"
                '  File "foo.py", line 10, in bar\n'
                "    result = baz()\n"
                "TypeError: unsupported operand"
            ),
            "TypeError: unsupported operand",
            id="prefixed_traceback",
        ),
        pytest.param(
            "  some error\n  ",
            "some error",
            id="strips_whitespace",
        ),
    ],
)
def test_extract_error_summary(error_message: str, expected: str) -> None:
    assert extract_error_summary(error_message) == expected
