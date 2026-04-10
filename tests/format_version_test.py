# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest

from ess.livedata import format_version


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('26.4.2', '26.4.2'),
        ('1.0.0', '1.0.0'),
        ('0.0.0', '0.0.0'),
        ('26.4.2.dev0+g68b165851.d20260410', '26.4.2-dev (68b16585)'),
        ('1.2.3.dev42+gabcdef012.d20250101', '1.2.3-dev (abcdef01)'),
    ],
)
def test_format_version(raw: str, expected: str) -> None:
    assert format_version(raw) == expected
