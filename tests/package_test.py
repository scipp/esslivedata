# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Tests of package integrity.

Note that additional imports need to be added for repositories that
contain multiple packages.
"""

import pytest

from ess import livedata as pkg
from ess.livedata import format_version


def test_has_version():
    assert hasattr(pkg, '__version__')


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


# This is for CI package tests. They need to run tests with minimal dependencies,
# that is, without installing pytest. This code does not affect pytest.
if __name__ == '__main__':
    test_has_version()
