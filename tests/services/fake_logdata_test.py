# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.services.fake_logdata import find_image_key_streams, image_key_at

PHASE_NS = int(10e9)


def test_image_key_cycles_through_dark_flat_field_and_sample() -> None:
    keys = [image_key_at(phase * PHASE_NS) for phase in range(4)]

    assert keys == [2, 1, 0, 2]


def test_image_key_is_constant_within_a_phase() -> None:
    assert image_key_at(1) == image_key_at(PHASE_NS - 1)


@pytest.mark.parametrize(
    ('instrument', 'expected'),
    [('tbl', ['orca_detector/image_key']), ('dummy', [])],
)
def test_finds_the_image_key_stream_of_an_instrument(
    instrument: str, expected: list[str]
) -> None:
    assert find_image_key_streams(instrument) == expected
