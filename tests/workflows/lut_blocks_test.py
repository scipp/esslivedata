# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the block layout of a streamed wavelength lookup table.

The producer decides how many blocks a table has and says so on the wire; the
consumer selects the one covering its flight path. These tests pin the two ends
of that contract against each other.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipp as sc

from ess.livedata.workflows.lut_blocks import (
    block_ranges,
    one_block,
    select_block,
)


def _range(lower: float, upper: float) -> tuple[sc.Variable, sc.Variable]:
    return (sc.scalar(lower, unit='m'), sc.scalar(upper, unit='m'))


def _table(
    *blocks: tuple[float, float],
    resolution: float = 0.1,
    blocked: frozenset[int] = frozenset(),
) -> sc.DataArray:
    """A table built the way the producer builds one: block per range.

    ``blocked`` names the blocks the cascade transmits nothing through, which
    the producer emits as all-NaN.
    """
    rows = []
    ids = []
    values = []
    for index, (lower, upper) in enumerate(blocks):
        count = round((upper - lower) / resolution) + 1
        rows.append(sc.linspace('distance', lower, upper, count, unit='m'))
        ids.append(sc.full(dims=['distance'], shape=[count], value=index))
        values.append(
            sc.full(
                dims=['distance', 'event_time_offset'],
                shape=[count, 2],
                value=np.nan if index in blocked else 0.0,
                unit='angstrom',
            )
        )
    distance = sc.concat(rows, 'distance')
    return sc.DataArray(
        sc.concat(values, 'distance'),
        coords={
            'distance': distance,
            'block': sc.concat(ids, 'distance'),
            'distance_resolution': sc.scalar(resolution, unit='m'),
        },
    )


class TestOneBlock:
    def test_spans_every_range(self) -> None:
        blocks = one_block([_range(30.9, 31.2), _range(30.8, 31.0)])

        assert blocks == [_range(30.8, 31.2)]

    def test_covers_the_gap_between_detached_ranges(self) -> None:
        # The detector layout: banks that tile a stretch of beamline share one
        # dense block rather than paying a block's padding each.
        assert one_block([_range(24.8, 25.4), _range(28.4, 43.8)]) == [
            _range(24.8, 43.8)
        ]


class TestSelectBlock:
    def test_recovers_the_producer_blocks(self) -> None:
        table = _table((6.7, 6.9), (15.1, 15.3), (23.7, 23.9))

        assert len(block_ranges(table)) == 3

    @pytest.mark.parametrize('ltotal', [6.7, 6.8, 6.9])
    def test_selects_the_covering_block(self, ltotal: float) -> None:
        table = _table((6.7, 6.9), (15.1, 15.3))

        block = select_block(table, sc.scalar(ltotal, unit='m'))

        assert block.coords['distance'][0].value == pytest.approx(6.7)
        assert block.coords['distance'][-1].value == pytest.approx(6.9)

    def test_a_single_block_table_selects_whole(self) -> None:
        table = _table((30.8, 31.2))

        block = select_block(table, sc.scalar(31.0, unit='m'))

        assert block.sizes == table.sizes

    def test_unit_conversion(self) -> None:
        table = _table((6.7, 6.9))

        block = select_block(table, sc.scalar(6800.0, unit='mm'))

        assert block.sizes['distance'] == 3

    def test_flight_path_outside_every_block_fails_loud(self) -> None:
        table = _table((6.7, 6.9), (15.1, 15.3))

        with pytest.raises(ValueError, match='No block'):
            select_block(table, sc.scalar(10.0, unit='m'))

    def test_prefers_a_transmitting_block_over_the_first_covering_one(
        self,
    ) -> None:
        # Two monitors whose padded blocks overlap, with a chopper between
        # them: the downstream block (emitted first) is shut, the upstream one
        # transmits. A flight path in the overlap belongs to the monitor that
        # still sees beam, so taking the first covering block would refuse a
        # job whose own cascade is fine.
        table = _table((6.7, 7.1), (6.5, 6.9), blocked=frozenset({0}))

        block = select_block(table, sc.scalar(6.8, unit='m'))

        assert block.coords['distance'][0].value == pytest.approx(6.5)
        assert block.coords['distance'][-1].value == pytest.approx(6.9)

    def test_all_covering_blocks_blocked_still_returns_one(self) -> None:
        # Nothing transmits at this flight path. Returning a block anyway lets
        # the caller report it as blocked; raising here would report it as
        # absent from the table, which is a different fault.
        table = _table((6.7, 7.1), (6.5, 6.9), blocked=frozenset({0, 1}))

        block = select_block(table, sc.scalar(6.8, unit='m'))

        assert block.coords['distance'][0].value == pytest.approx(6.7)

    def test_abutting_blocks_stay_distinct(self) -> None:
        # Two monitors close enough that their rows run into each other. The
        # wire says where the boundary is, so nothing has to be inferred from
        # the row spacing and the producer need not merge them.
        table = _table((6.7, 6.9), (6.9, 7.1))

        assert len(block_ranges(table)) == 2
        block = select_block(table, sc.scalar(7.0, unit='m'))
        assert block.coords['distance'][0].value == pytest.approx(6.9)
        assert block.coords['distance'][-1].value == pytest.approx(7.1)
