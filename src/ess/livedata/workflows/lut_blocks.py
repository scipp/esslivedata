# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Block structure of a streamed wavelength lookup table.

A lookup table is a uniform grid in ``distance``, so one table per component
would be the obvious layout -- and is a poor one: components an instrument
apart get near-identical tables (detectors sit within a couple of metres of
each other), while a single grid spanning everything would be mostly empty
(monitors sit tens to hundreds of metres upstream of the detectors).

A table instead carries one uniform *block* per group of components that sit
close together, concatenated along ``distance``. The detectors share one dense
block; each monitor gets its own, so the monitor table is a handful of rows at
each monitor's flight path and nothing in between.

The concatenation is deliberately *not* a uniform grid, and essreduce's
interpolator assumes one: ``interpolator_numba`` locates a row as
``int((ltotal - first) / (distance[1] - distance[0]))``, which reads the wrong
row -- silently, and only under numba, since the scipy fallback handles an
uneven axis correctly. A consumer must therefore select its own block with
:func:`select_block` and never hand a whole multi-block table to
``WavelengthInterpolator``.

What makes that recoverable from the wire is the gap: blocks are separated by
more than :data:`_SPLIT_FACTOR` resolution steps, while rows *within* a block
are exactly one step apart. The producer guarantees it by merging ranges that
would land closer (see :func:`blocks_by_gap`), leaving :data:`_MERGE_FACTOR`
resolution steps of headroom over what the consumer needs to see.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

import numpy as np
import scipp as sc

#: Flight-path interval a single block covers, ``(start, stop)``.
Range = tuple[sc.Variable, sc.Variable]

#: Ranges closer than this many resolution steps are merged into one block.
#: Larger than :data:`_SPLIT_FACTOR` because the table builder pads each block
#: by two steps at each end: the *rows* of two blocks are four steps closer
#: than their ranges.
_MERGE_FACTOR = 6.0

#: A jump of more than this many resolution steps between consecutive rows
#: starts a new block. Rows within a block are exactly one step apart, so
#: anything above one separates blocks; the margin absorbs float error in the
#: builder's ``arange``.
_SPLIT_FACTOR = 1.5


def one_block(ranges: Iterable[Range]) -> list[Range]:
    """Cover every range with a single dense block.

    The layout for detectors: they surround the sample, so their flight paths
    are clustered within metres of each other and the gaps between banks cost
    less than a per-bank block would.
    """
    lower, upper = zip(*ranges, strict=True)
    return [(min(lower, key=_metres), max(upper, key=_metres))]


def blocks_by_gap(ranges: Iterable[Range], resolution: sc.Variable) -> list[Range]:
    """One block per range, merging any that would sit too close to tell apart.

    The layout for monitors: they are strung out along the beamline, so a block
    each keeps the table to a few rows per monitor instead of a uniform grid
    over the tens or hundreds of metres between them.
    """
    min_gap = _MERGE_FACTOR * _metres(resolution)
    blocks: list[Range] = []
    for lower, upper in sorted(ranges, key=lambda r: _metres(r[0])):
        if blocks and _metres(lower) - _metres(blocks[-1][1]) <= min_gap:
            previous = blocks[-1]
            blocks[-1] = (previous[0], max(previous[1], upper, key=_metres))
        else:
            blocks.append((lower, upper))
    return blocks


def select_block(table: sc.DataArray, ltotal: sc.Variable) -> sc.DataArray:
    """The block of ``table`` covering the flight paths in ``ltotal``.

    Selects the block containing the midpoint of ``ltotal``, rather than the
    one containing all of it: a pixel beyond the table's range is a documented
    NaN (the range is padded, not guaranteed), whereas demanding full coverage
    would turn one stray pixel into a failed job.

    Raises
    ------
    ValueError:
        If no block covers the midpoint, i.e. the table was built for a
        different beamline than the one being reduced.
    """
    distance = table.coords['distance']
    resolution = table.coords['distance_resolution'].to(unit=distance.unit)
    midpoint = 0.5 * (
        ltotal.nanmin().to(unit=distance.unit) + ltotal.nanmax().to(unit=distance.unit)
    )
    for start, stop in _block_bounds(distance, resolution):
        block = table['distance', start:stop]
        bounds = block.coords['distance']
        if bounds[0] <= midpoint <= bounds[-1]:
            return block
    raise ValueError(
        f"No block of the streamed lookup table covers {midpoint:c}: the table "
        f"spans {_describe_blocks(distance, resolution)}. The table was built "
        "from a different geometry than the one this job reduces."
    )


def _block_bounds(
    distance: sc.Variable, resolution: sc.Variable
) -> list[tuple[int, int]]:
    """Half-open row index ranges of the table's uniform blocks."""
    if len(distance) == 0:
        return []
    steps = distance.values[1:] - distance.values[:-1]
    splits = np.flatnonzero(steps > _SPLIT_FACTOR * resolution.value) + 1
    edges = [0, *splits.tolist(), len(distance)]
    return list(itertools.pairwise(edges))


def _describe_blocks(distance: sc.Variable, resolution: sc.Variable) -> str:
    return ', '.join(
        f"[{distance[start].value}, {distance[stop - 1].value}] {distance.unit}"
        for start, stop in _block_bounds(distance, resolution)
    )


def _metres(value: sc.Variable) -> float:
    return value.to(unit='m').value


def block_ranges(table: sc.DataArray) -> Sequence[Range]:
    """The flight-path range of each block, for diagnostics and tests."""
    distance = table.coords['distance']
    resolution = table.coords['distance_resolution'].to(unit=distance.unit)
    return [
        (distance[start], distance[stop - 1])
        for start, stop in _block_bounds(distance, resolution)
    ]
