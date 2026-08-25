# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Block structure and wire format of a streamed wavelength lookup table.

A lookup table is a uniform grid in ``distance``, so one table per component
would be the obvious layout -- and is a poor one: components an instrument
apart get near-identical tables (detectors sit within a couple of metres of
each other), while a single grid spanning everything would be mostly empty.
Empty is not free: BIFROST's monitors are 155 m from its detectors, which at a
0.1 m resolution is 1550 rows of 286 event-time-offset bins, a 3.5 MB message
where a few tens of kilobytes carry the information -- and 35 MB at a 0.01 m
resolution. A broker's default message limit is 1 MB.

A table therefore carries rows only where components are: one uniform *block*
per group of components that sit close together, concatenated along
``distance``. The detectors share one dense block; each monitor gets its own,
so the monitor table is a handful of rows at each monitor's flight path and
nothing in between.

This module owns both halves of the wire format: :func:`pack_blocks` builds
the published ``DataArray`` from the blocks, :func:`unpack_block` takes it
apart again into one job's ``LookupTable``. They are inverses, and live
together so what the wire carries is written down once.

The concatenation is deliberately *not* a uniform grid, and essreduce's
interpolator assumes one: ``interpolator_numba`` locates a row as
``int((ltotal - first) / (distance[1] - distance[0]))``, which reads the wrong
row -- silently, and only under numba, since the scipy fallback handles an
uneven axis correctly. A consumer must therefore select its own block with
:func:`select_block` and never hand a whole multi-block table to
``WavelengthInterpolator``.

The wire says outright which rows form a block, in the :data:`_BLOCK_COORD`
coord, rather than leaving the consumer to infer it from where the row spacing
jumps. Inferring it would be a threshold on a float difference, tuned against
the padding the table builder happens to add, and it would forbid the producer
from ever emitting two blocks that overlap or abut.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import scipp as sc
from ess.reduce.unwrap import LookupTable

#: Flight-path interval a single block covers, ``(start, stop)``.
Range = tuple[sc.Variable, sc.Variable]

#: Coord numbering the rows of each block consecutively from zero. Not a
#: ``LookupTable`` field -- purely the wire's block structure, dropped again on
#: the way in.
_BLOCK_COORD = 'block'


def one_block(ranges: Iterable[Range]) -> list[Range]:
    """Cover every range with a single dense block.

    The layout for detectors: they surround the sample, so their flight paths
    are clustered within metres of each other and the gaps between banks cost
    less than a per-bank block would.
    """
    lower, upper = zip(*ranges, strict=True)
    return [(min(lower, key=_metres), max(upper, key=_metres))]


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
    midpoint = 0.5 * (
        ltotal.nanmin().to(unit=distance.unit) + ltotal.nanmax().to(unit=distance.unit)
    )
    for start, stop in _block_bounds(table):
        block = table['distance', start:stop]
        bounds = block.coords['distance']
        if bounds[0] <= midpoint <= bounds[-1]:
            return block
    raise ValueError(
        f"No block of the streamed lookup table covers {midpoint:c}: the table "
        f"spans {_describe_blocks(table)}. The table was built from a different "
        "geometry than the one this job reduces."
    )


def block_ranges(table: sc.DataArray) -> Sequence[Range]:
    """The flight-path range of each block, for diagnostics and tests."""
    distance = table.coords['distance']
    return [
        (distance[start], distance[stop - 1]) for start, stop in _block_bounds(table)
    ]


def _block_bounds(table: sc.DataArray) -> list[tuple[int, int]]:
    """Half-open row index ranges of the table's uniform blocks."""
    block = table.coords[_BLOCK_COORD].values
    if len(block) == 0:
        return []
    splits = np.flatnonzero(block[1:] != block[:-1]) + 1
    edges = [0, *splits.tolist(), len(block)]
    return list(itertools.pairwise(edges))


def _describe_blocks(table: sc.DataArray) -> str:
    distance = table.coords['distance']
    return ', '.join(
        f"[{distance[start].value}, {distance[stop - 1].value}] {distance.unit}"
        for start, stop in _block_bounds(table)
    )


def _metres(value: sc.Variable) -> float:
    return value.to(unit='m').value


#: Non-array ``LookupTable`` fields, carried on the wire as 0-D coords. The
#: dataclass cannot be serialized as such -- da00 transports a ``DataArray``,
#: and ``pulse_stride`` is an ``int`` -- so the array carries them itself,
#: making the published message self-describing.
_SCALAR_FIELDS = (
    'pulse_period',
    'pulse_stride',
    'distance_resolution',
    'time_resolution',
)


def pack_blocks(blocks: Sequence[LookupTable]) -> sc.DataArray:
    """Concatenate blocks into the table as published.

    Every coord comes from the built table, never from the job's parameters,
    because these fields describe the table that was built rather than what was
    asked for. The two differ: ``pulse_stride`` may be guessed from the choppers
    instead of supplied, and the builder honours the requested time resolution
    only up to fitting a whole number of bins into the frame period. Parameter
    provenance rides on the identity coord instead (ADR 0010).

    The blocks share every scalar field: they are built from one cascade with
    one set of parameters, differing only in the range they cover. Which
    component the upstream ``LookupTable`` type is parametrised by is irrelevant
    to what is published, since the consumer picks its rows by flight path.
    """
    first = blocks[0]
    table = sc.concat([block.array for block in blocks], 'distance')
    table.coords[_BLOCK_COORD] = sc.concat(
        [
            sc.full(dims=['distance'], shape=[block.array.sizes['distance']], value=i)
            for i, block in enumerate(blocks)
        ],
        'distance',
    )
    table.coords['pulse_period'] = first.pulse_period
    table.coords['pulse_stride'] = sc.scalar(int(first.pulse_stride))
    table.coords['distance_resolution'] = first.distance_resolution
    table.coords['time_resolution'] = first.time_resolution
    return table


def unpack_block(table: sc.DataArray, ltotal: sc.Variable) -> dict[str, Any]:
    """Split one job's block of a published table into ``LookupTable`` fields.

    The inverse of :func:`pack_blocks`, restricted to the block covering
    ``ltotal``. Returned as a field mapping rather than a ``LookupTable`` so the
    caller supplies the run and component the key is parametrised by.
    """
    expected = (*_SCALAR_FIELDS, _BLOCK_COORD)
    if missing := [name for name in expected if name not in table.coords]:
        raise ValueError(
            f"Streamed lookup table is missing coord(s) {missing}; got coords "
            f"{sorted(table.coords)}. The producer attaches these, so this "
            "indicates a table from an incompatible producer version."
        )
    block = select_block(table, ltotal)
    return {
        'array': block.drop_coords(list(expected)),
        'pulse_period': block.coords['pulse_period'],
        'pulse_stride': int(block.coords['pulse_stride'].value),
        'distance_resolution': block.coords['distance_resolution'],
        'time_resolution': block.coords['time_resolution'],
    }
