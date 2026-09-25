# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DREAM logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.

The logical structure of each bank is taken from the pixel-id formulas of the DREAM
detector ICD (ESS-5462547, section 4.4). Every bank is built from cassettes; a
cassette has one set of cathode strips between two planes of anode wires (the two
counters), and a voxel is the crossing of a wire with a strip:

- mantle: 5 mounting units (``module``) of 6 cassettes, 2x32 wires, 256 strips
- endcap: 11 (backward) or 5 (forward) sectors of 28 cassettes, 2x16 wires,
  16 strips. The cassettes of a sector belong to SUMO6, SUMO5, SUMO4 and SUMO3
  (10, 8, 6 and 4 cassettes).
- HR/SANS: 33 or 36 cuboids of 8 cassettes, 2x16 wires, 32 strips

A plain ``fold`` of the ``detector_number``-sorted pixels does not give this
structure for all banks:

- Endcap: the ICD numbers wires top-down (``y = 16 strip + 15 - wire``).
- HR/SANS: cuboids sit on a 7x7 grid of 16x16 cells with the centre (and for HR a
  slot below it) left empty, so the ids have gaps, and the cuboids of each quadrant
  are rotated by a multiple of 90 degrees, so wires run along a different pixel-grid
  axis per quadrant.

Instead, :func:`logical_detector_number` evaluates the ICD formula on the full logical
index grid and the transforms gather pixels into that order before folding.
"""

from functools import cache

import numpy as np
import scipp as sc

_ENDCAP_SECTORS = {'endcap_forward_detector': 5, 'endcap_backward_detector': 11}
_ENDCAP_OFFSET = {'endcap_forward_detector': 0, 'endcap_backward_detector': 71680}
#: Cassettes per SUMO within a sector, in ICD order (SUMO6, SUMO5, SUMO4, SUMO3).
#: The logical ``cassette`` dim of the endcaps runs over all 28 in this order.
_SUMO_CASSETTES = (10, 8, 6, 4)

_MANTLE_OFFSET = 229376

_CUBOID_OFFSET = {'sans_detector': 720896, 'high_resolution_detector': 1122304}
#: Cuboid occupancy of the 7x7 grid (ICD Fig. 11), row 0 at the top. Cuboids are
#: numbered row-major over the occupied cells.
_CUBOID_GRID = {
    'high_resolution_detector': (
        '..###..',
        '.#####.',
        '#######',
        '###.###',
        '###.###',
        '.##.##.',
        '..#.#..',
    ),
    'sans_detector': (
        '..###..',
        '.#####.',
        '#######',
        '###.###',
        '#######',
        '.#####.',
        '..###..',
    ),
}


def _cuboid_rotation(row: int, col: int) -> int:
    """Rotation of the cuboid in a grid cell in units of 90 degrees (ICD Fig. 11).

    The four quadrants around the empty centre cell form a pinwheel.
    """
    if row <= 3 and col <= 2:
        return 0
    if row <= 2:
        return 1
    if col >= 4:
        return 2
    return 3


def _endcap_detector_number(source_name: str) -> sc.Variable:
    n_sector = _ENDCAP_SECTORS[source_name]
    sector, cassette, counter, wire, strip = np.ogrid[
        :n_sector, : sum(_SUMO_CASSETTES), :2, :16, :16
    ]
    # ICD: x = 56 sector + sumo_offset + 2 cassette + counter, where sumo_offset is
    # twice the number of cassettes in the preceding SUMOs. The flat cassette index
    # already includes those, so x = 56 sector + 2 cassette + counter.
    x = 2 * sum(_SUMO_CASSETTES) * sector + 2 * cassette + counter
    y = 16 * strip + 15 - wire
    width = 2 * sum(_SUMO_CASSETTES) * n_sector
    return sc.array(
        dims=['sector', 'cassette', 'counter', 'wire', 'strip'],
        values=_ENDCAP_OFFSET[source_name] + width * y + x + 1,
        unit=None,
    )


def _mantle_detector_number() -> sc.Variable:
    wire, module, cassette, counter, strip = np.ogrid[:32, :5, :6, :2, :256]
    y = 60 * wire + 12 * module + 2 * cassette + counter
    return sc.array(
        dims=['wire', 'module', 'cassette', 'counter', 'strip'],
        values=_MANTLE_OFFSET + 256 * y + strip + 1,
        unit=None,
    )


def _cuboid_detector_number(source_name: str) -> sc.Variable:
    cells = [
        (row, col)
        for row, line in enumerate(_CUBOID_GRID[source_name])
        for col, char in enumerate(line)
        if char == '#'
    ]
    cassette, counter, wire, strip = np.ogrid[:8, :2, :16, :32]
    xl = 2 * cassette + counter
    yl = 15 - wire
    ids = []
    for row, col in cells:
        # ICD section 4.4.4: local cell coordinates of a rotated cuboid.
        x, y = {
            0: (xl, yl),
            1: (15 - yl, xl),
            2: (15 - xl, 15 - yl),
            3: (yl, 15 - xl),
        }[_cuboid_rotation(row, col)]
        x = 16 * col + x
        y = 112 * strip + 16 * row + y
        ids.append(_CUBOID_OFFSET[source_name] + 112 * y + x + 1)
    return sc.array(
        dims=['cuboid', 'cassette', 'counter', 'wire', 'strip'],
        values=np.stack(ids),
        unit=None,
    )


def logical_detector_number(source_name: str) -> sc.Variable:
    """Return the ICD ``detector_number`` of every voxel of a bank in logical order.

    Parameters
    ----------
    source_name:
        Name of the detector bank.

    Returns
    -------
    :
        Detector numbers with dims ``(<unit>, cassette, counter, wire, strip)``, where
        the unit is ``sector`` (endcaps) or ``cuboid`` (HR/SANS). The mantle has dims
        ``(wire, module, cassette, counter, strip)``, the order of its ids.
    """
    if source_name == 'mantle_detector':
        return _mantle_detector_number()
    if source_name in _ENDCAP_SECTORS:
        return _endcap_detector_number(source_name)
    return _cuboid_detector_number(source_name)


@cache
def _gather_indices(source_name: str) -> tuple[np.ndarray | None, dict[str, int]]:
    """Positions of the logical voxels in the ``detector_number``-sorted pixel list.

    Returns ``None`` instead of the positions if the logical order is the sorted
    order, so the gather can be skipped.
    """
    number = logical_detector_number(source_name)
    flat = number.values.ravel()
    indices = np.argsort(flat)
    # Invert the sort: position of each logical voxel in the sorted pixel list.
    positions = np.empty_like(indices)
    positions[indices] = np.arange(flat.size)
    identity = bool(np.all(positions == np.arange(flat.size)))
    return (None if identity else positions), dict(number.sizes)


def _to_logical(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Gather the ``detector_number``-sorted pixels of a bank into logical order."""
    positions, sizes = _gather_indices(source_name)
    n_voxel = int(np.prod(list(sizes.values())))
    if da.sizes[da.dim] != n_voxel:
        raise ValueError(
            f"{source_name} has {da.sizes[da.dim]} pixels, but its ICD layout "
            f"has {n_voxel} voxels."
        )
    if positions is not None:
        da = da[da.dim, positions]
    return da.fold(dim=da.dim, sizes=sizes)


def _unit_dims(folded: sc.DataArray) -> tuple[str, ...]:
    """Dims identifying a cassette: its unit (sector/module/cuboid) and index."""
    return tuple(d for d in folded.dims if d not in ('counter', 'wire', 'strip'))


def _image_order(source_name: str, component: str, group: str) -> tuple[str, str]:
    """Return the (y, x) order of the per-component dim and the grouping dim.

    The mantle puts the grouping (azimuth) on y, so that strips, which run along
    the beam, are horizontal as in ICD Fig. 15. The other banks follow the ICD
    figures with the grouping on x.
    """
    if source_name == 'mantle_detector':
        return group, component
    return component, group


def get_mantle_front_layer(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to extract mantle front layer."""
    return _to_logical(da, source_name)['wire', 0].flatten(
        ('module', 'cassette', 'counter'), to='module/cassette/counter'
    )


def get_wire_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to fold detector data for wire view.

    Gathers raw detector data into its logical structure and flattens all dims
    identifying a wire plane (unit, cassette, counter). The subsequent summing over
    ``strip`` is handled by the reduction_dim parameter in add_logical_view to
    preserve binned event structure for histogramming.

    Parameters
    ----------
    da:
        Raw detector data with a single dimension.
    source_name:
        Name of the detector bank.

    Returns
    -------
    :
        Data with dimensions ``strip``, ``wire`` and ``<unit>/cassette/counter``.
        After reduction over ``strip``, each wire of the bank is one pixel.
    """
    folded = _to_logical(da, source_name)
    planes = (*_unit_dims(folded), 'counter')
    flat = '/'.join(planes)
    return (
        folded.transpose(('strip', 'wire', *planes))
        .flatten(planes, to=flat)
        .transpose(('strip', *_image_order(source_name, 'wire', flat)))
    )


def get_strip_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to fold detector data for strip view.

    Gathers raw detector data into its logical structure and flattens all dims
    identifying a cassette (unit, cassette). The subsequent summing over ``wire``
    and ``counter`` is handled by the reduction_dim parameter in add_logical_view to
    preserve binned event structure for histogramming.

    Parameters
    ----------
    da:
        Raw detector data with a single dimension.
    source_name:
        Name of the detector bank.

    Returns
    -------
    :
        Data with dimensions ``counter``, ``wire``, ``strip`` and
        ``<unit>/cassette``. After reduction over ``counter`` and ``wire``, each
        strip of the bank is one pixel.
    """
    folded = _to_logical(da, source_name)
    cassettes = _unit_dims(folded)
    flat = '/'.join(cassettes)
    return (
        folded.transpose(('counter', 'wire', 'strip', *cassettes))
        .flatten(cassettes, to=flat)
        .transpose(('counter', 'wire', *_image_order(source_name, 'strip', flat)))
    )
