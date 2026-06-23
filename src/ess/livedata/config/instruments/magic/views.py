# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
MAGIC logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py. They operate purely on the logical voxel structure and require no
physical geometry.
"""

import scipp as sc

#: Logical voxel structure of the MAGIC detector banks.
#:
#: ``magic_detector_a`` (main bank) is derived from the NXoff_geometry voxel
#: centroids of the coda_magic geometry file (245760 voxels). It is a segment of
#: a vertical (Y-axis) cylinder. ``detector_number`` runs contiguously in C-order
#: over (wire, strip, segment), i.e. ``segment`` varies fastest and ``wire``
#: slowest. The axes, deduced from the centroids:
#:   - ``wire`` (32): anode wires into the detector depth; radius runs 1.02 -> 2.50 m
#:                    and is slightly slanted in Y (not purely radial).
#:   - ``strip`` (128): cathode strips along the vertical cylinder axis (~0.99 m).
#:   - ``segment`` (60): azimuthal segments around the cylinder (~59 deg, ~1 deg each).
#: The dict order is slowest-to-fastest so that ``fold`` reproduces this layout.
#:
#: ``magic_detector_b`` (polarization bank) has no geometry file yet; its sizes and
#: nesting order are provisional placeholders matching only the known voxel count.
DETECTOR_BANK_SIZES: dict[str, dict[str, int]] = {
    'magic_detector_a': {'wire': 32, 'strip': 128, 'segment': 60},
    # 32 cathode strips x 16 anode wires x 2 wire planes per cassette,
    # 16 cassettes per 15-degree module, 8 modules.
    'magic_detector_b': {
        'module': 8,
        'cassette': 16,
        'strip': 32,
        'wire': 16,
        'counter': 2,
    },
}


def get_wire_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to fold detector data for wire view.

    Folds raw detector data into its logical structure and flattens all
    dimensions except ``wire`` and ``strip`` into ``other``. The subsequent
    summing over ``strip`` is handled by the ``reduction_dim`` parameter in
    add_logical_view to preserve binned event structure for histogramming.

    Parameters
    ----------
    da:
        Raw detector data with a single dimension.
    source_name:
        Name of the detector bank.

    Returns
    -------
    :
        Folded data with dimensions (strip, wire, other).
        After reduction over ``strip``: (wire, other).
    """
    folded = da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES[source_name])
    other_dims = tuple(d for d in folded.dims if d not in ('wire', 'strip'))
    return folded.transpose(('strip', 'wire', *other_dims)).flatten(
        other_dims, to='other'
    )


def get_strip_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to fold detector data for strip view.

    Folds raw detector data into its logical structure and flattens all non-strip
    dimensions into ``other``. The subsequent summing over ``other`` is handled by
    the ``reduction_dim`` parameter in add_logical_view to preserve binned event
    structure for histogramming.

    Parameters
    ----------
    da:
        Raw detector data with a single dimension.
    source_name:
        Name of the detector bank.

    Returns
    -------
    :
        Folded data with dimensions (other, strip).
        After reduction over ``other``: (strip,).
    """
    folded = da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES[source_name])
    non_strip_dims = tuple(d for d in folded.dims if d != 'strip')
    return folded.transpose((*non_strip_dims, 'strip')).flatten(
        non_strip_dims, to='other'
    )
