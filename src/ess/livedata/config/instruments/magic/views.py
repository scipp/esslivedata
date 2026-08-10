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
#: Both banks are segments of a vertical (Y-axis) cylinder and share the same
#: axis semantics, deduced from the ``NXoff_geometry`` voxel centroids of the
#: coda_magic file. ``detector_number`` runs contiguously in C-order over
#: (wire, strip, segment), i.e. ``segment`` varies fastest and ``wire`` slowest;
#: the dict order is slowest-to-fastest so that ``fold`` reproduces this layout.
#:
#:   - ``wire``: anode wires into the detector depth. Radius is constant along
#:     strip and segment and spans ~0.50 m across the wires. The wires are
#:     slightly slanted, so radius is not purely radial: stepping along ``wire``
#:     also shifts Y and phi a little.
#:   - ``strip``: cathode strips along the vertical cylinder axis. Y is constant
#:     along segment and varies only along strip (and the wire slant).
#:   - ``segment``: azimuthal segments around the cylinder. Phi is constant along
#:     strip and varies only along segment (and the wire slant).
#:
#: Bank A spans ~1.47 m in Y over 62 deg of arc; bank B is short and wide,
#: ~0.15 m in Y over 120 deg of arc.
DETECTOR_BANK_SIZES: dict[str, dict[str, int]] = {
    'magic_detector_a': {'wire': 32, 'strip': 128, 'segment': 120},
    'magic_detector_b': {'wire': 32, 'strip': 16, 'segment': 256},
}


def get_wire_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to fold detector data for wire view.

    Folds raw detector data into its logical structure. The subsequent summing
    over ``strip`` is handled by the ``reduction_dim`` parameter in
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
        Folded data with dimensions (wire, strip, segment).
        After reduction over ``strip``: (wire, segment).
    """
    return da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES[source_name])


def get_strip_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to fold detector data for strip view.

    Folds raw detector data into its logical structure and flattens ``wire`` and
    ``segment`` into a single dimension. The subsequent summing over that
    dimension is handled by the ``reduction_dim`` parameter in add_logical_view
    to preserve binned event structure for histogramming.

    Parameters
    ----------
    da:
        Raw detector data with a single dimension.
    source_name:
        Name of the detector bank.

    Returns
    -------
    :
        Folded data with dimensions (wire/segment, strip).
        After reduction over ``wire/segment``: (strip,).
    """
    folded = da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES[source_name])
    return folded.transpose(('wire', 'segment', 'strip')).flatten(
        ('wire', 'segment'), to='wire/segment'
    )
