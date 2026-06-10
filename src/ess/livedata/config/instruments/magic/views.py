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
#: Both banks resemble the DREAM mantle: a vertical cylinder of cathode strips
#: (along the wire length) and anode wires (into the detector depth), with two
#: wire planes per detector element. Dimension names and the nesting order are
#: provisional placeholders chosen to match the known voxel counts; only the
#: product is currently confirmed, so the fold order may need adjustment once the
#: hardware numbering convention is known.
DETECTOR_BANK_SIZES: dict[str, dict[str, int]] = {
    # 128 cathode strips x 32 anode wires x 2 wire planes per segment, 60 segments.
    'magic_detector_a': {'segment': 60, 'strip': 128, 'wire': 32, 'counter': 2},
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
