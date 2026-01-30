# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DREAM logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import scipp as sc


def get_mantle_front_layer(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to extract mantle front layer."""
    from ess.dream.workflows import DETECTOR_BANK_SIZES

    return (
        da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES[source_name])
        .transpose(('wire', 'module', 'segment', 'counter', 'strip'))['wire', 0]
        .flatten(('module', 'segment', 'counter'), to='mod/seg/cntr')
    )


def get_wire_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to fold detector data for wire view.

    Folds raw detector data and flattens module/segment/counter dimensions.
    The subsequent summing over 'strip' is handled by the reduction_dim parameter
    in add_logical_view to preserve binned event structure for histogramming.

    Parameters
    ----------
    da:
        Raw detector data with a single dimension.
    source_name:
        Name of the detector bank.

    Returns
    -------
    :
        Folded data with dimensions (strip, wire, mod/seg/cntr).
    """
    from ess.dream.workflows import DETECTOR_BANK_SIZES

    return (
        da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES[source_name])
        # Transpose to make module/segment/counter contiguous for flattening.
        # After fold, dims are (wire, module, segment, strip, counter).
        .transpose(('strip', 'wire', 'module', 'segment', 'counter'))
        .flatten(('module', 'segment', 'counter'), to='mod/seg/cntr')
        # Result: (strip, wire, mod/seg/cntr)
        # After reduction over 'strip': (wire, mod/seg/cntr)
    )


def get_strip_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to fold detector data for strip view.

    Folds raw detector data into its logical structure and flattens all non-strip
    dimensions into 'other'. The subsequent summing over 'other' is handled by the
    reduction_dim parameter in add_logical_view to preserve binned event structure
    for histogramming.

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
    """
    from ess.dream.workflows import DETECTOR_BANK_SIZES

    folded = da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES[source_name])
    # Flatten all non-strip dims into 'other' for uniform reduction.
    # Move 'strip' to end first to make other dims contiguous for flattening.
    non_strip_dims = tuple(d for d in folded.dims if d != 'strip')
    if len(non_strip_dims) > 1:
        folded = folded.transpose((*non_strip_dims, 'strip'))
        folded = folded.flatten(non_strip_dims, to='other')
    return folded
