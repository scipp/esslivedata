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
    return da.transpose(('wire', 'module', 'segment', 'counter', 'strip'))[
        'wire', 0
    ].flatten(('module', 'segment', 'counter'), to='mod/seg/cntr')


def get_wire_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to fold detector data for wire view.

    Flattens module/segment/counter dimensions.
    The subsequent summing over 'strip' is handled by the reduction_dim parameter
    in add_logical_view to preserve binned event structure for histogramming.

    Parameters
    ----------
    da:
        Detector data with dimensions (wire, module, segment, strip, counter).
    source_name:
        Name of the detector bank.

    Returns
    -------
    :
        Data with dimensions (strip, wire, mod/seg/cntr).
    """
    return (
        da.transpose(('strip', 'wire', 'module', 'segment', 'counter')).flatten(
            ('module', 'segment', 'counter'), to='mod/seg/cntr'
        )
        # Result: (strip, wire, mod/seg/cntr)
        # After reduction over 'strip': (wire, mod/seg/cntr)
    )


def get_strip_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to fold detector data for strip view.

    Flattens all non-strip dimensions into 'other'. The subsequent summing over
    'other' is handled by the reduction_dim parameter in add_logical_view to
    preserve binned event structure for histogramming.

    Parameters
    ----------
    da:
        Detector data with dimensions (wire, module, segment, strip, counter).
    source_name:
        Name of the detector bank.

    Returns
    -------
    :
        Data with dimensions (other, strip).
    """
    # Flatten all non-strip dims into 'other' for uniform reduction.
    # Move 'strip' to end first to make other dims contiguous for flattening.
    non_strip_dims = tuple(d for d in da.dims if d != 'strip')
    if len(non_strip_dims) > 1:
        da = da.transpose((*non_strip_dims, 'strip'))
        da = da.flatten(non_strip_dims, to='other')
    return da
