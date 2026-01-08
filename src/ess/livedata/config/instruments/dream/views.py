# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DREAM logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import scipp as sc


def get_mantle_front_layer(da: sc.DataArray) -> sc.DataArray:
    """Transform to extract mantle front layer."""
    from ess.dream.workflows import DETECTOR_BANK_SIZES

    return (
        da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES['mantle_detector'])
        .transpose(('wire', 'module', 'segment', 'counter', 'strip'))['wire', 0]
        .flatten(('module', 'segment', 'counter'), to='mod/seg/cntr')
    )


def get_wire_view(da: sc.DataArray) -> sc.DataArray:
    """Transform to extract wire view."""
    from ess.dream.workflows import DETECTOR_BANK_SIZES

    return (
        da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES['mantle_detector'])
        .sum('strip')
        .flatten(('module', 'segment', 'counter'), to='mod/seg/cntr')
        # Transpose so that wire is the "x" dimension for more natural plotting.
        .transpose()
    )


def get_strip_view(da: sc.DataArray) -> sc.DataArray:
    """Transform to extract strip view (sum over all but strip)."""
    from ess.dream.workflows import DETECTOR_BANK_SIZES

    return da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES['mantle_detector']).sum(
        ('wire', 'module', 'segment', 'counter')
    )
