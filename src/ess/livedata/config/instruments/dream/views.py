# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DREAM logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import scipp as sc

# Bank sizes for mantle detector logical views
_bank_sizes = {
    'mantle_detector': {
        'wire': 32,
        'module': 5,
        'segment': 6,
        'strip': 256,
        'counter': 2,
    },
}


def get_mantle_front_layer(da: sc.DataArray) -> sc.DataArray:
    """Transform to extract mantle front layer."""
    return (
        da.fold(dim=da.dim, sizes=_bank_sizes['mantle_detector'])
        .transpose(('wire', 'module', 'segment', 'counter', 'strip'))['wire', 0]
        .flatten(('module', 'segment', 'counter'), to='mod/seg/cntr')
    )


def get_wire_view(da: sc.DataArray) -> sc.DataArray:
    """Transform to extract wire view."""
    return (
        da.fold(dim=da.dim, sizes=_bank_sizes['mantle_detector'])
        .sum('strip')
        .flatten(('module', 'segment', 'counter'), to='mod/seg/cntr')
        # Transpose so that wire is the "x" dimension for more natural plotting.
        .transpose()
    )


def get_strip_view(da: sc.DataArray) -> sc.DataArray:
    """Transform to extract strip view (sum over all but strip)."""
    return da.fold(dim=da.dim, sizes=_bank_sizes['mantle_detector']).sum(
        ('wire', 'module', 'segment', 'counter')
    )
